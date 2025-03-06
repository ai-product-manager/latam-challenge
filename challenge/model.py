import os
import pickle
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import yaml

# Import helper functions from an external utils module
from challenge.utils import compute_vectorized_features

# Configure logging: logs will be written to "app.log"
logging.basicConfig(level=logging.INFO, filename="challenge/app.log", filemode="a")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "challenge/default.yml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration parameters loaded from the file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logger.info("Configuration loaded from %s", config_path)
            return config
    except Exception as e:
        logger.error("Could not load config file (%s). Error: %s", config_path, e)
        raise FileNotFoundError(f"Configuration file {config_path} not found.") from e


class DelayModel:
    """
    DelayModel encapsulates the machine learning model used to predict flight delays.

    It provides methods for data preprocessing, model training, and prediction.

    Derived columns:
      - 'min_diff': Time difference (in minutes) between scheduled ('Fecha-I')
                    and actual flight time ('Fecha-O').
      - 'delay': Binary target; 1 if min_diff > threshold_in_minutes, else 0.
      - 'period_day': Period of the day (morning, afternoon, night) based on 'Fecha-I'.
      - 'high_season': 1 if 'Fecha-I' falls in high season, else 0.
      - 'MES': Extracted month from 'Fecha-I'.

    Categorical features ("OPERA", "TIPOVUELO", "MES") are one-hot encoded to produce final features.

    The model is trained using XGBoost with adjustments for class imbalance and is persisted to disk.
    """

    def __init__(self, config: Dict[str, Any] = None):
        # Load external configuration; if not provided, try to load from YAML.
        self.config = config if config is not None else load_config()
        # Raise error if no configuration is loaded.
        if not self.config:
            raise ValueError("Configuration could not be loaded.")
        # Use external configuration exclusively.
        self._test_set_rate: float = self.config["test_set_rate"]
        self._model_path: str = self.config["model_path"]
        self._model_version: str = self.config["model_version"]
        self._threshold_in_minutes: float = self.config["threshold_in_minutes"]
        self.categorical_features: List[str] = self.config["categorical_features"]
        self.expected_features: List[str] = self.config["expected_features"]
        self._default_model_params: Dict[str, Any] = self.config["default_model_params"]
        self._model: Union[xgb.XGBClassifier, None] = None

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Expects data with at least the columns: "OPERA", "TIPOVUELO", and "MES".
        If date columns ("Fecha-I", "Fecha-O") are present, derived features are computed.
        Otherwise, only one-hot encoding on the required columns is performed.

        Args:
            data (pd.DataFrame): Raw flight data.
            target_column (str, optional): If provided, returns a tuple (features, target); 
                                        otherwise, returns only the features DataFrame.

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]: The processed features (and target if specified).

        Raises:
            ValueError: If required columns are missing.
        """
        try:
            df = data.copy()
            # Check for required categorical fields
            required_columns = self.categorical_features
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(missing)}")

            # If date columns exist, compute derived features; otherwise, skip them.
            if "Fecha-I" in df.columns and "Fecha-O" in df.columns:
                df[["Fecha-I", "Fecha-O"]] = df[["Fecha-I", "Fecha-O"]].apply(pd.to_datetime, errors="coerce")
                df = df.dropna(subset=["Fecha-I", "Fecha-O"])
                df = compute_vectorized_features(self._threshold_in_minutes, df)
                # Also update "MES" if needed from "Fecha-I"
                df["MES"] = df["Fecha-I"].dt.month

            # One-hot encode the required categorical features
            dummies = pd.get_dummies(df[self.categorical_features], prefix_sep="_")
            # Ensure the final features contain exactly the expected columns
            features_df = dummies.reindex(columns=self.expected_features, fill_value=0)
            logger.info("Preprocessing complete: features generated.")
            
            if target_column and target_column in df.columns:
                target_df = df[[target_column]]
                logger.info("Preprocessing complete: target generated.")
                return features_df, target_df
            
            return features_df
        except Exception as e:
            logger.exception("Error in preprocessing: %s", e)
            raise

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): Preprocessed features.
            target (pd.DataFrame): Target values.
        """
        try:
            x_train, _, y_train, _ = self.__split_data(features, target)
            y_train_arr = y_train.values.ravel()

            # Count the number of negative (class 0) and positive (class 1) samples in the training set
            n_y1 = np.sum(y_train_arr == 1)
            n_y0 = np.sum(y_train_arr == 0)

            # Multiply ratio by a factor to further emphasize minority class
            scale_weight = (n_y0 / n_y1) * 1.3 if n_y1 > 0 else 1

            # Merge the computed scale weight into the hyperparameters
            params = {**self._default_model_params, "scale_pos_weight": scale_weight}
            self._model = xgb.XGBClassifier(**params)
            self._model.fit(x_train, y_train_arr)
            logger.info("Model training complete.")
            self.__save(params)
        except Exception as e:
            logger.exception("Error in fit: %s", e)
            raise

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new data.

        If the model is not in memory, attempts to load it from disk.
        Uses the default prediction behavior of XGBClassifier.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        try:
            if self._model is None:
                logger.info("Model not in memory; attempting to load from disk.")
                self.__load_model()
            if self._model is None:
                logger.warning(
                    "Model could not be loaded; returning default predictions (all 0)."
                )
                return [0] * features.shape[0]
            predictions = self._model.predict(features)
            logger.info("Prediction complete.")
            return predictions.tolist()
        except Exception as e:
            logger.exception("Error in predict: %s", e)
            raise

    def __split_data(
        self, features: pd.DataFrame, target: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and test sets.

        Returns:
            Tuple with (x_train, x_test, y_train, y_test)
        """
        try:
            return train_test_split(
                features, target, test_size=self._test_set_rate, random_state=42
            )
        except Exception as e:
            logger.exception("Error during data split: %s", e)
            raise

    def __save(self, configs: Dict[str, Any]) -> None:
        """
        Save the trained model and its configuration to disk.
        """
        try:
            model_dir = os.path.join(self._model_path, self._model_version)
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
                pickle.dump(self._model, f)
            with open(os.path.join(model_dir, "config.pkl"), "wb") as f:
                pickle.dump(configs, f)
            logger.info("Model and configuration saved to disk.")
        except Exception as e:
            logger.exception("Error saving model: %s", e)
            raise

    def __load_model(self) -> None:
        """
        Load the model and its configuration from disk.
        """
        try:
            model_dir = os.path.join(self._model_path, self._model_version)
            with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
                self._model = pickle.load(f)
            with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
                self._model_config = pickle.load(
                    f
                )  # pylint: disable=attribute-defined-outside-init
            logger.info("Model and configuration loaded from disk.")
        except Exception as e:
            logger.exception("Error loading model: %s", e)
            self._model = None
