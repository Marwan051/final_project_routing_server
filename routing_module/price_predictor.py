from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from joblib import load

import os
import json

from sklearn.pipeline import Pipeline


class TripPricePredictor(BaseEstimator, RegressorMixin):
    def __init__(self, coef=None, intercept=None, model=None):
        """
        Initialize predictor with either:
        - coef and intercept (extracted from model)
        - model (sklearn model to extract params from)
        """
        if model is not None:
            # Extract coefficients and intercept from the loaded model
            self.coef_ = self._extract_coef(model)
            self.intercept_ = self._extract_intercept(model)
        elif coef is not None and intercept is not None:
            self.coef_ = np.array(coef) if not isinstance(coef, np.ndarray) else coef
            self.intercept_ = float(intercept)
        else:
            raise ValueError("Must provide either model or (coef and intercept)")

    def __setstate__(self, state):
        """Handle unpickling of old TripPricePredictor instances that had self.model"""
        self.__dict__.update(state)

        # If this is an old instance with 'model' attribute, extract parameters
        if hasattr(self, "model") and not hasattr(self, "coef_"):
            self.coef_ = self._extract_coef(self.model)
            self.intercept_ = self._extract_intercept(self.model)
            # Remove the model to save memory
            delattr(self, "model")

    def _extract_coef(self, model):
        """Extract coefficient from sklearn model (handles Ridge, LinearRegression, etc.)"""
        if hasattr(model, "coef_"):
            coef = model.coef_
            # Handle both 1D and 2D coefficient arrays
            if coef.ndim == 2:
                return coef[0]  # Take first row for single-output regression
            return coef
        else:
            raise ValueError(f"Model {type(model)} does not have coef_ attribute")

    def _extract_intercept(self, model):
        """Extract intercept from sklearn model"""
        if hasattr(model, "intercept_"):
            intercept = model.intercept_
            # Handle both scalar and array intercepts
            if isinstance(intercept, np.ndarray):
                return float(intercept[0]) if len(intercept) > 0 else 0.0
            return float(intercept)
        else:
            raise ValueError(f"Model {type(model)} does not have intercept_ attribute")

    def _round_bus_style(self, vals):
        """Custom rounding logic for bus fare"""
        scalar = np.isscalar(vals)
        arr = np.array([vals]) if scalar else np.asarray(vals)
        out = []
        for v in arr:
            pounds = int(np.floor(v))
            dec = v - pounds
            if dec < 0.125:
                r = pounds + 0.0
            elif dec < 0.375:
                r = pounds + 0.25
            elif dec < 0.75:
                r = pounds + 0.5
            else:
                r = pounds + 1.0
            out.append(round(r, 2))
        return out[0] if scalar else np.array(out)

    def predict(self, distance_km):
        # 1. Preprocessing: Convert KM to Log Distance
        X = np.array(distance_km)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_log = np.log1p(X)

        # 2. Manual prediction using extracted coefficients
        # y = coef * x + intercept
        raw_pred = (X_log * self.coef_).sum(axis=1) + self.intercept_

        # 3. Post-processing: Custom Rounding
        return self._round_bus_style(raw_pred)


def load_model(model_path=None):
    """Load the trip price prediction model"""
    if model_path is None:
        # Default to data/utils/trip_price_model.joblib relative to this file
        model_path = os.path.join(
            os.path.dirname(__file__), "data", "utils", "trip_price_model.joblib"
        )
    # Load the model - it might already be a TripPricePredictor or just the sklearn model
    loaded_model = load(model_path)

    # If it's already a TripPricePredictor, return it directly
    if isinstance(loaded_model, TripPricePredictor):
        return loaded_model

    # Otherwise, wrap it in TripPricePredictor
    return TripPricePredictor(loaded_model)


def _estimator_to_dict(estimator):
    """Convert an sklearn estimator (or pipeline) to a serializable dict.

    - Uses `get_params()` for general params.
    - Adds `coef_`, `intercept_`, and `feature_importances_` if present.
    - If the estimator is a Pipeline, extracts step info recursively.
    """
    if estimator is None:
        return None

    # If wrapped in TripPricePredictor, use its extracted coefficients
    if isinstance(estimator, TripPricePredictor):
        return {
            "class": "TripPricePredictor",
            "coef": (
                estimator.coef_.tolist()
                if hasattr(estimator.coef_, "tolist")
                else estimator.coef_
            ),
            "intercept": float(estimator.intercept_),
        }

    result = {}

    # Basic params
    try:
        params = estimator.get_params(deep=False)
        # Convert any non-serializable values to strings
        safe_params = {}
        for k, v in params.items():
            try:
                json.dumps({k: v})
                safe_params[k] = v
            except Exception:
                safe_params[k] = str(v)
        result["params"] = safe_params
    except Exception:
        result["params"] = str(
            getattr(estimator, "__class__", type(estimator)).__name__
        )

    # Coefficients / intercept
    if hasattr(estimator, "coef_"):
        try:
            coef = estimator.coef_
            result["coef"] = coef.tolist() if hasattr(coef, "tolist") else coef
        except Exception:
            result["coef"] = str(getattr(estimator, "coef_", None))

    if hasattr(estimator, "intercept_"):
        try:
            intercept = estimator.intercept_
            result["intercept"] = (
                intercept.tolist() if hasattr(intercept, "tolist") else intercept
            )
        except Exception:
            result["intercept"] = str(getattr(estimator, "intercept_", None))

    # Feature importances (tree-based)
    if hasattr(estimator, "feature_importances_"):
        try:
            fi = estimator.feature_importances_
            result["feature_importances"] = fi.tolist() if hasattr(fi, "tolist") else fi
        except Exception:
            result["feature_importances"] = str(
                getattr(estimator, "feature_importances_", None)
            )

    # Pipeline handling
    if isinstance(estimator, Pipeline):
        steps_info = []
        for name, step in estimator.steps:
            steps_info.append(
                {
                    "name": name,
                    "class": step.__class__.__name__,
                    "estimator": _estimator_to_dict(step),
                }
            )
        result["pipeline"] = steps_info

    return result


def extract_model_params(model_path=None, out_path=None):
    """Load the saved model and return a JSON-serializable dictionary of its parameters.

    If `out_path` is provided, the resulting dict will be written to that file.
    """
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "data", "utils", "trip_price_model.joblib"
        )

    loaded = load(model_path)

    # If the saved object is the wrapper, use it; otherwise wrap
    if isinstance(loaded, TripPricePredictor):
        estimator = loaded
    else:
        estimator = TripPricePredictor(loaded)

    info = {
        "class": estimator.__class__.__name__,
        "wrapped_model": _estimator_to_dict(estimator),
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(info, f, indent=2)

    return info
