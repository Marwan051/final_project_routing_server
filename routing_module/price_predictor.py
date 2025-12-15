import numpy as np
import os
import json


class TripPricePredictor:
    def __init__(self, coef=None, intercept=None, params_path=None):
        if coef is None or intercept is None:
            if params_path is None:
                params_path = os.path.join(
                    os.path.dirname(__file__), "data", "utils", "model_params.json"
                )
            with open(params_path, "r") as f:
                params = json.load(f)
            coef = params["coef"]
            intercept = params["intercept"]

        self.coef_ = np.array(coef) if not isinstance(coef, np.ndarray) else coef
        self.intercept_ = float(intercept)

    def _round_bus_style(self, vals):
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
        X = np.array(distance_km)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_log = np.log1p(X)

        raw_pred = (X_log * self.coef_).sum(axis=1) + self.intercept_

        return self._round_bus_style(raw_pred)
