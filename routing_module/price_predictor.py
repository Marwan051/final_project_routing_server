from joblib import load
import warnings
import math
import os

warnings.filterwarnings("ignore")


class TripPricePredictor:
    def __init__(self, model_path=None, params_path=None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "data", "utils", "model.pkl"
            )
        self.model = load(model_path)
        self.intercept = self.model.intercept_
        self.beta_distance, self.beta_passengers = self.model.coef_

    def predict(self, distance, passengers):
        return math.ceil(
            self.intercept
            + self.beta_distance * distance
            + self.beta_passengers * passengers
        )
