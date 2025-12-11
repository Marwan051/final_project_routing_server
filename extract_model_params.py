"""CLI helper to extract and save model parameters.

Usage:
    python extract_model_params.py [--model-path PATH] [--out PATH]

Defaults:
    model_path: routing_module/data/utils/trip_price_model.joblib
    out: model_params.json
"""
import argparse
import os
import json

from routing_module.routing import extract_model_params, TripPricePredictor
import __main__

# Ensure TripPricePredictor is available in __main__ so joblib can unpickle
__main__.TripPricePredictor = TripPricePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Path to joblib model file', default=None)
    parser.add_argument('--out', help='Output JSON path', default='model_params.json')
    args = parser.parse_args()

    info = extract_model_params(model_path=args.model_path, out_path=args.out)
    print(f"Wrote model parameters to {args.out}")


if __name__ == '__main__':
    main()
