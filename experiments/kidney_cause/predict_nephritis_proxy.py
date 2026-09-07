# -*- coding: utf-8 -*-
"""Apply the packaged research proxy to a CSV with the exact locked feature schema."""
import argparse
import os
import sys

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nephritis_proxy import predict_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--model", default=os.path.join(HERE, "models", "immune_kidney_proxy.joblib"))
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.input_csv)
    predict_bundle(bundle, frame).to_csv(args.output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
