# SPDX-License-Identifier: GPL-3.0-only
# -*- coding: utf-8 -*-
"""Reproduce the final RMSE contrasts used in Tables 8 and 11."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


from rapeseed_ensemble.paths import input_path, output_path
ROOT = output_path('bootstrap')
CACHE_PATH = input_path('CACHE', 'predictions_cache.pkl')
TABLE8_CSV = ROOT / 'ensemble_rmse_comparisons.csv'
TABLE11_CSV = ROOT / 'base_model_rmse_comparisons.csv'
METADATA_JSON = ROOT / 'bootstrap_parameters.json'

SCALE = 15000.0
BOOTSTRAPS = 2000
SEED = 42
POWER = 3


def rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)) * SCALE)


def r2(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.sum((actual - actual.mean()) ** 2)
    return 1.0 - float(np.sum((actual - predicted) ** 2) / denominator)


def percentile_interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def main():
    with CACHE_PATH.open("rb") as handle:
        cache = pickle.load(handle)

    results = cache["results"]
    model_names = ["TCN", "BiLSTM", "RandomForest"]
    display_names = {"TCN": "TCN", "BiLSTM": "BiLSTM", "RandomForest": "RF"}
    reference = results[model_names[0]]
    actual = np.asarray(reference["acts"], dtype=float)
    drought = np.asarray(reference["drought"], dtype=bool)
    freeze = np.asarray(reference["freeze"], dtype=bool)
    waterlogging = np.asarray(reference["waterlog"], dtype=bool)
    predictions = {
        model: np.asarray(results[model]["preds"], dtype=float)
        for model in model_names
    }
    prediction_matrix = np.column_stack([predictions[model] for model in model_names])

    for model in model_names:
        if not np.allclose(results[model]["acts"], actual):
            raise RuntimeError(f"Observed values differ for {model}")

    regimes = [
        ("No hazard", ~drought & ~freeze & ~waterlogging),
        ("Freeze only", freeze & ~drought & ~waterlogging),
        ("Waterlogging only", waterlogging & ~drought & ~freeze),
        ("Drought + freeze", drought & freeze & ~waterlogging),
        ("Freeze + waterlogging", freeze & waterlogging & ~drought),
        ("Drought + waterlogging", drought & waterlogging & ~freeze),
        ("All three", drought & freeze & waterlogging),
    ]

    overall_mask = np.ones(len(actual), dtype=bool)
    overall_scores = np.asarray(
        [max(r2(actual, predictions[model]), 0.0) for model in model_names]
    )
    global_weights = overall_scores**POWER
    global_weights /= global_weights.sum()

    equal_prediction = prediction_matrix.mean(axis=1)
    global_prediction = prediction_matrix @ global_weights
    stratified_prediction = np.empty_like(actual)
    regime_weights = {}
    classified = np.zeros(len(actual), dtype=bool)
    for label, mask in regimes:
        scores = np.asarray(
            [
                max(r2(actual[mask], predictions[model][mask]), 0.0)
                for model in model_names
            ]
        )
        weights = scores**POWER
        weights = (
            weights / weights.sum()
            if weights.sum()
            else np.repeat(1.0 / len(model_names), len(model_names))
        )
        stratified_prediction[mask] = prediction_matrix[mask] @ weights
        regime_weights[label] = dict(zip(model_names, weights.tolist()))
        classified |= mask

    drought_only = drought & ~freeze & ~waterlogging
    stratified_prediction[drought_only] = global_prediction[drought_only]
    classified |= drought_only
    if not classified.all():
        stratified_prediction[~classified] = global_prediction[~classified]
    if not np.allclose(stratified_prediction, cache["ens"]["preds"], rtol=0, atol=1e-12):
        maximum = np.max(np.abs(stratified_prediction - cache["ens"]["preds"]))
        raise RuntimeError(
            f"Reconstructed hazard-stratified prediction differs from cache: {maximum}"
        )

    rng = np.random.RandomState(SEED)
    table8_rows = []
    scopes = [("Overall", overall_mask)] + regimes
    for label, mask in scopes:
        indices = np.flatnonzero(mask)
        equal_value = rmse(actual[indices], equal_prediction[indices])
        global_value = rmse(actual[indices], global_prediction[indices])
        stratified_value = rmse(actual[indices], stratified_prediction[indices])
        equal_bootstrap = np.empty(BOOTSTRAPS)
        global_bootstrap = np.empty(BOOTSTRAPS)
        for iteration in range(BOOTSTRAPS):
            sample = indices[rng.randint(0, len(indices), len(indices))]
            stratified_rmse = rmse(
                actual[sample], stratified_prediction[sample]
            )
            equal_bootstrap[iteration] = (
                rmse(actual[sample], equal_prediction[sample]) - stratified_rmse
            )
            global_bootstrap[iteration] = (
                rmse(actual[sample], global_prediction[sample]) - stratified_rmse
            )
        equal_low, equal_high = percentile_interval(equal_bootstrap)
        global_low, global_high = percentile_interval(global_bootstrap)
        table8_rows.append(
            {
                "Scope": label,
                "n": len(indices),
                "Equal_RMSE": equal_value,
                "Global_RMSE": global_value,
                "Stratified_RMSE": stratified_value,
                "Delta_equal_minus_stratified": equal_value - stratified_value,
                "Equal_CI_low": equal_low,
                "Equal_CI_high": equal_high,
                "Delta_global_minus_stratified": global_value - stratified_value,
                "Global_CI_low": global_low,
                "Global_CI_high": global_high,
            }
        )

    table11_rows = []
    for label, mask in regimes:
        indices = np.flatnonzero(mask)
        bootstrap_samples = [
            indices[rng.randint(0, len(indices), len(indices))]
            for _ in range(BOOTSTRAPS)
        ]
        row = {"Regime": label, "n": len(indices)}
        for model in model_names:
            base_value = rmse(actual[indices], predictions[model][indices])
            ensemble_value = rmse(
                actual[indices], stratified_prediction[indices]
            )
            bootstrap_values = np.asarray(
                [
                    rmse(actual[sample], predictions[model][sample])
                    - rmse(actual[sample], stratified_prediction[sample])
                    for sample in bootstrap_samples
                ]
            )
            low, high = percentile_interval(bootstrap_values)
            prefix = display_names[model]
            row[f"Delta_{prefix}_minus_ensemble"] = base_value - ensemble_value
            row[f"{prefix}_CI_low"] = low
            row[f"{prefix}_CI_high"] = high
        table11_rows.append(row)

    table8 = pd.DataFrame(table8_rows)
    table11 = pd.DataFrame(table11_rows)
    table8.to_csv(TABLE8_CSV, index=False, encoding="utf-8-sig")
    table11.to_csv(TABLE11_CSV, index=False, encoding="utf-8-sig")
    METADATA_JSON.write_text(
        json.dumps(
            {
                "cache": CACHE_PATH.name,
                "bootstrap_replicates": BOOTSTRAPS,
                "bootstrap_seed": SEED,
                "weight_power": POWER,
                "model_order": model_names,
                "global_weights": dict(zip(model_names, global_weights.tolist())),
                "regime_weights": regime_weights,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("TABLE 8")
    print(table8.to_string(index=False))
    print("\nTABLE 11")
    print(table11.to_string(index=False))


if __name__ == "__main__":
    main()
