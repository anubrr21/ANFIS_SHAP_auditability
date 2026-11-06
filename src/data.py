from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = ["pima", "early", "cdc"]
TITLES = {
    "pima": "Pima Indians Diabetes",
    "early": "Early Stage Diabetes Risk",
    "cdc": "CDC Diabetes Health Indicators",
}

PIMA_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
             "BMI", "DiabetesPedigree", "Age", "Outcome"]
PIMA_ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_raw(name):
    if name == "pima":
        df = pd.read_csv(RAW / "pima_indians_diabetes.csv", header=None, names=PIMA_COLS)
        X = df.drop(columns="Outcome").astype(float)
        X[PIMA_ZERO_IS_MISSING] = X[PIMA_ZERO_IS_MISSING].replace(0, np.nan)
        y = df["Outcome"]
    elif name == "early":
        df = pd.read_csv(RAW / "early_stage_diabetes.csv")
        df.columns = [c.strip().title().replace(" ", "") for c in df.columns]
        y = (df["Class"] == "Positive").astype(int)
        X = df.drop(columns="Class")
        X["Gender"] = (X["Gender"] == "Male").astype(int)
        for c in X.columns.drop(["Age", "Gender"]):
            X[c] = (X[c] == "Yes").astype(int)
        X = X.rename(columns={"Gender": "Male"}).astype(float)
    elif name == "cdc":
        df = pd.read_csv(RAW / "cdc_diabetes_health_indicators.csv")
        y = df["Diabetes_binary"].astype(int)
        X = df.drop(columns=["ID", "Diabetes_binary"]).astype(float)
    else:
        raise ValueError(f"unknown dataset {name}")
    return X, y.rename("target")


@dataclass
class Split:
    name: str
    features: list
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def to_original(self, Xs):
        return Xs * self.std + self.mean

    def select(self, idx):
        idx = list(idx)
        return Split(self.name, [self.features[i] for i in idx],
                     self.X_train[:, idx], self.y_train, self.X_val[:, idx], self.y_val,
                     self.X_test[:, idx], self.y_test,
                     self.median[idx], self.mean[idx], self.std[idx])


def make_split(name, seed):
    X, y = load_raw(name)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp,
                                              random_state=seed)
    median = X_tr.median()
    X_tr, X_va, X_te = X_tr.fillna(median), X_va.fillna(median), X_te.fillna(median)
    mean, std = X_tr.mean(), X_tr.std(ddof=0).replace(0, 1.0)
    scale = lambda d: ((d - mean) / std).to_numpy()
    return Split(name, list(X.columns),
                 scale(X_tr), y_tr.to_numpy(), scale(X_va), y_va.to_numpy(),
                 scale(X_te), y_te.to_numpy(),
                 median.to_numpy(), mean.to_numpy(), std.to_numpy())


def describe(name):
    X, y = load_raw(name)
    return {
        "dataset": TITLES[name],
        "records": len(X),
        "attributes": X.shape[1],
        "binary_attributes": int((X.nunique() <= 2).sum()),
        "missing_cells": int(X.isna().sum().sum()),
        "positive_rate": round(float(y.mean()), 4),
    }
