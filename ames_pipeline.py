import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
STRUCTURAL_NONE_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType"
]
ZERO_WHEN_MISSING_COLS = ["GarageYrBlt", "MasVnrArea"]
ORDINAL_MAPS = {
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"
]


class AmesFeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.input_columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        for col in STRUCTURAL_NONE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        for col in ZERO_WHEN_MISSING_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if "Electrical" in df.columns and not df["Electrical"].mode(dropna=True).empty:
            df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode(dropna=True)[0])

        for col in QUALITY_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("None").map(QUALITY_MAP)

        for col, mapping in ORDINAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].fillna("None").map(mapping).astype(float)

        if {"YrSold", "YearBuilt"}.issubset(df.columns):
            df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        if {"YrSold", "YearRemodAdd", "YearBuilt"}.issubset(df.columns):
            df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
            df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
        if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
            df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

        porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
        if set(porch_cols).issubset(df.columns):
            df["TotalPorchSF"] = df[porch_cols].sum(axis=1)

        bath_cols = ["FullBath", "BsmtFullBath", "HalfBath", "BsmtHalfBath"]
        if set(bath_cols).issubset(df.columns):
            df["TotalBaths"] = (
                df["FullBath"] + df["BsmtFullBath"] + 0.5 * df["HalfBath"] + 0.5 * df["BsmtHalfBath"]
            )

        if "PoolArea" in df.columns:
            df["HasPool"] = (df["PoolArea"] > 0).astype(int)
        if "GarageArea" in df.columns:
            df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        if "Fireplaces" in df.columns:
            df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
        if "2ndFlrSF" in df.columns:
            df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
        if "WoodDeckSF" in df.columns:
            df["HasWoodDeck"] = (df["WoodDeckSF"] > 0).astype(int)

        if {"OverallQual", "GrLivArea"}.issubset(df.columns):
            df["QualArea"] = df["OverallQual"] * df["GrLivArea"]
        if "MoSold" in df.columns:
            season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
            df["SeasonSold"] = df["MoSold"].map(season_map)
        if {"OverallQual", "OverallCond"}.issubset(df.columns):
            df["OverallScore"] = df["OverallQual"] + df["OverallCond"]
            df["QualCondScore"] = df["OverallQual"] * df["OverallCond"]
        if {"GrLivArea", "TotRmsAbvGrd"}.issubset(df.columns):
            df["LiveAreaPerRoom"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1)
        if {"BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF"}.issubset(df.columns):
            denom = df["TotalBsmtSF"].replace(0, np.nan)
            df["FinishedBsmtRatio"] = ((df["BsmtFinSF1"] + df["BsmtFinSF2"]) / denom).fillna(0)

        outdoor_cols = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch"]
        if set(outdoor_cols).issubset(df.columns):
            df["TotalOutdoorSF"] = df[outdoor_cols].sum(axis=1)
        if {"YearBuilt", "YrSold"}.issubset(df.columns):
            df["IsNewHouse"] = (df["YearBuilt"] >= df["YrSold"] - 2).astype(int)
        if {"TotalSF", "GrLivArea"}.issubset(df.columns):
            denom = df["TotalSF"].replace(0, np.nan)
            df["AboveGroundRatio"] = (df["GrLivArea"] / denom).fillna(0)
        if {"GarageCars", "GarageArea"}.issubset(df.columns):
            df["GarageScore"] = df["GarageCars"] * df["GarageArea"]
        if {"TotalBaths", "BedroomAbvGr"}.issubset(df.columns):
            df["BathsPerBedroom"] = df["TotalBaths"] / (df["BedroomAbvGr"] + 1)

        return df


class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=20.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        self.global_mean_ = float(np.mean(y))
        self.maps_ = {}
        y_series = pd.Series(y, index=X.index)

        for col in self.columns_:
            stats = pd.DataFrame({"x": X[col], "y": y_series}).groupby("x")["y"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (stats["count"] + self.smoothing)
            self.maps_[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        for col in self.columns_:
            out[f"{col}__te"] = X[col].map(self.maps_[col]).fillna(self.global_mean_)
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "columns_", [])
        return np.array([f"{col}__te" for col in input_features], dtype=object)


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.cols_ = [c for c in self.cols if c in pd.DataFrame(X).columns]
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col in self.cols_:
            df[col] = np.log1p(df[col].clip(lower=0))
        return df