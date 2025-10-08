import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cat_features = []
        self.num_features = []

    def set_features(self):
        self.num_features = self.df.select_dtypes(include='number').columns.tolist()
        self.cat_features = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def fill_missing(self):
        for col in self.num_features:
            if self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
        for col in self.cat_features:
            if self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    def new_features(self):
        new_features = pd.DataFrame({
            "life_full_sq": self.df["life_sq"] / (self.df["full_sq"] + 1),
            "kitchen_full_sq": self.df["kitch_sq"] / (self.df["full_sq"] + 1),
            "floor_ratio": self.df["floor"] / (self.df["max_floor"] + 1),
            "rooms_full_sq": self.df["num_room"] / (self.df["full_sq"] + 1),
            "floor_from_top": self.df["max_floor"] - self.df["floor"],
            "build_age": 2025 - self.df["build_year"],

            # Плотность и эффективность
            "sq_per_room": self.df["full_sq"] / (self.df["num_room"] + 1),
            "life_sq_per_room": self.df["life_sq"] / (self.df["num_room"] + 1),
            "kitch_sq_per_room": self.df["kitch_sq"] / (self.df["num_room"] + 1),
            "room_density": self.df["num_room"] / (self.df["full_sq"] + 1),
            "space_utilization": (self.df["life_sq"] + self.df["kitch_sq"]) / (self.df["full_sq"] + 1),
            "efficiency_gap": (self.df["full_sq"] - self.df["life_sq"]) / (self.df["full_sq"] + 1),

            # Отношения расстояний
            "center_to_mkad_ratio": self.df["kremlin_km"] / (self.df["mkad_km"] + 1),
            "ttk_to_mkad_ratio": self.df["ttk_km"] / (self.df["mkad_km"] + 1),
            "sadovoe_to_mkad_ratio": self.df["sadovoe_km"] / (self.df["mkad_km"] + 1),

            # Инфраструктура
            "education_density": (
                                         self.df["school_education_centers_raion"] + self.df[
                                     "preschool_education_centers_raion"]
                                 ) / (self.df["raion_popul"] + 1),
            "health_density": self.df["healthcare_centers_raion"] / (self.df["raion_popul"] + 1),
            "sport_density": self.df["sport_objects_raion"] / (self.df["raion_popul"] + 1),

            # --- Экология ---
            "green_to_industrial_ratio": self.df["green_zone_part"] / (self.df["indust_part"] + 0.001),
            "ecology_encoded": self.df["ecology"].astype("category").cat.codes,

            # --- Возраст и этаж ---
            "age_floor_interaction": (2025 - self.df["build_year"]) * self.df["floor"],
            "height_quality": self.df["max_floor"] * (2025 - self.df["build_year"]),
            # десятилетие постройки как категория
            "build_decade": ((self.df["build_year"] // 10) * 10).fillna(0).astype(int),
        })

        # --- Полиномиальные признаки и отобранные взаимодействия ---
        poly_features = pd.DataFrame({
            # оставим только минимальный набор полиномов
            "full_sq_squared": self.df["full_sq"] ** 2,
            "build_age_squared": (2025 - self.df["build_year"]) ** 2,
        })


        self.all_new_features = pd.concat([new_features, poly_features], axis=1)

        self.df = pd.concat([self.df, self.all_new_features], axis=1)

        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.fillna(0)


    def corr_drop(self):
        df_corr = self.df[self.num_features].corr()
        corrs = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
        col_for_drop = [col for col in corrs.columns if any(corrs[col] > 0.75)]
        self.df = self.df.drop(col_for_drop, axis=1, errors="ignore")

    def VarThreshold(self):
        current_num = self.df.select_dtypes(include='number').columns.tolist()
        if not current_num:
            return
        cutter = VarianceThreshold(threshold=0.1)
        cutter.fit(self.df[current_num])
        support_mask = cutter.get_support()
        kept_cols = [col for col, keep in zip(current_num, support_mask) if keep]
        drop_cols = [col for col in current_num if col not in kept_cols]
        if drop_cols:
            self.df = self.df.drop(columns=drop_cols, errors="ignore")

    def One_hot_encoder(self):
        ohe_list = []
        to_drop = []

        for col in self.cat_features:
            if self.df[col].nunique() <= 5:
                ohe = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                ohe_list.append(ohe)
                to_drop.append(col)
            else:
                mte = self.df.groupby(col)['price_doc'].transform('mean')
                self.df[col] = mte  # transform быстрее, чем map

        # Убираем исходные категориальные колонки одним разом
        self.df = self.df.drop(columns=to_drop)

        # Добавляем все OHE-таблицы одним concat
        if ohe_list:
            self.df = pd.concat([self.df] + ohe_list, axis=1)

    def quantile(self):
        top_quantile = self.df['price_doc'].quantile(0.92)
        low_quantile = self.df['price_doc'].quantile(0.08)
        self.df = self.df[(self.df['price_doc'] > low_quantile) & (self.df['price_doc'] < top_quantile)]

    def run(self):
        self.set_features()
        self.fill_missing()
        self.new_features()
        self.corr_drop()
        self.VarThreshold()
        self.One_hot_encoder()
        self.quantile()
        return self.df

