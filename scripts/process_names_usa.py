import datetime
import os
import sqlite3
from io import BytesIO
from urllib.request import urlopen
from collections import defaultdict
from zipfile import ZipFile

import pandas as pd
import numpy as np


class BabyNames:
    def __init__(self):
        self._df = self._retrieve_df()

        self._this_year = datetime.datetime.utcnow().year
        self._latest_year = self._df["year"].max()
        self._previous_year = self._latest_year - 1  # 1 years ago
        self._recent_year = self._latest_year - 3  # 4 years ago

        self._name_groupby = self._df.groupby("name")

        self._yearly_df = (
            self._get_count(self._df.groupby(["name", "year"]))
            .pipe(lambda df: df.reindex(pd.MultiIndex.from_product(df.index.levels)))
            .reset_index()
        ).fillna(0)
        self._latest_df = self._yearly_df.loc[
            self._yearly_df["year"] == self._latest_year
        ]
        self._previous_df = self._yearly_df.loc[
            self._yearly_df["year"] >= self._previous_year
        ]
        self._recent_df = self._yearly_df.loc[
            self._yearly_df["year"] >= self._recent_year
        ]

    @staticmethod
    def _retrieve_df():
        url = "https://www.ssa.gov/oact/babynames/state/namesbystate.zip"
        columns = ["state", "gender", "year", "name", "count"]
        print(f"opening {url}")
        with urlopen(url) as in_file:  # retrieve zip
            with BytesIO(in_file.read()) as buf:  # write to memory
                with ZipFile(buf) as zip_file:  # extract zip
                    df_list = []
                    for txt_file in sorted(zip_file.namelist()):  # loop contents
                        if not txt_file.lower().endswith(".txt"):  # skip not .txt
                            continue
                        with zip_file.open(txt_file) as txt_data:
                            print(f"reading {txt_file}")
                            df = pd.read_csv(txt_data, header=None, names=columns)
                            df_list.append(df)  # append to list
        df = pd.concat(df_list)  # combine

        return df.sort_values("year")

    @staticmethod
    def _get_count(name_groupby):
        return name_groupby["count"].sum()

    @staticmethod
    def _get_gender(df, gender):
        return df.loc[df["gender"] == gender]

    @staticmethod
    def _flatten_list(series):
        return series.astype(str).str.strip("[").str.strip("]").str.replace("'", "")

    @staticmethod
    def _get_mode(name_groupby, col):
        return name_groupby[col].agg(pd.Series.mode)

    @staticmethod
    def _get_rank(series):
        return series.rank(method="dense", ascending=False)

    @property
    def overall_count(self):
        return self._get_count(self._name_groupby)

    @property
    def overall_male_rank(self):
        return self._get_rank(
            self._get_count(self._get_gender(self._df, "M").groupby("name"))
        )

    @property
    def overall_female_rank(self):
        return self._get_rank(
            self._get_count(self._get_gender(self._df, "M").groupby("name"))
        )

    @property
    def latest_count(self):
        return self._get_count(self._latest_df.groupby("name"))

    @property
    def latest_male_rank(self):
        return self._get_rank(
            self._get_count(
                self._get_gender(
                    self._df.loc[self._df["year"] == self._latest_year], "M"
                ).groupby("name")
            )
        )

    @property
    def latest_female_rank(self):
        return self._get_rank(
            self._get_count(
                self._get_gender(
                    self._df.loc[self._df["year"] == self._latest_year], "F"
                ).groupby("name")
            )
        )

    @property
    def latest_change(self):
        return (
            self._previous_df.groupby("name")["count"]
            .diff()
            .to_frame()
            .set_index(self._previous_df["name"])["count"]
            .replace(np.inf, np.nan)
            .dropna()
        )

    @property
    def latest_percent_change(self):
        return (
            self._previous_df.groupby("name")["count"]
            .pct_change()
            .to_frame()
            .set_index(self._previous_df["name"])["count"]
            .replace(np.inf, np.nan)
            .dropna()
        ) * 100

    @property
    def recent_trend(self):
        return (
            self._recent_df.groupby("name")["count"]
            .rolling(3)
            .mean()
            .reset_index(level=1, drop=True)
            .groupby("name")
            .pct_change()
            .replace(np.inf, np.nan)
            .dropna()
        ) * 100

    @property
    def percent_male(self):
        return (
            self._get_count(self._get_gender(self._df, "M")) / self.overall_count
        ) * 100

    @property
    def percent_female(self):
        return 100 - self.percent_male.fillna(0).astype(int)

    @property
    def top_gender(self):
        return self._flatten_list(self._get_mode(self._name_groupby, "gender"))

    @property
    def top_state(self):
        return self._flatten_list(self._get_mode(self._name_groupby, "state"))

    @property
    def median_age(self):
        return self._this_year - self._name_groupby["year"].median()

    @property
    def average_year(self):
        return self._name_groupby["year"].mean()

    @property
    def average_age(self):
        return self._this_year - self.average_year

    @property
    def first_appearance(self):
        return self._name_groupby["year"].min()

    @property
    def final_appearance(self):
        return self._name_groupby["year"].max()

    @property
    def dataframe(self):
        print("aggregating stats")
        df = pd.concat(
            [
                getattr(self, prop).rename(prop)
                for prop in dir(self)
                if not prop.startswith("_") and prop not in ["dataframe", "export"]
            ],
            axis=1,
        )
        df = df.fillna(0)
        number_cols = list(df.select_dtypes(np.number).columns)
        number_cols.remove("overall_count")  # too big of a number
        number_cols.remove("latest_count")  # too big of a number
        df[number_cols] = df[number_cols].astype(np.int16)
        df[["overall_count", "latest_count"]] = df[
            ["latest_count", "overall_count"]
        ].astype(int)
        return df

    def export(self):
        # create data dir if not exists
        print("creating data dir")
        scripts_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(scripts_dir, "..", "data")
        os.makedirs(data_dir, exist_ok=True)

        stats_db = os.path.join(data_dir, "names_stats.db")
        if os.path.exists(stats_db):
            print(f"removing {stats_db}")
            os.remove(stats_db)

        with sqlite3.connect(stats_db) as con:
            print(f"exporting {stats_db}")
            self.dataframe.to_sql("names", con)

        # shard by starting letter
        for char in self._df["name"].str[0].unique():
            char_db = os.path.join(data_dir, f"{char}_names_recs.db".lower())
            if os.path.exists(char_db):
                print(f"removing {char_db}")
                os.remove(char_db)
            print(f"exporting {char_db}")
            char_conn = sqlite3.connect(char_db)
            char_df = self._df.loc[self._df["name"].str.startswith(char)]
            char_df.set_index("name").to_sql("names", char_conn)
            char_conn.execute("CREATE INDEX name ON names(name)")


if __name__ == "__main__":
    baby_names = BabyNames()
    baby_names.export()
