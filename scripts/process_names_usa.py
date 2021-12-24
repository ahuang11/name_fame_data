import datetime
import os
import sqlite3
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

SPECS = {
    "url": "https://www.ssa.gov/oact/babynames/state/namesbystate.zip",
    "columns": ["state", "gender", "year", "name", "count"],
}


def process_web_zip(url, columns):
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
    concat_df = pd.concat(df_list)  # combine

    # create data dir if not exists
    print("creating data dir")
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(scripts_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # create a database with just the names + stats
    print("computing stats")
    stat_list = []
    stat_list.append(
        df.groupby("name")["gender"].agg(pd.Series.mode).rename("top_gender")
    )
    stat_list.append(df.groupby("name")["count"].sum().rename("total_count"))
    for offset in [3, 5, 10, 25, 50, 100]:
        stat_list.append(
            df.loc[df["year"] >= datetime.datetime.utcnow().year - offset]
            .groupby("name")["count"]
            .max()
            .rename(f"max_count_{offset}_years")
        )

    stat_db = os.path.join(data_dir, "names_stats.db")
    print(f"exporting {stat_db}")
    if os.path.exists(stat_db):
        print(f"removing {stat_db}")
        os.remove(stat_db)
    stat_conn = sqlite3.connect(stat_db)
    stat_df = pd.concat(stat_list, axis=1)
    stat_df.to_sql("names", stat_conn)
    stat_conn.execute("CREATE INDEX name ON names(name)")

    # shard by starting letter
    for char in concat_df["name"].str[0].unique():
        char_db = os.path.join(data_dir, f"{char}_names_recs.db".lower())
        if os.path.exists(stat_db):
            print(f"removing {char_db}")
            os.remove(char_db)
        print(f"exporting {char_db}")
        char_conn = sqlite3.connect(char_db)
        char_df = concat_df.loc[concat_df["name"].str.startswith(char)]
        char_df.set_index("name").to_sql("names", char_conn)
        char_conn.execute("CREATE INDEX name ON names(name)")

    print("job complete!")


if __name__ == "__main__":
    process_web_zip(**SPECS)
