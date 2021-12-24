import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

FILE_SPECS = {
    "names_usa.csv": {
        "url": "https://www.ssa.gov/oact/babynames/names.zip",
        "columns": ["name", "gender", "count"],
        "ordering": ["name", "gender", "year", "count"],
    },
    "names_usa_states.csv": {
        "url": "https://www.ssa.gov/oact/babynames/state/namesbystate.zip",
        "columns": ["state", "gender", "year", "name", "count"],
        "ordering": ["state", "name", "gender", "year", "count"],
    },
}


def process_web_zip(out_file, url, columns, ordering):
    df_list = []
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(scripts_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, out_file)
    with urlopen(url) as in_file:  # retrieve zip
        with BytesIO(in_file.read()) as buf:  # write to memory
            with ZipFile(buf) as zip_file:  # extract zip
                df_list = []
                for txt_file in sorted(zip_file.namelist()):  # loop contents
                    if not txt_file.lower().endswith(".txt"):  # skip not .txt
                        continue
                    with zip_file.open(txt_file) as txt_data:
                        df = pd.read_csv(txt_data, header=None, names=columns)
                        if "year" not in columns:
                            df["year"] = int(txt_file[3:-4])  # add year col
                        df_list.append(df[ordering])  # reorder and add
            pd.concat(df_list).to_csv(out_path, index=None)
    return out_path


if __name__ == "__main__":
    for file, specs in FILE_SPECS.items():
        path = process_web_zip(file, **specs)
        print(f"{path} written!")
