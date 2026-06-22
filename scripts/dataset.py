import os
import requests
import glob
import shutil

from tqdm import tqdm
from zipfile import ZipFile
from pathlib import Path
import pandas as pd
from sklearn import preprocessing
from PIL import Image


def main():
    SKIN_LESIONS_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
    SKIN_LESIONS_PATH = Path("./data/skin-lesion-data/skin_lesions.zip")
    SKIN_LESIONS_CSV_PATH = Path("./data/skin-lesion-data/metadata.csv")
    SKIN_LESIONS_PICKLE_DIR = Path("./data/skin-lesion-data")
    SKIN_LESIONS_IMAGES_DIR = Path("./data/skin-lesion-data/images")

    download_file(SKIN_LESIONS_URL, SKIN_LESIONS_PATH)
    unzip_file(SKIN_LESIONS_PATH, create_dir=True)
    csv_to_pickle(SKIN_LESIONS_CSV_PATH, SKIN_LESIONS_PICKLE_DIR)
    preprocess_images(SKIN_LESIONS_IMAGES_DIR)


# NOTE: Credit: https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requestshttps://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
# NOTE: ChatGPT Propmt: Create a progress bar for the download of the file
def download_file(url: str, path: Path) -> None:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))
        block_size = 8192
        with open(path, "wb") as f:
            with tqdm(
                total=total_size, unit="iB", unit_scale=True, desc="Downloading"
            ) as bar:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    bar.update(len(chunk))


def unzip_file(file_path: Path, create_dir: bool = False) -> None:
    dir_path = file_path.with_suffix("") if create_dir else file_path.parent
    with ZipFile(file_path, "r") as z:
        z.extractall(path=dir_path)
        files = map(lambda n: Path(n), z.namelist())
        for f in files:
            if f.suffix.lower() != ".zip":
                continue
            unzip_file(dir_path.joinpath(f))


def csv_to_pickle(path_csv: Path, pickle_dir: Path) -> None:
    df = pd.read_csv(path_csv)
    # Convert types and fix typos
    df = convert_types(df)
    df = fix_typos(df)
    df.to_pickle(pickle_dir / "default.pickle")
    # Drop missing and duplicates
    df = df.dropna()
    df = remove_duplicates(df)
    df.to_pickle(pickle_dir / "dropped.pickle")
    # Convert target to binary
    df = binary_classification(df)
    df.to_pickle(pickle_dir / "binary.pickle")
    # Convert into numerical form
    numerical_df = numerical_encoding(df)
    numerical_df = numerical_feature_selection(numerical_df)
    numerical_df.to_pickle(pickle_dir / "numerical.pickle")
    # Convert into categorical form
    categorical_df = categorical_encoding(df)
    categorical_df = categorical_feature_selection(categorical_df)
    categorical_df.to_pickle(pickle_dir / "categorical.pickle")
    # Convert into ordinal form
    ordinal_df = ordinal_encoding(categorical_df)
    ordinal_df.to_pickle(pickle_dir / "ordinal.pickle")


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    to_string = ["patient_id", "img_id"]
    for c in to_string:
        df[c] = df[c].astype("string")

    to_boolean = [
        "smoke",
        "drink",
        "pesticide",
        "skin_cancer_history",
        "cancer_history",
        "has_piped_water",
        "has_sewage_system",
    ]
    for c in to_boolean:
        df[c] = df[c].astype("boolean")

    to_category = [
        "background_father",
        "background_mother",
        "gender",
        "region",
        "diagnostic",
    ]
    for c in to_category:
        df[c] = df[c].astype("category")

    replace_to_bool = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
    for c in replace_to_bool:
        df[c] = df[c].replace({"True": True, "False": False, "UNK": pd.NA})
        df[c] = df[c].astype("boolean")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    ignore = ["img_id"]
    filters = [c for c in df.columns if c not in ignore]
    df = df.loc[~df.duplicated(keep="first", subset=filters)]
    return df


def fix_typos(df: pd.DataFrame) -> pd.DataFrame:
    df["background_father"] = df["background_father"].replace("BRASIL", "BRAZIL")
    return df


def binary_classification(df: pd.DataFrame) -> pd.DataFrame:
    # Diagnostic to binary
    malignant = ["BCC", "MEL", "SCC"]
    df["malignant"] = df["diagnostic"].isin(malignant)
    return df


def numerical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    # Gender to binary
    df["gender"] = df["gender"].replace({"FEMALE": True, "MALE": False})
    df["gender"] = df["gender"].astype("boolean")
    # Column dummies one hot
    to_dummies = ["background_father", "background_mother", "region"]
    for c in to_dummies:
        dummies = pd.get_dummies(df[c], prefix=c)
        df = pd.concat([df, dummies], axis=1)
    # Convert bool to int
    boolean_cols = df.select_dtypes(include="boolean").columns
    df[boolean_cols] = df[boolean_cols].astype(int)
    # Min max columns
    scaler = preprocessing.MinMaxScaler()
    to_scale = ["age", "fitspatrick", "diameter_1", "diameter_2"]
    for c in to_scale:
        df[c] = scaler.fit_transform([[v] for v in df[c]])
    return df


def numerical_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [
        "patient_id",
        "lesion_id",
        "img_id",
        "diagnostic",
        "background_father",
        "background_mother",
        "region",
    ]
    for c in to_drop:
        df = df.drop(c, axis="columns")
    return df


def categorical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    # Binning
    to_bin = [
        ("age", ["young", "adult", "elderly"]),
        ("diameter_1", ["short", "medium", "long"]),
        ("diameter_2", ["short", "medium", "long"]),
    ]
    for c, l in to_bin:
        discretizer = preprocessing.KBinsDiscretizer(
            n_bins=len(l), encode="ordinal", strategy="uniform"
        )
        df[c] = discretizer.fit_transform(df[[c]])
        df[c] = df[c].map(dict(enumerate(l)))
        df[c] = df[c].astype("category")
    # Special binning for fitspatrick
    labels = ["I", "II", "III", "IV", "V", "VI"]
    df["fitspatrick"] = df["fitspatrick"].map(dict(enumerate(labels, 1)))
    df["fitspatrick"] = df["fitspatrick"].astype("category")
    df["fitspatrick"]
    return df


def categorical_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = ["patient_id", "lesion_id", "img_id", "diagnostic"]
    for c in to_drop:
        df = df.drop(c, axis="columns")
    return df


def ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    # Orindal encoding
    to_ordinal = [
        "age",
        "diameter_1",
        "diameter_2",
        "fitspatrick",
        "background_father",
        "background_mother",
        "gender",
        "region",
    ]
    for c in to_ordinal:
        encoder = preprocessing.OrdinalEncoder()
        df[c] = encoder.fit_transform(df[[c]])
        df[c] = df[c].astype(int)
    return df


def preprocess_images(path_images: Path):
    src = path_images / "imgs_part_*/*.png"
    dst = path_images / "train"

    copy_matching_files(src=src, dst=dst)
    resize_images(path=dst)


def copy_matching_files(src: Path, dst: Path):
    if not os.path.exists(dst):
        os.mkdir(dst)

    files = glob.glob(str(src))

    for f in files:
        shutil.copy(f, dst)


# Credit: https://stackoverflow.com/a/21518989
def resize_images(path: Path):
    for filename in os.listdir(path):
        filepath = path / filename
        if not filepath.is_file():
            continue
        im = Image.open(filepath)
        im_resized = im.resize((200, 200))
        im_resized.save(filepath, format="PNG", quality=90)


if __name__ == "__main__":
    main()
