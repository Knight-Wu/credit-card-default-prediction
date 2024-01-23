import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_column_names = {
        "ID": "id",
        "LIMIT_BAL": "credit_limit",
        "SEX": "gender",
        "EDUCATION": "education_level",
        "MARRIAGE": "marital_status",
        "AGE": "age",
        "PAY_0": "repay_status_sep",
        "PAY_2": "repay_status_aug",
        "PAY_3": "repay_status_jul",
        "PAY_4": "repay_status_jun",
        "PAY_5": "repay_status_may",
        "PAY_6": "repay_status_apr",
        "BILL_AMT1": "bill_amt_sep",
        "BILL_AMT2": "bill_amt_aug",
        "BILL_AMT3": "bill_amt_jul",
        "BILL_AMT4": "bill_amt_jun",
        "BILL_AMT5": "bill_amt_may",
        "BILL_AMT6": "bill_amt_apr",
        "PAY_AMT1": "pay_amt_sep",
        "PAY_AMT2": "pay_amt_aug",
        "PAY_AMT3": "pay_amt_jul",
        "PAY_AMT4": "pay_amt_jun",
        "PAY_AMT5": "pay_amt_may",
        "PAY_AMT6": "pay_amt_apr",
        "default.payment.next.month": "default_next_month",
    }

    return df.rename(columns=new_column_names)


def copy_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Copying the columns to avoid the SettingWithCopyWarning
    df["repay_label_sep"] = df["repay_status_sep"]
    df["repay_label_aug"] = df["repay_status_aug"]
    df["repay_label_jul"] = df["repay_status_jul"]
    df["repay_label_jun"] = df["repay_status_jun"]
    df["repay_label_may"] = df["repay_status_may"]
    df["repay_label_apr"] = df["repay_status_apr"]
    return df


def remap_categorical_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Remapping the categorical variables
    df["gender"] = df["gender"].map({1: "male", 2: "female"})
    df["education_level"] = df["education_level"].map(
        {
            1: "graduate_school",
            2: "university",
            3: "high_school",
            4: "others",
            5: "unknown",
            6: "unknown",
            0: "unknown",
        }
    )
    df["marital_status"] = df["marital_status"].map(
        {1: "married", 2: "single", 3: "others", 0: "others"}
    )

    return df


def remap_repayment_status(value):
    if value == -2:
        return "-2_no_consumption"
    elif value == -1:
        return "-1_paid_in_full"
    elif value == 0:
        return "0_paid_minimum"
    elif 1 <= value <= 8:
        return f"delay_{value}_mnths"
    else:
        return "delay_9+_mnths"


def apply_remap_repayment_status(df):
    # List of columns to apply the remapping
    repayment_status_columns = [
        "repay_status_sep",
        "repay_status_aug",
        "repay_status_jul",
        "repay_status_jun",
        "repay_status_may",
        "repay_status_apr",
    ]

    # Apply the remapping function to each column in the list
    for column in repayment_status_columns:
        df[column] = df[column].apply(lambda x: remap_repayment_status(x))

    return df


def bin_age(df: pd.DataFrame) -> pd.DataFrame:
    # Binning the age variable
    age_bins = [
        0,
        25,
        35,
        50,
        65,
        100,
    ]
    age_labels = ["21-25", "26-35", "36-50", "51-65", "66+"]
    df["age_group"] = pd.cut(
        df["age"],
        bins=age_bins,
        labels=age_labels,
        right=False,
        include_lowest=True,
    )
    return df


# Apply feature engineering


def combine_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    df["marital_gender"] = df["marital_status"] + "_" + df["gender"]
    df["education_gender"] = df["education_level"] + "_" + df["gender"]
    df["education_marital"] = df["education_level"] + "_" + df["marital_status"]
    df["education_marital_gender"] = (
        df["education_level"] + "_" + df["marital_status"] + "_" + df["gender"]
    )
    return df


def create_repay_label_sum(df: pd.DataFrame) -> pd.DataFrame:
    df["repay_label_sum"] = (
        df["repay_label_sep"]
        + df["repay_label_aug"]
        + df["repay_label_jul"]
        + df["repay_label_jun"]
        + df["repay_label_may"]
        + df["repay_label_apr"]
    )
    return df


def count_delayed_payments(row):
    pay_columns = [
        "repay_label_sep",
        "repay_label_aug",
        "repay_label_jul",
        "repay_label_jun",
        "repay_label_may",
        "repay_label_apr",
    ]
    return sum(1 for pay in row[pay_columns] if 1 <= pay <= 9)


def create_count_delayed_payments(df):
    # Apply the function to each row of the DataFrame
    df["count_of_delayed_payments"] = df.apply(count_delayed_payments, axis=1)
    return df


def convert_columns_to_object(df):
    # convert column to object
    df["count_of_delayed_payments"] = df["count_of_delayed_payments"].astype("category")
    return df


def one_hot_encode(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    ohe = OneHotEncoder(drop="first")
    encoded_data = ohe.fit_transform(df[cat_cols]).toarray()
    new_cols = ohe.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=new_cols)
    # Concatenate with the original DataFrame
    df = pd.concat([df.drop(cat_cols, axis=1), encoded_df], axis=1)
    return df


def drop_columns_basic(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "id",
    ]
    df = df.drop(columns=columns_to_drop)
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "id",
        "repay_label_sep",
        "repay_label_aug",
        "repay_label_jul",
        "repay_label_jun",
        "repay_label_may",
        "repay_label_apr",
    ]
    df = df.drop(columns=columns_to_drop)
    df = df.loc[:, ~df.columns.str.startswith("remainder_")]
    df = df.loc[:, ~df.columns.str.startswith("repay_label_")]

    return df


def apply_basic_data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_columns(df)
    df = remap_categorical_labels(df)
    df = apply_remap_repayment_status(df)
    df = drop_columns_basic(df)
    return df


def apply_data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_columns(df)
    df = copy_columns(df)
    df = remap_categorical_labels(df)
    df = apply_remap_repayment_status(df)
    df = bin_age(df)
    return df


def create_new_features(df):
    df = create_count_delayed_payments(df)
    df = convert_columns_to_object(df)
    df = combine_demographic_features(df)

    return df


def main():
    df = pd.read_csv("data/raw/credit_card_default.csv")
    df = apply_data_preprocessing(df)
    df = create_new_features(df)
    df = drop_columns(df)
    df = one_hot_encode(df)
    df.to_csv("data/processed/processed-data.csv", index=False)


if __name__ == "__main__":
    main()
