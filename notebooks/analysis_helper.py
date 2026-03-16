import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from enum import Enum
from typing import Literal, Optional, List
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold


MATPLOTLIB_STYLE_PATH = "../src/cider/matplotlibrc"


class FillnaMethod(Enum):
    MEDIAN = "median"
    MEAN = "mean"
    ZERO = "zero"


"""
Data descriptions
"""


def describe_data(
    data: pd.DataFrame, exclude_id_col: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Descriptive statistics for each column in dataframe except for columns specified in exclude_id_col.
    Also includes a completeness column with count of non-null values for each feature.

    Args
    ----
    data : pd.DataFrame
        The input dataframe to describe.
    exclude_id_col : list
        List of columns to exclude from the description.

    Returns
    -------
    pd.DataFrame
        A dataframe containing descriptive statistics and completeness for each feature.
    """
    if exclude_id_col is not None:
        completeness = data.drop(columns=exclude_id_col).notna().sum()
        description = data.drop(columns=exclude_id_col).describe().transpose()
    else:
        completeness = data.notna().sum()
        description = data.describe().transpose()
    description["completeness"] = completeness
    description = description.reset_index().rename(columns={"index": "feature"})
    return description


def describe_features(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    description_list = []
    for feat in features:
        description = describe_data(features[feat], exclude_id_col=["caller_id"])
        description["source"] = feat
        description_list.append(description)
    description_df = pd.concat(description_list, ignore_index=True)
    return description_df


"""
Visualizations
"""


def generate_boxplots(
    data: pd.DataFrame,
    features: List[str],
    groupby: Optional[str] = None,
    orient: Literal["v", "h"] = "v",
    showfliers: bool = True,
    ncol: int = 2,
    nrow: Optional[int] = None,
    title: Optional[str] = None,
    single_figsize: tuple = (4, 3),
    save_path: Optional[str] = None,
):
    """
    Generate boxplots for specified feature columns in the data, grouped by the groupby column.
    Args:
        data: DataFrame containing the data to plot
        features: List of column names to generate boxplots for
        groupby: Column name to group the data by for the boxplots
        orient: Orientation of the boxplots, 'v' for vertical and 'h' for horizontal
        showfliers: Whether to show outliers in the boxplots
        ncol: Number of columns for the subplot grid
        nrow: Number of rows for the subplot grid, if None it will be calculated based on the number of features and ncol
        title: Title for the entire figure
        single_figsize: Tuple specifying the size of each individual subplot (width, height)
        save_path: Path to save the figure, if None the figure will not be saved
    """
    if nrow is None:
        nrow = int(np.ceil(len(features) / ncol))

    with mpl.rc_context(mpl.rc_params_from_file(MATPLOTLIB_STYLE_PATH)):
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(single_figsize[0] * ncol, single_figsize[1] * nrow)
        )
        axes = np.array(axes).flatten()
        for i, feat in enumerate(features):
            if orient == "v":
                sns.boxplot(
                    data=data, y=feat, x=groupby, showfliers=showfliers, ax=axes[i]
                )
            elif orient == "h":
                sns.boxplot(
                    data=data, x=feat, y=groupby, showfliers=showfliers, ax=axes[i]
                )
        if title is not None:
            plt.suptitle(title)

        # hide unused subplots
        for ax in axes[len(features) :]:
            ax.set_visible(False)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()


def scatterplots(
    data: pd.DataFrame,
    features: list[str],
    label: str,
    ncol: int = 2,
    nrow: Optional[int] = None,
    title: Optional[str] = None,
    single_figsize: tuple = (4, 3),
    label_axis: Literal["x", "y"] = "y",
    save_path: Optional[str] = None,
):
    """
    Generate scatterplots for specified feature columns in the data, grouped by the groupby column.
    Args:
        data: DataFrame containing the data to plot
        features: List of column names to generate scatterplots for
        label: Column name to use as the label for the scatterplots
        ncol: Number of columns for the subplot grid
        nrow: Number of rows for the subplot grid, if None it will be calculated based on the number of features and ncol
        title: Title for the entire figure
        single_figsize: Tuple specifying the size of each individual subplot (width, height)
        label_axis: Axis to plot the label on, 'x' for x-axis and 'y' for y-axis
        save_path: Path to save the figure, if None the figure will not be saved
    """

    if nrow is None:
        nrow = int(np.ceil(len(features) / ncol))

    with mpl.rc_context(mpl.rc_params_from_file(MATPLOTLIB_STYLE_PATH)):
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(single_figsize[0] * ncol, single_figsize[1] * nrow)
        )
        axes = np.array(axes).flatten()
        for i, feat in enumerate(features):
            if label_axis == "x":
                sns.scatterplot(data=data, x=label, y=feat, ax=axes[i])
            elif label_axis == "y":
                sns.scatterplot(data=data, x=feat, y=label, ax=axes[i])
        if title is not None:
            fig.suptitle(title)

        # hide unused subplots
        for ax in axes[len(features) :]:
            ax.set_visible(False)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()


"""
Processing raw data
"""


def process_raw_data(column_map: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns to match the expected schema.
    Args:
        column_map: dict mapping expected column names in schema to column names in the raw data.
        df: DataFrame containing raw data.

    Returns:
        df: DataFrame with mapped columns renamed to match the expected schema.
    """
    # missing_columns = [key for key, value in column_map.items() if value is None]
    # if len(missing_columns) > 0:
    #     for col in missing_columns:
    #         logging.info(f"Column {col} is required but not mapped to any column in input dataframe. Filling with null values.")
    #         df[col] = None
    rename_col = {value: key for key, value in column_map.items() if value is not None}
    cols = {key: value for key, value in column_map.items() if value is not None}
    df = df.rename(columns=rename_col)
    df = df[cols.keys()].copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


"""
Preprocessing data for modelling, running k-fold cross validation, and evaluating model performance
"""


# NEXT TO DO: Need to define which features have which best missing value imputation methods.
def process_data_for_modelling(
    X,
    drop_zero_variance=False,
    null_max_threshold=None,
    fillna_method: Optional[FillnaMethod] = None,
    scale=False,
):
    """
    Data preprocessing steps for Lasso regression
    1. Drop rows with all null values
    2. Select features with nonzero variance
    3. Drop features with more null values than null_max_threshold, if specified
    4. Impute remaining null values with specified method (median, mean, or zero), if specified
    5. Scale features with RobustScaler, if specified
    """
    logging.info("Starting data processing for modelling... ")
    logging.info(
        f"Initial number of features: {X.shape[1]}, number of rows: {X.shape[0]}"
    )
    # 1. Drop rows with all null values
    X = X.loc[X.notna().sum(axis=1) > 0]
    logging.info(
        f"{X.shape[0]} rows remaining after dropping rows with all null values."
    )

    # 2. Select features with nonzero variance
    if drop_zero_variance:
        X = X.loc[:, X.std() > 0]
        logging.info(
            f"{len(X.columns)} features remaining after dropping features with zero variance."
        )

    # 3. Drop features with more null values than null_max_threshold, if specified
    if null_max_threshold is not None:
        temp = (X.isna().sum() / len(X)).reset_index(name="nulls")
        logging.info(
            f"Null value distribution for features: {temp['nulls'].describe()}"
        )
        keep_features = temp[temp["nulls"] < null_max_threshold][
            "index"
        ].tolist()  # keep features with less than null_threshold null values
        if len(keep_features) == 0:
            raise ValueError(
                "No features left after dropping features with too many null values. Consider increasing null_max_threshold."
            )
        else:
            logging.info(
                f"{len(keep_features)} features remaining after dropping features with more than {null_max_threshold} null values."
            )
            X = X[keep_features]

    # 4. Impute null values according to method specified in fillna_method argument
    match fillna_method:
        case FillnaMethod.MEDIAN:
            X = X.apply(
                lambda col: col.fillna(col.median()), axis=0
            )  # impute remaining null values with median
        case FillnaMethod.MEAN:
            X = X.apply(
                lambda col: col.fillna(col.mean()), axis=0
            )  # impute remaining null values with mean
        case FillnaMethod.ZERO:
            X = X.fillna(0)  # impute remaining null values with zero
        case None:
            pass

    # 5. Scale features
    if scale:
        X = pd.DataFrame(
            RobustScaler().fit_transform(X), columns=X.columns, index=X.index
        )

    logging.info(
        f"Post-processing number of features: {X.shape[1]}, number of rows: {X.shape[0]}"
    )
    return X


def precision_recall_at_k(y_true, y_pred, k, tail: Literal["upper", "lower"]):
    """
    Calculates precision and recall at top/bottom k% of predictions.
    Args:
        tail: if 'lower', calculates precision and recall for bottom k% of predicted values.
              if 'upper', calculates precision and recall for top k% of predicted values.
    Returns:
        dict with precision and recall at k% for specified tail
    """
    data = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    if tail == "upper":
        threshold_pred = np.percentile(y_pred, 100 - k)
        threshold_true = np.percentile(y_true, 100 - k)
        selected_pred = data[data["y_pred"] >= threshold_pred].index
        selected_true = data[data["y_true"] >= threshold_true].index

    elif tail == "lower":
        threshold_pred = np.percentile(y_pred, k)
        threshold_true = np.percentile(y_true, k)
        selected_pred = data[data["y_pred"] <= threshold_pred].index
        selected_true = data[data["y_true"] <= threshold_true].index
    else:
        raise ValueError("tail must be either 'upper' or 'lower'")

    if len(selected_pred) == 0:
        raise ValueError(
            "No predictions selected. Consider increasing k or check if predictions are valid."
        )
    if len(selected_true) == 0:
        raise ValueError(
            "No true values selected. Consider increasing k or check if true values are valid."
        )

    precision = len(set(selected_pred).intersection(set(selected_true))) / len(
        selected_pred
    )
    recall = len(set(selected_pred).intersection(set(selected_true))) / len(
        selected_true
    )
    return {f"precision_{tail}_{k}": precision, f"recall_{tail}_{k}": recall}


def evaluation_metrics(
    X_test, y_test, model, percentile_k=50, precision_recall_tail="lower"
):
    r_squared = model.score(X_test, y_test)
    adj_r_squared = 1 - (1 - r_squared) * (
        (X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)
    )
    # mean_abs_error = np.mean(np.abs(model.predict(X_test) - y_test))
    mean_squared_error = np.mean((model.predict(X_test) - y_test) ** 2)
    mean_abs_perc_error = np.mean(np.abs(model.predict(X_test) - y_test) / y_test)
    spearman_rho, spearman_pvalue = spearmanr(model.predict(X_test), y_test)
    precision_recall_k = precision_recall_at_k(
        y_test, model.predict(X_test), k=percentile_k, tail=precision_recall_tail
    )
    precision_k = precision_recall_k[
        f"precision_{precision_recall_tail}_{percentile_k}"
    ]
    recall_k = precision_recall_k[f"recall_{precision_recall_tail}_{percentile_k}"]

    return {
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        # 'mean_abs_error': mean_abs_error,
        "mean_squared_error": mean_squared_error,
        "mean_abs_perc_error": mean_abs_perc_error,
        "spearman_rho": spearman_rho,
        "spearman_pvalue": spearman_pvalue,
        f"precision_{precision_recall_tail}_{percentile_k}": precision_k,
        f"recall_{precision_recall_tail}_{percentile_k}": recall_k,
    }


def run_kfold(
    X: np.ndarray,
    y: np.ndarray,
    k_fold: int,
    model,
    random_state,
    features: np.ndarray,
    percentile_k: int = 50,
    precision_recall_tail: Literal["upper", "lower"] = "lower",
):
    """
    Args:
        X: array with shape (n_samples, n_features) containing feature values
        y: array with shape (n_samples,) containing label values
        model: linear model to fit
        k_fold: number of folds for cross validation
        features: array of feature names used for model training
        percentile_k: k value for precision and recall at top/bottom k%
        precision_recall_tail: whether to calculate precision and recall for upper or lower tail of predictions
    """
    assert len(X) == len(y), "Length of X and y must be the same."

    kfold = KFold(k_fold, shuffle=True, random_state=random_state)
    train_k_eval_dict: dict[str, list[float]] = {
        "r_squared": [],
        "adj_r_squared": [],
        "mean_squared_error": [],
        "mean_abs_perc_error": [],
        "spearman_rho": [],
        "spearman_pvalue": [],
        f"precision_{precision_recall_tail}_{percentile_k}": [],
        f"recall_{precision_recall_tail}_{percentile_k}": [],
    }
    test_k_eval_dict: dict[str, list[float]] = {
        "r_squared": [],
        "adj_r_squared": [],
        "mean_squared_error": [],
        "mean_abs_perc_error": [],
        "spearman_rho": [],
        "spearman_pvalue": [],
        f"precision_{precision_recall_tail}_{percentile_k}": [],
        f"recall_{precision_recall_tail}_{percentile_k}": [],
    }

    k_eval_metrics = {}
    num_of_features_selected = []
    features_selected_list = []
    model_coef_list = []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)

        train_eval_dict = evaluation_metrics(
            X_train,
            y_train,
            model,
            percentile_k=percentile_k,
            precision_recall_tail=precision_recall_tail,
        )
        test_eval_dict = evaluation_metrics(
            X_test,
            y_test,
            model,
            percentile_k=percentile_k,
            precision_recall_tail=precision_recall_tail,
        )
        num_of_features = np.sum(model.coef_ != 0)
        features_selected = features[model.coef_ != 0].tolist()
        model_coef = [coef for coef in model.coef_ if coef != 0]
        assert len(features_selected) == len(
            model_coef
        ), "Number of features selected should match number of nonzero coefficients"

        num_of_features_selected.append(num_of_features)
        features_selected_list.append(features_selected)
        model_coef_list.append(model_coef)

        for i in train_eval_dict:
            train_k_eval_dict[i].append(train_eval_dict[i])
            test_k_eval_dict[i].append(test_eval_dict[i])

    for x in train_k_eval_dict:
        k_eval_metrics["mean_train_" + x] = np.mean(train_k_eval_dict[x], axis=0)
        k_eval_metrics["std_train_" + x] = np.std(train_k_eval_dict[x], axis=0)
    for x in test_k_eval_dict:
        k_eval_metrics["mean_test_" + x] = np.mean(test_k_eval_dict[x], axis=0)
        k_eval_metrics["std_test_" + x] = np.std(test_k_eval_dict[x], axis=0)
    k_eval_metrics["mean_num_of_features_selected"] = np.mean(num_of_features_selected)

    return {
        "train": train_k_eval_dict,
        "test": test_k_eval_dict,
        "eval_metrics": k_eval_metrics,
        "num_of_features": num_of_features_selected,
        "features_selected": features_selected_list,
        "model_coef": model_coef_list,
    }


def kfold_performance_across_alphas(
    X: np.ndarray,
    y: np.ndarray,
    alpha_options: list[float],
    features: np.ndarray,
    model_type: Literal["lasso", "elasticnet"],
):
    """
    Runs k-fold cross validation for different alpha options for Lasso or ElasticNet regression.
    Collects evaluation metrics for each iteration.
    """
    rows = []
    for alpha in alpha_options:
        if model_type == "lasso":
            model = Lasso(alpha=alpha)
        elif model_type == "elasticnet":
            model = ElasticNet(alpha=alpha, l1_ratio=0.9)
        kfold_results = run_kfold(
            X, np.array(y), k_fold=5, model=model, random_state=100, features=features
        )
        mean_results = kfold_results["eval_metrics"]
        mean_results["alpha"] = alpha
        mean_results["features_selected"] = kfold_results["features_selected"]
        mean_results["model_coef"] = kfold_results["model_coef"]
        rows.append(mean_results)
    return pd.DataFrame(rows)
