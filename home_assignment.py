import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, preprocessing
from calendar import monthrange
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import AgglomerativeClustering

# Snapshot - sample ID, ignore
# Snapshot Date - sample date
# Checkin Date - date to check in
# Days - duration of stay
# Original price - original room price, $
# Discount Price - price after discount, $, depends on Discount Code
# Discount Code - 1-4, affects Discount Price
# Available Rooms - num of available room on Checkin Date, -1 unknown
# Hotel Name - name
# Hotel Stars - rating


class Hotels:
    def __init__(self, df):
        self.df = df

    # Part 1
    def get_df(self):
        return self.df

    def get_col(self, colname):
        return self.df[colname]

    def update_df(self, cols_to_keep):
        self.df = self.get_df()[cols_to_keep]

    def str_to_datetime(self, colnames):
        for col in colnames:
            self.df[col] = pd.to_datetime(self.df[col])

    def timedelta_to_int(self, colname):
        col = self.get_col(colname)
        self.df[colname] = col.dt.days

    def add_diff_col(self, larger, smaller, diff_colname):
        larger_col = self.get_col(larger)
        smaller_col = self.get_col(smaller)
        self.df[diff_colname] = larger_col - smaller_col

    def datetime_to_week_day(self, date_col, colname):
        col = self.get_col(date_col)
        self.df[colname] = col.dt.strftime('%a')

    def add_perc_change(self, original_val, diff, colname):
        original_col = self.get_col(original_val)
        diff_col = self.get_col(diff)
        self.df[colname] = diff_col / original_col * 100

    # Part 3
    def filter_by_entries_limit_in_col(self, colname, limit):
        df = self.get_df()
        value_count = pd.DataFrame(df[colname].value_counts())
        value_count = value_count[:limit]
        to_keep = list(value_count.index)
        self.df = df[df[colname].isin(to_keep)]

    def prep_df_for_clustering(self, main_val, min_val, group, apply_func, apply_colname, fill_val):
        df = self.get_df()
        # Keep only necessary columns
        cols_to_keep = group + [min_val]
        df = df[cols_to_keep]
        # Group vals by group s.t. val in min_val is the min possible for the group
        grouped = df.groupby(by = group).min(min_val)
        # Normalize norm_by using norm_func
        grouped[apply_colname] = grouped.groupby(by=[main_val]).apply(apply_func)

        # Get all combinations of vals in group
        unique_levels = [set(df[col].values) for col in group]
        all_combo_index = pd.MultiIndex.from_product(unique_levels, names = group)

        # Update group to include all combos
        grouped = grouped.reindex(all_combo_index, fill_value=fill_val)
        # Drop min_val, only need apply_colname
        grouped.drop(min_val, inplace=True, axis=1)
        # Unstack to get the df - main val as rows, cols are combos of the rest of group
        group.remove(main_val)
        ungroup = grouped.unstack(level = group)
        # Drop rows with NAs (= all non -1 vals are identical)
        ungroup.dropna(inplace=True)
        return ungroup

    def clustering(self, source_df, show):
        data_for_clusters = source_df.iloc[:, :].values
        fig, ax = plt.subplots(figsize=(7, 10))
        hier_clust = sch.linkage(data_for_clusters, method='ward')
        dendrogram = sch.dendrogram(hier_clust, labels=source_df.index,
                                    leaf_font_size=5, color_threshold=839,
                                    orientation="left")
        plt.tight_layout()
        if show: plt.show()
        return dendrogram

    def group_by_max_param(self, params, max_param):
        df = self.get_df()
        best_code_df = df.groupby(by = params).max(max_param)
        return best_code_df



def encode_to_cyclic(df, col, max_vals):
    df.insert(len(df.columns), col + ' sin', np.sin(2 * np.pi * df[col]/max_vals))
    df.insert(len(df.columns), col + ' cos', np.cos(2 * np.pi * df[col] / max_vals))
    # df[col + ' sin'] = np.sin(2 * np.pi * df[col]/max_vals)
    # df[col + ' cos'] = np.cos(2 * np.pi * df[col]/max_vals)
    return df

def convert_dates(df, colname, cyclic_trans):
    # Add columns for year, month, day from colname (datetime)
    years = df[colname].dt.year
    months = df[colname].dt.month
    df.insert(len(df.columns), colname + " year", years)
    df.insert(len(df.columns), colname + " month", months)
    df.insert(len(df.columns), colname + " day", df[colname].dt.day)
    # Add columns for month and day to represent the cyclical nature of these vars
    if cyclic_trans:
        def days_in_month(year, month):
            return monthrange(year, month)[1]
        days_in_month = np.vectorize(days_in_month)(years, months)
        df = encode_to_cyclic(df, colname + " month", 12)
        df = encode_to_cyclic(df, colname + " day", days_in_month)
    return df

def print_model_scores(y_test, y_pred, y_pred_prob, print_cm):
    print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 3))
    print("ROC AUC:", round(metrics.roc_auc_score(y_test, y_pred_prob, multi_class='ovo'), 3))
    if print_cm:
        cm = metrics.confusion_matrix(y_test, y_pred)
        print(cm)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        print("FP =", FP)
        print("FN =", FN)
        print("TP =", TP)
        print("TN =", TN, "\n")


def classifier(df, feature_cols, y_col, model, print_cm):
    # Split data
    df_filtered = df[feature_cols]
    y = df_filtered.pop(y_col)
    X = df_filtered
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    model.fit(X_train, y_train)
    # For tree
    # print(model.feature_importances_)
    # print(model.feature_names_in_)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    print_model_scores(y_test, y_pred, y_pred_prob, print_cm)


def naive_bayes_classifier(df, feature_cols, y_col, print_cm):
    classifier(df, feature_cols, y_col, GaussianNB(), print_cm)


def decision_tree_classifier(df, feature_cols, y_col, print_cm):
    classifier(df, feature_cols, y_col, DecisionTreeClassifier(), print_cm)




if __name__ == '__main__':

    """
    1.
    1st part of the assignment
    """

    hotels_df = pd.read_csv("hotels_data.csv")
    hotels = Hotels(hotels_df)

    # check types of cols
    # print(hotels.get_df().dtypes)
    # Dates and Hotel Name are objects, rest are int

    # Convert Dates to datetime
    date_cols = ["Snapshot Date", "Checkin Date"]
    hotels.str_to_datetime(date_cols)

    # Add DayDiff (timedelta)
    hotels.add_diff_col("Checkin Date", "Snapshot Date", "DayDiff")
    hotels.timedelta_to_int("DayDiff")

    # Add WeekDay
    hotels.datetime_to_week_day("Checkin Date", "WeekDay")

    # Add DiscountDiff (int)
    hotels.add_diff_col("Original Price", "Discount Price", "DiscountDiff")

    # Add DiscountPerc
    hotels.add_perc_change("Original Price", "DiscountDiff", "DiscountPerc")

    # Save file
    hotels_data_changed = hotels.get_df()
    hotels_data_changed.to_csv("Hotels_data_Changed.csv")


    """
    2.
    2nd part of the assignment
    """

    # Upload df and convert dates to Datetime
    hotels_data_changed = pd.read_csv("Hotels_data_Changed.csv")
    hotels = Hotels(hotels_data_changed)
    hotels.str_to_datetime(["Snapshot Date", "Checkin Date"])

    # Filter df by these cols
    params = ["Snapshot Date", "Checkin Date", "DayDiff", "Hotel Name", "WeekDay"]
    cols_to_keep = params + ["DiscountPerc", "Discount Code"]
    hotels.update_df(cols_to_keep)

    # Get DiscountCode per params with max DiscountPerc
    best_code_df = hotels.group_by_max_param(params, "DiscountPerc")
    best_code_df = hotels.get_df()

    # Separate dates to year, month, day columns, True if should convert the month and day to cyclic vars
    convert_dates(best_code_df, "Snapshot Date", True)
    convert_dates(best_code_df, "Checkin Date", True)

    cols_for_classifier = ["DayDiff", "Discount Code", "Snapshot Date", "Checkin Date",
                           "Hotel Name", "WeekDay",
                           "Snapshot Date year", "Snapshot Date month", "Snapshot Date day",
                           "Checkin Date year", "Checkin Date month", "Checkin Date day",
                           "Snapshot Date month sin", "Snapshot Date month cos",
                           "Snapshot Date day sin", "Snapshot Date day cos",
                           "Checkin Date month sin", "Checkin Date month cos",
                           "Checkin Date day sin", "Checkin Date day cos"]

    # Prep df to encode ordinal vars
    best_code_df = best_code_df[cols_for_classifier]
    cols_to_encode = ["Snapshot Date", "Checkin Date", "Hotel Name", "WeekDay"]
    df_to_encode = pd.concat([best_code_df.pop(x) for x in cols_to_encode], axis=1)

    # Encode categorical features
    enc = preprocessing.OrdinalEncoder()

    # test encoding
    # days = pd.DataFrame({"days": ["Wed", "Thu", "Wed", "Fri"], "fds":[1, 0, 3, 1]})
    # vs = pd.DataFrame(enc.fit_transform(days), columns=["vds", "csd"])
    # print(vs)

    encoded_df = pd.DataFrame(enc.fit_transform(df_to_encode[cols_to_encode]), columns=cols_to_encode)

    classifier_df = pd.concat([best_code_df, encoded_df], axis=1)

    # Naive Bayes classifier
    NB_features = ["DayDiff", "Discount Code", "Snapshot Date", "Checkin Date", "Hotel Name", "WeekDay",
                   "Snapshot Date year", "Snapshot Date month", "Snapshot Date day",
                   "Checkin Date year", "Checkin Date month", "Checkin Date day"]

    naive_bayes_classifier(classifier_df, NB_features, "Discount Code", True)

    # subset of vars for NB
    # NB_features = ["DayDiff", "Discount Code", "Snapshot Date", "Checkin Date", "Hotel Name", "WeekDay",
    #                "Checkin Date year", "Checkin Date month"]
    # naive_bayes_classifier(classifier_df, NB_features, "Discount Code", True)

    # Decision tree classifier
    tree_features = ["Hotel Name", "DayDiff", "Discount Code", "Snapshot Date", "Checkin Date", "WeekDay",
                     "Snapshot Date year", "Snapshot Date month", "Snapshot Date day",
                     "Checkin Date year", "Checkin Date month", "Checkin Date day"]

    decision_tree_classifier(classifier_df, tree_features, "Discount Code", True)
    # decision_tree_classifier(classifier_df, ["Discount Code", "Checkin Date", "Hotel Name", "WeekDay"], "Discount Code", True)
    # decision_tree_classifier(classifier_df, ["Discount Code", "Hotel Name", "Snapshot Date", "DayDiff"], "Discount Code", True)


    # check individual features
    # individual_features = ["DayDiff", "Snapshot Date", "Checkin Date", "Hotel Name", "WeekDay",
    #                        "Snapshot Date year", "Snapshot Date month", "Snapshot Date day",
    #                        "Checkin Date year", "Checkin Date month", "Checkin Date day"]
    #                        # "Snapshot Date month sin", "Snapshot Date month cos",
    #                        # "Snapshot Date day sin", "Snapshot Date day cos",
    #                        # "Checkin Date month sin", "Checkin Date month cos",
    #                        # "Checkin Date day sin", "Checkin Date day cos"]
    # for feature in individual_features:
    #     print(feature)
    #     naive_bayes_classifier(classifier_df, ["Discount Code", feature], "Discount Code", True)
        # decision_tree_classifier(classifier_df, ["Discount Code", feature], "Discount Code", True)


    """
    3.
    3rd part of the assignment
    """
    # For each Hotel Name, vector describing normalized price for the date, for the time period.
    hotels_data_changed = pd.read_csv("Hotels_data_Changed.csv")

    hotels = Hotels(hotels_data_changed)
    hotels.filter_by_entries_limit_in_col("Hotel Name", 150)
    # print(len(set(hotels.get_col("Hotel Name")))) # 150
    hotels.filter_by_entries_limit_in_col("Checkin Date", 40)
    # print(len(set(hotels.get_col("Hotel Name")))) # 149, one was eliminated by the date filtering
    # print(len(set(hotels.get_col("Checkin Date")))) # 40

    # Group
    norm_func = lambda x: 100 * (x - x.min()) / (x.max() - x.min())
    group_by = ["Hotel Name", "Checkin Date", "Discount Code"]
    prices_per_date_for_hotel = hotels.prep_df_for_clustering("Hotel Name", "Discount Price", group_by, norm_func, "norm", -1.)
    # print(prices_per_date_for_hotel.shape) # (148, 160), Hotel Name is the index

    # Build dendogram
    dendogram = hotels.clustering(prices_per_date_for_hotel, show = False)

    cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

    prices_per_date_for_hotel["cluster"] = cluster.fit_predict(prices_per_date_for_hotel)
    cluster_group = prices_per_date_for_hotel.groupby(by="cluster").mean()
    print(cluster_group)


