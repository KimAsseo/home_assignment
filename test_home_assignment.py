import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from home_assignment import Hotels

class TestPart1(unittest.TestCase):
    def setUp(self):
        df_test = pd.DataFrame({"early_date": ["7/17/2015 0:00", "2/2/2015 00:00"],
                                "later_date": ["08-12-15 00:00", "2-10-2015 0:00"],
                                "og_price": [1550, 200],
                                "discount_price": [1300, 20],
                                "perc_disc_true": [(1550-1300)/15.50, 90]})
        self.df = Hotels(df_test)

    def test_add_diff_col(self):
        date_diff = pd.Series([26, 8])
        self.df.str_to_datetime(["early_date", "later_date"])
        Hotels.add_diff_col(self.df, "later_date", "early_date", "date_diff_col")
        Hotels.timedelta_to_int(self.df, "date_diff_col")
        date_diff_col = self.df.get_col("date_diff_col")
        assert (all(date_diff_col.values == date_diff.values))
        # self.assertEqual(date_diff_col[0], date_diff[0], "should be 26")

        # Price diff
        price_diff = pd.Series([250, 180])
        Hotels.add_diff_col(self.df, "og_price", "discount_price", "price_diff_col")
        price_diff_col = Hotels.get_col(self.df, "price_diff_col")
        assert (all(price_diff_col.values == price_diff.values))


    def test_add_WeekDay(self):
        week_days = pd.Series(["Fri", "Mon"])
        self.df.str_to_datetime(["early_date"])
        Hotels.datetime_to_week_day(self.df, "early_date", "week_day")
        week_days_col = Hotels.get_col(self.df, "week_day")
        # assert (all(week_days_col.values == week_days.values))
        self.assertEqual(week_days_col[0], "Fri", "Pass week_day")
        self.assertEqual(week_days_col[1], "Mon", "Pass week_day")


    def test_add_perc_change(self):
        Hotels.add_diff_col(self.df, "og_price", "discount_price", "price_diff_col")
        Hotels.add_perc_change(self.df, "og_price", "price_diff_col", "perc_disc")
        perc_disc = Hotels.get_col(self.df, "perc_disc")
        perc_disc_true = Hotels.get_col(self.df, "perc_disc_true")
        self.assertEqual(perc_disc[0], perc_disc_true[0], "should be 16.129..")
        self.assertEqual(perc_disc[1], perc_disc_true[1], "should be 90")
        assert (all(perc_disc.values == perc_disc_true.values))


class TestPart3(unittest.TestCase):
    def setUp(self):
        df_test = pd.DataFrame({"name": ["a", "d", "b", "b", "c", "a", "a"],
                                "x": [1, 2, 3, 1, 1, 3, 1],
                                "y": pd.to_datetime(["7/17/2015 0:00", "2/2/2015 00:00",
                                      "7/17/2015 0:00", "2/2/2015 00:00",
                                      "7/17/2015 0:00", "2/8/2015 00:00",
                                      "7/17/2015 00:00"])})
        self.df = Hotels(df_test)

    def test_filter_by_entries_limit_in_col(self):
        # first filter by name
        name = pd.Series(["a", "b", "b", "a", "a"])
        Hotels.filter_by_entries_limit_in_col(self.df, "name", 2)
        df = Hotels.get_df(self.df)
        # print(df)
        # print(df.name.values)
        self.assertTrue(df.shape == (5, 3), "should be (5,3)")
        assert (all(df.name.values == name.values))
        # filter by x
        name = pd.Series(["a", "b", "a"])
        Hotels.filter_by_entries_limit_in_col(self.df, "x", 1)
        df = Hotels.get_df(self.df)
        self.assertTrue(df.shape == (3, 3), "should be (3,3)")
        assert (all(df.name.values == name.values))
        # filter by y
        name = pd.Series(["a", "a"])
        Hotels.filter_by_entries_limit_in_col(self.df, "y", 1)
        df = Hotels.get_df(self.df)
        self.assertTrue(df.shape == (2, 3), "should be (2,3)")
        assert (all(df.name.values == name.values))


class TestPart3Group(unittest.TestCase):
    def setUp(self):
        df_test = pd.DataFrame({"name": ["a", "a", "b", "b", "b", "a", "a", "b"],
                                "code": [ 1,   1,   2,   1,   3,   2,   3,   2],
                                "num":  [ 7,   2,   5,   1,   4,   4,   6,   5],
                                "date": [11,  11,  11,  11,  11,  11,  22,  22]})
        self.df = Hotels(df_test)

    def test_prep_df_for_clustering(self):
        # test group_and_normalize
        group = ["name", "date", "code"]
        norm_func = lambda x: 100 * (x - x.min()) / (x.max() - x.min())
        ungroup = Hotels.prep_df_for_clustering(self.df, "name", "num", group, norm_func, "norm", -1)
        # print(group)
        norm_a = pd.Series([0, 50, -1, -1, -1, 100])
        norm_b = pd.Series([0, 100, 75, -1, 100, -1])
        assert (all(ungroup.loc["a"].values == norm_a.values))
        assert (all(ungroup.loc["b"].values == norm_b.values))



if __name__ == '__main__':
    unittest.main()