import re
from typing import Dict, List
import numpy as np
import pandas as pd


"""
To do:
    - transform generalisation
    - saving method
    - optimize the fit
    - fit_transform methods
    - test with pipeline
    - apply memory reduction function
"""


class SchemaBuilder(object):

    __slots__ = [
        "data",
        "featNum",
        "featBin",
        "featCat",
        "featDate",
        "featID",
        "featUnk",
    ]

    def __init__(self):
        """
        Schema_builder takes a dataframe and returns 6 lists
        - featNum: Returns numeric features
        - featCat: Returns categorical features
        - featBin: Returns binary features
        - featDate: Returns date features
        - featID: Returns id features
        - featUnk: Returns features that don't fit anywhere
        """

    @staticmethod
    def is_binary(series: List, allow_na: bool = False) -> bool:
        """
        Test for binary series
        """
        if allow_na:
            series.replace(" ", np.nan, inplace=True)
            series.replace("", np.nan, inplace=True)
            series.replace(None, np.nan, inplace=True)
            series.dropna(inplace=True)
        return sorted(series.unique()) == [0, 1]

    @staticmethod
    def is_continuous(series: List) -> bool:
        """
        Test for continuous series
        """

        if series.dtype in [
            np.int16,
            np.int32,
            np.int64,
            np.float16,
            np.float32,
            np.float64,
            int,
            float,
        ]:
            if (
                len(series.astype(int).unique()) / len(series) == 1
                or "id" == series.name.lower()
            ):
                return False

            elif sorted(series.unique()) == [0, 1]:
                return False
            elif len(series.unique()) == 1:
                return False

            else:
                return True
        else:

            return False

    @staticmethod
    def is_categoric(series: List) -> bool:
        """
        Test for categoric series
        """
        if series.dtype == str or series.dtype == np.object:
            try:
                if (
                    int(re.split(r"[^\w\s]", series[0])[0]) >= 1900
                    and len(re.split(r"[^\w\s]", series[0])) >= 3
                ):
                    return False
                else:
                    return True
            except:
                if (
                    len(series.unique()) / len(series) == 1
                    or "id" in series.name.lower()
                ):
                    return False
                elif (
                    True in series.unique().tolist()
                    and False in series.unique().tolist()
                ):
                    return False
                elif (
                    "True" in series.unique().tolist()
                    and "False" in series.unique().tolist()
                ):
                    return False
                else:
                    return True
        else:
            return False

    @staticmethod
    def is_date(series: List) -> bool:
        """
        Test for date
        """
        try:

            if (
                int(re.split(r"[^\w\s]", series[0])[0]) >= 1900
                and len(re.split(r"[^\w\s]", series[0])) >= 3
            ):

                return True

            elif (
                series.dtype == pd._libs.tslibs.timestamps.Timestamp
                or series.dtype == "<M8[ns]"
            ):

                return True

            else:
                return False
        except:
            return False

    @staticmethod
    def is_id(series: List) -> bool:
        """
        Test for id columns
        """

        return (
            len(series.astype(int).unique()) / len(series) == 1
            or "id" == series.name.lower()
        )

    @property
    def show_featNum(self):
        """
        Show Continuous features
        """
        try:
            return self.featNum
        except:
            return None

    def fit(self, data: Dict):
        """
        Build the schema
        """
        self.featNum = list(
            data.apply(self.is_continuous)[data.apply(self.is_continuous) == True].index
        )
        self.featBin = list(
            data.apply(self.is_binary)[data.apply(self.is_binary) == True].index
        )
        self.featCat = list(
            data.apply(self.is_categoric)[data.apply(self.is_categoric) == True].index
        )
        self.featDate = list(
            data.apply(self.is_date)[data.apply(self.is_date) == True].index
        )
        self.featID = list(data.apply(self.is_id)[data.apply(self.is_id) == True].index)
        self.featUnk = list(
            set(data.columns.tolist())
            - set(self.featNum)
            - set(self.featBin)
            - set(self.featCat)
            - set(self.featID)
        )

    def transform(self, data: Dict) -> Dict:

        """
        Applies the schema to a new dataset
        """

        for c in data.columns:
            if c in self.featBin:
                data[c] = data[c].astype(int)
                if data[c].max() > 1:
                    data.loc[data[c] > 1, c] = 1
                elif data[c].min() < 0:
                    data.loc[data[c] < 0] = 1
                else:
                    pass
            elif c in self.featNum:
                data[c] = np.abs(data[c])

            else:
                pass

        return data
