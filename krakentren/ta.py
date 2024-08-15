"""Technical analysis toolbox"""
from math import log10
from sys import float_info

import numpy as np
import pandas as pd


def sma_indicator_data(df: pd.DataFrame, period: int, column_name: str):
    """Simple Moving Average
    Adds simple moving average series to the Dataframe
    https://www.investopedia.com/terms/m/movingaverage.asp

    Args:
        df (pd.DataFrame): ohlc data Dataframe
        period (int): simple moving average period
        column_name (str): name on Dataframe column
    """
    df[column_name] = df["Close price"].rolling(window=period).mean()
    df[column_name] = df[column_name].round(2)


def mfi_indicator_data(df: pd.DataFrame, period: int, column_name: str):
    """Money Flow Index
    Adds money flow index series to the Dataframe
    https://www.investopedia.com/terms/m/mfi.asp

    Args:
        df (pd.DataFrame): ohlc data Dataframe
        period (int): money flow index period
        column_name (str): name on Dataframe column
    """
    # Calculates Typical price for every row
    df["Typical price"] = (df["High"] + df["Low"] + df["Close price"]) / 3
    # Calculates if period has positive or negative flow
    # Positive = 1 , Negative = -1
    df["Period compare"] = np.where(
        df['Typical price'] > df["Typical price"].shift(1), 1, -1)
    # Calculates Raw money flow for every row
    df["Raw money flow"] = df["Typical price"] * df["volume"]
    # Calculates positive money_flow for every row
    df["positive_money_flow"] = df[df['Period compare'] != -1]['Raw money flow']
    df.fillna({"positive_money_flow": 0}, inplace=True)
    df["positive_money_flow"] = df["positive_money_flow"].rolling(
        window=period).sum()
    # Calculates negative money_flow for every row
    df["negative_money_flow"] = df[df['Period compare']
                                   == -1]['Raw money flow']
    df.fillna({"negative_money_flow": 0}, inplace=True)
    df["negative_money_flow"] = df["negative_money_flow"].rolling(
        window=period).sum()
    # Calculates money flow ratio for every row
    df["money_flow_ratio"] = (df["positive_money_flow"]
                              / df["negative_money_flow"])
    # Calculates money flow index for every row
    df[column_name] = 100 - (100/(1 + df["money_flow_ratio"]))
    df.drop(["Typical price", "Raw money flow",
             "Period compare", "positive_money_flow",
             "negative_money_flow", "money_flow_ratio"], axis=1, inplace=True)
    df[column_name] = df[column_name].shift(1)  # Remove most recent value
    df[column_name] = df[column_name].round(2)


def psl_indicator_data(df: pd.DataFrame, period: int, column_name: str):
    """Psychological Line
    Adds psychological line series to the Dataframe
    https://library.tradingtechnologies.com/trade/chrt-ti-psychological-line.html

    Args:
        df (pd.DataFrame): ohlc data Dataframe
        period (int): psychological line period
        column_name (str): name on Dataframe column
    """
    df["bar_price_compare"] = np.where(
        df["Close price"] > df["Close price"].shift(1), 1, 0)
    df[column_name] = (df["bar_price_compare"].rolling(
        window=period).sum() / period) * 100
    df.drop("bar_price_compare", axis=1, inplace=True)
    df[column_name] = df[column_name].round(2)


def chop_indicator_data(df: pd.DataFrame, period: int, column_name: str):
    """Choppiness Index
    Adds choppiness index series to the Dataframe
    https://www.motivewave.com/studies/choppiness_index.htm

    Args:
        df (pd.DataFrame): ohlc data Dataframe
        period (int): choppiness index period
        column_name (str): name on Dataframe column
    """
    # calculate True range
    df["tr1"] = df["High"] - df["Low"]
    df["tr2"] = abs(df["High"] - df["Close price"].shift(1))
    df["tr3"] = abs(df["Low"] - df["Close price"].shift(1))
    df["True range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    # calculate Choppiness index
    df["TR SUM"] = df["True range"].rolling(window=period).sum()
    df["CHOP1"] = df["TR SUM"] / (df["High"].rolling(window=period).max()
                                  - df["Low"].rolling(window=period).min())
    df["CHOP1"] = df["CHOP1"].apply(log10)
    df[column_name] = (df["CHOP1"] / log10(period)) * 100
    df.drop(["tr1", "tr2", "tr3", "TR SUM", "CHOP1",
            "True range"], axis=1, inplace=True)
    df[column_name] = df[column_name].round(2)


def roc_indicator_data(df: pd.DataFrame, period: int, column_name: str):
    """Price Rate Of Change Indicator
    Adds price rate of change series to the Dataframe
    https://www.investopedia.com/terms/p/pricerateofchange.asp

    Args:
        df (pd.DataFrame): ohlc data Dataframe
        period (int): price rate of change period
        column_name (str): name on Dataframe column
    """
    df[column_name] = df["Close price"] / df["Close price"].shift(period-1)
    df[column_name] = (df[column_name] * 100) - 100
    df[column_name] = df[column_name].round(2)


def adl_indicator_data(df: pd.DataFrame, column_name: str):
    """Accumulation/Distribution Indicator
    Adds accumulation/distribution indicator series to the Dataframe
    https://www.investopedia.com/terms/a/accumulationdistribution.asp

    Args:
        df (pd.DataFrame): ohlc data Dataframe
        column_name (str): name on Dataframe column
    """

    df[column_name] = ((df["Close price"] - df["Low"])
                       - (df["High"] - df["Close price"]))
    # To avoid dividing by zero
    df["diff"] = np.where((df["High"] - df["Low"]) == 0,
                          float_info.epsilon, (df["High"] - df["Low"]))
    df[column_name] = df[column_name] / df["diff"]
    df[column_name] = (df[column_name] * df["volume"]).cumsum()
    df[column_name] = df[column_name].round(3)
    df.drop(["diff"], axis=1, inplace=True)


def psar_indicator_data(df: pd.DataFrame, iaf: float, max_af: float, column_name: str):
    """Parabolic SAR
    Adds parabolic SAR series to the Dataframe
    https://www.investopedia.com/trading/introduction-to-parabolic-sar/

    Args:
        df (pd.DataFrame): ohlc data Dataframe
        iaf (float): initial value
        max_af (float): max af value
        column_name (str): name on Dataframe column
    """
    af = iaf
    df["uptrend"] = None
    df["reverse"] = False
    df.loc[1, "uptrend"] = True
    df[column_name] = df["Close price"]
    uptrend_high = df["High"][0]
    downtrend_low = df["Low"][0]
    for row in range(2, len(df)):
        if df["uptrend"][row-1]:  # Calculate uptrend psar
            df.loc[row, column_name] = (df[column_name][row-1]
                                        + af
                                        * (uptrend_high - df[column_name][row-1]))
            # Check trend reverse status
            df.loc[row, "uptrend"] = True
            if df["Low"][row] < df[column_name][row]:  # Uptrend stop/reverse
                df.loc[row, "uptrend"] = False
                df.loc[row, column_name] = uptrend_high
                downtrend_low = df["Low"][row]
                df.loc[row, "reverse"] = True
                af = iaf
            if not df["reverse"][row]:  # If not reverse is occuring
                if df["High"][row] > uptrend_high:
                    uptrend_high = df["High"][row]
                    af = min(af + iaf, max_af)
                if df["Low"][row-1] < df[column_name][row]:
                    df.loc[row, column_name] = df["Low"][row-1]
                if df["Low"][row-2] < df[column_name][row]:
                    df.loc[row, column_name] = df["Low"][row-2]
        else:  # Calculate downtrend psar
            df.loc[row, column_name] = (df[column_name][row-1]
                                        + af
                                        * (downtrend_low - df[column_name][row-1]))
            # Check trend reverse status
            df.loc[row, "uptrend"] = False
            if df["High"][row] > df[column_name][row]:  # Downtrend stop/reverse
                df.loc[row, "uptrend"] = True
                df.loc[row, column_name] = downtrend_low
                uptrend_high = df["High"][row]
                df.loc[row, "reverse"] = True
                af = iaf
            if not df["reverse"][row]:  # If not reverse is occuring
                if df["Low"][row] < downtrend_low:
                    downtrend_low = df["Low"][row]
                    af = min(af + iaf, max_af)
                if df["High"][row-1] > df[column_name][row]:
                    df.loc[row, column_name] = df["High"][row-1]
                if df["High"][row-2] > df[column_name][row]:
                    df.loc[row, column_name] = df["High"][row-2]
    df[column_name + " trend"] = np.where(df["uptrend"] is True,
                                          "Uptrend",
                                          "downtrend")
    df.loc[0, column_name] = None
    df.loc[1, column_name] = None
    df.drop(["uptrend", "reverse"], axis=1, inplace=True)
    df[column_name] = df[column_name].round(2)
