""" Kraken_trade_enhancer """
from math import floor, log10
from time import time
from urllib.parse import urlencode
from sys import float_info
import multiprocessing as mp
import base64
import hashlib
import hmac
import numpy as np
import pandas as pd
import requests


def round_down_decimals(number: float, decimals: int) -> float:
    """Rounds number down to requested decimal

    Args:
        number (float): Number to round down decimals
        decimals (int): Number of decimals to keep

    Returns:
        float: Number with rounded down decimal
    """
    number = floor(number * (10 ** decimals)) / (10 ** decimals)
    return number


def contact_kraken(method,
                   parameters: dict = None,
                   public_key="",
                   private_key=""
                   ) -> dict:
    """Contacts Kraken's REST API through HTTP requests.
    For Api methods check: https://www.kraken.com/features/api

    Args:
        method (str): API method to call
        parameters (dict, optional): Dict of parameters. Defaults to {}.

    Returns:
        json: Dict with requested data
    """
    api_public = {"Time", "Assets", "AssetPairs",
                  "Ticker", "OHLC", "Depth", "Trades", "Spread"}
    api_private = {"Balance", "TradeBalance", "OpenOrders", "ClosedOrders",
                   "QueryOrders", "TradesHistory", "QueryTrades",
                   "OpenPositions", "Ledgers", "QueryLedgers", "TradeVolume",
                   "AddOrder", "CancelOrder"}
    api_domain = "https://api.kraken.com"
    if method in api_private or method == "AssetPairs":
        timeout_sec = 30
    elif method == "AddOrder" or method == "CancelOrder":
        timeout_sec = None
    else:
        timeout_sec = 10
    if method in api_public:
        api_url = api_domain + "/0/public/" + method
        try:
            api_data = requests.get(api_url, params=parameters,
                                    timeout=timeout_sec)
        except Exception as error:
            raise Exception('krakentren ApiError: ' + str(error)) from error
    elif method in api_private:
        api_method = "/0/private/" + method
        api_url = api_domain + api_method
        api_key = public_key
        api_secret = base64.b64decode(private_key)
        api_nonce = str(int(time()*1000))
        api_postdata = (urlencode(parameters) + "&nonce=" + api_nonce)
        api_postdata = api_postdata.encode('utf-8')
        api_sha256 = hashlib.sha256(
            api_nonce.encode('utf-8') + api_postdata).digest()
        api_hmacsha512 = hmac.new(api_secret, api_method.encode(
            'utf-8') + api_sha256, hashlib.sha512)
        headers = {"API-Key": api_key,
                   "API-Sign": base64.b64encode(api_hmacsha512.digest()),
                   "User-Agent": "Kraken REST API"}
        try:
            api_data = requests.post(api_url, headers=headers,
                                     data=api_postdata,
                                     timeout=timeout_sec)
        except Exception as error:
            raise Exception('krakentren ApiError: ' + str(error)) from error
    api_data = api_data.json()
    if api_data["error"] == []:
        return api_data["result"]
    else:
        raise Exception('krakentren ApiError: ' + str(api_data["error"]))


def get_server_time() -> dict:
    """Gets server time

    Returns:
        dict: Server's time
    """
    return contact_kraken("Time")


def get_tradable_assets_pairs() -> list:
    """Gets all tradable crypto pairs

    Returns:
        list: Pair names
    """
    return list(contact_kraken("AssetPairs"))


def get_asset_pair_info(pair: str) -> dict:
    """Gets selected pair's trading info

    Args:
        pair (str): Name of selected pair

    Returns:
        dict: Dict with asset's info
    """
    return contact_kraken("AssetPairs", {"pair=": pair})[pair]


def get_account_balance(public_key: str, private_key: str) -> pd.DataFrame:
    """Returns pandas dataframe of assets and estimated value

    Returns:
        Dataframe: Pandas dataframe with account's balance info
    """
    pairs = get_tradable_assets_pairs()
    account = contact_kraken("Balance",
                             {},
                             public_key,
                             private_key)
    account_dict = {}
    sum_acc = 0
    for key in account.keys():
        if key != "ZEUR":
            if key + "ZEUR" in pairs:
                pair = str(key + "ZEUR")
            elif key + "EUR" in pairs:
                pair = str(key + "EUR")
            price = contact_kraken("Ticker", {"pair": pair})[pair]["b"][0]
            amount = round_down_decimals(float(account[key]), 5)
            value = round_down_decimals(float(price) * amount, 2)
            account_dict[key + "EUR"] = {"amount": amount, "value": value}
            sum_acc += value
    account_dict["Total asssets value"] = round_down_decimals(sum_acc, 2)
    account_dict["EUR"] = round_down_decimals(float(account["ZEUR"]), 2)
    acount_df = pd.DataFrame.from_dict(account_dict)
    acount_df.at["amount", "Total asssets value"] = None
    return acount_df


def get_order_info(txid: str, public_key: str, private_key: str) -> dict:
    """Gets selected order's info

    Args:
        txid (str): Order's id

    Returns:
        dict: Dictionary with order's info
    """
    return contact_kraken("QueryOrders",
                          {"txid": txid, "trades": True},
                          public_key,
                          private_key)


def order_status(txid: str, public_key: str, private_key: str) -> str:
    """Gets selected order's status

    Args:
        txid (str): Order's id

    Returns:
        str: Order's status
    """
    return get_order_info(txid, public_key, private_key)[txid]['status']


def trade_fee(order_txid: str, public_key: str, private_key: str) -> float:
    """Gets the total fee paid for the trade associated with the
    selected order

    Args:
        order_txid (str): Order's id

    Returns:
        float: Trade's total fee
    """
    order_trade_id = get_order_info(order_txid,
                                    public_key,
                                    private_key)[order_txid]["trades"][0]
    trade = contact_kraken("QueryTrades",
                           {"txid": order_trade_id},
                           public_key,
                           private_key)
    trade_id = list(trade.keys())[0]
    fee = float(trade[trade_id]["fee"])
    return fee


def trade_cost(order_txid: str, public_key: str, private_key: str) -> float:
    """"Gets the total cost for the trade associated with the
    selected order

    Args:
        order_txid (str): Order's id

    Returns:
        float: Trade's total cost
    """
    order_trade_id = get_order_info(order_txid,
                                    public_key,
                                    private_key)[order_txid]["trades"][0]
    trade = contact_kraken("QueryTrades",
                           {"txid": order_trade_id},
                           public_key,
                           private_key)
    trade_id = list(trade.keys())[0]
    cost = float(trade[trade_id]["cost"])
    return cost


def order_price(txid: str, public_key: str, private_key: str) -> float:
    """Gets the order's price

    Args:
        txid (str): Order's id

    Returns:
        float: Order's price
    """
    return float(get_order_info(txid, public_key, private_key)[txid]['price'])


def order_volume(txid: str, public_key: str, private_key: str) -> float:
    """Gets the order's volume

    Args:
        txid (str): Order's id

    Returns:
        float: Order's volume
    """
    return float(get_order_info(txid, public_key, private_key)[txid]['vol'])


def cancel_order(txid: str, public_key: str, private_key: str):
    """Cancel an open order

    Args:
        txid (str): Order's id
    """
    contact_kraken("CancelOrder",
                   {"txid": txid},
                   public_key,
                   private_key)


def sma_indicator_data(queue, df: pd.DataFrame, period: int, column_name: str):
    """Simple Moving Average
    Adds simple moving average series to the Dataframe
    https://www.investopedia.com/terms/m/movingaverage.asp

    Args:
        queue (method): multiprocessing.queue
        df (pd.DataFrame): ohlc data Dataframe
        period (int): simple moving average period
        column_name (str): name on Dataframe column
    """
    df[column_name] = df["Close price"].rolling(window=period).mean()
    df[column_name] = df[column_name].round(2)
    queue.put(df[column_name])


def mfi_indicator_data(queue, df: pd.DataFrame, period: int, column_name: str):
    """Money Flow Index
    Adds money flow index series to the Dataframe
    https://www.investopedia.com/terms/m/mfi.asp

    Args:
        queue (method): multiprocessing.queue
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
    df["positive_money_flow"] = df[df['Period compare']
                                   != -1]['Raw money flow']
    df["positive_money_flow"].fillna(0, inplace=True)
    df["positive_money_flow"] = df["positive_money_flow"].rolling(
        window=period).sum()
    # Calculates negative money_flow for every row
    df["negative_money_flow"] = df[df['Period compare']
                                   == -1]['Raw money flow']
    df["negative_money_flow"].fillna(0, inplace=True)
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
    queue.put(df[column_name])


def psl_indicator_data(queue, df: pd.DataFrame, period: int, column_name: str):
    """Psychological Line
    Adds psychological line series to the Dataframe
    https://library.tradingtechnologies.com/trade/chrt-ti-psychological-line.html

    Args:
        queue (method): multiprocessing.queue
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
    queue.put(df[column_name])


def chop_indicator_data(queue, df: pd.DataFrame, period: int, column_name: str):
    """Choppiness Index
    Adds choppiness index series to the Dataframe
    https://www.motivewave.com/studies/choppiness_index.htm

    Args:
        queue (method): multiprocessing.queue
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
    queue.put(df[column_name])


def roc_indicator_data(queue, df: pd.DataFrame, period: int, column_name: str):
    """Price Rate Of Change Indicator
    Adds price rate of change series to the Dataframe
    https://www.investopedia.com/terms/p/pricerateofchange.asp

    Args:
        queue (method): multiprocessing.queue
        df (pd.DataFrame): ohlc data Dataframe
        period (int): price rate of change period
        column_name (str): name on Dataframe column
    """
    df[column_name] = df["Close price"] / df["Close price"].shift(period-1)
    df[column_name] = (df[column_name] * 100) - 100
    df[column_name] = df[column_name].round(2)
    queue.put(df[column_name])


def adl_indicator_data(queue, df: pd.DataFrame, column_name: str):
    """Accumulation/Distribution Indicator
    Adds accumulation/distribution indicator series to the Dataframe
    https://www.investopedia.com/terms/a/accumulationdistribution.asp

    Args:
        queue (method): multiprocessing.queue
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
    queue.put(df[column_name])


def psar_indicator_data(queue, df: pd.DataFrame, iaf: float, max_af: float, column_name: str):
    """Parabolic SAR
    Adds parabolic SAR series to the Dataframe
    https://www.investopedia.com/trading/introduction-to-parabolic-sar/

    Args:
        queue (method): multiprocessing.queue
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
    df[column_name + " trend"] = np.where(df["uptrend"] == True,
                                          "Uptrend",
                                          "downtrend")
    df.loc[0, column_name] = None
    df.loc[1, column_name] = None
    df.drop(["uptrend", "reverse"], axis=1, inplace=True)
    df[column_name] = df[column_name].round(2)
    queue.put(df[[column_name, column_name + " trend"]])


def add_ta(ohlc_data, **kwargs):
    """Adds technical data indicators to the ohlc data Dataframe

    Args:
        ohlc_data (Dataframe): ohlc data Dataframe
    """
    indicators = ['sma', 'mfi', 'psl', 'chop', 'roc', 'adl', 'psar']
    error = None
    queue = mp.Queue()
    processes = []
    for name in kwargs.items():
        if ('indicator' not in name[1]
                or name[1]['indicator'] not in indicators):
            error = name
            break
        else:
            if name[1]['indicator'] == 'sma':
                if ('period' not in name[1] or name[1]['period'] < 0):
                    error = name
                    break
                prs = mp.Process(target=sma_indicator_data,
                                 args=(queue,
                                       ohlc_data,
                                       name[1]['period'],
                                       name[0],))
                processes.append(prs)
            if name[1]['indicator'] == 'mfi':
                if ('period' not in name[1] or name[1]['period'] < 0):
                    error = name
                    break
                prs = mp.Process(target=mfi_indicator_data,
                                 args=(queue,
                                       ohlc_data,
                                       name[1]['period'],
                                       name[0],))
                processes.append(prs)
            if name[1]['indicator'] == 'psl':
                if ('period' not in name[1] or name[1]['period'] < 0):
                    error = name
                    break
                prs = mp.Process(target=psl_indicator_data,
                                 args=(queue,
                                       ohlc_data,
                                       name[1]['period'],
                                       name[0],))
                processes.append(prs)
            if name[1]['indicator'] == 'chop':
                if ('period' not in name[1] or name[1]['period'] < 0):
                    error = name
                    break
                prs = mp.Process(target=chop_indicator_data,
                                 args=(queue,
                                       ohlc_data,
                                       name[1]['period'],
                                       name[0],))
                processes.append(prs)
            if name[1]['indicator'] == 'roc':
                if ('period' not in name[1] or name[1]['period'] < 0):
                    error = name
                    break
                prs = mp.Process(target=roc_indicator_data,
                                 args=(queue,
                                       ohlc_data,
                                       name[1]['period'],
                                       name[0],))
                processes.append(prs)
            if name[1]['indicator'] == 'adl':
                prs = mp.Process(target=adl_indicator_data,
                                 args=(queue,
                                       ohlc_data,
                                       name[0],))
                processes.append(prs)
            if name[1]['indicator'] == 'psar':
                if ('af' not in name[1]
                        or 'max_af' not in name[1]
                        or name[1]['af'] < 0
                        or name[1]['max_af'] < 0
                        or name[1]['af'] > 0.08
                        or name[1]['max_af'] > 0.8):
                    error = name
                    break
                prs = mp.Process(target=psar_indicator_data,
                                 args=(queue,
                                       ohlc_data,
                                       name[1]['af'],
                                       name[1]['max_af'],
                                       name[0],))
                processes.append(prs)
    for process in processes:
        process.start()
        ta_ind = queue.get()
        if isinstance(ta_ind, pd.Series):
            ohlc_data[str(ta_ind.name)] = ta_ind
        elif isinstance(ta_ind, pd.DataFrame):
            for column in ta_ind:
                ohlc_data[str(column)] = ta_ind[column]
    for process in processes:
        process.join()
    if error != None:
        raise Exception("krakentren- add_ta -Missing "
                        "or faulty T.A. indicator elements: "
                        + str(error))


class Coin:
    """Creates trade pair objects
    """

    def __init__(self, pair: str):
        """Creates instance of treadable asset, example: XBT/EUR,
        in order to trade or receive info

        Args:
            pair (str): The coin trading pair name
        """
        self.pair = pair
        self.info = get_asset_pair_info(self.pair)

    def get_ticker_info(self) -> dict:
        """Gets the coin tick info

        Returns:
            dict: Dictionary with all ticker info
        """
        return contact_kraken("Ticker", {"pair": self.pair})[self.pair]

    def get_ohlc_data(self, interval_minutes="1", num_of_last_bars=0,) -> pd.DataFrame:
        """Gets the coin's OHLC data

        Args:
            interval_minutes (str, optional): OHLC interval.
            Defaults to "1".
            num_of_last_bars (int, optional): Bars to return
            0 value returns last 720 bars. Defaults to 0.

        Returns:
            pandas Dataframe: Pandas Dataframe with OHLC data
        """
        ohlc_data = contact_kraken("OHLC",
                                   {"pair": self.pair,
                                    "interval": str(interval_minutes)
                                    })[self.pair]
        ohlc_data = pd.DataFrame.from_dict(
            ohlc_data).astype(float).round(5)
        ohlc_data.columns = ["DateTime", "Open price", "High", "Low",
                             "Close price", "vwap", "volume", "count"]
        ohlc_data["DateTime"] = pd.to_datetime(
            ohlc_data["DateTime"], unit='s')
        return ohlc_data[-num_of_last_bars:].reset_index(drop=True)

    def place_order(self, order_details: dict, public_key: str, private_key: str) -> str:
        """Places trade orders

        Args:
            order_details (dict): Dict of order parameters.
            public_key (str): Kraken API public key
            private_key (str): Kraken API private key

        Returns:
            str: str: Order's transaction id in case of successfull placement
        """
        order_details["pair"] = self.pair
        order = contact_kraken("AddOrder",
                               order_details,
                               public_key,
                               private_key)
        if "validate" in order_details:
            return order
        else:
            return order['txid'][0]
