""" Kraken_trade_enhancer """
from math import floor
from time import time
from urllib.parse import urlencode
import base64
import hashlib
import hmac
import pandas as pd
import requests
from .ta import (sma_indicator_data,
                 mfi_indicator_data,
                 psl_indicator_data,
                 chop_indicator_data,
                 roc_indicator_data,
                 adl_indicator_data,
                 psar_indicator_data)


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
                   parameters=None,
                   public_key="",
                   private_key=""
                   ) -> dict:
    """Contacts Kraken's REST API through HTTP requests.
    For Api methods check: https://docs.kraken.com/rest/

    Args:
        method (str): API method to call
        parameters (dict, optional): Dict of parameters. Defaults to {}.

    Returns:
        json: Dict with requested data
    """
    if not parameters:
        parameters = {}
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
            api_data = requests.get(api_url,
                                    params=parameters,
                                    timeout=timeout_sec)
        except Exception as error:
            raise Exception('krakentren ApiError: ' + str(error)) from error
    elif method in api_private:
        api_method = "/0/private/" + method
        api_url = api_domain + api_method
        api_key = public_key
        api_secret = base64.b64decode(private_key)
        api_nonce = str(int(time()*1000))
        api_postdata = urlencode(parameters) + "&nonce=" + api_nonce
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
    if api_data["error"]:  # If api returns error raise Exception
        raise Exception('krakentren ApiError: ' + str(api_data["error"]))
    return api_data["result"]


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
    pair = None
    for key in account.keys():
        if key != "ZEUR":
            if key + "ZEUR" in pairs:
                pair = str(key + "ZEUR")
            elif key + "EUR" in pairs:
                pair = str(key + "EUR")
            if pair is not None:
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


def add_ta(ohlc_data, **kwargs):
    """Adds technical data indicators to the ohlc data Dataframe

    Args:
        ohlc_data (Dataframe): ohlc data Dataframe
    """
    indicators = ['sma', 'mfi', 'psl', 'chop', 'roc', 'adl', 'psar']
    error = None
    for name in kwargs.items():
        if ('indicator' not in name[1]
                or name[1]['indicator'] not in indicators):
            error = name
            break
        if name[1]['indicator'] == 'sma':
            if ('period' not in name[1] or name[1]['period'] < 0):
                error = name
                break
            sma_indicator_data(ohlc_data, name[1]['period'], name[0])
        if name[1]['indicator'] == 'mfi':
            if ('period' not in name[1] or name[1]['period'] < 0):
                error = name
                break
            mfi_indicator_data(ohlc_data, name[1]['period'], name[0])
        if name[1]['indicator'] == 'psl':
            if ('period' not in name[1] or name[1]['period'] < 0):
                error = name
                break
            psl_indicator_data(ohlc_data, name[1]['period'], name[0])
        if name[1]['indicator'] == 'chop':
            if ('period' not in name[1] or name[1]['period'] < 0):
                error = name
                break
            chop_indicator_data(ohlc_data, name[1]['period'], name[0])
        if name[1]['indicator'] == 'roc':
            if ('period' not in name[1] or name[1]['period'] < 0):
                error = name
                break
            roc_indicator_data(ohlc_data, name[1]['period'], name[0])
        if name[1]['indicator'] == 'adl':
            adl_indicator_data(ohlc_data, name[0])
        if name[1]['indicator'] == 'psar':
            if ('af' not in name[1]
                    or 'max_af' not in name[1]
                    or name[1]['af'] < 0
                    or name[1]['max_af'] < 0
                    or name[1]['af'] > 0.08
                    or name[1]['max_af'] > 0.8):
                error = name
                break
            psar_indicator_data(
                ohlc_data, name[1]['af'], name[1]['max_af'], name[0])
    if error:
        raise ValueError("krakentren - add_ta -"
                         + " Problematic indicator values: "
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

    def get_ohlc_data(self, interval_minutes="1", date_conv=True,
                      since=0, num_of_last_bars=0) -> pd.DataFrame:
        """Gets the coin's OHLC data

        Args:
            interval_minutes (str, optional): OHLC interval.
            Defaults to "1".
            date_conv (optional): If True converts timestab
            Defaults to True
            since (float,optional): unix timestab to get data since
            Defaults to 0
            num_of_last_bars (int, optional): Bars to return
            0 value returns last 720 bars. Defaults to 0.

        Returns:
            pandas Dataframe: Pandas Dataframe with OHLC data
        """
        if since != 0:
            info = {"pair": self.pair, "interval": str(interval_minutes),
                    "since": str(since)}
        else:
            info = {"pair": self.pair, "interval": str(interval_minutes)}
        ohlc_data = contact_kraken("OHLC", info)[self.pair]
        ohlc_data = pd.DataFrame.from_dict(
            ohlc_data).astype(float).round(5)
        ohlc_data.columns = ["DateTime", "Open price", "High", "Low",
                             "Close price", "vwap", "volume", "count"]
        if date_conv:
            ohlc_data["DateTime"] = pd.to_datetime(
                ohlc_data["DateTime"], unit='s')
        return ohlc_data[-num_of_last_bars:].reset_index(drop=True)

    def place_order(self, order_details: dict, public_key: str, private_key: str) -> dict:
        """Places trade orders

        Args:
            order_details (dict): Dict of order parameters.
            public_key (str): Kraken API public key
            private_key (str): Kraken API private key

        Returns:
            dict: Order's transaction id in case of successfull placement
        """
        order_details["pair"] = self.pair
        order = contact_kraken("AddOrder",
                               order_details,
                               public_key,
                               private_key)
        if "validate" in order_details:
            return order
        return order['txid'][0]
