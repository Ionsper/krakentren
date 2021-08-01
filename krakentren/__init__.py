""" Kraken_trade_enhancer

A python package to interact with Kraken.com REST API

"""
from .krakentren import (round_down_decimals,
                         get_server_time,
                         get_tradable_assets_pairs,
                         get_asset_pair_info,
                         get_account_balance,
                         get_order_info,
                         order_status,
                         trade_fee,
                         trade_cost,
                         order_price,
                         order_volume,
                         cancel_order,
                         add_ta,
                         Coin
                         )
