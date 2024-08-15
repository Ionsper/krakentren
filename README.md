# Kraken trade enhancer (krakentren)

A python package to interact with Kraken.com REST API.
(<https://docs.kraken.com/rest/>)
***

## Use

The package contacts Kraken's REST API in order to get market data and trade.

It can also add several trading tecnhical indicators to the ohlc data received by the api.

***

## Dependencies - Installation

I recommend using a virtual environment [(like venv)](https://docs.python.org/3/library/venv.html) to install/run the package.

To install, type:

 ```console
test@something:~$ pip install krakentren
```

***

## Examples

```python
import krakentren as kr

# Get all tradable crypto pairs
kr.get_tradable_assets_pairs()

# Create trading pair
pair = kr.Coin("XXBTZEUR")

# Get ohlc data for created pair (as a pandas dataframe)
ohlc = pair.get_ohlc_data()

# Add ta indicators to pandas dataframe of ohlc data
kr.add_ta(ohlc,
          sma1={'indicator': 'sma', 'period': 9},
          sma2={'indicator': 'sma', 'period': 25},
          mfi={'indicator': 'mfi', 'period': 8},
          psl={'indicator': 'psl', 'period': 10},
          chop={'indicator': 'chop', 'period': 15},
          adl={'indicator': 'adl'},
          roc={'indicator': 'roc', 'period': 10},
          psar={'indicator': 'psar', 'af': 0.035, 'max_af': 0.035*10}
          )

# Place order ("validate": True is testing/dummy order)
order = pair.place_order({"type": "buy",
                          "ordertype": "market",
                          "volume": 0.25,
                          "oflags": "fciq",
                          "validate": True},
                         "API public key",
                         "API private key"
                         )

# Get account balance info
kr.get_account_balance("API public key", "API private key")

```
***
## Disclaimer - Warning

This is a work in progress. Use at your own risk.

### The developer of the application is not in any way affiliated with Kraken exchange or Payward Inc
