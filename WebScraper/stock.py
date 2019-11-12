from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
from datetime import datetime


def getStock(code, year, month, day):
    start = datetime(year, month, day)
    stock = get_historical_data(code, start, start, token='pk_3fc4f2751a6746f3b1cdc30763095572')
    dict = stock[year + '-' + moneth + '-' + day]
    print(dict['close'] - dict['open'])

