#purely for testing purposes

from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
from datetime import datetime


def getStock(code, year, month, day):
    start = datetime(year, month, day)
    end = datetime(year, month, day+1)

    stock = get_historical_data(code, start, end, token='pk_3fc4f2751a6746f3b1cdc30763095572')
    print(stock)
    dict = stock[str(year) + '-' + str(month) + '-' + str(day)]
    print(dict['close'] - dict['open'])

if __name__ == '__init__':
    getStock('TSLA', 2019, 10, 10)