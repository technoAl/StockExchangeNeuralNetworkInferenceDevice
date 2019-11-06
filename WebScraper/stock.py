from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
from datetime import datetime


def main(code):
    start = datetime(2019, 10, 31)
    stock = get_historical_data(code, start, start, token='pk_3fc4f2751a6746f3b1cdc30763095572')
    dict = stock['2019-10-31']
    print(dict['close'] - dict['open'])

if __name__ == '__main__':
    main('TSLA')