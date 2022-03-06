
import pandas as pd
from currency_converter import CurrencyConverter,ECB_URL

def process_df(df):
    df['Close Date']= pd.to_datetime(df['Close Date'], format='%d/%m/%Y %H:%M:%S')#infer_datetime_format=True)
    df['Open Date']= pd.to_datetime(df['Open Date'], format='%d/%m/%Y %H:%M:%S')#infer_datetime_format=True)

    c = CurrencyConverter(ECB_URL, fallback_on_missing_rate=True,fallback_on_missing_rate_method="last_known")
    df["Profit Euro"] =  df.apply( lambda row:  c.convert(row["Profit"], 'USD', 'EUR', row["Close Date"]),  axis=1)


    return df.sort_values(by=['Close Date'],ascending=True)