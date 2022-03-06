import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objects as go
import numpy.polynomial.polynomial as poly
import matplotlib.dates as mdates
import plotly.io as pio
# pio.renderers.default = "browser"

palette = sns.color_palette("Set2").as_hex()
palette_desat = sns.color_palette("Set2",desat=0.1).as_hex()

def ema(values, period):
    values = np.array(values)
    return pd.ewma(values, span=period)[-1]



def plot_profit(df, period = 5, colname="Profit"):
    """
    Plot for each PI the profit on the colosed trades given by tge dataframe df.

    Args:
        df (pandas.DataFrame): Pandas DataFrame with the information from the eToro Account statement. This dataframe is assumed to be sorted by closed positions.


    Returns:
        plotly.graph_objects.Figure: Returns the pointer to the current figure.
    """
     


    fig = go.Figure()
    annotations = {}
    buttons = []
    N_METRICS = 3

    vecLen = len(df['Copied From'].unique()) * N_METRICS + 2


    for idx, PI in enumerate(df['Copied From'].unique()):
        close_date = df['Close Date'].loc[df['Copied From']== PI].reset_index(drop=True)
        profit = df[colname].loc[df['Copied From']==PI].cumsum().reset_index(drop=True)
        PICap = PI.capitalize()
        visVec = np.full(vecLen, False)
        visVec[idx * N_METRICS : idx * N_METRICS + N_METRICS] = True
        # print(visVec)
        x = mdates.date2num(close_date)
        # xlinspace = np.linspace(x[0], x[-1], num=len(x))
        date_range = pd.date_range(min(close_date), max(close_date), periods=len(close_date))
        date_num = mdates.date2num(date_range)
        coefs = poly.polyfit(x, profit.values, 10)
        ffit = poly.polyval(date_num, coefs)
        # lema = ema(profit, date_num)


        fig.add_trace(
            go.Scatter( x=close_date,
                        y=profit,
                        name=PICap,
                        line=dict(color=palette[idx])))
        fig.add_trace(
            go.Scatter( x=close_date,
                        y=[profit.mean()] * len(close_date.index),
                        name= PICap + " Average",
                        visible=False,
                        line=dict(color=palette[idx], dash="dash")))
        fig.add_trace(
            go.Scatter( x=date_range,
                        # y=ffit,
                        y = profit.ewm(span = period, min_periods=1, adjust = False).mean(),
                        name= PICap + " Fit",
                        visible=False,
                        line=dict(color=palette_desat[idx])))

        annotations[PI] = [dict(x=close_date[[0]].item(),
                                y=profit.mean(),
                                xref="x", yref="y",
                                text= "Average:<br> %.3f" % profit.mean(),
                                ax=0, ay=-40),
                        dict(x=close_date[[profit.idxmax()]].item(),
                                y=profit.max(),
                                xref="x", yref="y",
                                text="Max:<br> %.3f" % profit.max(),
                                ax=0, ay=-40)]
        buttons.append(
            dict(label=PICap,
                method="update",
                args=[{"visible": visVec},
                    {"title": PICap,
                    "annotations": annotations[PI]}])
        )




## ALL

    visVec = np.full(vecLen, False)
    visVec[:-2:N_METRICS] = True
    buttons.append(
        dict(label='All',
                method="update",
                args=[{"visible": visVec},
                    {"title": 'All',
                    "annotations": None}])
    )


## TOTAL 

    close_date = df['Close Date']
    profit = df[colname].cumsum()
    PICap = PI.capitalize()
    x = mdates.date2num(close_date)
    date_range = pd.date_range(min(close_date), max(close_date), periods=len(close_date))
    date_num = mdates.date2num(date_range)
    coefs = poly.polyfit(x, profit.values, 10)
    ffit = poly.polyval(date_num, coefs)

    fig.add_trace(
            go.Scatter( x=df['Close Date'],
                        y=df[colname].cumsum(),
                        name='Total',
                        visible=False,
                        line=dict(color=palette[0])))
    
    fig.add_trace(
            go.Scatter( x=date_range,
                        y=ffit,
                        name= "Total Fit",
                        visible=False,
                        line=dict(color=palette_desat[idx])))

    annotations['Total'] = [dict(text= "Total profit:<br> %.3f" % df[colname].sum(), showarrow=False)]
    visVec = np.full(vecLen, False)
    visVec[-2:] = True


    buttons.append(
        dict(label='Total',
                method="update",
                args=[{"visible": visVec},
                    {"title": 'Total',
                    "annotations": annotations['Total']}])
    )

    fig.update_layout(
        updatemenus=[
            dict(
                active=vecLen ,
                buttons=buttons
            )
        ])

    # Set title
    fig.update_layout(title_text="eToro Profits")

    # fig.show()

    return fig

