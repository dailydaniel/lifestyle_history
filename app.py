import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib
from random import choices
import requests

import warnings
warnings.simplefilter('ignore')


st.set_page_config(
    page_title="Lifestyle History",
    page_icon="🧑‍💻",
    layout="wide",
)

key_life = '1CZqJ-z0i2yGqFr5dTOG2YOYngKx8ngVVqsnPbja95-A'
key_poopee = '1GfLQYA0g04Rjib0xMtOzwGA_vnJP0NSGuZLENI71EgI'
weight_goal = 69.0


def get_gb(df: pd.DataFrame, weight_goal: float = weight_goal) -> pd.DataFrame:
    df_gb = df.groupby(['Type', pd.Grouper(key='Date', freq='D')], as_index=False).agg(
        Kk=('Kk', 'sum'),
        Kg=('Kg', 'first'),
        Hours=('Hours', 'sum'),
    ).sort_values(by='Date')

    df_gb = df_gb.fillna(0)

    for column in ['Kk', 'Hours']:
        df_gb[column + '_num'] = df_gb[column] / df_gb[column].max()

    df_gb['Kg_num'] = df_gb['Kg'] / weight_goal

    # idx = df_gb[df_gb['Type'] == 'Sleep'].index
    # df_gb.loc[idx, 'Hours_num'] = df_gb.loc[idx, 'Hours'] / df_gb.loc[idx, 'Hours'].max()
    #
    # idx = df_gb[df_gb['Type'] == 'Train'].index
    # df_gb.loc[idx, 'Hours_num'] = df_gb.loc[idx, 'Hours'] / df_gb.loc[idx, 'Hours'].max()

    idx_col = 'Type'

    all_dates = pd.date_range(start=df_gb['Date'].min(), end=df_gb['Date'].max(), freq='D')
    all_codes = df_gb[idx_col].unique()
    full_index = pd.MultiIndex.from_product([all_dates, all_codes], names=['Date', idx_col])
    full_df = pd.DataFrame(index=full_index).reset_index()

    merged_df = pd.merge(full_df, df_gb, how='left', on=['Date', idx_col]).sort_values(by=['Date', idx_col])

    return merged_df


def h2n(s: str) -> str:
    if isinstance(s, str):
        if ':' not in s:
            return s

        h, m = s.split(':')
        m = str(int(m) / 60).split('.')[-1]

        return h + '.' + m
    else:
        return s


# @st.cache_data
def get_data(filter_period: int = None, key: str = key_life) -> pd.DataFrame:
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/' +
                     key +
                     '/export?gid=0&format=csv',
                     parse_dates=['Date'],
                     dayfirst=True)

    df = df.replace('-', np.NaN)

    df['Kk'] = df['Kk'].apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
    df['Kk'] = pd.to_numeric(df['Kk'], errors='coerce')

    df['Kg'] = df['Kg'].apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
    df['Kg'] = pd.to_numeric(df['Kg'], errors='coerce')

    df['Hours'] = df['Hours'].apply(lambda x: h2n(x))
    df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce')

    if type(filter_period) == int:
        date = pd.Timestamp.now()
        delta_date = pd.to_timedelta(f'{filter_period * 7} days')
        period_date = date - delta_date
        df = df[df['Date'].dt.date >= period_date.date()]

    return df


def get_poopee_data(filter_period: int = None, key: str = key_poopee) -> pd.DataFrame:
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/' +
                     key +
                     '/export?gid=0&format=csv',
                     parse_dates=['Date'],
                     dayfirst=True)

    if type(filter_period) == int:
        date = pd.Timestamp.now()
        delta_date = pd.to_timedelta(f'{filter_period * 7} days')
        period_date = date - delta_date
        df = df[df['Date'].dt.date >= period_date.date()]

    return df


colors = dict(matplotlib.colors.cnames.items())
hex_colors = tuple(colors.values())

st.title("Eats, Trains, Weights, Sleep")
st.markdown("Start: 2024-02-16. Logbook of my Health.")
st.markdown("Powered by google sheet and siri shortcuts.")
url_tg = "https://t.me/mandanya77"
st.markdown("made by Daniel Zholkovsky [telegram](%s)" % url_tg)
st.markdown("Version 1.2")

filter_period = st.selectbox("Select num weeks:", ["All Sync", "All", 4, 1])
# filter_period = None if filter_period == "All" else filter_period

placeholder = st.empty()

while True:
    df = get_data(filter_period=filter_period)
    df_poopee = get_poopee_data(filter_period=filter_period)

    if filter_period == "All Sync":
        df_poopee = df_poopee[df_poopee['Date'] >= df['Date'].min()]

    type2color = {'Poope': 'saddlebrown', 'Pee': 'yellowgreen'}

    with placeholder.container():
        df_gb = get_gb(df)

        kpi_list = st.columns(4)

        val = df_gb[df_gb['Type'] == 'Sleep']['Hours'].mean()
        d = val - (df_gb[df_gb['Type'] == 'Sleep']['Hours'][:-1].mean()
                   if len(df_gb[df_gb['Type'] == 'Sleep']) > 1
                   else val)
        kpi_list[0].metric(label=f"Sleep hours mean by day", value=round(val, 2), delta=round(d, 2))

        val = df_gb[df_gb['Type'] == 'Train']['Hours'].mean()
        d = val - (df_gb[df_gb['Type'] == 'Train']['Hours'][:-1].mean()
                   if len(df_gb[df_gb['Type'] == 'Train']) > 1
                   else val)
        kpi_list[1].metric(label=f"Train hours mean by day", value=round(val, 2), delta=round(d, 2))

        val = df_gb[df_gb['Type'] == 'Eat']['Kk'].dropna().mean()
        d = val - (df_gb[df_gb['Type'] == 'Eat']['Kk'][:-1].dropna().mean()
                   if len(df_gb[df_gb['Type'] == 'Eat']) > 1
                   else val)
        kpi_list[2].metric(label=f"Calories mean by day", value=int(val), delta=round(d, 2))

        val = df['Kg'].dropna().values[-1]
        d = df['Kg'].max() - df['Kg'].min()
        kpi_list[3].metric(label="current weight", value=val, delta=round(d, 2))

        col1, col2 = st.columns(2)

        with col1:
            txt = f"Health chart by {filter_period} weeks" if type(filter_period) == int else "Health chart"
            st.markdown(f"<h4 style='text-align: center;'>{txt}</h1>", unsafe_allow_html=True)

            fig1 = go.Figure(
                layout=dict(
                    xaxis=dict(categoryorder="category descending"),
                    yaxis=dict(range=[-0.05, 1.05]),
                    scattermode="group",
                    legend=dict(groupclick="toggleitem"),
                )
            )

            fig1.add_trace(
                go.Bar(
                    x=df_gb[df_gb['Type'] == 'Eat']['Date'],
                    y=df_gb[df_gb['Type'] == 'Eat']['Kk_num'],
                    name="Eat (kcal)",
                    hovertext=df_gb[df_gb['Type'] == 'Eat']['Kk'],
                    marker_color="rosybrown",
                )
            )

            fig1.add_trace(
                go.Bar(
                    x=df_gb[df_gb['Type'] == 'Sleep']['Date'],
                    y=df_gb[df_gb['Type'] == 'Sleep']['Hours_num'],
                    name="Sleep (h-s)",
                    hovertext=df_gb[df_gb['Type'] == 'Sleep']['Hours'],
                    marker_color="darkcyan",
                )
            )

            fig1.add_trace(
                go.Bar(
                    x=df_gb[df_gb['Type'] == 'Train']['Date'],
                    y=df_gb[df_gb['Type'] == 'Train']['Hours_num'],
                    name="Train (h-s)",
                    hovertext=df_gb[df_gb['Type'] == 'Train']['Hours'],
                    marker_color="sandybrown",
                )
            )

            fig1.add_trace(
                go.Scatter(
                    x=df_gb[df_gb['Type'] == 'Weight']['Date'],
                    y=df_gb[df_gb['Type'] == 'Weight']['Kg_num'],
                    name="Weight (kg)",
                    hovertext=df_gb[df_gb['Type'] == 'Weight']['Kg'],
                    mode='lines+markers',
                    marker_color="purple",
                )
            )

            fig1.add_hline(
                y=1, line_width=3,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Goal: {int(weight_goal)}kg",
            )

            fig1.update_xaxes(tickformat="%d %b")
            fig1.update_layout(
                autosize=False,
                width=550,
                height=400
            )

            st.write(fig1)

        with col2:
            txt = f"Poopee chart by {filter_period} weeks" if type(filter_period) == int else "Poopee chart"
            st.markdown(f"<h4 style='text-align: center;'>{txt}</h1>", unsafe_allow_html=True)

            df_gb_poopee = df_poopee.groupby(pd.Grouper(key='Date', freq='D'))['Type'].value_counts().reset_index()
            fig2 = px.bar(data_frame=df_gb_poopee, y='count', x='Date', color='Type')
            fig2.update_layout(legend=dict(yanchor="top", y=1.2, xanchor="left", x=0.01))
            fig2.update_layout(margin=dict(l=50, r=50))
            fig2.update_yaxes(nticks=5)

            for i in range(len(fig2.data)):
                name = fig2.data[i].name
                fig2.data[i].marker.color = type2color[name]

            fig2.update_layout(
                autosize=False,
                width=550,
                height=400
            )

            st.write(fig2)

        df_col1, df_col2 = st.columns([0.4, 0.6])

        with df_col1:
            st.markdown("<h4 style='text-align: center;'>Full Table Tail</h4>", unsafe_allow_html=True)
            st.dataframe(df.tail())

        with df_col2:
            st.markdown("<h4 style='text-align: center;'>Grouped Table Tail</h4>", unsafe_allow_html=True)
            st.dataframe(df_gb.tail())

    time.sleep(15)
