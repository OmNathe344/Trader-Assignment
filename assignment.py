#Trader Analysis Assignment

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets

trades = pd.read_csv("historical_data.csv")
sentiment = pd.read_csv("fear_greed_index.csv")

print("="*60)
print("DATASET SHAPES")
print("="*60)

print("Trades :", trades.shape)
print("Sentiment :", sentiment.shape)

# Dataset Information

print("\nTrades Columns")
print(trades.columns)

print("\nSentiment Columns")
print(sentiment.columns)

print("\nMissing Values (Trades)")
print(trades.isnull().sum())

print("\nMissing Values (Sentiment)")
print(sentiment.isnull().sum())

# Data Cleaning

trades.drop_duplicates(inplace=True)
sentiment.drop_duplicates(inplace=True)

# Convert date columns

trades["Timestamp IST"] = pd.to_datetime(
    trades["Timestamp IST"],
    format="%d-%m-%Y %H:%M"
)

sentiment["date"] = pd.to_datetime(sentiment["date"])

# Common Date column

trades["Date"] = trades["Timestamp IST"].dt.date
sentiment["Date"] = sentiment["date"].dt.date

# Merge datasets

df = pd.merge(
    trades,
    sentiment[["Date","classification","value"]],
    on="Date",
    how="left"
)

print("\nMerged Dataset Shape :", df.shape)

# Feature Engineering

df["Win"] = df["Closed PnL"] > 0

df["Loss"] = df["Closed PnL"] < 0

# Overall Statistics

print("\nOverall Statistics")

print(df.describe())

# Trades by Sentiment

trade_count = df.groupby("classification").size()

print("\nTrades by Sentiment")

print(trade_count)

# Average Profit

avg_profit = df.groupby("classification")["Closed PnL"].mean()

print("\nAverage Profit")

print(avg_profit)

# Total Profit

total_profit = df.groupby("classification")["Closed PnL"].sum()

print("\nTotal Profit")

print(total_profit)

# Win Rate

win_rate = df.groupby("classification")["Win"].mean()*100

print("\nWin Rate (%)")

print(win_rate)

# Average Trade Size

avg_size = df.groupby("classification")["Size USD"].mean()

print("\nAverage Trade Size (USD)")

print(avg_size)

# Profit by Coin

coin_profit = df.groupby("Coin")["Closed PnL"].sum()

print("\nTop 10 Coins")

print(coin_profit.sort_values(ascending=False).head(10))

# Profit by Side

side_profit = df.groupby("Side")["Closed PnL"].mean()

print("\nProfit by Side")

print(side_profit)

# Top Traders

top_traders = df.groupby("Account")["Closed PnL"].sum()

print("\nTop 10 Traders")

print(top_traders.sort_values(ascending=False).head(10))

# Save Summary

summary = pd.DataFrame({

    "Trades": trade_count,

    "Average PnL": avg_profit,

    "Total PnL": total_profit,

    "Win Rate (%)": win_rate,

    "Average Trade Size": avg_size

})

summary.to_csv("Summary_Report.csv")

print("\nSummary_Report.csv Saved")

# VISUALIZATION

sns.set_style("whitegrid")

# --------------------------------
# Chart 1
# --------------------------------

plt.figure(figsize=(8,5))

sns.countplot(data=df,x="classification")

plt.title("Market Sentiment Distribution")

plt.tight_layout()

plt.savefig("chart1_sentiment_distribution.png")

plt.show()

# --------------------------
# Chart 2
# -------------------------

plt.figure(figsize=(8,5))

avg_profit.plot(kind="bar")

plt.title("Average Profit by Sentiment")

plt.ylabel("Average Closed PnL")

plt.tight_layout()

plt.savefig("chart2_average_profit.png")

plt.show()

# --------------------
# Chart 3
# --------------------

plt.figure(figsize=(8,5))

total_profit.plot(kind="bar")

plt.title("Total Profit by Sentiment")

plt.ylabel("Total Profit")

plt.tight_layout()

plt.savefig("chart3_total_profit.png")

plt.show()

# ------------------
# Chart 4
# ------------------

plt.figure(figsize=(8,5))

win_rate.plot(kind="bar")

plt.title("Win Rate by Sentiment")

plt.ylabel("Win Rate (%)")

plt.tight_layout()

plt.savefig("chart4_winrate.png")

plt.show()

# ----------------
# Chart 5
# ----------------

plt.figure(figsize=(10,6))

sns.boxplot(
    x="classification",
    y="Closed PnL",
    data=df
)

plt.title("Closed PnL Distribution")

plt.tight_layout()

plt.savefig("chart5_boxplot.png")

plt.show()

# ----------------
# Chart 6
# --------------

plt.figure(figsize=(8,5))

plt.hist(df["Closed PnL"],bins=50)

plt.title("Closed PnL Histogram")

plt.xlabel("Closed PnL")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("chart6_histogram.png")

plt.show()

# -----------------
# Chart 7
# ----------------

numeric = df.select_dtypes(include="number")

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("chart7_heatmap.png")

plt.show()

# -------------------
# Chart 8
# ------------------

plt.figure(figsize=(10,5))

coin_profit.sort_values(ascending=False).head(10).plot(kind="bar")

plt.title("Top 10 Coins by Profit")

plt.ylabel("Total Profit")

plt.tight_layout()

plt.savefig("chart8_topcoins.png")

plt.show()

print("\nAssignment Completed Successfully!")