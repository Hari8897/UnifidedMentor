import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Ferry Operational Efficiency Dashboard")

# Load data
uploaded_file = st.sidebar.file_uploader("Upload Ferry Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload the dataset")
    st.stop()
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sidebar
st.sidebar.header("Filters")

min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()

start = st.sidebar.date_input("Start Date", min_date)
end = st.sidebar.date_input("End Date", max_date)

run_analysis = st.sidebar.button("Run Analysis")

# Only run when button clicked
if run_analysis:

    mask = (df['Timestamp'].dt.date >= start) & (df['Timestamp'].dt.date <= end)

    df = df.loc[mask]

    if df.empty:
        st.warning("No data available for selected date range.")
        st.stop()

    # Capacity settings
    capacity = 400

    df['Utilization'] = df['Redemption Count'] / capacity
    df['Idle_Capacity'] = capacity - df['Redemption Count']

    # Time features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day_name()
    df['Month'] = df['Timestamp'].dt.month

    # KPIs
    avg_utilization = df['Utilization'].mean()
    idle_capacity = df['Idle_Capacity'].mean()
    total_passengers = df['Redemption Count'].sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Capacity Utilization", round(avg_utilization, 2))
    col2.metric("Idle Capacity", round(idle_capacity, 2))
    col3.metric("Passengers", int(total_passengers))

    # Timeline
    st.subheader("Capacity Utilization Timeline")
    fig, ax = plt.subplots(figsize=(8,5))
    

    ax.plot(df['Timestamp'], df['Utilization'])

    ax.set_xlabel("Time")
    ax.set_ylabel("Utilization")

    st.pyplot(fig)

    # Heatmap
    st.subheader("Congestion Heatmap")

    heatmap = df.pivot_table(
        values='Redemption Count',
        index='Day',
        columns='Hour',
        aggfunc='mean'
    )

    if heatmap.empty:
        st.info("Not enough data to display heatmap.")
    else:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(heatmap, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Seasonal analysis
    summer = df[df['Month'].isin([6,7,8])]
    winter = df[df['Month'].isin([12,1,2])]

    st.write("Summer Avg Demand:", summer['Redemption Count'].mean())
    st.write("Winter Avg Demand:", winter['Redemption Count'].mean())

    # Congestion detection
    threshold = df['Redemption Count'].quantile(0.9)

    congestion = df[df['Redemption Count'] > threshold]

    st.warning(f"{len(congestion)} congestion intervals detected")