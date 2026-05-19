import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

# Use correct absolute path based on your project structure
DB_PATH = Path(__file__).resolve().parent / "agentic_memory.db"

@st.cache_data
def load_extractions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM extractions", conn)
    conn.close()
    return df

st.set_page_config(page_title="Agentic Memory Viewer", layout="wide")
st.title("🧠 Agentic Memory - Extractions Viewer")

try:
    df = load_extractions()
    st.success(f"✅ Loaded {len(df)} records from `extractions` table.")

    # Sidebar filter options
    st.sidebar.header("🔍 Filter Options")

    if "patient_id" in df.columns:
        selected = st.sidebar.multiselect("Filter by Patient ID", sorted(df["patient_id"].dropna().unique()))
        if selected:
            df = df[df["patient_id"].isin(selected)]

    column = st.sidebar.selectbox("Search in column:", df.columns)
    query = st.sidebar.text_input("Search for:")
    if query:
        df = df[df[column].astype(str).str.contains(query, case=False, na=False)]

    st.dataframe(df, use_container_width=True)

    # Export button
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Filtered Data", csv, "filtered_extractions.csv", "text/csv")

except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
