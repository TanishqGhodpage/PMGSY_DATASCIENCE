import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Loading", page_icon="📁", layout="wide")

st.title("📁 Data Loading")
st.markdown("Upload and explore your PMGSY dataset")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

# File uploader
st.markdown("### Upload Dataset")
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your PMGSY dataset in CSV or Excel format"
)

if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.data = df
        st.session_state.file_name = uploaded_file.name

        st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")

        # Dataset Overview
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        with col4:
            st.metric("Duplicate Rows", df.duplicated().sum())

        # Display sample data
        st.markdown("### 🔍 Data Preview")
        tab1, tab2, tab3 = st.tabs(["First Rows", "Last Rows", "Random Sample"])

        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with tab2:
            st.dataframe(df.tail(10), use_container_width=True)
        with tab3:
            st.dataframe(df.sample(min(10, len(df))), use_container_width=True)

        # Column Information
        st.markdown("### 📋 Column Information")
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2).values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

        # Basic Statistics
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        # Download section
        st.markdown("---")
        st.markdown("### 💾 Export Data Info")
        col1, col2 = st.columns(2)

        with col1:
            csv = col_info.to_csv(index=False)
            st.download_button(
                label="📥 Download Column Info",
                data=csv,
                file_name="column_info.csv",
                mime="text/csv"
            )

        with col2:
            csv_stats = df.describe().to_csv()
            st.download_button(
                label="📥 Download Statistics",
                data=csv_stats,
                file_name="statistics.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

elif st.session_state.data is not None:
    st.info(f"📂 Currently loaded: {st.session_state.file_name}")
    st.markdown(f"**Records:** {st.session_state.data.shape[0]} | **Features:** {st.session_state.data.shape[1]}")

    if st.button("🗑️ Clear Data"):
        st.session_state.data = None
        st.session_state.file_name = None
        st.rerun()
else:
    st.info("👆 Please upload a dataset to begin")

    # Sample dataset format guide
    st.markdown("### 📝 Expected Dataset Format")
    st.markdown("""
    Your PMGSY dataset should contain columns such as:
    - State/District information
    - Road specifications (length, type, etc.)
    - Budget and expenditure details
    - Timeline information
    - Completion status
    - And other relevant features
    """)