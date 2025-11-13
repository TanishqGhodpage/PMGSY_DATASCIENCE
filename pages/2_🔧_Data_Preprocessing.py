import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

st.set_page_config(page_title="Data Preprocessing", page_icon="🔧", layout="wide")

st.title("🔧 Data Preprocessing")
st.markdown("Clean and prepare your data for analysis")

# Check if data is loaded
if st.session_state.get('data') is None:
    st.warning("⚠️ Please load data first from the Data Loading page")
    st.stop()

df = st.session_state.data.copy()

st.success(f"✅ Working with: {st.session_state.file_name}")
st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

st.markdown("---")

# Preprocessing options
st.markdown("### 🛠️ Preprocessing Operations")

tabs = st.tabs([
    "Missing Values",
    "Duplicates",
    "Outliers",
    "Encoding",
    "Scaling",
    "Data Types"
])

# Tab 1: Missing Values
with tabs[0]:
    st.markdown("#### Handle Missing Values")

    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    if len(missing_data) > 0:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Columns with Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Percentage': (missing_data.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)

        with col2:
            st.markdown("**Handling Strategy:**")
            selected_col = st.selectbox("Select column", missing_data.index)

            strategy = st.radio(
                "Choose strategy",
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value"]
            )

            if strategy == "Fill with custom value":
                custom_value = st.text_input("Enter custom value")

            if st.button("Apply", key="missing"):
                st.info("✓ Missing value handling applied (demo mode)")
    else:
        st.success("✅ No missing values found!")

# Tab 2: Duplicates
with tabs[1]:
    st.markdown("#### Handle Duplicate Rows")

    duplicates = df.duplicated().sum()
    st.metric("Duplicate Rows Found", duplicates)

    if duplicates > 0:
        if st.button("Remove Duplicates"):
            st.info("✓ Duplicates removed (demo mode)")
    else:
        st.success("✅ No duplicates found!")

# Tab 3: Outliers
with tabs[2]:
    st.markdown("#### Detect and Handle Outliers")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Select numeric column", numeric_cols, key="outlier_col")

        col1, col2 = st.columns(2)

        with col1:
            method = st.radio(
                "Detection method",
                ["IQR Method", "Z-Score Method"]
            )

        with col2:
            action = st.radio(
                "Action",
                ["Remove outliers", "Cap outliers", "No action"]
            )

        if st.button("Analyze Outliers"):
            st.info("✓ Outlier analysis complete (demo mode)")
    else:
        st.warning("No numeric columns found")

# Tab 4: Encoding
with tabs[3]:
    st.markdown("#### Encode Categorical Variables")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        st.markdown(f"**Categorical Columns:** {len(categorical_cols)}")

        selected_cols = st.multiselect(
            "Select columns to encode",
            categorical_cols
        )

        if selected_cols:
            encoding_method = st.radio(
                "Encoding method",
                ["Label Encoding", "One-Hot Encoding", "Target Encoding"]
            )

            if st.button("Apply Encoding"):
                st.info("✓ Encoding applied (demo mode)")
    else:
        st.info("No categorical columns found")

# Tab 5: Scaling
with tabs[4]:
    st.markdown("#### Scale Numeric Features")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        selected_cols = st.multiselect(
            "Select columns to scale",
            numeric_cols,
            key="scale_cols"
        )

        if selected_cols:
            scaling_method = st.radio(
                "Scaling method",
                ["Standard Scaler (Z-score)", "Min-Max Scaler", "Robust Scaler"]
            )

            if st.button("Apply Scaling"):
                st.info("✓ Scaling applied (demo mode)")
    else:
        st.warning("No numeric columns found")

# Tab 6: Data Types
with tabs[5]:
    st.markdown("#### Convert Data Types")

    st.dataframe(
        pd.DataFrame({
            'Column': df.columns,
            'Current Type': df.dtypes.values
        }),
        use_container_width=True
    )

    col_to_convert = st.selectbox("Select column to convert", df.columns)
    new_type = st.selectbox(
        "Convert to",
        ["int", "float", "string", "datetime", "category"]
    )

    if st.button("Convert Type"):
        st.info("✓ Data type converted (demo mode)")

# Summary and Save
st.markdown("---")
st.markdown("### 💾 Save Preprocessed Data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Save to Session", type="primary"):
        st.session_state.preprocessed_data = df
        st.success("✅ Saved to session!")

with col2:
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="preprocessed_data.csv",
        mime="text/csv"
    )

with col3:
    if st.button("🔄 Reset Changes"):
        st.info("Data reset to original (demo mode)")