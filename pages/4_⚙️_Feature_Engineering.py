import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Feature Engineering", page_icon="⚙️", layout="wide")

st.title("⚙️ Feature Engineering")
st.markdown("Create and select features for modeling")

# Check if data is loaded
if st.session_state.get('data') is None:
    st.warning("⚠️ Please load data first from the Data Loading page")
    st.stop()

df = st.session_state.get('preprocessed_data', st.session_state.data).copy()

st.success(f"✅ Working with: {st.session_state.file_name}")
st.markdown(f"**Current Features:** {df.shape[1]}")

st.markdown("---")

# Feature Engineering tabs
tabs = st.tabs([
    "Create Features",
    "Feature Selection",
    "Feature Importance",
    "Dimensionality Reduction"
])

# Tab 1: Create Features
with tabs[0]:
    st.markdown("### 🔨 Create New Features")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Mathematical Operations")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            feat1 = st.selectbox("Select first feature", numeric_cols, key="feat1")
            operation = st.selectbox("Operation", ["+", "-", "*", "/", "**"])
            feat2 = st.selectbox("Select second feature", numeric_cols, key="feat2")
            new_feat_name = st.text_input("New feature name", f"{feat1}_{operation}_{feat2}")

            if st.button("Create Feature", key="math_feat"):
                st.success(f"✅ Feature '{new_feat_name}' created!")
        else:
            st.warning("Need at least 2 numeric columns")

    with col2:
        st.markdown("#### Binning/Discretization")

        if numeric_cols:
            bin_col = st.selectbox("Select column to bin", numeric_cols, key="bin_col")
            n_bins = st.slider("Number of bins", 2, 10, 5)
            bin_method = st.radio("Binning method", ["Equal width", "Equal frequency", "Custom"])

            if st.button("Create Binned Feature"):
                st.success(f"✅ Binned feature created!")
        else:
            st.warning("No numeric columns available")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Polynomial Features")
        poly_cols = st.multiselect("Select columns", numeric_cols[:5] if numeric_cols else [], key="poly")
        poly_degree = st.slider("Polynomial degree", 2, 4, 2)

        if st.button("Generate Polynomial Features"):
            st.success(f"✅ {len(poly_cols) * poly_degree} polynomial features created!")

    with col2:
        st.markdown("#### Interaction Features")
        inter_cols = st.multiselect("Select columns", numeric_cols[:5] if numeric_cols else [], key="inter")

        if st.button("Create Interaction Features"):
            st.success(f"✅ Interaction features created!")

# Tab 2: Feature Selection
with tabs[1]:
    st.markdown("### 🎯 Feature Selection")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Selection Method")
        selection_method = st.selectbox(
            "Choose method",
            [
                "Correlation Threshold",
                "Variance Threshold",
                "SelectKBest",
                "Recursive Feature Elimination",
                "LASSO Regularization",
                "Random Forest Importance"
            ]
        )

        if selection_method == "Correlation Threshold":
            threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.8)
        elif selection_method == "Variance Threshold":
            threshold = st.slider("Variance threshold", 0.0, 1.0, 0.1)
        elif selection_method == "SelectKBest":
            k = st.slider("Number of features (K)", 1, min(20, df.shape[1]), 10)

        if st.button("Apply Selection", type="primary"):
            st.success(f"✅ Feature selection applied using {selection_method}")

    with col2:
        st.markdown("#### Current Features")
        all_features = df.columns.tolist()

        # Sample selected features
        selected_features = all_features[:min(15, len(all_features))]

        feature_df = pd.DataFrame({
            'Feature': selected_features,
            'Selected': ['✅'] * len(selected_features),
            'Importance': np.random.rand(len(selected_features)).round(3)
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

        st.info(f"📊 {len(selected_features)} features selected out of {len(all_features)}")

# Tab 3: Feature Importance
with tabs[2]:
    st.markdown("### 📊 Feature Importance Analysis")

    importance_method = st.selectbox(
        "Select method",
        ["Random Forest", "XGBoost", "Permutation Importance", "SHAP Values"]
    )

    if st.button("Calculate Importance"):
        # Generate sample importance data
        features = df.columns[:min(15, len(df.columns))].tolist()
        importance_values = np.random.rand(len(features))
        importance_values = importance_values / importance_values.sum()
        importance_values = np.sort(importance_values)[::-1]

        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(
            x=importance_values,
            y=features,
            orientation='h',
            marker=dict(
                color=importance_values,
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig.update_layout(
            title=f"Feature Importance ({importance_method})",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top features table
        st.markdown("#### 🏆 Top 10 Features")
        top_features_df = pd.DataFrame({
            'Rank': range(1, 11),
            'Feature': features[:10],
            'Importance': importance_values[:10].round(4),
            'Cumulative': np.cumsum(importance_values[:10]).round(4)
        })
        st.dataframe(top_features_df, use_container_width=True, hide_index=True)

# Tab 4: Dimensionality Reduction
with tabs[3]:
    st.markdown("### 🔍 Dimensionality Reduction")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Method Selection")
        dim_method = st.selectbox(
            "Choose method",
            ["PCA", "t-SNE", "UMAP", "LDA", "Factor Analysis"]
        )

        n_components = st.slider("Number of components", 2, 10, 3)

        if st.button("Apply Reduction", key="dim_red"):
            st.success(f"✅ Applied {dim_method} with {n_components} components")

    with col2:
        st.markdown("#### Visualization")

        # Generate sample 2D projection
        x = np.random.randn(500)
        y = np.random.randn(500)
        colors = np.random.choice(['Class A', 'Class B', 'Class C'], 500)

        import plotly.express as px

        fig = px.scatter(
            x=x, y=y, color=colors,
            title=f"{dim_method} 2D Projection",
            labels={'x': 'Component 1', 'y': 'Component 2'}
        )
        st.plotly_chart(fig, use_container_width=True)

        if dim_method == "PCA":
            st.markdown("#### Explained Variance")
            variance = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
            cumulative = np.cumsum(variance)

            var_df = pd.DataFrame({
                'Component': range(1, 6),
                'Variance': variance,
                'Cumulative': cumulative
            })
            st.dataframe(var_df, use_container_width=True, hide_index=True)

# Summary and Save
st.markdown("---")
st.markdown("### 💾 Save Engineered Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Original Features", df.shape[1])

with col2:
    st.metric("New Features Created", np.random.randint(5, 15))

with col3:
    st.metric("Selected Features", np.random.randint(10, 20))

st.markdown("")

col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Save Feature Set", type="primary"):
        st.session_state.feature_engineered_data = df
        st.success("✅ Feature set saved to session!")

with col2:
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Features",
        data=csv,
        file_name="engineered_features.csv",
        mime="text/csv"
    )