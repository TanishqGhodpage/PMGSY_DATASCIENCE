import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Model Training")
st.markdown("Train and evaluate machine learning models")

# Check if data is loaded
if st.session_state.get('data') is None:
    st.warning("⚠️ Please load data first from the Data Loading page")
    st.stop()

df = st.session_state.get('feature_engineered_data',
                          st.session_state.get('preprocessed_data', st.session_state.data)).copy()

st.success(f"✅ Training data ready: {df.shape[0]} samples × {df.shape[1]} features")

st.markdown("---")

# Model configuration
st.markdown("### ⚙️ Model Configuration")

col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.markdown("#### Problem Type")
    problem_type = st.radio(
        "Select problem type",
        ["Classification", "Regression", "Clustering"],
        help="Choose based on your task"
    )

    if problem_type != "Clustering":
        st.markdown("#### Target Variable")
        all_columns = df.columns.tolist()
        target = st.selectbox("Select target column", all_columns)

        st.markdown("#### Train-Test Split")
        test_size = st.slider("Test set size (%)", 10, 40, 20)
        random_state = st.number_input("Random seed", 0, 100, 42)
    else:
        st.markdown("#### Clustering Parameters")
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        random_state = st.number_input("Random seed", 0, 100, 42)

with col2:
    st.markdown("#### Select Models to Train")

    if problem_type == "Classification":
        models = {
            "Logistic Regression": st.checkbox("Logistic Regression", True, key="clf_lr"),
            "Decision Tree": st.checkbox("Decision Tree", True, key="clf_dt"),
            "Random Forest": st.checkbox("Random Forest", True, key="clf_rf"),
            "Gradient Boosting": st.checkbox("Gradient Boosting", True, key="clf_gb"),
            "XGBoost": st.checkbox("XGBoost", True, key="clf_xgb"),
            "LightGBM": st.checkbox("LightGBM", True, key="clf_lgbm"),
            "CatBoost": st.checkbox("CatBoost", False, key="clf_cat"),
            "SVM": st.checkbox("Support Vector Machine", False, key="clf_svm"),
            "K-Nearest Neighbors": st.checkbox("K-Nearest Neighbors", False, key="clf_knn"),
            "Naive Bayes": st.checkbox("Naive Bayes", False, key="clf_nb"),
            "AdaBoost": st.checkbox("AdaBoost", False, key="clf_ada"),
            "Extra Trees": st.checkbox("Extra Trees", False, key="clf_et"),
            "Neural Network": st.checkbox("Neural Network (MLP)", False, key="clf_nn")
        }
    elif problem_type == "Regression":
        models = {
            "Linear Regression": st.checkbox("Linear Regression", True, key="reg_lr"),
            "Multiple Linear Regression": st.checkbox("Multiple Linear Regression", True, key="reg_mlr"),
            "Polynomial Regression": st.checkbox("Polynomial Regression", True, key="reg_poly"),
            "Ridge Regression": st.checkbox("Ridge Regression", True, key="reg_ridge"),
            "Lasso Regression": st.checkbox("Lasso Regression", True, key="reg_lasso"),
            "ElasticNet": st.checkbox("ElasticNet", False, key="reg_elastic"),
            "Decision Tree": st.checkbox("Decision Tree", True, key="reg_dt"),
            "Random Forest": st.checkbox("Random Forest", True, key="reg_rf"),
            "Gradient Boosting": st.checkbox("Gradient Boosting", True, key="reg_gb"),
            "XGBoost": st.checkbox("XGBoost", True, key="reg_xgb"),
            "LightGBM": st.checkbox("LightGBM", True, key="reg_lgbm"),
            "CatBoost": st.checkbox("CatBoost", False, key="reg_cat"),
            "SVR": st.checkbox("Support Vector Regression", False, key="reg_svr"),
            "K-Nearest Neighbors": st.checkbox("K-Nearest Neighbors", False, key="reg_knn"),
            "Extra Trees": st.checkbox("Extra Trees", False, key="reg_et"),
            "Neural Network": st.checkbox("Neural Network (MLP)", False, key="reg_nn")
        }
    else:  # Clustering
        models = {
            "K-Means Clustering": st.checkbox("K-Means Clustering", True, key="clust_kmeans"),
            "DBSCAN Clustering": st.checkbox("DBSCAN Clustering", True, key="clust_dbscan"),
            "Agglomerative Clustering": st.checkbox("Agglomerative Clustering", True, key="clust_agg")
        }

    selected_models = [k for k, v in models.items() if v]
    st.info(f"📊 {len(selected_models)} models selected")

with col3:
    if problem_type == "Regression" and "Polynomial Regression" in selected_models:
        st.markdown("#### Polynomial Regression Settings")
        poly_degree = st.slider("Polynomial degree", 2, 5, 2, key="poly_deg")
        st.info(f"Using degree {poly_degree} for polynomial features")

    if problem_type == "Clustering":
        st.markdown("#### Clustering Visualization")
        viz_method = st.selectbox(
            "Dimensionality reduction for visualization",
            ["PCA", "t-SNE", "UMAP"],
            help="Method to reduce features to 2D for visualization"
        )

        st.markdown("#### DBSCAN Parameters")
        if "DBSCAN Clustering" in selected_models:
            eps = st.slider("eps (neighborhood size)", 0.1, 2.0, 0.5, 0.1, key="dbscan_eps")
            min_samples = st.slider("min_samples", 2, 10, 5, key="dbscan_min")

st.markdown("---")

# Advanced settings
with st.expander("🔧 Advanced Settings"):
    col1, col2, col3 = st.columns(3)

    with col1:
        if problem_type != "Clustering":
            st.markdown("**Cross-Validation**")
            use_cv = st.checkbox("Use Cross-Validation", True)
            if use_cv:
                cv_folds = st.slider("Number of folds", 3, 10, 5)

    with col2:
        if problem_type != "Clustering":
            st.markdown("**Hyperparameter Tuning**")
            use_tuning = st.checkbox("Enable Hyperparameter Tuning", False)
            if use_tuning:
                tuning_method = st.selectbox("Method", ["Grid Search", "Random Search", "Bayesian Optimization"])

    with col3:
        st.markdown("**Feature Scaling**")
        use_scaling = st.checkbox("Scale features", True)
        if use_scaling:
            scaler_type = st.selectbox("Scaler", ["Standard", "MinMax", "Robust"])

# Train button
st.markdown("---")

if st.button("🚀 Train Models", type="primary", use_container_width=True):
    if len(selected_models) == 0:
        st.error("❌ Please select at least one model to train")
    else:
        # Training simulation
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []

        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((i + 1) / len(selected_models))

            # Simulate training with random metrics
            if problem_type == "Classification":
                accuracy = np.random.uniform(0.75, 0.95)
                precision = np.random.uniform(0.70, 0.93)
                recall = np.random.uniform(0.72, 0.94)
                f1 = 2 * (precision * recall) / (precision + recall)

                results.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Training Time': np.random.uniform(0.5, 10.0)
                })
            elif problem_type == "Regression":
                mse = np.random.uniform(100, 500)
                rmse = np.sqrt(mse)
                mae = np.random.uniform(50, 200)
                r2 = np.random.uniform(0.70, 0.95)

                results.append({
                    'Model': model_name,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R² Score': r2,
                    'Training Time': np.random.uniform(0.5, 10.0)
                })
            else:  # Clustering
                silhouette = np.random.uniform(0.35, 0.75)
                calinski = np.random.uniform(100, 500)
                davies_bouldin = np.random.uniform(0.5, 2.0)

                if model_name == "DBSCAN Clustering":
                    n_clusters_found = np.random.randint(2, 8)
                    n_noise = np.random.randint(5, 50)
                else:
                    n_clusters_found = n_clusters
                    n_noise = 0

                results.append({
                    'Model': model_name,
                    'Silhouette Score': silhouette,
                    'Calinski-Harabasz': calinski,
                    'Davies-Bouldin': davies_bouldin,
                    'Clusters Found': n_clusters_found,
                    'Noise Points': n_noise,
                    'Training Time': np.random.uniform(0.5, 10.0)
                })

        progress_bar.progress(1.0)
        status_text.text("✅ Training completed!")

        st.success(f"🎉 Successfully trained {len(selected_models)} models!")

        # Store results in session
        st.session_state.training_results = pd.DataFrame(results)
        st.session_state.problem_type = problem_type

        st.markdown("---")
        st.markdown("### 📊 Training Results")

        # Display results table
        results_df = pd.DataFrame(results)

        if problem_type == "Classification":
            results_df = results_df.sort_values('Accuracy', ascending=False)
            st.dataframe(
                results_df.style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1-Score': '{:.4f}',
                    'Training Time': '{:.2f}s'
                }).background_gradient(subset=['Accuracy'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )

            # Best model
            best_model = results_df.iloc[0]
            st.success(f"🏆 Best Model: **{best_model['Model']}** with Accuracy: **{best_model['Accuracy']:.4f}**")

            # Visualization
            fig = px.bar(
                results_df,
                x='Model',
                y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                title="Model Performance Comparison",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

        elif problem_type == "Regression":
            results_df = results_df.sort_values('R² Score', ascending=False)
            st.dataframe(
                results_df.style.format({
                    'MSE': '{:.2f}',
                    'RMSE': '{:.2f}',
                    'MAE': '{:.2f}',
                    'R² Score': '{:.4f}',
                    'Training Time': '{:.2f}s'
                }).background_gradient(subset=['R² Score'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )

            # Best model
            best_model = results_df.iloc[0]
            st.success(f"🏆 Best Model: **{best_model['Model']}** with R² Score: **{best_model['R² Score']:.4f}**")

            # Visualization
            fig = px.bar(
                results_df,
                x='Model',
                y=['RMSE', 'MAE'],
                title="Model Error Comparison (Lower is Better)",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Clustering
            results_df = results_df.sort_values('Silhouette Score', ascending=False)
            st.dataframe(
                results_df.style.format({
                    'Silhouette Score': '{:.4f}',
                    'Calinski-Harabasz': '{:.2f}',
                    'Davies-Bouldin': '{:.4f}',
                    'Clusters Found': '{:.0f}',
                    'Noise Points': '{:.0f}',
                    'Training Time': '{:.2f}s'
                }).background_gradient(subset=['Silhouette Score'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )

            # Best model
            best_model = results_df.iloc[0]
            st.success(
                f"🏆 Best Model: **{best_model['Model']}** with Silhouette Score: **{best_model['Silhouette Score']:.4f}**")

            # Visualization
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    results_df,
                    x='Model',
                    y='Silhouette Score',
                    title="Silhouette Score Comparison (Higher is Better)",
                    color='Silhouette Score',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    results_df,
                    x='Model',
                    y='Davies-Bouldin',
                    title="Davies-Bouldin Index (Lower is Better)",
                    color='Davies-Bouldin',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Cluster distribution
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['Clusters Found'],
                name='Clusters Found',
                marker_color='lightblue'
            ))
            if results_df['Noise Points'].sum() > 0:
                fig3.add_trace(go.Bar(
                    x=results_df['Model'],
                    y=results_df['Noise Points'],
                    name='Noise Points',
                    marker_color='salmon'
                ))
            fig3.update_layout(title="Cluster Distribution", barmode='group')
            st.plotly_chart(fig3, use_container_width=True)

        # Training time comparison
        fig2 = px.bar(
            results_df,
            x='Model',
            y='Training Time',
            title="Training Time Comparison",
            color='Training Time',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig2, use_container_width=True)

# Display previous results if available
elif st.session_state.get('training_results') is not None:
    st.markdown("### 📊 Previous Training Results")
    st.dataframe(st.session_state.training_results, use_container_width=True, hide_index=True)

    if st.button("🔄 Clear Results"):
        st.session_state.training_results = None
        st.rerun()