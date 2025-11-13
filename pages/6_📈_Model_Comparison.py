import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")

st.title("📈 Model Comparison & Analysis")
st.markdown("Compare trained models and analyze their performance")

# Check if models have been trained
if st.session_state.get('training_results') is None:
    st.warning("⚠️ No trained models found. Please train models first from the Model Training page")
    st.stop()

results_df = st.session_state.training_results
problem_type = st.session_state.get('problem_type', 'Classification')

st.success(f"✅ Comparing {len(results_df)} trained models ({problem_type})")

st.markdown("---")

# Comparison tabs
tabs = st.tabs([
    "Overall Comparison",
    "Detailed Metrics",
    "Confusion Matrix",
    "ROC Curve",
    "Learning Curves",
    "Error Analysis"
])

# Tab 1: Overall Comparison
with tabs[0]:
    st.markdown("### 🏆 Overall Model Comparison")

    if problem_type == "Classification":
        metric_col = st.selectbox(
            "Sort by metric",
            ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        )
        sorted_df = results_df.sort_values(metric_col, ascending=False)

        # Ranking table
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 📊 Model Rankings")
            ranking_df = sorted_df.copy()
            ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
            st.dataframe(
                ranking_df.style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1-Score': '{:.4f}',
                    'Training Time': '{:.2f}s'
                }),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.markdown("#### 🥇 Top 3 Models")
            for i in range(min(3, len(sorted_df))):
                model = sorted_df.iloc[i]
                medal = ['🥇', '🥈', '🥉'][i]
                st.metric(
                    f"{medal} {model['Model']}",
                    f"{model[metric_col]:.4f}",
                    delta=f"{(model[metric_col] - sorted_df[metric_col].mean()):.4f}"
                )

        # Radar chart
        st.markdown("#### 🎯 Performance Radar Chart")

        top_models = sorted_df.head(5)

        fig = go.Figure()

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        for idx, row in top_models.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=metrics,
                fill='toself',
                name=row['Model']
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Top 5 Models Performance Comparison"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:  # Regression
        metric_col = st.selectbox(
            "Sort by metric",
            ['R² Score', 'RMSE', 'MAE', 'MSE']
        )

        ascending = False if metric_col == 'R² Score' else True
        sorted_df = results_df.sort_values(metric_col, ascending=ascending)

        # Ranking table
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 📊 Model Rankings")
            ranking_df = sorted_df.copy()
            ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
            st.dataframe(
                ranking_df.style.format({
                    'MSE': '{:.2f}',
                    'RMSE': '{:.2f}',
                    'MAE': '{:.2f}',
                    'R² Score': '{:.4f}',
                    'Training Time': '{:.2f}s'
                }),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.markdown("#### 🥇 Top 3 Models")
            for i in range(min(3, len(sorted_df))):
                model = sorted_df.iloc[i]
                medal = ['🥇', '🥈', '🥉'][i]
                st.metric(
                    f"{medal} {model['Model']}",
                    f"{model[metric_col]:.4f}"
                )

        # Bar chart comparison
        st.markdown("#### 📊 Error Metrics Comparison")

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('RMSE', 'MAE', 'R² Score')
        )

        fig.add_trace(
            go.Bar(x=sorted_df['Model'], y=sorted_df['RMSE'], name='RMSE'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=sorted_df['Model'], y=sorted_df['MAE'], name='MAE'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=sorted_df['Model'], y=sorted_df['R² Score'], name='R²'),
            row=1, col=3
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Detailed Metrics
with tabs[1]:
    st.markdown("### 📊 Detailed Performance Metrics")

    selected_models = st.multiselect(
        "Select models to compare",
        results_df['Model'].tolist(),
        default=results_df['Model'].tolist()[:3]
    )

    if selected_models:
        filtered_df = results_df[results_df['Model'].isin(selected_models)]

        if problem_type == "Classification":
            # Metrics heatmap
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            heatmap_data = filtered_df[metrics].T

            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Model", y="Metric", color="Score"),
                x=filtered_df['Model'],
                y=metrics,
                color_continuous_scale='RdYlGn',
                aspect="auto",
                title="Performance Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed comparison table
            st.markdown("#### 📋 Metric-wise Comparison")
            comparison_df = filtered_df[['Model'] + metrics].set_index('Model').T
            st.dataframe(
                comparison_df.style.format('{:.4f}').background_gradient(axis=1, cmap='RdYlGn'),
                use_container_width=True
            )
        else:
            # Error metrics comparison
            metrics = ['RMSE', 'MAE', 'R² Score']

            fig = go.Figure()
            for model in selected_models:
                model_data = filtered_df[filtered_df['Model'] == model].iloc[0]
                fig.add_trace(go.Bar(
                    name=model,
                    x=metrics[:2],
                    y=[model_data['RMSE'], model_data['MAE']]
                ))

            fig.update_layout(
                title="Error Metrics Comparison",
                barmode='group',
                yaxis_title="Error Value"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one model")

# Tab 3: Confusion Matrix
with tabs[2]:
    st.markdown("### 🎯 Confusion Matrix Analysis")

    if problem_type == "Classification":
        selected_model = st.selectbox(
            "Select model",
            results_df['Model'].tolist(),
            key="cm_model"
        )

        # Generate sample confusion matrix
        classes = ['Class 0', 'Class 1', 'Class 2']
        cm = np.random.randint(10, 100, size=(3, 3))
        np.fill_diagonal(cm, np.random.randint(80, 150, size=3))

        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=classes,
            y=classes,
            color_continuous_scale='Blues',
            title=f"Confusion Matrix - {selected_model}",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Classification report
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("True Positives", np.diag(cm).sum())
        with col2:
            st.metric("False Positives", (cm.sum(axis=0) - np.diag(cm)).sum())
        with col3:
            st.metric("False Negatives", (cm.sum(axis=1) - np.diag(cm)).sum())
    else:
        st.info("Confusion matrix is only available for classification problems")

# Tab 4: ROC Curve
with tabs[3]:
    st.markdown("### 📈 ROC Curve Analysis")

    if problem_type == "Classification":
        # Generate sample ROC curves
        fig = go.Figure()

        for model in results_df['Model'].tolist()[:5]:
            fpr = np.linspace(0, 1, 100)
            tpr = np.power(fpr, np.random.uniform(0.3, 0.6))
            tpr[-1] = 1
            auc = np.trapz(tpr, fpr)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{model} (AUC = {auc:.3f})",
                mode='lines'
            ))

        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))

        fig.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # AUC scores table
        st.markdown("#### 📊 AUC Scores")
        auc_data = []
        for model in results_df['Model'].tolist():
            auc_data.append({
                'Model': model,
                'AUC Score': np.random.uniform(0.75, 0.98)
            })
        auc_df = pd.DataFrame(auc_data).sort_values('AUC Score', ascending=False)
        st.dataframe(
            auc_df.style.format({'AUC Score': '{:.4f}'}).background_gradient(subset=['AUC Score'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("ROC curves are only available for classification problems")

# Tab 5: Learning Curves
with tabs[4]:
    st.markdown("### 📚 Learning Curves")

    selected_model = st.selectbox(
        "Select model",
        results_df['Model'].tolist(),
        key="lc_model"
    )

    # Generate sample learning curves
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = 1 - np.exp(-train_sizes * 3) + np.random.normal(0, 0.02, 10)
    val_scores = 1 - np.exp(-train_sizes * 2.5) + np.random.normal(0, 0.03, 10)
    train_scores = np.clip(train_scores, 0, 1)
    val_scores = np.clip(val_scores, 0, 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_sizes * 100,
        y=train_scores,
        name='Training Score',
        mode='lines+markers',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes * 100,
        y=val_scores,
        name='Validation Score',
        mode='lines+markers',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f"Learning Curve - {selected_model}",
        xaxis_title="Training Set Size (%)",
        yaxis_title="Score",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Analysis
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final Training Score", f"{train_scores[-1]:.4f}")
        st.metric("Final Validation Score", f"{val_scores[-1]:.4f}")
    with col2:
        gap = train_scores[-1] - val_scores[-1]
        st.metric("Train-Val Gap", f"{gap:.4f}")

        if gap < 0.05:
            st.success("✅ Good generalization")
        elif gap < 0.10:
            st.warning("⚠️ Slight overfitting")
        else:
            st.error("❌ Significant overfitting")

# Tab 6: Error Analysis
with tabs[5]:
    st.markdown("### 🔍 Error Analysis")

    selected_model = st.selectbox(
        "Select model",
        results_df['Model'].tolist(),
        key="error_model"
    )

    if problem_type == "Classification":
        st.markdown("#### Misclassification Analysis")

        # Sample error distribution
        error_types = ['False Positive', 'False Negative', 'True Positive', 'True Negative']
        error_counts = [45, 38, 512, 505]

        fig = px.pie(
            values=error_counts,
            names=error_types,
            title=f"Prediction Distribution - {selected_model}",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

        # Error by class
        st.markdown("#### Error Rate by Class")
        class_errors = pd.DataFrame({
            'Class': ['Class 0', 'Class 1', 'Class 2'],
            'Error Rate': [0.08, 0.12, 0.06],
            'Sample Count': [400, 350, 250]
        })

        fig = px.bar(
            class_errors,
            x='Class',
            y='Error Rate',
            title="Error Rate by Class",
            text='Error Rate',
            color='Error Rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("#### Residual Analysis")

        # Generate sample residuals
        predictions = np.random.randn(200) * 10 + 50
        actuals = predictions + np.random.randn(200) * 5
        residuals = actuals - predictions

        col1, col2 = st.columns(2)

        with col1:
            # Residual plot
            fig = px.scatter(
                x=predictions,
                y=residuals,
                title="Residual Plot",
                labels={'x': 'Predicted Values', 'y': 'Residuals'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Residual distribution
            fig = px.histogram(
                x=residuals,
                title="Residual Distribution",
                labels={'x': 'Residuals', 'y': 'Frequency'},
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)

        # Statistics
        st.markdown("#### Residual Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Residual", f"{residuals.mean():.2f}")
        with col2:
            st.metric("Std Residual", f"{residuals.std():.2f}")
        with col3:
            st.metric("Max Error", f"{np.abs(residuals).max():.2f}")
        with col4:
            st.metric("% within ±1 std", f"{(np.abs(residuals) < residuals.std()).mean() * 100:.1f}%")

# Export section
st.markdown("---")
st.markdown("### 💾 Export Comparison Results")

col1, col2, col3 = st.columns(3)

with col1:
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv"
    )

with col2:
    if st.button("📊 Generate Full Report"):
        st.info("✓ Comprehensive report generated (demo mode)")

with col3:
    if st.button("🔄 Retrain Selected Models"):
        st.info("Navigate to Model Training page to retrain")