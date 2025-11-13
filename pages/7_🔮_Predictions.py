import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

st.title("🔮 Model Predictions")
st.markdown("Make predictions on new data using trained models")

# Check if models have been trained
if st.session_state.get('training_results') is None:
    st.warning("⚠️ No trained models found. Please train models first from the Model Training page")
    st.stop()

results_df = st.session_state.training_results
problem_type = st.session_state.get('problem_type', 'Classification')

st.success(f"✅ {len(results_df)} trained models available for predictions")

st.markdown("---")

# Model selection
st.markdown("### 🎯 Select Model for Predictions")

col1, col2 = st.columns([2, 1])

with col1:
    # Sort by best metric
    if problem_type == "Classification":
        best_model_idx = results_df['Accuracy'].idxmax()
    else:
        best_model_idx = results_df['R² Score'].idxmax()

    selected_model = st.selectbox(
        "Choose model",
        results_df['Model'].tolist(),
        index=int(best_model_idx)
    )

    model_info = results_df[results_df['Model'] == selected_model].iloc[0]

with col2:
    st.markdown("#### Model Performance")
    if problem_type == "Classification":
        st.metric("Accuracy", f"{model_info['Accuracy']:.4f}")
        st.metric("F1-Score", f"{model_info['F1-Score']:.4f}")
    else:
        st.metric("R² Score", f"{model_info['R² Score']:.4f}")
        st.metric("RMSE", f"{model_info['RMSE']:.2f}")

st.markdown("---")

# Prediction input methods
st.markdown("### 📊 Input Data for Predictions")

input_method = st.radio(
    "Select input method",
    ["Single Prediction", "Batch Prediction (Upload File)", "Use Test Set"],
    horizontal=True
)

if input_method == "Single Prediction":
    st.markdown("#### Enter Feature Values")

    # Simulate feature input form
    col1, col2, col3 = st.columns(3)

    input_data = {}

    with col1:
        input_data['Feature_1'] = st.number_input("Feature 1", value=0.0)
        input_data['Feature_2'] = st.number_input("Feature 2", value=0.0)
        input_data['Feature_3'] = st.number_input("Feature 3", value=0.0)

    with col2:
        input_data['Feature_4'] = st.number_input("Feature 4", value=0.0)
        input_data['Feature_5'] = st.selectbox("Feature 5", ["Option A", "Option B", "Option C"])
        input_data['Feature_6'] = st.slider("Feature 6", 0, 100, 50)

    with col3:
        input_data['Feature_7'] = st.number_input("Feature 7", value=0.0)
        input_data['Feature_8'] = st.checkbox("Feature 8")
        input_data['Feature_9'] = st.number_input("Feature 9", value=0.0)

    st.markdown("---")

    if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
        # Simulate prediction
        with st.spinner("Making prediction..."):
            if problem_type == "Classification":
                # Simulated classification
                classes = ["Class 0", "Class 1", "Class 2"]
                probabilities = np.random.dirichlet(np.ones(3))
                predicted_class = classes[np.argmax(probabilities)]

                st.success(f"### 🎯 Prediction: **{predicted_class}**")

                # Show probabilities
                st.markdown("#### Class Probabilities")
                prob_df = pd.DataFrame({
                    'Class': classes,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)

                fig = px.bar(
                    prob_df,
                    x='Class',
                    y='Probability',
                    title="Prediction Probabilities",
                    color='Probability',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Confidence
                confidence = probabilities.max()
                st.metric("Prediction Confidence", f"{confidence * 100:.2f}%")

                if confidence > 0.8:
                    st.success("✅ High confidence prediction")
                elif confidence > 0.6:
                    st.warning("⚠️ Moderate confidence prediction")
                else:
                    st.error("❌ Low confidence prediction")

            else:
                # Simulated regression
                predicted_value = np.random.uniform(50, 150)
                prediction_interval = np.random.uniform(5, 15)

                st.success(f"### 🎯 Predicted Value: **{predicted_value:.2f}**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", f"{predicted_value:.2f}")
                with col2:
                    st.metric("Lower Bound (95% CI)", f"{predicted_value - prediction_interval:.2f}")
                with col3:
                    st.metric("Upper Bound (95% CI)", f"{predicted_value + prediction_interval:.2f}")

                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted_value,
                    title={'text': "Predicted Value"},
                    delta={'reference': 100},
                    gauge={
                        'axis': {'range': [None, 200]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 150], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 175
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

elif input_method == "Batch Prediction (Upload File)":
    st.markdown("#### Upload Data File")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                pred_df = pd.read_csv(uploaded_file)
            else:
                pred_df = pd.read_excel(uploaded_file)

            st.success(f"✅ File uploaded: {pred_df.shape[0]} records")

            st.markdown("#### Data Preview")
            st.dataframe(pred_df.head(10), use_container_width=True)

            if st.button("🔮 Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    # Simulate predictions
                    if problem_type == "Classification":
                        predictions = np.random.choice(["Class 0", "Class 1", "Class 2"], size=len(pred_df))
                        probabilities = np.random.rand(len(pred_df))
                    else:
                        predictions = np.random.uniform(50, 150, size=len(pred_df))

                    pred_df['Prediction'] = predictions

                    if problem_type == "Classification":
                        pred_df['Confidence'] = probabilities

                    st.success("✅ Predictions completed!")

                    st.markdown("#### Results Preview")
                    st.dataframe(pred_df.head(20), use_container_width=True)

                    # Summary statistics
                    st.markdown("#### Prediction Summary")

                    if problem_type == "Classification":
                        col1, col2 = st.columns(2)

                        with col1:
                            # Class distribution
                            class_dist = pred_df['Prediction'].value_counts()
                            fig = px.pie(
                                values=class_dist.values,
                                names=class_dist.index,
                                title="Predicted Class Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # Confidence distribution
                            fig = px.histogram(
                                pred_df,
                                x='Confidence',
                                title="Prediction Confidence Distribution",
                                nbins=30
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    else:
                        col1, col2 = st.columns(2)

                        with col1:
                            # Prediction distribution
                            fig = px.histogram(
                                pred_df,
                                x='Prediction',
                                title="Prediction Distribution",
                                nbins=30
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # Statistics
                            st.markdown("**Statistics**")
                            st.metric("Mean Prediction", f"{pred_df['Prediction'].mean():.2f}")
                            st.metric("Std Deviation", f"{pred_df['Prediction'].std():.2f}")
                            st.metric("Min Prediction", f"{pred_df['Prediction'].min():.2f}")
                            st.metric("Max Prediction", f"{pred_df['Prediction'].max():.2f}")

                    # Download results
                    st.markdown("---")
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    else:
        st.info("👆 Please upload a file to make batch predictions")

else:  # Use Test Set
    st.markdown("#### Predictions on Test Set")

    st.info("Using the test set from model training")

    n_samples = st.slider("Number of samples to display", 10, 100, 20)

    if st.button("🔮 Generate Test Set Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            # Simulate test set predictions
            if problem_type == "Classification":
                actual = np.random.choice(["Class 0", "Class 1", "Class 2"], size=n_samples)
                predicted = actual.copy()
                # Add some errors
                error_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
                predicted[error_indices] = np.random.choice(["Class 0", "Class 1", "Class 2"], size=len(error_indices))

                results = pd.DataFrame({
                    'Sample ID': range(1, n_samples + 1),
                    'Actual': actual,
                    'Predicted': predicted,
                    'Correct': actual == predicted
                })

            else:
                actual = np.random.uniform(50, 150, size=n_samples)
                predicted = actual + np.random.normal(0, 10, size=n_samples)
                error = np.abs(actual - predicted)

                results = pd.DataFrame({
                    'Sample ID': range(1, n_samples + 1),
                    'Actual': actual,
                    'Predicted': predicted,
                    'Error': error,
                    'Relative Error %': (error / actual * 100)
                })

            st.success("✅ Predictions completed!")

            st.markdown("#### Results")
            st.dataframe(results, use_container_width=True, hide_index=True)

            # Metrics
            st.markdown("#### Performance Metrics")

            if problem_type == "Classification":
                accuracy = (results['Correct'].sum() / len(results)) * 100
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Accuracy", f"{accuracy:.2f}%")
                with col2:
                    st.metric("Correct Predictions", results['Correct'].sum())
                with col3:
                    st.metric("Incorrect Predictions", (~results['Correct']).sum())

            else:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    mae = results['Error'].mean()
                    st.metric("MAE", f"{mae:.2f}")
                with col2:
                    rmse = np.sqrt((results['Error'] ** 2).mean())
                    st.metric("RMSE", f"{rmse:.2f}")
                with col3:
                    mape = results['Relative Error %'].mean()
                    st.metric("MAPE", f"{mape:.2f}%")
                with col4:
                    r2 = 1 - (results['Error'] ** 2).sum() / ((results['Actual'] - results['Actual'].mean()) ** 2).sum()
                    st.metric("R²", f"{r2:.4f}")

                # Actual vs Predicted plot
                fig = px.scatter(
                    results,
                    x='Actual',
                    y='Predicted',
                    title="Actual vs Predicted Values",
                    trendline="ols"
                )
                fig.add_trace(go.Scatter(
                    x=[results['Actual'].min(), results['Actual'].max()],
                    y=[results['Actual'].min(), results['Actual'].max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                st.plotly_chart(fig, use_container_width=True)

            # Download
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Test Results",
                data=csv,
                file_name="test_predictions.csv",
                mime="text/csv"
            )