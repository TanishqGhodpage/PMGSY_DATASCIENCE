import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis")
st.markdown("Visualize and understand your data")

# Check if data is loaded
if st.session_state.get('data') is None:
    st.warning("⚠️ Please load data first from the Data Loading page")
    st.stop()

df = st.session_state.get('preprocessed_data', st.session_state.data).copy()

# Fix duplicate column names and save back to session state
if df.columns.duplicated().any():
    st.warning("⚠️ Duplicate column names detected. Renaming them...")
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_indices):
            if i > 0:
                cols[idx] = f"{dup}_{i}"
    df.columns = cols

    # Save the fixed dataframe back to session state
    if 'preprocessed_data' in st.session_state:
        st.session_state.preprocessed_data = df.copy()
    else:
        st.session_state.data = df.copy()

    st.success("✅ Duplicate columns renamed")
    st.rerun()  # Rerun to use the fixed dataframe

st.success(f"✅ Analyzing: {st.session_state.file_name}")
st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

st.markdown("---")

# Visualization tabs
tabs = st.tabs([
    "Distribution",
    "Correlation",
    "Relationships",
    "Categorical",
    "Advanced Plots",
    "Time Series",
    "Statistical Tests"
])

# Tab 1: Distribution Analysis
with tabs[0]:
    st.markdown("### 📈 Distribution Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        col1, col2 = st.columns([1, 3])

        with col1:
            selected_col = st.selectbox("Select column", numeric_cols, key="dist_col")
            plot_type = st.radio("Plot type", [
                "Histogram",
                "Box Plot",
                "Violin Plot",
                "Density Plot (KDE)",
                "Cumulative Distribution",
                "Combined Distribution"
            ])

        with col2:
            # Use actual data from selected column
            if selected_col in df.columns and df[selected_col].notna().sum() > 0:
                sample_data = df[selected_col].dropna().values
            else:
                st.warning("No valid data available for this column")
                sample_data = np.array([])

            if len(sample_data) > 0:
                if plot_type == "Histogram":
                    # Interactive histogram with customization
                    nbins = st.slider("Number of bins", 10, 100, 30, key="hist_bins")
                    fig = px.histogram(
                        x=sample_data,
                        nbins=nbins,
                        title=f"Distribution of {selected_col}",
                        labels={'x': selected_col, 'y': 'Frequency'},
                        marginal="box"
                    )
                    fig.update_traces(marker=dict(line=dict(width=1, color='white')))

                elif plot_type == "Box Plot":
                    # Interactive box plot with points
                    show_points = st.checkbox("Show all points", value=False, key="box_points")
                    fig = px.box(
                        y=sample_data,
                        title=f"Box Plot of {selected_col}",
                        labels={'y': selected_col},
                        points="all" if show_points else "outliers"
                    )

                elif plot_type == "Violin Plot":
                    # Violin plot with box inside
                    fig = px.violin(
                        y=sample_data,
                        title=f"Violin Plot of {selected_col}",
                        labels={'y': selected_col},
                        box=True,
                        points="all"
                    )

                elif plot_type == "Density Plot (KDE)":
                    # KDE plot
                    fig = go.Figure()
                    density = stats.gaussian_kde(sample_data)
                    xs = np.linspace(sample_data.min(), sample_data.max(), 200)
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=density(xs),
                        fill='tozeroy',
                        name='Density',
                        line=dict(color='rgb(31, 119, 180)', width=3)
                    ))
                    fig.update_layout(
                        title=f"Density Plot of {selected_col}",
                        xaxis_title=selected_col,
                        yaxis_title="Density"
                    )

                elif plot_type == "Cumulative Distribution":
                    # CDF plot
                    sorted_data = np.sort(sample_data)
                    y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=sorted_data,
                        y=y_vals,
                        mode='lines',
                        name='CDF',
                        line=dict(color='rgb(31, 119, 180)', width=2)
                    ))
                    fig.update_layout(
                        title=f"Cumulative Distribution of {selected_col}",
                        xaxis_title=selected_col,
                        yaxis_title="Cumulative Probability"
                    )

                else:  # Combined Distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=sample_data,
                        nbinsx=30,
                        name='Histogram',
                        opacity=0.7
                    ))
                    fig.add_trace(go.Box(
                        x=sample_data,
                        name='Distribution',
                        boxmean='sd'
                    ))
                    fig.update_layout(
                        title=f"Combined Distribution Plot - {selected_col}",
                        xaxis_title=selected_col,
                        showlegend=True
                    )

                st.plotly_chart(fig, use_container_width=True)

                # Statistics
                col_stats = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis', 'Q1', 'Q3',
                               'IQR'],
                    'Value': [
                        f"{len(sample_data)}",
                        f"{sample_data.mean():.2f}",
                        f"{np.median(sample_data):.2f}",
                        f"{sample_data.std():.2f}",
                        f"{sample_data.min():.2f}",
                        f"{sample_data.max():.2f}",
                        f"{pd.Series(sample_data).skew():.2f}",
                        f"{pd.Series(sample_data).kurtosis():.2f}",
                        f"{np.percentile(sample_data, 25):.2f}",
                        f"{np.percentile(sample_data, 75):.2f}",
                        f"{np.percentile(sample_data, 75) - np.percentile(sample_data, 25):.2f}"
                    ]
                })
                st.dataframe(col_stats, use_container_width=True, hide_index=True)
    else:
        st.warning("No numeric columns available")

# Tab 2: Correlation Analysis
with tabs[1]:
    st.markdown("### 🔗 Correlation Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        # Calculate actual correlation matrix
        corr_matrix = df[numeric_cols].corr()

        # Limit to reasonable number of columns for visualization
        display_cols = numeric_cols[:min(len(numeric_cols), 15)]
        corr_display = corr_matrix.loc[display_cols, display_cols]

        fig = px.imshow(
            corr_display,
            labels=dict(x="Features", y="Features", color="Correlation"),
            x=display_cols,
            y=display_cols,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            title="Correlation Heatmap",
            zmin=-1,
            zmax=1
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations
        st.markdown("#### 🔝 Top Positive Correlations")

        # Get upper triangle of correlation matrix
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        corr_pairs = corr_matrix.where(upper_triangle).stack().reset_index()
        corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']

        # Sort by absolute correlation and get top positive
        top_positive = corr_pairs[corr_pairs['Correlation'] > 0].nlargest(5, 'Correlation')

        if len(top_positive) > 0:
            st.dataframe(
                top_positive.style.format({'Correlation': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No strong positive correlations found")

        st.markdown("#### 🔻 Top Negative Correlations")
        top_negative = corr_pairs[corr_pairs['Correlation'] < 0].nsmallest(5, 'Correlation')

        if len(top_negative) > 0:
            st.dataframe(
                top_negative.style.format({'Correlation': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No strong negative correlations found")
    else:
        st.warning("Need at least 2 numeric columns for correlation analysis")

# Tab 3: Relationships
with tabs[2]:
    st.markdown("### 🔄 Feature Relationships")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="x_col")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="y_col")

        plot_type = st.radio("Plot type", ["Scatter", "Line", "3D Scatter"], horizontal=True)

        # Use actual data - create a clean dataframe
        if x_col == y_col:
            st.warning("Please select different columns for X and Y axes")
        else:
            # Ensure no duplicate columns in selection
            cols_to_use = [x_col, y_col]
            plot_df = df[cols_to_use].copy()

            # Double check for duplicates and remove
            if plot_df.columns.duplicated().any():
                plot_df = plot_df.loc[:, ~plot_df.columns.duplicated(keep='first')]

            plot_df = plot_df.dropna()

            if len(plot_df) > 0:
                if plot_type == "Scatter":
                    fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        title=f"{x_col} vs {y_col}",
                        trendline="ols"
                    )
                elif plot_type == "Line":
                    fig = px.line(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        title=f"{x_col} vs {y_col}"
                    )
                else:  # 3D Scatter
                    if len(numeric_cols) >= 3:
                        z_col = st.selectbox("Z-axis", [c for c in numeric_cols if c not in [x_col, y_col]],
                                             key="z_col")
                        plot_df_3d = df[[x_col, y_col, z_col]].copy()

                        # Remove duplicate columns if any
                        if plot_df_3d.columns.duplicated().any():
                            plot_df_3d = plot_df_3d.loc[:, ~plot_df_3d.columns.duplicated(keep='first')]

                        plot_df_3d = plot_df_3d.dropna()

                        if len(plot_df_3d) > 0:
                            fig = px.scatter_3d(
                                plot_df_3d,
                                x=x_col,
                                y=y_col,
                                z=z_col,
                                title=f"3D Relationship: {x_col}, {y_col}, {z_col}"
                            )
                        else:
                            st.warning("No valid data for 3D plot")
                            fig = None
                    else:
                        st.warning("Need at least 3 numeric columns for 3D scatter")
                        fig = None

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid data available for selected columns")
    else:
        st.warning("Need at least 2 numeric columns")

# Tab 4: Categorical Analysis
with tabs[3]:
    st.markdown("### 📊 Categorical Analysis")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_cols:
        selected_col = st.selectbox("Select categorical column", categorical_cols, key="cat_col")

        # Get actual value counts
        value_counts = df[selected_col].value_counts()

        # Limit to top categories if too many
        if len(value_counts) > 20:
            st.info(f"Showing top 20 categories out of {len(value_counts)} total")
            value_counts = value_counts.head(20)

        categories = value_counts.index.tolist()
        values = value_counts.values.tolist()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                x=categories,
                y=values,
                title=f"Count by {selected_col}",
                labels={'x': selected_col, 'y': 'Count'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                values=values,
                names=categories,
                title=f"Distribution of {selected_col}"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Show statistics
        st.markdown("#### Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Unique Values', 'Most Common', 'Most Common Count', 'Least Common', 'Least Common Count'],
            'Value': [
                len(df[selected_col].unique()),
                value_counts.index[0],
                value_counts.values[0],
                value_counts.index[-1],
                value_counts.values[-1]
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("No categorical columns available")

# Tab 5: Advanced Plots
with tabs[4]:
    st.markdown("### 🎨 Advanced Visualizations")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    adv_plot_type = st.selectbox(
        "Select visualization type",
        ["Multi-variable Comparison", "2D Density Heatmap", "Sankey Diagram", "Dumbbell Plot"]
    )

    if adv_plot_type == "Multi-variable Comparison":
        if numeric_cols:
            selected_cols = st.multiselect(
                "Select columns to compare",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))],
                key="multi_var_cols"
            )

            if selected_cols and len(selected_cols) > 0:
                fig = go.Figure()
                for col in selected_cols:
                    if col in df.columns:
                        data = df[col].dropna().values
                        if len(data) > 0:
                            fig.add_trace(go.Violin(
                                y=data,
                                name=col,
                                box_visible=True,
                                meanline_visible=True
                            ))

                fig.update_layout(
                    title="Multi-variable Distribution Comparison",
                    yaxis_title="Values",
                    showlegend=True,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one column")
        else:
            st.warning("No numeric columns available")

    elif adv_plot_type == "2D Density Heatmap":
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="heatmap_x")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, key="heatmap_y")

            # Use actual data
            valid_data = df[[x_col, y_col]].dropna()
            if len(valid_data) > 0:
                x_data = valid_data[x_col].values
                y_data = valid_data[y_col].values

                fig = go.Figure(go.Histogram2d(
                    x=x_data,
                    y=y_data,
                    colorscale='Blues',
                    showscale=True
                ))
                fig.update_layout(
                    title=f"2D Density Heatmap: {x_col} vs {y_col}",
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid data available for selected columns")
        else:
            st.warning("Need at least 2 numeric columns")

    elif adv_plot_type == "Sankey Diagram":
        st.markdown("**Sankey Diagram - Flow Between Categories**")

        if len(categorical_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                source_col = st.selectbox("Source column", categorical_cols, key="sankey_source")
            with col2:
                target_col = st.selectbox("Target column", [c for c in categorical_cols if c != source_col],
                                          key="sankey_target")

            if st.button("Generate Sankey Diagram"):
                # Create flow data from actual dataset
                sankey_data = df.groupby([source_col, target_col]).size().reset_index(name='count')
                sankey_data = sankey_data.nlargest(50, 'count')  # Limit to top 50 flows

                if len(sankey_data) > 0:
                    # Get unique sources and targets
                    source_labels = sankey_data[source_col].unique().tolist()
                    target_labels = sankey_data[target_col].unique().tolist()

                    # Create combined label list
                    all_labels = source_labels + [t for t in target_labels if t not in source_labels]
                    label_dict = {label: idx for idx, label in enumerate(all_labels)}

                    # Create source and target indices
                    source_indices = [label_dict[src] for src in sankey_data[source_col]]
                    target_indices = [label_dict[tgt] for tgt in sankey_data[target_col]]
                    values = sankey_data['count'].tolist()

                    # Create colors
                    colors = [
                        f'rgba({np.random.randint(50, 250)}, {np.random.randint(50, 250)}, {np.random.randint(50, 250)}, 0.6)'
                        for _ in range(len(all_labels))]

                    fig = go.Figure(data=[go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=all_labels,
                            color=colors
                        ),
                        link=dict(
                            source=source_indices,
                            target=target_indices,
                            value=values
                        )
                    )])

                    fig.update_layout(
                        title=f"Sankey Diagram: {source_col} → {target_col}",
                        font_size=10,
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.info(f"💡 Showing top {len(sankey_data)} flows between {source_col} and {target_col}")
                else:
                    st.warning("No data available for Sankey diagram")
        else:
            st.warning("Need at least 2 categorical columns for Sankey diagram")

    else:  # Dumbbell Plot
        st.markdown("**Dumbbell Plot - Compare Two Values**")

        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                before_col = st.selectbox("Before/Start values", numeric_cols, key="dumbbell_before")
            with col2:
                after_col = st.selectbox("After/End values", [c for c in numeric_cols if c != before_col],
                                         key="dumbbell_after")
            with col3:
                if categorical_cols:
                    category_col = st.selectbox("Category labels (optional)", ["Use Index"] + categorical_cols,
                                                key="dumbbell_cat")
                else:
                    category_col = "Use Index"

            # Get data for dumbbell plot
            plot_data = df[[before_col, after_col]].dropna()
            if category_col != "Use Index" and category_col in df.columns:
                plot_data['category'] = df.loc[plot_data.index, category_col]
            else:
                plot_data['category'] = plot_data.index.astype(str)

            # Limit to reasonable number of items
            if len(plot_data) > 20:
                st.info("Showing top 20 items based on the difference")
                plot_data['diff'] = abs(plot_data[after_col] - plot_data[before_col])
                plot_data = plot_data.nlargest(20, 'diff')

            if len(plot_data) > 0:
                fig = go.Figure()

                # Add lines connecting before and after
                for idx, row in plot_data.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row[before_col], row[after_col]],
                        y=[row['category'], row['category']],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # Add before points
                fig.add_trace(go.Scatter(
                    x=plot_data[before_col],
                    y=plot_data['category'],
                    mode='markers',
                    name=before_col,
                    marker=dict(size=12, color='red', symbol='circle'),
                    text=plot_data[before_col].round(2),
                    hovertemplate='%{y}<br>Before: %{x:.2f}<extra></extra>'
                ))

                # Add after points
                fig.add_trace(go.Scatter(
                    x=plot_data[after_col],
                    y=plot_data['category'],
                    mode='markers',
                    name=after_col,
                    marker=dict(size=12, color='green', symbol='circle'),
                    text=plot_data[after_col].round(2),
                    hovertemplate='%{y}<br>After: %{x:.2f}<extra></extra>'
                ))

                fig.update_layout(
                    title=f"Dumbbell Plot: {before_col} vs {after_col}",
                    xaxis_title="Values",
                    yaxis_title="Categories",
                    height=max(400, len(plot_data) * 30),
                    showlegend=True,
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show summary statistics
                st.markdown("#### Summary Statistics")
                summary_df = pd.DataFrame({
                    'Metric': ['Average Before', 'Average After', 'Average Change', 'Max Increase', 'Max Decrease'],
                    'Value': [
                        f"{plot_data[before_col].mean():.2f}",
                        f"{plot_data[after_col].mean():.2f}",
                        f"{(plot_data[after_col] - plot_data[before_col]).mean():.2f}",
                        f"{(plot_data[after_col] - plot_data[before_col]).max():.2f}",
                        f"{(plot_data[after_col] - plot_data[before_col]).min():.2f}"
                    ]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No valid data available for dumbbell plot")
        else:
            st.warning("Need at least 2 numeric columns for dumbbell plot")

# Tab 6: Time Series
with tabs[5]:
    st.markdown("### 📅 Time Series Analysis")

    # Check for datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Also check for columns that might be dates but stored as strings
    potential_date_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_datetime(df[col].dropna().head(100))
            potential_date_cols.append(col)
        except:
            pass

    all_date_cols = datetime_cols + potential_date_cols

    if all_date_cols:
        selected_date_col = st.selectbox("Select date column", all_date_cols, key="date_col")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_value_col = st.selectbox("Select value column", numeric_cols, key="ts_value")

            # Prepare data
            try:
                ts_data = df[[selected_date_col, selected_value_col]].copy()
                ts_data[selected_date_col] = pd.to_datetime(ts_data[selected_date_col])
                ts_data = ts_data.dropna().sort_values(selected_date_col)

                if len(ts_data) > 0:
                    fig = px.line(
                        ts_data,
                        x=selected_date_col,
                        y=selected_value_col,
                        title=f"Time Series: {selected_value_col} over {selected_date_col}",
                        labels={selected_date_col: 'Date', selected_value_col: 'Value'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Time series statistics
                    st.markdown("#### Time Series Statistics")
                    ts_stats = pd.DataFrame({
                        'Metric': ['Start Date', 'End Date', 'Number of Records', 'Mean Value', 'Trend'],
                        'Value': [
                            ts_data[selected_date_col].min().strftime('%Y-%m-%d'),
                            ts_data[selected_date_col].max().strftime('%Y-%m-%d'),
                            len(ts_data),
                            f"{ts_data[selected_value_col].mean():.2f}",
                            "Increasing" if ts_data[selected_value_col].iloc[-1] > ts_data[selected_value_col].iloc[
                                0] else "Decreasing"
                        ]
                    })
                    st.dataframe(ts_stats, use_container_width=True, hide_index=True)
                else:
                    st.warning("No valid time series data available")
            except Exception as e:
                st.error(f"Error processing time series: {str(e)}")
        else:
            st.warning("No numeric columns available for time series values")
    else:
        st.info("💡 No datetime columns detected in your dataset")

# Tab 7: Statistical Tests
with tabs[6]:
    st.markdown("### 🧪 Statistical Tests")

    test_type = st.selectbox(
        "Select test",
        ["Normality Test", "T-Test", "ANOVA", "Chi-Square Test"]
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if test_type == "Normality Test" and numeric_cols:
        selected_col = st.selectbox("Select column", numeric_cols, key="norm_test_col")

        if st.button("Run Normality Test"):
            data = df[selected_col].dropna().values
            if len(data) > 0:
                # Shapiro-Wilk test
                statistic, p_value = stats.shapiro(data)

                st.success("✅ Normality Test completed")
                st.markdown("**Results:**")
                result_df = pd.DataFrame({
                    'Test': ['Shapiro-Wilk'],
                    'Statistic': [f"{statistic:.4f}"],
                    'P-Value': [f"{p_value:.4f}"],
                    'Conclusion': ['Normal distribution' if p_value > 0.05 else 'Not normal distribution']
                })
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                # Histogram with normal curve
                fig = px.histogram(
                    x=data,
                    nbins=30,
                    title=f"Distribution of {selected_col}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid data available")

    elif test_type == "T-Test" and numeric_cols and len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            group1_col = st.selectbox("Group 1", numeric_cols, key="ttest_g1")
        with col2:
            group2_col = st.selectbox("Group 2", numeric_cols, key="ttest_g2")

        if st.button("Run T-Test"):
            data1 = df[group1_col].dropna().values
            data2 = df[group2_col].dropna().values

            if len(data1) > 0 and len(data2) > 0:
                statistic, p_value = stats.ttest_ind(data1, data2)

                st.success("✅ T-Test completed")
                result_df = pd.DataFrame({
                    'Test': ['Independent T-Test'],
                    'Statistic': [f"{statistic:.4f}"],
                    'P-Value': [f"{p_value:.4f}"],
                    'Conclusion': ['Significant difference' if p_value < 0.05 else 'No significant difference']
                })
                st.dataframe(result_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Insufficient data for T-Test")

    elif test_type == "Chi-Square Test" and categorical_cols and len(categorical_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            cat1 = st.selectbox("Categorical Variable 1", categorical_cols, key="chi_c1")
        with col2:
            cat2 = st.selectbox("Categorical Variable 2", categorical_cols, key="chi_c2")

        if st.button("Run Chi-Square Test"):
            contingency_table = pd.crosstab(df[cat1], df[cat2])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            st.success("✅ Chi-Square Test completed")
            result_df = pd.DataFrame({
                'Test': ['Chi-Square Test'],
                'Chi2 Statistic': [f"{chi2:.4f}"],
                'P-Value': [f"{p_value:.4f}"],
                'Degrees of Freedom': [dof],
                'Conclusion': ['Variables are dependent' if p_value < 0.05 else 'Variables are independent']
            })
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            st.markdown("**Contingency Table:**")
            st.dataframe(contingency_table, use_container_width=True)

    else:
        st.info("Configure test parameters and run analysis")

# Export section
st.markdown("---")
st.markdown("### 💾 Export Visualizations")

col1, col2 = st.columns(2)
with col1:
    if st.button("📊 Export All Plots as PDF"):
        st.info("Plots exported (demo mode)")
with col2:
    if st.button("📈 Generate EDA Report"):
        st.info("Report generated (demo mode)")