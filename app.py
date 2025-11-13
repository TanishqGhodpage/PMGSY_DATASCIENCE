import streamlit as st

# Page configuration
st.set_page_config(
    page_title="PMGSY Data Analysis",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">🛣️ PMGSY Data Analysis Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pradhan Mantri Gram Sadak Yojana - Comprehensive ML Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ### Welcome to the PMGSY Analysis Dashboard
    
    This application provides a complete pipeline for analyzing PMGSY (Pradhan Mantri Gram Sadak Yojana) data, 
    from data loading to advanced machine learning predictions.
    """)
    
    # Features Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📊 Data Management</h3>
            <ul>
                <li>Load CSV/Excel files</li>
                <li>View data statistics</li>
                <li>Handle missing values</li>
                <li>Data type conversions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🔍 EDA & Visualization</h3>
            <ul>
                <li>Statistical summaries</li>
                <li>Distribution plots</li>
                <li>Correlation analysis</li>
                <li>Interactive charts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 ML Models</h3>
            <ul>
                <li>10+ ML algorithms</li>
                <li>Model comparison</li>
                <li>Performance metrics</li>
                <li>Predictions & exports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Guide
    st.markdown("### 📍 Navigation Guide")
    
    st.info("""
    **Use the sidebar** to navigate between different sections:
    
    1. **Data Loading** - Upload and explore your PMGSY dataset
    2. **Data Preprocessing** - Clean and prepare your data
    3. **Exploratory Data Analysis** - Visualize patterns and insights
    4. **Feature Engineering** - Create and select features
    5. **Model Training** - Train multiple ML models
    6. **Model Comparison** - Compare model performances
    7. **Predictions** - Make predictions on new data
    """)
    
    # Quick Stats (placeholder)
    st.markdown("### 📈 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset Status", "Not Loaded", "Upload Data")
    with col2:
        st.metric("Records", "0", "")
    with col3:
        st.metric("Features", "0", "")
    with col4:
        st.metric("Models Trained", "0", "")
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("### 🚀 Getting Started")
    st.success("""
    **Step 1:** Navigate to the **Data Loading** page from the sidebar
    
    **Step 2:** Upload your PMGSY dataset (CSV or Excel format)
    
    **Step 3:** Follow the pipeline through preprocessing, EDA, and model training
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 2rem;">
        <p>PMGSY Data Analysis Platform | Built with Streamlit</p>
        <p>For rural road connectivity analysis and prediction</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()