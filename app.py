import streamlit as st
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
from typing import Dict, List
from agents.claims_analysis_agent import ClaimsAnalysisAgent
from agents.policy_validation_agent import PolicyValidationAgent
from utils.data_loader import DataLoader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
# Load environment variables
load_dotenv()

# Get API key from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')

# Set page config
st.set_page_config(
    page_title="Insurance Claims Advisor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .claim-stats {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-approved {
        color: green;
        font-weight: bold;
    }
    .status-denied {
        color: red;
        font-weight: bold;
    }
    .status-pending {
        color: orange;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_agents():
    """Initialize all required agents"""
    if 'claims_agent' not in st.session_state:
        with st.spinner("Initializing Claims Analysis System..."):
            try:
                claims_agent = ClaimsAnalysisAgent(groq_api_key)
                
                # Load initial data
                claims_data = claims_agent.data_loader.load_claims_history()
                if claims_data:
                    st.session_state.claims_agent = claims_agent
                    st.success("Claims Analysis System initialized successfully!")
                else:
                    st.error("No claims data found. Please check your data files.")
                    return
                    
            except Exception as e:
                st.error(f"Error initializing Claims Analysis System: {str(e)}")
                return
    
    if 'policy_agent' not in st.session_state:
        st.session_state.policy_agent = PolicyValidationAgent(groq_api_key)

    if 'claims_agent' not in st.session_state:
        st.session_state.claims_agent = ClaimsAnalysisAgent(groq_api_key)
    
    if 'policy_agent' not in st.session_state:
        st.session_state.policy_agent = PolicyValidationAgent(groq_api_key)
    
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()

def render_historical_analysis():
    """Render historical claims analysis"""
    st.subheader("📊 Historical Claims Analysis")
    
    # Get claims data using data loader
    claims_data = st.session_state.data_loader.load_claims_history()
    
    if not claims_data:
        st.info("No historical claims data available")
        return
    
    df = pd.DataFrame(claims_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Claims",
            len(df),
            f"{len(df[df['date_filed'] > '2024-01-01'])} this year"
        )
    
    with col2:
        approval_rate = (len(df[df['status'] == 'Approved']) / len(df) * 100)
        st.metric(
            "Approval Rate",
            f"{approval_rate:.1f}%"
        )
    
    with col3:
        avg_amount = df['amount'].mean()
        st.metric(
            "Average Claim",
            f"${avg_amount:,.2f}"
        )
    
    with col4:
        avg_processing = df['processing_time'].mean()
        st.metric(
            "Avg Processing Time",
            f"{avg_processing:.1f} days"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Claims by Type")
        fig_type = px.pie(df, names='claim_type', values='amount')
        st.plotly_chart(fig_type)
    
    with col2:
        st.subheader("Claims Status")
        status_counts = df['status'].value_counts()
        fig_status = px.bar(
            x=status_counts.index, 
            y=status_counts.values,
            title="Claim Status Distribution"
        )
        st.plotly_chart(fig_status)
    
    # Filtering options
    st.subheader("🔍 Filter Claims")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_policy = st.selectbox(
            "Filter by Policy",
            ["All"] + list(df['policy_number'].unique())
        )
    
    with col2:
        selected_type = st.selectbox(
            "Filter by Type",
            ["All"] + list(df['claim_type'].unique())
        )
    
    with col3:
        selected_status = st.selectbox(
            "Filter by Status",
            ["All"] + list(df['status'].unique())
        )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_policy != "All":
        filtered_df = filtered_df[filtered_df['policy_number'] == selected_policy]
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df['claim_type'] == selected_type]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df['status'] == selected_status]
    
    # Detailed claims table
    st.subheader("📋 Claims Details")
    st.dataframe(
        filtered_df[[
            'claim_id', 'policy_number', 'claim_type', 
            'amount', 'status', 'settlement_amount', 
            'processing_time', 'date_filed'
        ]].sort_values('date_filed', ascending=False),
        use_container_width=True
    )
    
    # Download option
    if st.button("Download Claims Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="claims_data.csv",
            mime="text/csv"
        )

def render_claim_form():
    """Render the claim submission form"""
    with st.form("claim_form"):
        st.subheader("📝 Claim Details")
        
        col1, col2 = st.columns(2)
        with col1:
            policy_number = st.text_input("Policy Number (e.g., POL001)")
            claim_id = st.text_input(
                "Claim ID", 
                value=f"CLM{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
        
        with col2:
            claim_type = st.selectbox(
                "Claim Type",
                ["Emergency Care", "Prescription", "Specialist Visit", "Preventive Care"]
            )
            amount = st.number_input("Claim Amount ($)", min_value=0.0, step=100.0)
        
        date_filed = st.date_input("Date Filed")
        description = st.text_area("Claim Description")
        
        st.subheader("📎 Required Documents")
        if claim_type:
            required_docs = st.session_state.claims_agent.get_required_documents(claim_type)
            doc_cols = st.columns(len(required_docs))
            provided_docs = []
            
            for i, doc in enumerate(required_docs):
                with doc_cols[i]:
                    if st.checkbox(doc):
                        provided_docs.append(doc)
        
        submitted = st.form_submit_button("Submit Claim")
        
        if submitted:
            if not policy_number:
                st.error("Please enter a policy number")
                return None
                
            return {
                "claim_id": claim_id,
                "policy_number": policy_number,
                "claim_type": claim_type,
                "amount": amount,
                "date_filed": date_filed.strftime("%Y-%m-%d"),
                "description": description,
                "documents_provided": provided_docs
            }
    
    return None

def render_claim_analysis(claim_details: Dict):
    """Render claim analysis results"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.expander("🔍 Claim Analysis", expanded=True):
            with st.spinner("Analyzing claim..."):
                analysis = st.session_state.claims_agent.analyze_claim(claim_details)
                st.markdown(analysis)
        
        with st.expander("📊 Similar Claims"):
            with st.spinner("Finding similar claims..."):
                similar_claims = st.session_state.claims_agent.get_similar_claims(claim_details)
                st.markdown(similar_claims)
    
    with col2:
        st.subheader("⚡ Quick Actions")
        
        # Fraud check
        with st.spinner("Checking for fraud indicators..."):
            fraud_flags = st.session_state.claims_agent.detect_fraud_indicators(claim_details)
            if fraud_flags:
                st.error("⚠️ Potential Risk Indicators:")
                for flag in fraud_flags:
                    st.warning(f"• {flag}")
            else:
                st.success("✅ No risk indicators detected")
        
        # Settlement suggestion
        with st.spinner("Calculating suggested settlement..."):
            try:
                suggested_settlement, explanation = st.session_state.claims_agent.suggest_settlement_amount(claim_details)
                
                st.info(f"""
                💰 Suggested Settlement:
                ${suggested_settlement:,.2f}
                
                Original Claim: ${claim_details['amount']:,.2f}
                Settlement Ratio: {(suggested_settlement/claim_details['amount']*100):.1f}%
                """)
                
                with st.expander("📝 Settlement Analysis"):
                    st.markdown(explanation)
                
                # Get additional metrics
                metrics = st.session_state.claims_agent.get_settlement_metrics(claim_details)
                
                if metrics and not metrics.get("error"):
                    st.success("📊 Settlement Metrics")
                    
                    if metrics.get("average_processing_time"):
                        st.metric(
                            "Avg. Processing Time",
                            f"{metrics['average_processing_time']:.1f} days"
                        )
                    
                    if metrics.get("approval_rate"):
                        st.metric(
                            "Approval Rate",
                            f"{metrics['approval_rate']:.1f}%"
                        )
                    
                    if metrics.get("settlement_range"):
                        range_data = metrics["settlement_range"]
                        if all(v is not None for v in range_data.values()):
                            st.write("Settlement Range:")
                            st.write(f"Min: ${range_data['min']:,.2f}")
                            st.write(f"Avg: ${range_data['avg']:,.2f}")
                            st.write(f"Max: ${range_data['max']:,.2f}")
                    
                    if metrics.get("confidence_score"):
                        st.metric(
                            "Confidence Score",
                            f"{metrics['confidence_score']:.2f}"
                        )
                
            except Exception as e:
                st.error(f"Error calculating settlement: {str(e)}")
        
        # Document check
        required_docs = st.session_state.claims_agent.get_required_documents(claim_details['claim_type'])
        provided_docs = claim_details.get('documents_provided', [])
        missing_docs = set(required_docs) - set(provided_docs)
        
        if missing_docs:
            st.warning("📎 Missing Documents:")
            for doc in missing_docs:
                st.markdown(f"• {doc}")
        else:
            st.success("📎 All required documents provided")

def main():
    st.title("🏥 Insurance Claims Advisor")
    
    # Initialize agents
    initialize_agents()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["New Claim", "Historical Analysis", "Policy Lookup"]
    )
    
    if page == "New Claim":
        claim_details = render_claim_form()
        if claim_details:
            render_claim_analysis(claim_details)
    
    elif page == "Historical Analysis":
        render_historical_analysis()
    
    elif page == "Policy Lookup":
        st.subheader("📋 Policy Lookup")
        policy_number = st.text_input("Enter Policy Number")
        
        if st.button("Look Up Policy"):
            if policy_number:
                policy_summary = st.session_state.policy_agent.get_policy_summary(policy_number)
                st.markdown(policy_summary)
            else:
                st.error("Please enter a policy number")

if __name__ == "__main__":
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        st.stop()
    
    main()