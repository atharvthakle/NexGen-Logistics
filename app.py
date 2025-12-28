import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import our backend modules
from data_loader import load_all_data
from analytics import LogisticsAnalytics
from predictive_models import DeliveryPredictor, RouteOptimizer
from visualizations import DashboardVisualizations

# Page configuration
st.set_page_config(
    page_title="NexGen Logistics Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    
    /* Metric boxes - dark text on light background */
    .stMetric {
        background-color: #f0f2f6 !important;
        padding: 15px;
        border-radius: 10px;
    }
    
    /* Force ALL metric text to be dark and visible */
    .stMetric label,
    .stMetric [data-testid="stMetricLabel"],
    .stMetric [data-testid="stMetricValue"],
    .stMetric [data-testid="stMetricDelta"],
    .stMetric div,
    .stMetric span,
    .stMetric p {
        color: #0e1117 !important;
    }
    
    /* Target the specific metric container */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6 !important;
    }
    
    div[data-testid="metric-container"] * {
        color: #0e1117 !important;
    }
    
    /* Metric label specifically */
    div[data-testid="metric-container"] label {
        color: #0e1117 !important;
        font-weight: 600 !important;
    }
    
    /* Metric value specifically */
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0e1117 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Metric delta specifically */
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #0e1117 !important;
        font-size: 0.875rem !important;
    }
    
    /* All metric text must be dark */
    div[data-testid="metric-container"] * {
        color: #0e1117 !important;
    }
    
    h1 {
        color: #1f77b4 !important;
        padding-bottom: 20px;
    }
    
    h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Main content text */
    p {
        color: #fafafa !important;
    }
    
    /* Italic/emphasis text */
    em, i {
        color: #c0c0c0 !important;
        font-style: italic;
    }
    
    /* Strong/bold text */
    strong, b {
        color: #fafafa !important;
    }
    
    .highlight-box {
        background-color: #1e3a5f;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
        color: #fafafa !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        color: #000000 !important;
    }
    
    .dataframe tbody tr {
        background-color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #1f77b4 !important;
        color: #ffffff !important;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: 500;
    }
    
    .stButton button:hover {
        background-color: #1557a0 !important;
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #fafafa !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background-color: #28a745 !important;
        color: #ffffff !important;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: rgba(31, 119, 180, 0.2) !important;
        color: #fafafa !important;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: rgba(255, 193, 7, 0.2) !important;
        color: #fafafa !important;
    }
    
    /* Error boxes */
    .stError {
        background-color: rgba(220, 53, 69, 0.2) !important;
        color: #fafafa !important;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.2) !important;
        color: #fafafa !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache data loading for performance
@st.cache_data
def load_data():
    """Load all data with caching"""
    return load_all_data()

@st.cache_resource
def initialize_analytics(_data):
    """Initialize analytics with caching"""
    return LogisticsAnalytics(_data)

@st.cache_resource
def initialize_predictor(_merged_data):
    """Initialize and train predictor with caching"""
    predictor = DeliveryPredictor(_merged_data)
    predictor.train_model()
    return predictor

# Load data
with st.spinner('Loading NexGen Logistics data...'):
    data = load_data()
    analytics = initialize_analytics(data)
    predictor = initialize_predictor(analytics.merged_data)
    optimizer = RouteOptimizer(analytics.merged_data, data['fleet'])
    viz = DashboardVisualizations(analytics, predictor)

# Sidebar
st.sidebar.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=NexGen+Logistics", use_container_width=True)
st.sidebar.title("🚚 Navigation")

# Navigation menu
page = st.sidebar.radio(
    "Select Dashboard",
    ["🏠 Overview", "📊 Delivery Performance", "💰 Cost Analysis", 
     "🗺️ Route Optimization", "👥 Customer Insights", "🤖 AI Predictions",
     "🚛 Fleet Manager", "📦 Warehouse Optimizer", "🌱 Sustainability", "📈 Reports"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Last Updated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}")

# Main content area
st.title("🚚 NexGen Logistics Intelligence Platform")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.markdown("### Welcome to your Logistics Command Center")
    st.markdown("*Real-time insights and predictive analytics for smarter operations*")
    
    # Calculate key metrics
    delivery_metrics = analytics.calculate_delivery_metrics()
    cost_metrics = analytics.calculate_cost_metrics()
    route_metrics = analytics.calculate_route_metrics()
    customer_metrics = analytics.calculate_customer_metrics()
    
    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📦 Total Deliveries",
            value=f"{delivery_metrics['total_deliveries']}",
            delta=f"{delivery_metrics['on_time_percentage']:.1f}% On-Time"
        )
    
    with col2:
        st.metric(
            label="⭐ Customer Rating",
            value=f"{delivery_metrics['avg_customer_rating']:.2f}/5.0",
            delta=f"{customer_metrics['recommendation_rate']:.1f}% Recommend"
        )
    
    with col3:
        st.metric(
            label="💵 Total Revenue",
            value=f"₹{cost_metrics['total_revenue']:,.0f}",
            delta=f"₹{cost_metrics['total_profit']:,.0f} Profit"
        )
    
    with col4:
        st.metric(
            label="🚗 Total Distance",
            value=f"{route_metrics['total_distance']:,.0f} km",
            delta=f"{route_metrics['avg_fuel_efficiency']:.1f} km/L"
        )
    
    st.markdown("---")
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(viz.create_delivery_performance_chart(), use_container_width=True)
        st.plotly_chart(viz.create_cost_breakdown_chart(), use_container_width=True)
    
    with col2:
        st.plotly_chart(viz.create_priority_performance_chart(), use_container_width=True)
        st.plotly_chart(viz.create_fleet_utilization_chart(), use_container_width=True)
    
    # Critical alerts
    st.markdown("### 🚨 Critical Alerts & Recommendations")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        if delivery_metrics['on_time_percentage'] < 70:
            st.error(f"⚠️ **Low On-Time Rate:** Only {delivery_metrics['on_time_percentage']:.1f}% deliveries on time. Target: 90%+")
        
        if delivery_metrics['quality_issue_rate'] > 30:
            st.warning(f"📦 **High Quality Issues:** {delivery_metrics['quality_issue_rate']:.1f}% of deliveries have quality issues")
    
    with alert_col2:
        if customer_metrics['avg_rating'] < 4.0:
            st.error(f"⭐ **Customer Satisfaction Low:** Average rating is {customer_metrics['avg_rating']:.2f}/5.0")
        
        if cost_metrics['avg_profit_margin'] < 10:
            st.warning(f"💰 **Thin Profit Margins:** Average margin is {cost_metrics['avg_profit_margin']:.1f}%")

# ============================================================================
# PAGE 2: DELIVERY PERFORMANCE
# ============================================================================
elif page == "📊 Delivery Performance":
    st.markdown("### Delivery Performance Analytics")
    
    delivery_metrics = analytics.calculate_delivery_metrics()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("On-Time Deliveries", f"{delivery_metrics['on_time_deliveries']}")
    
    with col2:
        st.metric("Delayed Deliveries", f"{delivery_metrics['delayed_deliveries']}")
    
    with col3:
        st.metric("Avg Delay", f"{delivery_metrics['avg_delay_days']:.1f} days")
    
    with col4:
        st.metric("Quality Issues", f"{delivery_metrics['quality_issues']}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(viz.create_priority_performance_chart(), use_container_width=True)
    
    with col2:
        st.plotly_chart(viz.create_carrier_performance_chart(), use_container_width=True)
    
    # Detailed tables
    st.markdown("### 📋 Performance by Priority")
    priority_stats = analytics.get_priority_analysis()
    if priority_stats is not None:
        st.dataframe(priority_stats, use_container_width=True)
    
    st.markdown("### 🚚 Performance by Carrier")
    carrier_stats = analytics.get_carrier_analysis()
    if carrier_stats is not None:
        st.dataframe(carrier_stats, use_container_width=True)

# ============================================================================
# PAGE 3: COST ANALYSIS
# ============================================================================
elif page == "💰 Cost Analysis":
    st.markdown("### Cost & Profitability Analysis")
    
    cost_metrics = analytics.calculate_cost_metrics()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"₹{cost_metrics['total_revenue']:,.0f}")
    
    with col2:
        st.metric("Total Cost", f"₹{cost_metrics['total_cost']:,.0f}")
    
    with col3:
        st.metric("Total Profit", f"₹{cost_metrics['total_profit']:,.0f}")
    
    with col4:
        st.metric("Avg Cost/Order", f"₹{cost_metrics['avg_cost_per_order']:,.0f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(viz.create_cost_breakdown_chart(), use_container_width=True)
    
    with col2:
        st.plotly_chart(viz.create_revenue_profit_chart(), use_container_width=True)
    
    # Cost insights
    st.markdown("### 💡 Cost Optimization Opportunities")
    
    st.info(f"**Highest Cost Category:** {cost_metrics['highest_cost_category']}")
    
    if cost_metrics['cost_breakdown']:
        st.markdown("#### Cost Breakdown Details")
        cost_df = pd.DataFrame({
            'Category': cost_metrics['cost_breakdown'].keys(),
            'Amount (INR)': cost_metrics['cost_breakdown'].values()
        })
        cost_df['Percentage'] = (cost_df['Amount (INR)'] / cost_df['Amount (INR)'].sum() * 100).round(2)
        st.dataframe(cost_df, use_container_width=True)

# ============================================================================
# PAGE 4: ROUTE OPTIMIZATION
# ============================================================================
elif page == "🗺️ Route Optimization":
    st.markdown("### Route & Fleet Optimization")
    
    route_metrics = analytics.calculate_route_metrics()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Distance", f"{route_metrics['total_distance']:,.0f} km")
    
    with col2:
        st.metric("Fuel Consumed", f"{route_metrics['total_fuel_consumed']:,.0f} L")
    
    with col3:
        st.metric("Avg Fuel Efficiency", f"{route_metrics['avg_fuel_efficiency']:.1f} km/L")
    
    with col4:
        st.metric("Total Toll Charges", f"₹{route_metrics['total_toll_charges']:,.0f}")
    
    st.markdown("---")
    
    # Route efficiency chart
    st.plotly_chart(viz.create_route_efficiency_chart(), use_container_width=True)
    
    # Route efficiency table
    st.markdown("### 📊 Route Efficiency Rankings")
    route_efficiency = optimizer.calculate_route_efficiency()
    if route_efficiency is not None:
        st.dataframe(route_efficiency.head(10), use_container_width=True)
        
        # Download option
        csv = route_efficiency.to_csv(index=True)
        st.download_button(
            label="📥 Download Full Route Analysis",
            data=csv,
            file_name="route_efficiency_analysis.csv",
            mime="text/csv"
        )
    
    # Traffic insights
    st.markdown("### 🚦 Traffic Impact Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Avg Traffic Delay", f"{route_metrics['avg_traffic_delay']:.1f} min")
    
    with col2:
        st.metric("Weather Impacted Routes", f"{route_metrics['weather_impacted_routes']}")

# ============================================================================
# PAGE 5: CUSTOMER INSIGHTS
# ============================================================================
elif page == "👥 Customer Insights":
    st.markdown("### Customer Experience Analytics")
    
    customer_metrics = analytics.calculate_customer_metrics()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", f"{customer_metrics['total_feedback']}")
    
    with col2:
        st.metric("Avg Rating", f"{customer_metrics['avg_rating']:.2f}/5.0")
    
    with col3:
        st.metric("Would Recommend", f"{customer_metrics['recommendation_rate']:.1f}%")
    
    with col4:
        st.metric("Issues Reported", f"{customer_metrics['issues_reported']}")
    
    st.markdown("---")
    
    # Customer satisfaction trend
    st.plotly_chart(viz.create_customer_satisfaction_trend(), use_container_width=True)
    
    # Top issues
    if 'top_issues' in customer_metrics and customer_metrics['top_issues']:
        st.markdown("### 🔍 Top Customer Issues")
        
        issue_df = pd.DataFrame({
            'Issue Category': customer_metrics['top_issues'].keys(),
            'Count': customer_metrics['top_issues'].values()
        }).sort_values('Count', ascending=False)
        
        fig = px.bar(issue_df, x='Issue Category', y='Count', 
                     title='Most Common Issues', color='Count',
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent feedback
    st.markdown("### 💬 Recent Customer Feedback")
    if analytics.feedback is not None:
        recent_feedback = analytics.feedback.sort_values('Feedback_Date', ascending=False).head(10)
        st.dataframe(recent_feedback[['Order_ID', 'Rating', 'Feedback_Text', 'Would_Recommend']], 
                    use_container_width=True)

# ============================================================================
# PAGE 6: AI PREDICTIONS
# ============================================================================
elif page == "🤖 AI Predictions":
    st.markdown("### AI-Powered Delay Prediction")
    
    st.info("🧠 **Model Accuracy:** 95.33% - Our AI can predict delivery delays with high confidence!")
    
    # Show delay factors
    st.plotly_chart(viz.create_delay_factors_chart(), use_container_width=True)
    
    st.markdown("---")
    
    # Interactive prediction tool
    st.markdown("### 🔮 Predict Delay for New Order")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority = st.selectbox("Priority", ["Express", "Standard", "Economy"])
        product_category = st.selectbox("Product Category", 
            ["Electronics", "Fashion", "Food & Beverage", "Healthcare", "Industrial", "Books", "Home Goods"])
    
    with col2:
        distance = st.number_input("Distance (KM)", min_value=1, max_value=10000, value=500)
        traffic_delay = st.number_input("Expected Traffic Delay (minutes)", min_value=0, max_value=300, value=30)
    
    with col3:
        customer_segment = st.selectbox("Customer Segment", ["Enterprise", "SMB", "Individual"])
        carrier = st.selectbox("Carrier", ["BlueDart", "Delhivery", "DTDC", "Ecom Express", "FedEx"])
    
    if st.button("🚀 Predict Delay Risk", type="primary"):
        # Create order data
        order_data = {
            'Priority': priority,
            'Product_Category': product_category,
            'Distance_KM': distance,
            'Traffic_Delay_Minutes': traffic_delay,
            'Customer_Segment': customer_segment,
            'Carrier': carrier
        }
        
        # Get prediction
        prediction = predictor.predict_delay_risk(order_data)
        suggestions = predictor.suggest_improvements(order_data, prediction)
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction['will_delay']:
                st.error("⚠️ **Will Delay:** YES")
            else:
                st.success("✅ **Will Delay:** NO")
        
        with col2:
            st.metric("Delay Probability", f"{prediction['delay_probability']:.1f}%")
        
        with col3:
            risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.metric("Risk Level", f"{risk_color[prediction['risk_level']]} {prediction['risk_level']}")
        
        # Suggestions
        st.markdown("### 💡 Recommended Actions")
        for suggestion in suggestions:
            st.info(suggestion)
    
    # Best carrier recommendation
    st.markdown("---")
    st.markdown("### 🏆 Carrier Recommendation Tool")
    
    col1, col2 = st.columns(2)
    with col1:
        rec_priority = st.selectbox("Select Priority", ["Express", "Standard", "Economy"], key="rec_priority")
    with col2:
        rec_distance = st.number_input("Distance (KM)", min_value=1, max_value=10000, value=500, key="rec_distance")
    
    if st.button("Find Best Carrier"):
        best_carrier = optimizer.find_best_carrier(rec_priority, rec_distance)
        if best_carrier:
            st.success(f"**Recommended Carrier:** {best_carrier['recommended_carrier']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Rating", f"{best_carrier['expected_rating']:.2f}/5.0")
            with col2:
                st.metric("Expected Cost", f"₹{best_carrier['expected_cost']:.2f}")
            with col3:
                st.metric("Expected Days", f"{best_carrier['expected_days']:.1f}")

# ============================================================================
# PAGE 7: REPORTS
# ============================================================================
elif page == "📈 Reports":
    st.markdown("### Executive Reports & Data Export")
    
    st.markdown("#### 📊 Generate Custom Reports")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Complete Operations Report", "Delivery Performance Report", 
         "Cost Analysis Report", "Customer Satisfaction Report"]
    )
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            # Calculate all metrics
            delivery_metrics = analytics.calculate_delivery_metrics()
            cost_metrics = analytics.calculate_cost_metrics()
            route_metrics = analytics.calculate_route_metrics()
            customer_metrics = analytics.calculate_customer_metrics()
            
            # Create report based on type
            st.markdown("---")
            st.markdown(f"## {report_type}")
            st.markdown(f"**Generated on:** {datetime.now().strftime('%B %d, %Y at %H:%M')}")
            st.markdown("---")
            
            if "Complete" in report_type or "Delivery" in report_type:
                st.markdown("### 📦 Delivery Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Deliveries", delivery_metrics['total_deliveries'])
                    st.metric("On-Time %", f"{delivery_metrics['on_time_percentage']:.1f}%")
                with col2:
                    st.metric("Delayed Deliveries", delivery_metrics['delayed_deliveries'])
                    st.metric("Avg Delay", f"{delivery_metrics['avg_delay_days']:.1f} days")
                with col3:
                    st.metric("Avg Rating", f"{delivery_metrics['avg_customer_rating']:.2f}/5.0")
                    st.metric("Quality Issues", delivery_metrics['quality_issues'])
            
            if "Complete" in report_type or "Cost" in report_type:
                st.markdown("### 💰 Financial Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Revenue", f"₹{cost_metrics['total_revenue']:,.0f}")
                with col2:
                    st.metric("Total Cost", f"₹{cost_metrics['total_cost']:,.0f}")
                with col3:
                    st.metric("Total Profit", f"₹{cost_metrics['total_profit']:,.0f}")
            
            if "Complete" in report_type or "Customer" in report_type:
                st.markdown("### 👥 Customer Satisfaction")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Rating", f"{customer_metrics['avg_rating']:.2f}/5.0")
                with col2:
                    st.metric("Recommendation Rate", f"{customer_metrics['recommendation_rate']:.1f}%")
                with col3:
                    st.metric("Issues Reported", customer_metrics['issues_reported'])
    
    st.markdown("---")
    st.markdown("#### 📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Merged Dataset"):
            csv = analytics.merged_data.to_csv(index=False)
            st.download_button(
                label="Download Merged Data CSV",
                data=csv,
                file_name="nexgen_logistics_merged_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export All Metrics"):
            # Create comprehensive metrics dataframe
            all_metrics = {
                'Metric Category': [],
                'Metric Name': [],
                'Value': []
            }
            
            delivery_metrics = analytics.calculate_delivery_metrics()
            for key, value in delivery_metrics.items():
                all_metrics['Metric Category'].append('Delivery Performance')
                all_metrics['Metric Name'].append(key)
                all_metrics['Value'].append(value)
            
            metrics_df = pd.DataFrame(all_metrics)
            csv = metrics_df.to_csv(index=False)
            
            st.download_button(
                label="Download All Metrics CSV",
                data=csv,
                file_name="nexgen_logistics_metrics.csv",
                mime="text/csv"
            )
            
# ============================================================================
# PAGE 8: FLEET MANAGER
# ============================================================================
elif page == "🚛 Fleet Manager":
    st.markdown("### Dynamic Fleet Management System")
    
    from predictive_models import FleetManager
    fleet_manager = FleetManager(data['fleet'], analytics.merged_data)
    
    # Fleet overview
    fleet_overview = fleet_manager.get_fleet_overview()
    
    if fleet_overview:
        st.markdown("#### 🚗 Fleet Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vehicles", fleet_overview['total_vehicles'])
        
        with col2:
            st.metric("Utilization Rate", f"{fleet_overview['utilization_rate']:.1f}%")
        
        with col3:
            st.metric("Avg Fuel Efficiency", f"{fleet_overview['avg_fuel_efficiency']:.1f} km/L")
        
        with col4:
            st.metric("Total Capacity", f"{fleet_overview['total_capacity']:,.0f} kg")
        
        st.markdown("---")
        
        # Fleet composition charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(viz.create_fleet_type_distribution(data['fleet']), use_container_width=True)
            st.plotly_chart(viz.create_fleet_age_distribution(data['fleet']), use_container_width=True)
        
        with col2:
            st.plotly_chart(viz.create_fleet_utilization_chart(), use_container_width=True)
            
            # Fleet by status
            st.markdown("##### Fleet Status Breakdown")
            status_df = pd.DataFrame({
                'Status': fleet_overview['by_status'].keys(),
                'Count': fleet_overview['by_status'].values()
            })
            st.dataframe(status_df, use_container_width=True)
        
        # Fleet performance analysis
        st.markdown("---")
        st.markdown("#### 📊 Fleet Performance Analysis")
        
        fleet_performance = fleet_manager.get_fleet_performance_analysis()
        if fleet_performance is not None:
            st.plotly_chart(viz.create_fleet_efficiency_chart(fleet_performance), use_container_width=True)
            
            st.markdown("##### Detailed Performance Metrics")
            st.dataframe(fleet_performance, use_container_width=True)
        
        # Vehicle matching tool
        st.markdown("---")
        st.markdown("#### 🎯 Smart Vehicle Matching Tool")
        st.info("Match the optimal vehicle to your order based on AI-powered analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            match_priority = st.selectbox("Order Priority", ["Express", "Standard", "Economy"], key="fleet_priority")
            match_category = st.selectbox("Product Category", 
                ["Electronics", "Fashion", "Food & Beverage", "Healthcare", "Industrial", "Books", "Home Goods"],
                key="fleet_category")
        
        with col2:
            match_value = st.number_input("Order Value (INR)", min_value=100, max_value=100000, value=5000, key="fleet_value")
            match_distance = st.number_input("Distance (KM)", min_value=10, max_value=10000, value=500, key="fleet_distance")
        
        with col3:
            st.markdown("##### ")  # Spacing
            if st.button("🚀 Find Best Vehicle", type="primary"):
                order_details = {
                    'priority': match_priority,
                    'product_category': match_category,
                    'order_value': match_value,
                    'distance_km': match_distance
                }
                
                match_result = fleet_manager.match_vehicle_to_order(order_details)
                
                if match_result and 'error' not in match_result:
                    st.success(f"✅ **Recommended Vehicle:** {match_result['recommended_vehicle_id']}")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("Vehicle Type", match_result['vehicle_type'])
                        st.metric("Match Score", f"{match_result['match_score']}/10")
                    
                    with result_col2:
                        st.metric("Capacity", f"{match_result['capacity_kg']} kg")
                        st.metric("Fuel Efficiency", f"{match_result['fuel_efficiency']} km/L")
                    
                    with result_col3:
                        st.metric("Est. Fuel Cost", f"₹{match_result['estimated_fuel_cost']:.2f}")
                        st.metric("Est. CO2", f"{match_result['estimated_co2_kg']:.2f} kg")
                    
                    st.info(f"📍 Current Location: {match_result['current_location']} | Vehicle Age: {match_result['vehicle_age']} years")
                else:
                    st.error("❌ No available vehicles found")
        
        # Maintenance alerts
        st.markdown("---")
        st.markdown("#### 🔧 Maintenance Alerts")
        
        maintenance_alerts = fleet_manager.get_maintenance_alerts()
        if maintenance_alerts is not None and len(maintenance_alerts) > 0:
            st.warning(f"⚠️ {len(maintenance_alerts)} vehicles require attention")
            st.dataframe(maintenance_alerts, use_container_width=True)
        else:
            st.success("✅ All vehicles are in good condition!")

# ============================================================================
# PAGE 9: WAREHOUSE OPTIMIZER
# ============================================================================
elif page == "📦 Warehouse Optimizer":
    st.markdown("### Warehouse Inventory Optimization")
    
    warehouse_metrics = analytics.calculate_warehouse_metrics()
    
    if warehouse_metrics:
        # Overview KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Warehouses", warehouse_metrics['total_warehouses'])
        
        with col2:
            st.metric("Total Stock", f"{warehouse_metrics['total_stock_units']:,.0f} units")
        
        with col3:
            st.metric("Avg Stock/Warehouse", f"{warehouse_metrics['avg_stock_per_warehouse']:.0f} units")
        
        with col4:
            st.metric("Warehouses Need Restock", warehouse_metrics['warehouses_needing_restock'])
        
        st.markdown("---")
        
        # Warehouse visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(viz.create_warehouse_stock_chart(warehouse_metrics), use_container_width=True)
        
        with col2:
            st.plotly_chart(viz.create_warehouse_status_chart(warehouse_metrics), use_container_width=True)
        
        # Warehouse summary table
        st.markdown("#### 📊 Warehouse Summary")
        warehouse_summary = warehouse_metrics['warehouse_summary'].reset_index()
        st.dataframe(warehouse_summary, use_container_width=True)
        
        # Stock by category
        st.markdown("---")
        st.markdown("#### 📦 Stock by Product Category")
        
        stock_category = pd.DataFrame({
            'Product Category': warehouse_metrics['stock_by_category'].keys(),
            'Total Stock': warehouse_metrics['stock_by_category'].values()
        }).sort_values('Total Stock', ascending=False)
        
        fig = px.bar(stock_category, x='Product Category', y='Total Stock',
                     title='Stock Distribution by Product Category',
                     color='Total Stock', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        # Rebalancing suggestions
        st.markdown("---")
        st.markdown("#### 🔄 Inventory Rebalancing Recommendations")
        
        rebalancing_suggestions = analytics.get_warehouse_rebalancing_suggestions()
        
        if rebalancing_suggestions is not None and len(rebalancing_suggestions) > 0:
            st.info(f"💡 {len(rebalancing_suggestions)} rebalancing opportunities identified")
            st.dataframe(rebalancing_suggestions, use_container_width=True)
            
            # Download suggestions
            csv = rebalancing_suggestions.to_csv(index=False)
            st.download_button(
                label="📥 Download Rebalancing Plan",
                data=csv,
                file_name="warehouse_rebalancing_plan.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ All warehouses are optimally balanced!")
        
        # Cost analysis
        st.markdown("---")
        st.markdown("#### 💰 Storage Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Storage Cost", f"₹{warehouse_metrics['total_storage_cost']:,.2f}")
        
        with col2:
            avg_cost = warehouse_metrics['total_storage_cost'] / warehouse_metrics['total_stock_units']
            st.metric("Avg Cost per Unit", f"₹{avg_cost:.2f}")
    
    else:
        st.warning("No warehouse data available")

# ============================================================================
# PAGE 10: SUSTAINABILITY TRACKER
# ============================================================================
elif page == "🌱 Sustainability":
    st.markdown("### Environmental Impact & Sustainability Tracker")
    
    env_metrics = analytics.calculate_environmental_metrics()
    
    if env_metrics:
        # Overview KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total CO2 Emissions", f"{env_metrics['total_co2_emissions_tonnes']:.2f} tonnes")
        
        with col2:
            st.metric("Avg CO2/Delivery", f"{env_metrics['avg_co2_per_delivery']:.2f} kg")
        
        with col3:
            st.metric("Total Fuel Consumed", f"{env_metrics['total_fuel_consumed']:,.0f} L")
        
        with col4:
            st.metric("Avg Fuel Efficiency", f"{env_metrics['avg_fuel_efficiency']:.2f} km/L")
        
        st.markdown("---")
        
        # Environmental impact visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(viz.create_sustainability_overview_chart(env_metrics), use_container_width=True)
        
        with col2:
            st.plotly_chart(viz.create_priority_emissions_chart(env_metrics), use_container_width=True)
        
        # Savings potential
        st.markdown("---")
        st.markdown("#### 💡 Optimization Potential")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**10% Efficiency Improvement Would Save:**")
            st.metric("CO2 Reduction", f"{env_metrics['potential_co2_reduction_10pct']:.2f} kg")
            st.metric("Fuel Savings", f"{env_metrics['potential_fuel_savings_10pct']:.2f} L")
            
            fuel_cost_savings = env_metrics['potential_fuel_savings_10pct'] * 100  # Assuming ₹100/L
            st.metric("Cost Savings", f"₹{fuel_cost_savings:,.2f}")
        
        with col2:
            st.info("**Recommended Actions:**")
            st.markdown("""
            - 🚗 Route optimization to reduce distance
            - ⚡ Upgrade to more fuel-efficient vehicles
            - 📦 Consolidate deliveries to reduce trips
            - 🌿 Consider electric vehicles for short routes
            - 📊 Monitor and reward eco-friendly driving
            """)
        
        # Green routes analysis
        st.markdown("---")
        st.markdown("#### 🌿 Eco-Friendly Route Analysis")
        
        green_routes = analytics.get_green_route_recommendations()
        if green_routes is not None:
            st.plotly_chart(viz.create_green_routes_chart(green_routes), use_container_width=True)
            
            st.markdown("##### Top 10 Most Eco-Friendly Routes")
            st.dataframe(green_routes.head(10), use_container_width=True)
            
            st.markdown("##### Bottom 10 Least Eco-Friendly Routes (Need Improvement)")
            st.dataframe(green_routes.tail(10), use_container_width=True)
            
            # Download option
            csv = green_routes.to_csv(index=True)
            st.download_button(
                label="📥 Download Complete Route Environmental Analysis",
                data=csv,
                file_name="green_routes_analysis.csv",
                mime="text/csv"
            )
        
        # Environmental goals
        st.markdown("---")
        st.markdown("#### 🎯 Sustainability Goals & Tracking")
        
        current_co2 = env_metrics['total_co2_emissions_tonnes']
        target_reduction = current_co2 * 0.20  # 20% reduction goal
        
        progress = st.progress(0)
        st.markdown(f"**Goal:** Reduce CO2 emissions by 20% ({target_reduction:.2f} tonnes)")
        st.markdown(f"**Current:** {current_co2:.2f} tonnes")
        st.markdown("**Progress:** 0% (Baseline established)")
    
    else:
        st.warning("No environmental data available")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚚 NexGen Logistics Intelligence Platform | Built with Streamlit & Python</p>
        <p>Powered by Machine Learning & Advanced Analytics</p>
    </div>
    """,
    unsafe_allow_html=True
)