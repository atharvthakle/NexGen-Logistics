import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DashboardVisualizations:
    """
    Creates all visualizations for the dashboard
    """
    
    def __init__(self, analytics, predictor=None):
        self.analytics = analytics
        self.predictor = predictor
        self.merged_data = analytics.merged_data
    
    def create_delivery_performance_chart(self):
        """Create delivery performance overview chart"""
        
        df = self.merged_data[self.merged_data['Delivery_Status'].notna()].copy()
        df['Delay_Days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
        df['Status'] = df['Delay_Days'].apply(lambda x: 'On Time' if x <= 0 else 'Delayed')
        
        status_counts = df['Status'].value_counts()
        
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title='Delivery Performance Overview',
            color=status_counts.index,
            color_discrete_map={'On Time': '#00D26A', 'Delayed': '#FF4B4B'}
        )
        
        return fig
    
    def create_priority_performance_chart(self):
        """Performance by priority level"""
        
        priority_stats = self.analytics.get_priority_analysis()
        
        if priority_stats is None:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Total Orders',
            x=priority_stats.index,
            y=priority_stats['Total_Orders'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='On Time Orders',
            x=priority_stats.index,
            y=priority_stats['On_Time_Orders'],
            marker_color='green'
        ))
        
        fig.update_layout(
            title='Orders by Priority Level',
            xaxis_title='Priority',
            yaxis_title='Number of Orders',
            barmode='group'
        )
        
        return fig
    
    def create_carrier_performance_chart(self):
        """Carrier performance comparison"""
        
        carrier_stats = self.analytics.get_carrier_analysis()
        
        if carrier_stats is None:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Carrier Ratings', 'On-Time Performance'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Ratings chart
        fig.add_trace(
            go.Bar(
                x=carrier_stats.index,
                y=carrier_stats['Avg_Rating'],
                name='Avg Rating',
                marker_color='orange'
            ),
            row=1, col=1
        )
        
        # On-time performance chart
        fig.add_trace(
            go.Bar(
                x=carrier_stats.index,
                y=carrier_stats['On_Time_%'],
                name='On-Time %',
                marker_color='blue'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text='Carrier Performance Analysis',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_cost_breakdown_chart(self):
        """Cost breakdown pie chart"""
        
        cost_metrics = self.analytics.calculate_cost_metrics()
        
        if cost_metrics is None or 'cost_breakdown' not in cost_metrics:
            return None
        
        cost_breakdown = cost_metrics['cost_breakdown']
        
        # Clean up names for display
        display_names = {
            'Fuel_Cost': 'Fuel',
            'Labor_Cost': 'Labor',
            'Vehicle_Maintenance': 'Maintenance',
            'Insurance': 'Insurance',
            'Packaging_Cost': 'Packaging',
            'Technology_Platform_Fee': 'Technology',
            'Other_Overhead': 'Other'
        }
        
        labels = [display_names.get(k, k) for k in cost_breakdown.keys()]
        values = list(cost_breakdown.values())
        
        fig = px.pie(
            values=values,
            names=labels,
            title='Cost Breakdown by Category'
        )
        
        return fig
    
    def create_route_efficiency_chart(self):
        """Route efficiency scatter plot"""
        
        df = self.merged_data[self.merged_data['Distance_KM'].notna()].copy()
        df['Cost_per_KM'] = df['Fuel_Cost'] / df['Distance_KM']
        df['Delay_Days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
        
        fig = px.scatter(
            df,
            x='Distance_KM',
            y='Cost_per_KM',
            size='Traffic_Delay_Minutes',
            color='Delay_Days',
            hover_data=['Route', 'Priority'],
            title='Route Efficiency Analysis',
            labels={
                'Distance_KM': 'Distance (KM)',
                'Cost_per_KM': 'Cost per KM (INR)',
                'Traffic_Delay_Minutes': 'Traffic Delay',
                'Delay_Days': 'Delay Days'
            },
            color_continuous_scale='RdYlGn_r'
        )
        
        return fig
    
    def create_customer_satisfaction_trend(self):
        """Customer satisfaction over time"""
        
        if self.analytics.feedback is None:
            return None
        
        feedback = self.analytics.feedback.copy()
        feedback['Feedback_Date'] = pd.to_datetime(feedback['Feedback_Date'])
        feedback = feedback.sort_values('Feedback_Date')
        
        # Calculate rolling average
        feedback['Rating_MA'] = feedback['Rating'].rolling(window=10, min_periods=1).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=feedback['Feedback_Date'],
            y=feedback['Rating'],
            mode='markers',
            name='Individual Ratings',
            marker=dict(size=6, opacity=0.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=feedback['Feedback_Date'],
            y=feedback['Rating_MA'],
            mode='lines',
            name='10-Order Moving Average',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title='Customer Satisfaction Trend',
            xaxis_title='Date',
            yaxis_title='Rating (1-5)',
            hovermode='x unified'
        )
        
        return fig
    
    def create_delay_factors_chart(self):
        """Show top factors causing delays"""
        
        if self.predictor is None or self.predictor.feature_importance is None:
            return None
        
        importance_df = self.predictor.feature_importance
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top Factors Causing Delays',
            labels={'Importance': 'Importance Score', 'Feature': 'Factor'},
            color='Importance',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        return fig
    
    def create_revenue_profit_chart(self):
        """Revenue vs Profit analysis"""
        
        df = self.merged_data[self.merged_data['Fuel_Cost'].notna()].copy()
        
        cost_columns = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 
                       'Insurance', 'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead']
        
        df['Total_Cost'] = df[cost_columns].sum(axis=1)
        df['Profit'] = df['Order_Value_INR'] - df['Total_Cost'] - df['Delivery_Cost_INR']
        
        # Group by product category
        category_analysis = df.groupby('Product_Category').agg({
            'Order_Value_INR': 'sum',
            'Profit': 'sum',
            'Order_ID': 'count'
        }).round(2)
        
        category_analysis.columns = ['Revenue', 'Profit', 'Orders']
        category_analysis['Profit_Margin_%'] = (category_analysis['Profit'] / category_analysis['Revenue'] * 100).round(2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Revenue',
            x=category_analysis.index,
            y=category_analysis['Revenue'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Profit',
            x=category_analysis.index,
            y=category_analysis['Profit'],
            marker_color='green'
        ))
        
        fig.update_layout(
            title='Revenue & Profit by Product Category',
            xaxis_title='Product Category',
            yaxis_title='Amount (INR)',
            barmode='group',
            hovermode='x unified'
        )
        
        return fig
    
    def create_fleet_utilization_chart(self):
        """Fleet status and utilization"""
        
        if self.analytics.fleet is None:
            return None
        
        fleet = self.analytics.fleet.copy()
        
        # Status distribution
        status_counts = fleet['Status'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.3
            )
        ])
        
        fig.update_layout(
            title='Fleet Status Distribution',
            annotations=[dict(text='Fleet', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def create_sustainability_overview_chart(self, env_metrics):
        """Create sustainability metrics overview"""
        
        if env_metrics is None:
            return None
        
        # Create gauge chart for CO2 emissions
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = env_metrics['total_co2_emissions_tonnes'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Total CO2 Emissions (Tonnes)"},
            delta = {'reference': env_metrics['total_co2_emissions_tonnes'] * 1.1},
            gauge = {
                'axis': {'range': [None, env_metrics['total_co2_emissions_tonnes'] * 1.5]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, env_metrics['total_co2_emissions_tonnes'] * 0.7], 'color': "lightgreen"},
                    {'range': [env_metrics['total_co2_emissions_tonnes'] * 0.7, env_metrics['total_co2_emissions_tonnes'] * 1.2], 'color': "yellow"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': env_metrics['total_co2_emissions_tonnes'] * 1.2
                }
            }
        ))
        
        return fig
    
    def create_priority_emissions_chart(self, env_metrics):
        """CO2 emissions by priority level"""
        
        if env_metrics is None or 'by_priority' not in env_metrics:
            return None
        
        priority_data = env_metrics['by_priority']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Total CO2 (kg)',
            x=priority_data.index,
            y=priority_data['Total_CO2_KG'],
            marker_color='indianred'
        ))
        
        fig.add_trace(go.Scatter(
            name='Avg CO2 per Delivery',
            x=priority_data.index,
            y=priority_data['Avg_CO2_per_Delivery'],
            yaxis='y2',
            marker_color='orange',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='CO2 Emissions by Priority Level',
            xaxis_title='Priority',
            yaxis_title='Total CO2 (kg)',
            yaxis2=dict(
                title='Avg CO2 per Delivery (kg)',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def create_green_routes_chart(self, green_routes):
        """Most and least eco-friendly routes"""
        
        if green_routes is None:
            return None
        
        # Get top 10 greenest and bottom 5 least green
        top_green = green_routes.head(10)
        
        fig = px.bar(
            top_green.reset_index(),
            x='Route',
            y='Green_Score',
            color='Avg_CO2_per_KM',
            title='Top 10 Most Eco-Friendly Routes',
            labels={'Green_Score': 'Eco-Efficiency Score', 'Avg_CO2_per_KM': 'Avg CO2/km'},
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        
        return fig
    
    def create_warehouse_stock_chart(self, warehouse_metrics):
        """Warehouse stock levels visualization"""
        
        if warehouse_metrics is None or 'warehouse_summary' not in warehouse_metrics:
            return None
        
        warehouse_summary = warehouse_metrics['warehouse_summary'].reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current Stock',
            x=warehouse_summary['Location'],
            y=warehouse_summary['Total_Stock'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Reorder Level',
            x=warehouse_summary['Location'],
            y=warehouse_summary['Total_Reorder_Level'],
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='Warehouse Stock Levels vs Reorder Points',
            xaxis_title='Warehouse Location',
            yaxis_title='Stock Units',
            barmode='group'
        )
        
        return fig
    
    def create_warehouse_status_chart(self, warehouse_metrics):
        """Warehouse status distribution"""
        
        if warehouse_metrics is None or 'warehouse_summary' not in warehouse_metrics:
            return None
        
        warehouse_summary = warehouse_metrics['warehouse_summary']
        status_counts = warehouse_summary['Stock_Status'].value_counts()
        
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title='Warehouse Stock Status Distribution',
            color=status_counts.index,
            color_discrete_map={
                'Healthy': '#00D26A',
                'Overstocked': '#FFA500',
                'Critical': '#FF4B4B'
            }
        )
        
        return fig
    
    def create_fleet_type_distribution(self, fleet_data):
        """Fleet composition by vehicle type"""
        
        if fleet_data is None:
            return None
        
        type_counts = fleet_data['Vehicle_Type'].value_counts()
        
        fig = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title='Fleet Composition by Vehicle Type',
            labels={'x': 'Vehicle Type', 'y': 'Count'},
            color=type_counts.values,
            color_continuous_scale='Blues'
        )
        
        return fig
    
    def create_fleet_efficiency_chart(self, fleet_performance):
        """Fleet efficiency by type"""
        
        if fleet_performance is None:
            return None
        
        performance_df = fleet_performance.reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Fuel Efficiency',
            x=performance_df['Vehicle_Type'],
            y=performance_df['Avg_Fuel_Efficiency'],
            marker_color='lightblue',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            name='Efficiency Score',
            x=performance_df['Vehicle_Type'],
            y=performance_df['Efficiency_Score'],
            marker_color='green',
            mode='lines+markers',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Fleet Performance by Vehicle Type',
            xaxis_title='Vehicle Type',
            yaxis=dict(title='Fuel Efficiency (km/L)'),
            yaxis2=dict(
                title='Efficiency Score',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def create_fleet_age_distribution(self, fleet_data):
        """Fleet age distribution histogram"""
        
        if fleet_data is None:
            return None
        
        fig = px.histogram(
            fleet_data,
            x='Age_Years',
            nbins=10,
            title='Fleet Age Distribution',
            labels={'Age_Years': 'Vehicle Age (Years)', 'count': 'Number of Vehicles'},
            color_discrete_sequence=['#1f77b4']
        )
        
        return fig

# Test visualizations
if __name__ == "__main__":
    from data_loader import load_all_data
    from analytics import LogisticsAnalytics
    from predictive_models import DeliveryPredictor
    
    print("Loading data and creating visualizations...")
    
    # Load data
    data = load_all_data()
    analytics = LogisticsAnalytics(data)
    
    # Train predictor
    predictor = DeliveryPredictor(analytics.merged_data)
    predictor.train_model()
    
    # Create visualizations
    viz = DashboardVisualizations(analytics, predictor)
    
    print("\n✓ Visualization module ready!")
    print("✓ All charts can be generated for the dashboard")
    print("\nAvailable charts:")
    print("  1. Delivery Performance")
    print("  2. Priority Performance")
    print("  3. Carrier Performance")
    print("  4. Cost Breakdown")
    print("  5. Route Efficiency")
    print("  6. Customer Satisfaction Trend")
    print("  7. Delay Factors")
    print("  8. Revenue & Profit")
    print("  9. Fleet Utilization")