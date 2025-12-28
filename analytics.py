import pandas as pd
import numpy as np
from datetime import datetime

class LogisticsAnalytics:
    """
    Core analytics engine for NexGen Logistics
    """
    
    def __init__(self, data):
        """Initialize with loaded data dictionary"""
        self.orders = data['orders']
        self.delivery = data['delivery']
        self.routes = data['routes']
        self.fleet = data['fleet']
        self.inventory = data['inventory']
        self.feedback = data['feedback']
        self.costs = data['costs']
        
        # Merge data for comprehensive analysis
        self.merged_data = self._merge_datasets()
    
    def _merge_datasets(self):
        """Merge all datasets on Order_ID"""
        print("Merging datasets...")
        
        # Start with orders
        merged = self.orders.copy()
        
        # Merge delivery performance
        if self.delivery is not None:
            merged = merged.merge(self.delivery, on='Order_ID', how='left')
        
        # Merge routes
        if self.routes is not None:
            merged = merged.merge(self.routes, on='Order_ID', how='left')
        
        # Merge costs
        if self.costs is not None:
            merged = merged.merge(self.costs, on='Order_ID', how='left')
        
        print(f"✓ Merged dataset created: {len(merged)} records")
        return merged
    
    def calculate_delivery_metrics(self):
        """Calculate key delivery performance metrics"""
        
        df = self.merged_data.copy()
        
        # Filter only delivered orders
        delivered = df[df['Delivery_Status'].notna()].copy()
        
        if len(delivered) == 0:
            return None
        
        # Calculate delay
        delivered['Delay_Days'] = delivered['Actual_Delivery_Days'] - delivered['Promised_Delivery_Days']
        delivered['Is_Delayed'] = delivered['Delay_Days'] > 0
        
        metrics = {
            'total_deliveries': len(delivered),
            'on_time_deliveries': len(delivered[delivered['Is_Delayed'] == False]),
            'delayed_deliveries': len(delivered[delivered['Is_Delayed'] == True]),
            'on_time_percentage': (len(delivered[delivered['Is_Delayed'] == False]) / len(delivered)) * 100,
            'avg_delay_days': delivered[delivered['Is_Delayed'] == True]['Delay_Days'].mean(),
            'avg_customer_rating': delivered['Customer_Rating'].mean(),
            'quality_issues': delivered['Quality_Issue'].notna().sum(),
            'quality_issue_rate': (delivered['Quality_Issue'].notna().sum() / len(delivered)) * 100
        }
        
        return metrics
    
    def calculate_cost_metrics(self):
        """Calculate cost-related metrics"""
        
        df = self.merged_data.copy()
        
        # Filter orders with cost data
        cost_data = df[df['Fuel_Cost'].notna()].copy()
        
        if len(cost_data) == 0:
            return None
        
        # Calculate total cost per order
        cost_columns = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 
                       'Insurance', 'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead']
        
        cost_data['Total_Cost'] = cost_data[cost_columns].sum(axis=1)
        cost_data['Profit'] = cost_data['Order_Value_INR'] - cost_data['Total_Cost'] - cost_data['Delivery_Cost_INR']
        cost_data['Profit_Margin_%'] = (cost_data['Profit'] / cost_data['Order_Value_INR']) * 100
        
        metrics = {
            'total_revenue': cost_data['Order_Value_INR'].sum(),
            'total_cost': cost_data['Total_Cost'].sum(),
            'total_delivery_cost': cost_data['Delivery_Cost_INR'].sum(),
            'total_profit': cost_data['Profit'].sum(),
            'avg_profit_margin': cost_data['Profit_Margin_%'].mean(),
            'avg_cost_per_order': cost_data['Total_Cost'].mean(),
            'highest_cost_category': None
        }
        
        # Find highest cost category
        cost_breakdown = {}
        for col in cost_columns:
            cost_breakdown[col] = cost_data[col].sum()
        
        metrics['highest_cost_category'] = max(cost_breakdown, key=cost_breakdown.get)
        metrics['cost_breakdown'] = cost_breakdown
        
        return metrics
    
    def calculate_route_metrics(self):
        """Calculate route and efficiency metrics"""
        
        df = self.merged_data.copy()
        route_data = df[df['Distance_KM'].notna()].copy()
        
        if len(route_data) == 0:
            return None
        
        # Calculate efficiency metrics
        route_data['Fuel_Efficiency'] = route_data['Distance_KM'] / route_data['Fuel_Consumption_L']
        route_data['Cost_per_KM'] = route_data['Fuel_Cost'] / route_data['Distance_KM']
        
        metrics = {
            'total_distance': route_data['Distance_KM'].sum(),
            'total_fuel_consumed': route_data['Fuel_Consumption_L'].sum(),
            'avg_fuel_efficiency': route_data['Fuel_Efficiency'].mean(),
            'total_toll_charges': route_data['Toll_Charges_INR'].sum(),
            'avg_traffic_delay': route_data['Traffic_Delay_Minutes'].mean(),
            'weather_impacted_routes': route_data['Weather_Impact'].notna().sum(),
            'avg_cost_per_km': route_data['Cost_per_KM'].mean()
        }
        
        return metrics
    
    def calculate_customer_metrics(self):
        """Calculate customer satisfaction metrics"""
        
        if self.feedback is None or len(self.feedback) == 0:
            return None
        
        feedback_df = self.feedback.copy()
        
        metrics = {
            'total_feedback': len(feedback_df),
            'avg_rating': feedback_df['Rating'].mean(),
            'would_recommend_count': feedback_df['Would_Recommend'].sum() if feedback_df['Would_Recommend'].dtype == 'bool' else len(feedback_df[feedback_df['Would_Recommend'] == 'Yes']),
            'recommendation_rate': None,
            'issues_reported': feedback_df['Issue_Category'].notna().sum(),
            'issue_rate': (feedback_df['Issue_Category'].notna().sum() / len(feedback_df)) * 100
        }
        
        # Calculate recommendation rate
        if feedback_df['Would_Recommend'].dtype == 'bool':
            metrics['recommendation_rate'] = (feedback_df['Would_Recommend'].sum() / len(feedback_df)) * 100
        else:
            yes_count = len(feedback_df[feedback_df['Would_Recommend'] == 'Yes'])
            metrics['recommendation_rate'] = (yes_count / len(feedback_df)) * 100
        
        # Issue breakdown
        if metrics['issues_reported'] > 0:
            issue_counts = feedback_df['Issue_Category'].value_counts().to_dict()
            metrics['top_issues'] = issue_counts
        
        return metrics
    
    def get_priority_analysis(self):
        """Analyze performance by priority level"""
        
        df = self.merged_data.copy()
        delivered = df[df['Delivery_Status'].notna()].copy()
        
        if len(delivered) == 0:
            return None
        
        delivered['Delay_Days'] = delivered['Actual_Delivery_Days'] - delivered['Promised_Delivery_Days']
        delivered['Is_Delayed'] = delivered['Delay_Days'] > 0
        
        priority_stats = delivered.groupby('Priority').agg({
            'Order_ID': 'count',
            'Is_Delayed': lambda x: (x == False).sum(),
            'Customer_Rating': 'mean',
            'Delivery_Cost_INR': 'mean'
        }).round(2)
        
        priority_stats.columns = ['Total_Orders', 'On_Time_Orders', 'Avg_Rating', 'Avg_Delivery_Cost']
        priority_stats['On_Time_%'] = ((priority_stats['On_Time_Orders'] / priority_stats['Total_Orders']) * 100).round(2)
        
        return priority_stats
    
    def get_carrier_analysis(self):
        """Analyze performance by carrier"""
        
        df = self.merged_data.copy()
        delivered = df[df['Carrier'].notna()].copy()
        
        if len(delivered) == 0:
            return None
        
        delivered['Delay_Days'] = delivered['Actual_Delivery_Days'] - delivered['Promised_Delivery_Days']
        delivered['Is_Delayed'] = delivered['Delay_Days'] > 0
        
        carrier_stats = delivered.groupby('Carrier').agg({
            'Order_ID': 'count',
            'Is_Delayed': lambda x: (x == False).sum(),
            'Customer_Rating': 'mean',
            'Delivery_Cost_INR': 'mean',
            'Quality_Issue': lambda x: x.notna().sum()
        }).round(2)
        
        carrier_stats.columns = ['Total_Deliveries', 'On_Time_Deliveries', 'Avg_Rating', 'Avg_Cost', 'Quality_Issues']
        carrier_stats['On_Time_%'] = ((carrier_stats['On_Time_Deliveries'] / carrier_stats['Total_Deliveries']) * 100).round(2)
        
        return carrier_stats.sort_values('Avg_Rating', ascending=False)
    
    def calculate_environmental_metrics(self):
        """Calculate environmental and sustainability metrics"""
        
        df = self.merged_data.copy()
        route_data = df[df['Distance_KM'].notna()].copy()
        
        if len(route_data) == 0:
            return None
        
        # Calculate CO2 emissions
        # Average CO2 emission: 2.68 kg per liter of diesel
        CO2_PER_LITER = 2.68
        route_data['CO2_Emissions_KG'] = route_data['Fuel_Consumption_L'] * CO2_PER_LITER
        
        # Get fleet CO2 data if available
        total_fleet_co2 = 0
        if self.fleet is not None and 'CO2_Emissions_Kg_per_KM' in self.fleet.columns:
            avg_fleet_co2_per_km = self.fleet['CO2_Emissions_Kg_per_KM'].mean()
        else:
            avg_fleet_co2_per_km = 0.15  # Default estimate
        
        metrics = {
            'total_co2_emissions_kg': route_data['CO2_Emissions_KG'].sum(),
            'total_co2_emissions_tonnes': route_data['CO2_Emissions_KG'].sum() / 1000,
            'avg_co2_per_delivery': route_data['CO2_Emissions_KG'].mean(),
            'avg_co2_per_km': (route_data['CO2_Emissions_KG'].sum() / route_data['Distance_KM'].sum()),
            'total_fuel_consumed': route_data['Fuel_Consumption_L'].sum(),
            'avg_fuel_efficiency': route_data['Distance_KM'].sum() / route_data['Fuel_Consumption_L'].sum(),
            'total_distance': route_data['Distance_KM'].sum(),
            'deliveries_analyzed': len(route_data)
        }
        
        # Calculate potential savings with 10% efficiency improvement
        metrics['potential_co2_reduction_10pct'] = metrics['total_co2_emissions_kg'] * 0.10
        metrics['potential_fuel_savings_10pct'] = metrics['total_fuel_consumed'] * 0.10
        
        # Calculate by priority (to see which priority is most/least green)
        if 'Priority' in route_data.columns:
            priority_emissions = route_data.groupby('Priority').agg({
                'CO2_Emissions_KG': 'sum',
                'Distance_KM': 'sum',
                'Fuel_Consumption_L': 'sum',
                'Order_ID': 'count'
            }).round(2)
            priority_emissions.columns = ['Total_CO2_KG', 'Total_Distance_KM', 'Total_Fuel_L', 'Deliveries']
            priority_emissions['Avg_CO2_per_Delivery'] = (priority_emissions['Total_CO2_KG'] / priority_emissions['Deliveries']).round(2)
            metrics['by_priority'] = priority_emissions
        
        return metrics
    
    def get_green_route_recommendations(self):
        """Identify most and least eco-friendly routes"""
        
        df = self.merged_data.copy()
        route_data = df[df['Route'].notna() & df['Distance_KM'].notna()].copy()
        
        if len(route_data) == 0:
            return None
        
        # Calculate CO2 per route
        CO2_PER_LITER = 2.68
        route_data['CO2_Emissions_KG'] = route_data['Fuel_Consumption_L'] * CO2_PER_LITER
        route_data['CO2_per_KM'] = route_data['CO2_Emissions_KG'] / route_data['Distance_KM']
        
        route_analysis = route_data.groupby('Route').agg({
            'CO2_Emissions_KG': 'sum',
            'CO2_per_KM': 'mean',
            'Distance_KM': 'mean',
            'Fuel_Consumption_L': 'mean',
            'Order_ID': 'count'
        }).round(2)
        
        route_analysis.columns = ['Total_CO2_KG', 'Avg_CO2_per_KM', 'Avg_Distance', 'Avg_Fuel_L', 'Deliveries']
        
        # Calculate efficiency score (lower is better/greener)
        route_analysis['Green_Score'] = (
            (1 / route_analysis['Avg_CO2_per_KM']) * 100
        ).round(2)
        
        return route_analysis.sort_values('Green_Score', ascending=False)
    
    def calculate_warehouse_metrics(self):
        """Calculate warehouse inventory and optimization metrics"""
        
        if self.inventory is None or len(self.inventory) == 0:
            return None
        
        warehouse_data = self.inventory.copy()
        
        # Group by warehouse location
        warehouse_summary = warehouse_data.groupby('Location').agg({
            'Current_Stock_Units': 'sum',
            'Reorder_Level': 'sum',
            'Storage_Cost_per_Unit': 'mean',
            'Product_Category': 'count'
        }).round(2)
        
        warehouse_summary.columns = ['Total_Stock', 'Total_Reorder_Level', 'Avg_Storage_Cost', 'Product_Categories']
        
        # Calculate stock health
        warehouse_summary['Stock_Status'] = warehouse_summary.apply(
            lambda row: 'Overstocked' if row['Total_Stock'] > row['Total_Reorder_Level'] * 1.5 
            else ('Critical' if row['Total_Stock'] < row['Total_Reorder_Level'] 
            else 'Healthy'), axis=1
        )
        
        # Overall metrics
        metrics = {
            'total_warehouses': len(warehouse_summary),
            'total_stock_units': warehouse_data['Current_Stock_Units'].sum(),
            'avg_stock_per_warehouse': warehouse_data.groupby('Location')['Current_Stock_Units'].sum().mean(),
            'total_storage_cost': (warehouse_data['Current_Stock_Units'] * warehouse_data['Storage_Cost_per_Unit']).sum(),
            'warehouses_needing_restock': len(warehouse_data[warehouse_data['Current_Stock_Units'] < warehouse_data['Reorder_Level']]),
            'warehouse_summary': warehouse_summary
        }
        
        # Identify rebalancing opportunities
        stock_by_category = warehouse_data.groupby('Product_Category')['Current_Stock_Units'].sum().to_dict()
        metrics['stock_by_category'] = stock_by_category
        
        return metrics
    
    def get_warehouse_rebalancing_suggestions(self):
        """Suggest inventory rebalancing between warehouses"""
        
        if self.inventory is None:
            return None
        
        warehouse_data = self.inventory.copy()
        
        # Calculate stock levels vs reorder levels
        warehouse_data['Stock_Ratio'] = warehouse_data['Current_Stock_Units'] / warehouse_data['Reorder_Level']
        warehouse_data['Needs_Action'] = warehouse_data['Stock_Ratio'].apply(
            lambda x: 'Transfer Out' if x > 2.0 else ('Restock' if x < 0.8 else 'Optimal')
        )
        
        suggestions = []
        
        # Group by product category
        for category in warehouse_data['Product_Category'].unique():
            category_data = warehouse_data[warehouse_data['Product_Category'] == category]
            
            # Find overstocked and understocked warehouses
            overstocked = category_data[category_data['Stock_Ratio'] > 2.0]
            understocked = category_data[category_data['Stock_Ratio'] < 0.8]
            
            for _, over_row in overstocked.iterrows():
                for _, under_row in understocked.iterrows():
                    if over_row['Location'] != under_row['Location']:
                        transfer_qty = min(
                            over_row['Current_Stock_Units'] - over_row['Reorder_Level'],
                            under_row['Reorder_Level'] - under_row['Current_Stock_Units']
                        )
                        
                        if transfer_qty > 0:
                            suggestions.append({
                                'Product_Category': category,
                                'From_Warehouse': over_row['Location'],
                                'To_Warehouse': under_row['Location'],
                                'Suggested_Transfer_Qty': int(transfer_qty),
                                'Priority': 'High' if under_row['Stock_Ratio'] < 0.5 else 'Medium'
                            })
        
        if suggestions:
            return pd.DataFrame(suggestions)
        return None


# Test the analytics
if __name__ == "__main__":
    from data_loader import load_all_data
    
    # Load data
    data = load_all_data()
    
    # Initialize analytics
    analytics = LogisticsAnalytics(data)
    
    # Calculate and display metrics
    print("\n" + "="*60)
    print("DELIVERY PERFORMANCE METRICS")
    print("="*60)
    delivery_metrics = analytics.calculate_delivery_metrics()
    if delivery_metrics:
        for key, value in delivery_metrics.items():
            print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("COST METRICS")
    print("="*60)
    cost_metrics = analytics.calculate_cost_metrics()
    if cost_metrics:
        for key, value in cost_metrics.items():
            if key != 'cost_breakdown':
                print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("ROUTE METRICS")
    print("="*60)
    route_metrics = analytics.calculate_route_metrics()
    if route_metrics:
        for key, value in route_metrics.items():
            print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("CUSTOMER METRICS")
    print("="*60)
    customer_metrics = analytics.calculate_customer_metrics()
    if customer_metrics:
        for key, value in customer_metrics.items():
            if key != 'top_issues':
                print(f"{key}: {value}")