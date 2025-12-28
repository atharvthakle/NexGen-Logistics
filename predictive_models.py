import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class DeliveryPredictor:
    """
    Predicts delivery delays and suggests corrective actions
    """
    
    def __init__(self, merged_data):
        self.data = merged_data.copy()
        self.model = None
        self.label_encoders = {}
        self.feature_importance = None
        
    def prepare_prediction_data(self):
        """Prepare data for machine learning"""
        
        # Filter delivered orders only
        df = self.data[self.data['Delivery_Status'].notna()].copy()
        
        if len(df) == 0:
            return None, None
        
        # Create target variable (1 = delayed, 0 = on time)
        df['Delay_Days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
        df['Is_Delayed'] = (df['Delay_Days'] > 0).astype(int)
        
        # Select features for prediction
        feature_columns = ['Priority', 'Product_Category', 'Distance_KM', 
                          'Traffic_Delay_Minutes', 'Customer_Segment', 'Carrier']
        
        # Remove rows with missing values in feature columns
        df_clean = df[feature_columns + ['Is_Delayed']].dropna()
        
        if len(df_clean) < 20:  # Need minimum data for training
            return None, None
        
        # Encode categorical variables
        for col in ['Priority', 'Product_Category', 'Customer_Segment', 'Carrier']:
            if col in df_clean.columns:
                self.label_encoders[col] = LabelEncoder()
                df_clean[col] = self.label_encoders[col].fit_transform(df_clean[col].astype(str))
        
        # Separate features and target
        X = df_clean[feature_columns]
        y = df_clean['Is_Delayed']
        
        return X, y
    
    def train_model(self):
        """Train the delay prediction model"""
        
        X, y = self.prepare_prediction_data()
        
        if X is None:
            print("✗ Insufficient data for training")
            return False
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        self.model.fit(X, y)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"✓ Model trained successfully on {len(X)} orders")
        print(f"✓ Model accuracy: {self.model.score(X, y)*100:.2f}%")
        
        return True
    
    def predict_delay_risk(self, order_data):
        """
        Predict if an order will be delayed
        order_data: dict with keys like Priority, Product_Category, etc.
        """
        
        if self.model is None:
            return None
        
        # Prepare single order data
        order_df = pd.DataFrame([order_data])
        
        # Encode categorical variables
        for col in ['Priority', 'Product_Category', 'Customer_Segment', 'Carrier']:
            if col in order_df.columns and col in self.label_encoders:
                try:
                    order_df[col] = self.label_encoders[col].transform(order_df[col].astype(str))
                except:
                    order_df[col] = 0  # Default for unseen categories
        
        # Predict
        delay_probability = self.model.predict_proba(order_df)[0][1]
        will_delay = self.model.predict(order_df)[0]
        
        # Risk level
        if delay_probability < 0.3:
            risk_level = "Low"
        elif delay_probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'will_delay': bool(will_delay),
            'delay_probability': delay_probability * 100,
            'risk_level': risk_level
        }
    
    def suggest_improvements(self, order_data, prediction):
        """Suggest corrective actions based on prediction"""
        
        suggestions = []
        
        if prediction['risk_level'] == "High":
            suggestions.append("⚠️ HIGH RISK: Consider upgrading to Express shipping")
            
            if order_data.get('Traffic_Delay_Minutes', 0) > 30:
                suggestions.append("🚗 Heavy traffic expected - Consider alternate routes")
            
            if order_data.get('Distance_KM', 0) > 500:
                suggestions.append("📍 Long distance - Use faster carrier or direct route")
            
            suggestions.append("📞 Proactive customer communication recommended")
        
        elif prediction['risk_level'] == "Medium":
            suggestions.append("⚡ Monitor closely for potential delays")
            suggestions.append("🔄 Have backup carrier ready")
        
        else:
            suggestions.append("✅ Low risk - Standard processing recommended")
        
        return suggestions
    
    def get_top_delay_factors(self):
        """Return the most important factors causing delays"""
        
        if self.feature_importance is None:
            return None
        
        return self.feature_importance


class RouteOptimizer:
    """
    Suggests optimal routes and carriers
    """
    
    def __init__(self, merged_data, fleet_data):
        self.data = merged_data
        self.fleet = fleet_data
    
    def find_best_carrier(self, priority, distance):
        """Find best carrier based on priority and distance"""
        
        df = self.data[self.data['Carrier'].notna()].copy()
        
        if len(df) == 0:
            return None
        
        # Calculate carrier performance
        carrier_performance = df.groupby('Carrier').agg({
            'Customer_Rating': 'mean',
            'Delivery_Cost_INR': 'mean',
            'Actual_Delivery_Days': 'mean'
        }).round(2)
        
        carrier_performance.columns = ['Avg_Rating', 'Avg_Cost', 'Avg_Days']
        
        # Score carriers (higher is better)
        carrier_performance['Score'] = (
            carrier_performance['Avg_Rating'] * 0.5 -  # Rating is important
            (carrier_performance['Avg_Days'] / 10) * 0.3 -  # Faster is better
            (carrier_performance['Avg_Cost'] / 1000) * 0.2  # Lower cost is better
        )
        
        best_carrier = carrier_performance.sort_values('Score', ascending=False).iloc[0]
        
        return {
            'recommended_carrier': carrier_performance.sort_values('Score', ascending=False).index[0],
            'expected_rating': best_carrier['Avg_Rating'],
            'expected_cost': best_carrier['Avg_Cost'],
            'expected_days': best_carrier['Avg_Days']
        }
    
    def calculate_route_efficiency(self):
        """Calculate which routes are most/least efficient"""
        
        df = self.data[self.data['Route'].notna()].copy()
        
        if len(df) == 0:
            return None
        
        df['Cost_per_KM'] = df['Fuel_Cost'] / df['Distance_KM']
        df['Delay_Days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
        
        route_efficiency = df.groupby('Route').agg({
            'Order_ID': 'count',
            'Cost_per_KM': 'mean',
            'Distance_KM': 'mean',
            'Traffic_Delay_Minutes': 'mean',
            'Delay_Days': 'mean'
        }).round(2)
        
        route_efficiency.columns = ['Total_Orders', 'Avg_Cost_per_KM', 'Avg_Distance', 'Avg_Traffic_Delay', 'Avg_Delay']
        
        # Efficiency score (higher is better)
        route_efficiency['Efficiency_Score'] = (
            100 / (route_efficiency['Avg_Cost_per_KM'] * 100) * 0.4 +
            100 / (route_efficiency['Avg_Traffic_Delay'] + 1) * 0.3 +
            100 / (route_efficiency['Avg_Delay'].abs() + 1) * 0.3
        )
        
        return route_efficiency.sort_values('Efficiency_Score', ascending=False)
    
class FleetManager:
    """
    Dynamic fleet management system
    Matches vehicles to orders based on multiple constraints
    """
    
    def __init__(self, fleet_data, merged_data):
        self.fleet = fleet_data
        self.orders = merged_data
    
    def get_fleet_overview(self):
        """Get comprehensive fleet overview"""
        
        if self.fleet is None:
            return None
        
        fleet_df = self.fleet.copy()
        
        overview = {
            'total_vehicles': len(fleet_df),
            'by_type': fleet_df['Vehicle_Type'].value_counts().to_dict(),
            'by_status': fleet_df['Status'].value_counts().to_dict(),
            'avg_age': fleet_df['Age_Years'].mean(),
            'total_capacity': fleet_df['Capacity_KG'].sum(),
            'avg_fuel_efficiency': fleet_df['Fuel_Efficiency_KM_per_L'].mean(),
            'avg_co2_per_km': fleet_df['CO2_Emissions_Kg_per_KM'].mean()
        }
        
        # Calculate utilization
        available_vehicles = len(fleet_df[fleet_df['Status'] == 'Available'])
        overview['utilization_rate'] = ((overview['total_vehicles'] - available_vehicles) / overview['total_vehicles']) * 100
        
        return overview
    
    def match_vehicle_to_order(self, order_details):
        """
        Match the best vehicle to an order based on:
        - Product category and special handling
        - Order value and priority
        - Distance and fuel efficiency
        - Vehicle availability
        """
        
        if self.fleet is None:
            return None
        
        available_fleet = self.fleet[self.fleet['Status'] == 'Available'].copy()
        
        if len(available_fleet) == 0:
            return {'error': 'No vehicles available'}
        
        # Score each vehicle
        available_fleet['Match_Score'] = 0
        
        # Factor 1: Capacity match (assume order weight based on value)
        # High value orders = heavier (rough estimate)
        estimated_weight = min(order_details.get('order_value', 1000) / 10, 5000)
        available_fleet['Capacity_Score'] = available_fleet['Capacity_KG'].apply(
            lambda cap: 10 if cap >= estimated_weight * 1.2 else (5 if cap >= estimated_weight else 1)
        )
        
        # Factor 2: Fuel efficiency (important for long distances)
        distance = order_details.get('distance_km', 100)
        if distance > 500:
            available_fleet['Efficiency_Score'] = (available_fleet['Fuel_Efficiency_KM_per_L'] / available_fleet['Fuel_Efficiency_KM_per_L'].max()) * 10
        else:
            available_fleet['Efficiency_Score'] = 5  # Less important for short distances
        
        # Factor 3: Vehicle type match
        priority = order_details.get('priority', 'Standard')
        product_category = order_details.get('product_category', 'General')
        
        def get_type_score(vehicle_type):
            if priority == 'Express' and vehicle_type in ['Van', 'Express Bike']:
                return 10
            elif product_category in ['Food & Beverage', 'Healthcare'] and vehicle_type == 'Refrigerated Truck':
                return 10
            elif vehicle_type == 'Truck':
                return 7
            else:
                return 5
        
        available_fleet['Type_Score'] = available_fleet['Vehicle_Type'].apply(get_type_score)
        
        # Factor 4: Vehicle age (newer is better)
        available_fleet['Age_Score'] = 10 - (available_fleet['Age_Years'] / available_fleet['Age_Years'].max() * 5)
        
        # Factor 5: Environmental score (lower emissions better)
        available_fleet['Env_Score'] = (1 - (available_fleet['CO2_Emissions_Kg_per_KM'] / available_fleet['CO2_Emissions_Kg_per_KM'].max())) * 5
        
        # Calculate total match score
        available_fleet['Match_Score'] = (
            available_fleet['Capacity_Score'] * 0.3 +
            available_fleet['Efficiency_Score'] * 0.25 +
            available_fleet['Type_Score'] * 0.25 +
            available_fleet['Age_Score'] * 0.1 +
            available_fleet['Env_Score'] * 0.1
        )
        
        # Get best match
        best_match = available_fleet.loc[available_fleet['Match_Score'].idxmax()]
        
        # Calculate estimated costs
        fuel_cost = (distance / best_match['Fuel_Efficiency_KM_per_L']) * 100  # Assuming ₹100 per liter
        co2_emissions = distance * best_match['CO2_Emissions_Kg_per_KM']
        
        return {
            'recommended_vehicle_id': best_match['Vehicle_ID'],
            'vehicle_type': best_match['Vehicle_Type'],
            'capacity_kg': best_match['Capacity_KG'],
            'fuel_efficiency': best_match['Fuel_Efficiency_KM_per_L'],
            'match_score': round(best_match['Match_Score'], 2),
            'estimated_fuel_cost': round(fuel_cost, 2),
            'estimated_co2_kg': round(co2_emissions, 2),
            'current_location': best_match['Current_Location'],
            'vehicle_age': best_match['Age_Years']
        }
    
    def get_fleet_performance_analysis(self):
        """Analyze fleet performance metrics"""
        
        if self.fleet is None:
            return None
        
        fleet_df = self.fleet.copy()
        
        # Performance by vehicle type
        type_analysis = fleet_df.groupby('Vehicle_Type').agg({
            'Vehicle_ID': 'count',
            'Capacity_KG': 'mean',
            'Fuel_Efficiency_KM_per_L': 'mean',
            'CO2_Emissions_Kg_per_KM': 'mean',
            'Age_Years': 'mean'
        }).round(2)
        
        type_analysis.columns = ['Count', 'Avg_Capacity', 'Avg_Fuel_Efficiency', 'Avg_CO2', 'Avg_Age']
        
        # Calculate efficiency score
        type_analysis['Efficiency_Score'] = (
            (type_analysis['Avg_Fuel_Efficiency'] / type_analysis['Avg_Fuel_Efficiency'].max()) * 50 +
            (1 - (type_analysis['Avg_CO2'] / type_analysis['Avg_CO2'].max())) * 50
        ).round(2)
        
        return type_analysis.sort_values('Efficiency_Score', ascending=False)
    
    def get_maintenance_alerts(self):
        """Identify vehicles needing maintenance"""
        
        if self.fleet is None:
            return None
        
        fleet_df = self.fleet.copy()
        
        alerts = []
        
        for _, vehicle in fleet_df.iterrows():
            if vehicle['Age_Years'] > 8:
                alerts.append({
                    'Vehicle_ID': vehicle['Vehicle_ID'],
                    'Alert_Type': 'High Age',
                    'Priority': 'High',
                    'Message': f"Vehicle is {vehicle['Age_Years']} years old - Consider replacement"
                })
            
            if vehicle['Fuel_Efficiency_KM_per_L'] < fleet_df['Fuel_Efficiency_KM_per_L'].mean() * 0.7:
                alerts.append({
                    'Vehicle_ID': vehicle['Vehicle_ID'],
                    'Alert_Type': 'Low Efficiency',
                    'Priority': 'Medium',
                    'Message': f"Fuel efficiency ({vehicle['Fuel_Efficiency_KM_per_L']} km/L) below average"
                })
            
            if vehicle['CO2_Emissions_Kg_per_KM'] > fleet_df['CO2_Emissions_Kg_per_KM'].mean() * 1.3:
                alerts.append({
                    'Vehicle_ID': vehicle['Vehicle_ID'],
                    'Alert_Type': 'High Emissions',
                    'Priority': 'Medium',
                    'Message': f"CO2 emissions ({vehicle['CO2_Emissions_Kg_per_KM']} kg/km) above average"
                })
        
        if alerts:
            return pd.DataFrame(alerts)
        return None

# Test the predictive models
if __name__ == "__main__":
    from data_loader import load_all_data
    from analytics import LogisticsAnalytics
    
    # Load data
    data = load_all_data()
    analytics = LogisticsAnalytics(data)
    
    # Test Delay Predictor
    print("\n" + "="*60)
    print("TRAINING DELAY PREDICTION MODEL")
    print("="*60)
    
    predictor = DeliveryPredictor(analytics.merged_data)
    predictor.train_model()
    
    print("\n" + "="*60)
    print("TOP DELAY FACTORS")
    print("="*60)
    if predictor.feature_importance is not None:
        print(predictor.feature_importance)
    
    # Test prediction on sample order
    print("\n" + "="*60)
    print("SAMPLE PREDICTION")
    print("="*60)
    
    sample_order = {
        'Priority': 'Express',
        'Product_Category': 'Electronics',
        'Distance_KM': 450,
        'Traffic_Delay_Minutes': 45,
        'Customer_Segment': 'Enterprise',
        'Carrier': 'BlueDart'
    }
    
    prediction = predictor.predict_delay_risk(sample_order)
    if prediction:
        print(f"Will Delay: {prediction['will_delay']}")
        print(f"Delay Probability: {prediction['delay_probability']:.2f}%")
        print(f"Risk Level: {prediction['risk_level']}")
        print("\nSuggested Actions:")
        for suggestion in predictor.suggest_improvements(sample_order, prediction):
            print(f"  {suggestion}")
    
    # Test Route Optimizer
    print("\n" + "="*60)
    print("ROUTE EFFICIENCY ANALYSIS")
    print("="*60)
    
    optimizer = RouteOptimizer(analytics.merged_data, data['fleet'])
    route_efficiency = optimizer.calculate_route_efficiency()
    if route_efficiency is not None:
        print("\nTop 5 Most Efficient Routes:")
        print(route_efficiency.head())