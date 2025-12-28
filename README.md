# 🚚 NexGen Logistics Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52.2-red.svg)
![ML](https://img.shields.io/badge/ML-95.33%25%20Accuracy-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

An end-to-end analytics solution designed to transform logistics operations from reactive to predictive. Built with Python and Streamlit, it provides real-time insights, AI-powered predictions, and actionable recommendations to optimize delivery performance, reduce costs, and improve customer satisfaction.

---

## Preview



---

## Features

### **1. Overview Dashboard**
- **Real-time KPIs**: Total deliveries, customer ratings, revenue, distance covered
- **Performance at a Glance**: Delivery success rates, fleet utilization
- **Critical Alerts**: Automated identification of operational issues
- **Interactive Visualizations**: Pie charts, bar graphs, and trend analysis

### **2. Delivery Performance Analytics**
- On-time vs delayed delivery tracking
- Performance breakdown by priority level (Express/Standard/Economy)
- Carrier performance comparison with ratings and costs
- Quality issue monitoring and root cause analysis

### **3. Cost Intelligence Platform**
- Revenue and profit tracking across product categories
- Detailed cost breakdown (Fuel, Labor, Maintenance, Insurance, etc.)
- Profit margin analysis
- Cost optimization opportunities identification
- Identifies highest cost categories for targeted improvements

### **4. Route Optimization**
- Route efficiency rankings based on cost, time, and distance
- Fuel consumption analysis per route
- Traffic impact assessment
- Best carrier recommendations
- Weather impact tracking
- Downloadable route analysis reports

### **5. Customer Experience Dashboard**
- Customer satisfaction trend analysis with moving averages
- Feedback sentiment tracking
- Issue category breakdown (Delivery, Damage, Wrong Item, etc.)
- Recommendation rate monitoring
- At-risk customer identification

### **6. AI-Powered Predictions**
- **95.33% Accurate** delay prediction model
- Risk level assessment (Low/Medium/High)
- Automated corrective action suggestions
- Interactive prediction tool for new orders
- Feature importance analysis (Distance, Product Category, Traffic, etc.)
- Smart carrier recommendation engine

### **7. Dynamic Fleet Manager**
- Real-time fleet status monitoring
- Vehicle utilization tracking
- **Smart Vehicle Matching**: AI-powered vehicle-to-order assignment based on:
  - Product category and special handling requirements
  - Order priority and value
  - Distance and fuel efficiency
  - Vehicle availability and capacity
- Fleet performance analysis by vehicle type
- Maintenance alerts and recommendations
- Age and efficiency tracking

### **8. Warehouse Optimization Tool**
- Multi-warehouse inventory tracking (5 locations)
- Stock level monitoring vs reorder points
- Warehouse health status (Healthy/Overstocked/Critical)
- **Intelligent Rebalancing**: Automated suggestions for inventory transfers
- Stock distribution by product category
- Storage cost analysis
- Downloadable rebalancing plans

### **9. Sustainability Tracker**
- Carbon footprint calculation (CO2 emissions tracking)
- Fuel consumption monitoring
- Environmental impact by priority level
- **Eco-friendly route identification**: Green score ranking
- Optimization potential calculator (10% improvement scenarios)
- Sustainability goals and progress tracking
- Actionable green recommendations

### **10. Reports & Data Export**
- Custom report generation (Operations, Delivery, Cost, Customer)
- Complete dataset export functionality
- Comprehensive metrics download
- Executive summary generation

---

## Tech Stack

### Core Technologies
- **Python 3.8+**: Backend logic and data processing
- **Streamlit 1.52.2**: Interactive web application framework
- **Pandas 2.2.2**: Data manipulation and analysis
- **NumPy 1.26.4**: Numerical computing

### Data Visualization
- **Plotly 6.5.0**: Interactive charts and graphs
- **Matplotlib**: Additional plotting capabilities

### Machine Learning
- **Scikit-learn 1.8.0**: Predictive modeling (Random Forest Classifier)
- **SciPy 1.13.0**: Statistical analysis

### Additional Libraries
- **OpenPyXL 3.1.5**: Excel file handling
- **Python-dotenv**: Environment variable management

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/nexgen-logistics-platform.git
cd nexgen-logistics-platform
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy plotly scipy scikit-learn openpyxl
```

### Step 3: Prepare Data

Ensure all CSV files are in the `data/` folder:
- orders.csv
- delivery_performance.csv
- routes_distance.csv
- vehicle_fleet.csv
- warehouse_inventory.csv
- customer_feedback.csv
- cost_breakdown.csv

### Step 4: Run the Application
```bash
streamlit run app.py
```

Or if `streamlit` command is not recognized:
```bash
python -m streamlit run app.py
```

The dashboard will automatically open at `http://localhost:8501`

---

## Usage

### Navigation

The platform features a sidebar with 10 main sections:

1. **🏠 Overview**: Start here for a high-level view of operations
2. **📊 Delivery Performance**: Analyze delivery metrics and carrier performance
3. **💰 Cost Analysis**: Understand revenue, costs, and profitability
4. **🗺️ Route Optimization**: Identify efficient routes and reduce costs
5. **👥 Customer Insights**: Monitor satisfaction and identify issues
6. **🤖 AI Predictions**: Predict delays and get recommendations
7. **🚛 Fleet Manager**: Optimize vehicle assignments and track maintenance
8. **📦 Warehouse Optimizer**: Balance inventory across warehouses
9. **🌱 Sustainability**: Track environmental impact and get green recommendations
10. **📈 Reports**: Generate custom reports and export data

### Key Workflows

#### Predicting Delivery Delays
1. Navigate to **AI Predictions**
2. Enter order details (priority, category, distance, etc.)
3. Click "Predict Delay Risk"
4. Review risk level and automated recommendations

#### Smart Vehicle Matching
1. Go to **Fleet Manager**
2. Use the "Smart Vehicle Matching Tool"
3. Input order requirements
4. Get AI-powered vehicle recommendation with cost estimates

#### Warehouse Rebalancing
1. Open **Warehouse Optimizer**
2. Review stock status across warehouses
3. Check "Rebalancing Recommendations" section
4. Download rebalancing plan CSV

#### Environmental Impact Analysis
1. Visit **Sustainability** page
2. Review current CO2 emissions and fuel consumption
3. Explore eco-friendly routes
4. Check optimization potential

---

## Project Structure
```
NexGen_Logistics/
├── .streamlit/
│   └── config.toml              # Streamlit theme configuration
├── data/
│   ├── orders.csv               # Order information (200 records)
│   ├── delivery_performance.csv # Delivery metrics (150 records)
│   ├── routes_distance.csv      # Route data (150 records)
│   ├── vehicle_fleet.csv        # Fleet information (50 vehicles)
│   ├── warehouse_inventory.csv  # Inventory data (35 records)
│   ├── customer_feedback.csv    # Customer feedback (83 records)
│   └── cost_breakdown.csv       # Cost details (150 records)
├── app.py                       # Main Streamlit application (Frontend)
├── analytics.py                 # Core analytics engine (Backend)
├── data_loader.py               # Data loading utilities
├── predictive_models.py         # ML models and optimization algorithms
├── visualizations.py            # Chart generation functions
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
```

---

## Key Insights

### Current Performance Metrics

Based on analysis of 200 orders and 150 completed deliveries:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| On-Time Delivery Rate | 53.3% | 90%+ | 🔴 Critical |
| Customer Rating | 3.64/5.0 | 4.5+ | 🟡 Needs Improvement |
| Total Revenue | ₹282,805 | - | 📊 Baseline |
| Total Profit | ₹102,160 | - | 📊 Baseline |
| Total Distance | 300,552 km | - | 📊 Baseline |
| Fuel Efficiency | 8.31 km/L | 10+ | 🟡 Below Target |
| CO2 Emissions | 96.7 tonnes | - | 🌱 Track & Reduce |

### Top Operational Issues

1. **High Delay Rate**: 70 out of 150 deliveries (46.7%) are delayed
2. **Quality Issues**: 100% of deliveries have some quality concerns flagged
3. **Fuel Costs**: Highest cost category at 27.4% of total operational costs
4. **Traffic Impact**: Average 34 minutes delay per delivery
5. **Customer Satisfaction**: Only 73.5% would recommend the service

### Optimization Opportunities

- **Route Optimization**: Potential 15-20% fuel savings through better routing
- **Fleet Efficiency**: Upgrading older vehicles could reduce emissions by 10%+
- **Warehouse Rebalancing**: Identified multiple rebalancing opportunities
- **Predictive Maintenance**: Several vehicles flagged for maintenance

---

## AI Model Performance

### Delay Prediction Model

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 95.33%
- **Training Data**: 150 completed deliveries
- **Features**: 6 input variables

#### Feature Importance

| Feature | Importance | Impact |
|---------|-----------|---------|
| Distance (KM) | 31.3% | Primary delay factor |
| Product Category | 19.3% | Significant impact |
| Traffic Delay | 18.6% | Controllable factor |
| Carrier | 14.9% | Performance variance |
| Customer Segment | 8.0% | Minor influence |
| Priority | 7.8% | Minor influence |

#### Prediction Capabilities

- ✅ Predicts delivery delays before they occur
- ✅ Provides risk level classification (Low/Medium/High)
- ✅ Suggests corrective actions automatically
- ✅ Works with real-time order data

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
