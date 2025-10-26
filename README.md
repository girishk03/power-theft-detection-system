# Power Theft Detection in Smart Grids Using AI-Based Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Project Overview

This project implements an **AI-based Intrusion Detection System (IDS)** for detecting power theft in smart grid networks. Using advanced deep learning techniques including **CNN-LSTM**, **LSTM**, and **Neural Networks**, the system analyzes smart meter consumption patterns to identify anomalies and potential electricity theft in real-time.

### 🎯 Objectives

- **Develop an AI-based Intrusion Detection System** using deep learning techniques
- **Handle data-related challenges** effectively (missing values, class imbalance)
- **Enhance detection performance** through advanced feature engineering
- **Provide real-time monitoring** and automated alert generation
- **Support energy security** and grid sustainability

### ⚡ Key Features

- **Real-time Theft Detection**: Continuous monitoring of power consumption patterns
- **Multiple AI Models**: CNN-LSTM, LSTM, Neural Networks, and traditional ML models
- **Advanced Feature Engineering**: Time-domain and frequency-domain feature extraction
- **Anomaly Detection**: Statistical and ML-based anomaly detection
- **Class Imbalance Handling**: SMOTE implementation for balanced training
- **Interactive Dashboard**: Web-based monitoring interface
- **Automated Alerts**: Risk-based alert system (HIGH/MEDIUM/LOW)
- **Comprehensive Visualization**: Training metrics, ROC curves, confusion matrices

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Smart Meter Data                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preprocessing Module                      │
│  • Missing Value Handling (Interpolation)                   │
│  • Feature Engineering (Time & Statistical Features)        │
│  • SMOTE for Class Imbalance                                │
│  • Normalization & Scaling                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Deep Learning Models                           │
│  • CNN-LSTM Hybrid Model                                    │
│  • LSTM Model                                               │
│  • Deep Neural Network                                      │
│  • Traditional ML (RF, SVM, GB, DT)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Intrusion Detection System (IDS)                    │
│  • Real-time Theft Detection                                │
│  • Risk Classification (HIGH/MEDIUM/LOW)                    │
│  • Anomaly Detection                                        │
│  • Alert Generation & Management                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Web Dashboard & Monitoring                     │
│  • Real-time Metrics                                        │
│  • Alert Management                                         │
│  • Visualization & Reports                                  │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework
- **PyTorch**: Alternative deep learning framework
- **Scikit-learn**: Machine learning algorithms

### Data Processing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Imbalanced-learn**: SMOTE implementation
- **SciPy**: Statistical analysis

### Visualization
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical plots
- **Plotly**: Interactive dashboards

### Web Framework
- **Flask**: Web application framework
- **Flask-CORS**: Cross-origin resource sharing

### Additional Libraries
- **LightGBM**: Gradient boosting
- **XGBoost**: Extreme gradient boosting

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster training

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd power-theft-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Train the Models
Run the complete training pipeline:

```bash
python main.py
```

This will:
- Generate sample smart meter data (or load your own)
- Preprocess the data
- Train multiple models (Neural Network, LSTM, Random Forest, etc.)
- Evaluate model performance
- Generate visualizations
- Save trained models

### 2. Launch the Web Dashboard
Start the Flask web application:

```bash
python app.py
```

Access the dashboard at: **http://localhost:8105**

### 3. Use the Dashboard
1. Click **"Initialize System"** to load the trained model
2. View real-time statistics and metrics
3. Click **"Simulate Detection"** to test the system
4. Monitor alerts and acknowledge them
5. Export reports and detection logs

## 📊 Dataset

### Sample Data Generation
The system includes a built-in sample data generator that creates realistic smart meter consumption data:

```python
from src.data_preprocessing import generate_sample_data

# Generate 10,000 samples with 10% theft rate
df = generate_sample_data(n_samples=10000, theft_ratio=0.1)
```

### Using Your Own Data
To use your own smart meter data, ensure it has the following structure:

| Column | Description | Type |
|--------|-------------|------|
| `timestamp` | Date and time of reading | datetime |
| `user_id` | Unique user identifier | int/string |
| `consumption` | Power consumption (kWh) | float |
| `is_theft` | Theft label (0=Normal, 1=Theft) | int |

Place your CSV file in `data/raw/` and modify `main.py`:

```python
df = pd.read_csv('data/raw/your_data.csv')
```

## 🧠 Models

### 1. CNN-LSTM Hybrid Model
Combines Convolutional layers for feature extraction with LSTM for temporal pattern recognition.

**Architecture:**
- CNN layers: Feature extraction from consumption patterns
- LSTM layers: Temporal sequence learning
- Dense layers: Classification

**Best for:** Sequential time-series data with complex patterns

### 2. LSTM Model
Specialized for sequential data analysis.

**Architecture:**
- Multiple LSTM layers with dropout
- Batch normalization
- Dense output layer

**Best for:** Time-series consumption patterns

### 3. Deep Neural Network
Fully connected neural network for tabular data.

**Architecture:**
- Multiple hidden layers (256→128→64→32)
- Dropout and batch normalization
- Binary classification output

**Best for:** Preprocessed feature vectors

### 4. Traditional ML Models
- **Random Forest**: Ensemble decision trees
- **Gradient Boosting**: Boosted decision trees
- **SVM**: Support Vector Machine
- **Decision Tree**: Single decision tree

## 📈 Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to theft cases
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: True/False positives and negatives

## 🔍 Feature Engineering

### Time-Based Features
- Hour, day, month, year
- Day of week, weekend indicator
- Peak hour indicator
- Cyclical encoding (sin/cos)

### Statistical Features
- Rolling mean, std, min, max (24h, 1 week, 1 month)
- Rate of change
- Percentage change
- Skewness and kurtosis

### User Profile Features
- User average consumption
- Deviation from user baseline
- User consumption patterns

### Anomaly Features
- Z-score based anomaly detection
- Sudden drop detection
- Consumption pattern irregularities

## 🚨 Intrusion Detection System

### Detection Process
1. **Data Collection**: Receive consumption reading
2. **Preprocessing**: Extract features
3. **Prediction**: Run through trained model
4. **Risk Classification**: Classify as HIGH/MEDIUM/LOW/NORMAL
5. **Alert Generation**: Create alerts for theft cases
6. **Logging**: Record all detections

### Risk Levels
- **HIGH**: Probability ≥ 80% - Immediate investigation required
- **MEDIUM**: Probability ≥ 50% - Investigation recommended
- **LOW**: Probability ≥ 30% - Monitor closely
- **NORMAL**: Probability < 30% - No action needed

## 📱 Web Dashboard Features

- **Real-time Monitoring**: Live system status and metrics
- **Detection Statistics**: Total detections, theft rate, accuracy
- **Alert Management**: View and acknowledge alerts
- **Risk Distribution**: Visualization of risk levels
- **Simulation Mode**: Test the system with random data
- **Export Functionality**: Download alerts and detection logs

## 📁 Project Structure

```
power-theft-detection/
├── src/
│   ├── data_preprocessing.py    # Data preprocessing and feature engineering
│   ├── models.py                # Deep learning and ML models
│   ├── intrusion_detection.py  # IDS implementation
│   └── visualization.py         # Plotting and visualization
├── templates/
│   └── index.html              # Web dashboard template
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Preprocessed data
├── models/                     # Saved trained models
├── results/
│   ├── plots/                  # Visualization outputs
│   └── reports/                # Training reports
├── static/                     # Static files for web app
├── config.py                   # Configuration settings
├── main.py                     # Main training script
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model parameters
MODEL_PARAMS = {
    'neural_network': {
        'hidden_layers': [256, 128, 64, 32],
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 100
    }
}

# Detection thresholds
ANOMALY_THRESHOLD = 0.7
THEFT_PROBABILITY_THRESHOLD = 0.5

# SMOTE configuration
SMOTE_SAMPLING_STRATEGY = 0.5
```

## 📊 Results & Visualization

The system generates comprehensive visualizations:

1. **Training History**: Loss and accuracy curves
2. **ROC Curves**: Model performance comparison
3. **Confusion Matrices**: Classification results
4. **Precision-Recall Curves**: Trade-off analysis
5. **Feature Importance**: Most influential features
6. **Alert Timeline**: Temporal distribution of alerts
7. **Consumption Patterns**: Normal vs. theft patterns

## 🎓 Research References

Based on literature survey including:

1. **Smart grids based on deep learning** (Noor Mahmoud Ibrahim et al., 2021)
2. **Electricity Theft Detection in Smart Grid Systems: A CNN-LSTM Based Approach** (Md. Nazmul Hasan et al., 2019)
3. **An Ensemble Deep Convolutional Neural Network Model for Electricity Theft Detection in Smart Grids** (Hossein Mohammadi Rouzbahani et al., 2021)
4. **An Intelligent Framework for Electricity Theft Detection in Smart Grid** (Yogesh Kulkarni et al., 2021)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Ravisha**
- AI/ML Engineer & Researcher
- Specialization: Smart Grid Security & Intrusion Detection Systems

## 🙏 Acknowledgments

- Anurag University - Department of Computer Science & Engineering
- Research papers and authors cited in the literature survey
- Open-source community for libraries and frameworks

## 📧 Contact

For questions, suggestions, or collaboration opportunities:
- Email: rohit@example.com
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [Your GitHub Profile]

---

**Built with ❤️ for a smarter and more secure energy future**

*Last Updated: October 2025*
