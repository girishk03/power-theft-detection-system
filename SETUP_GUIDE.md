# 🚀 Power Theft Detection System - Setup Guide

## 📦 For Teammates: Quick Setup

### Option 1: Using the Zip File

1. **Extract the zip file**:
   ```bash
   unzip power-theft-detection-v1.0.zip
   cd power-theft-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the dashboard**:
   - Open browser: http://localhost:8105
   - Login with:
     - Username: `admin`
     - Password: `password`

### Option 2: Clone from GitHub

1. **Clone the repository**:
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   cd power-theft-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

---

## 🌐 Railway Deployment (Already Configured)

The project is configured for automatic Railway deployment:

### Files for Railway:
- ✅ `Procfile` - Tells Railway how to start the app
- ✅ `runtime.txt` - Specifies Python version
- ✅ `railway.json` - Railway configuration
- ✅ `requirements.txt` - Python dependencies

### Deployment Steps:

1. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Railway will automatically**:
   - Detect the Python app
   - Install dependencies
   - Start the server
   - Provide a public URL

3. **Environment Variables** (if needed):
   - `PORT` - Automatically set by Railway
   - `FLASK_ENV` - Set to `production` for production mode

---

## 📁 Project Structure

```
power-theft-detection/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── Procfile                  # Railway/Heroku deployment
├── runtime.txt              # Python version
├── railway.json             # Railway configuration
├── README.md                # Full documentation
├── SETUP_GUIDE.md          # This file
├── .gitignore              # Git ignore rules
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── intrusion_detection.py
│   ├── models.py
│   └── visualization.py
│
├── templates/              # HTML templates
│   ├── index_with_timeline.html
│   └── login.html
│
├── data/                   # Dataset
│   └── raw/
│       └── Electricity_Theft_Data.csv
│
├── models/                 # Trained ML models
│   ├── sgcc_random_forest_model.pkl
│   ├── sgcc_scaler.pkl
│   └── minimal_rf_model.pkl
│
└── results/               # Performance metrics
    ├── training_results.json
    ├── MODEL_PERFORMANCE_SUMMARY.md
    └── plots/
```

---

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 2GB (4GB recommended)
- **Storage**: 50MB free space
- **OS**: Windows, macOS, or Linux

---

## 📊 Features

- ✅ Real-time power theft detection
- ✅ Interactive dashboard with timeline (2015-2025)
- ✅ Multiple ML models (Random Forest, Neural Networks)
- ✅ Beautiful visualizations
- ✅ User authentication
- ✅ Year-by-year analysis
- ✅ Risk-based alerts (HIGH/MEDIUM/LOW)

---

## 🐛 Troubleshooting

### Port already in use:
```bash
# Kill existing process on port 8105
lsof -ti:8105 | xargs kill -9
```

### Missing dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset not found:
The app will use simulated data if the dataset is missing. This is normal for development.

---

## 👥 Team Collaboration

### Sharing the Project:

1. **Via Zip File**:
   - Share `power-theft-detection-v1.0.zip` (13 MB)
   - Teammates extract and run

2. **Via GitHub**:
   - Push to GitHub repository
   - Teammates clone and run

3. **Via Railway**:
   - Deploy once
   - Share the public URL
   - Everyone can access without setup!

---

## 🔐 Default Credentials

- **Username**: `admin`
- **Password**: `password`

⚠️ **Important**: Change these credentials in production!

Edit in `app.py`:
```python
ADMIN_CREDENTIALS = {
    'admin': 'your_new_password'
}
```

---

## 📞 Support

If you encounter any issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure port 8105 is available
4. Check Python version (3.8+)

---

## 🎉 Quick Start Summary

```bash
# 1. Extract/Clone
unzip power-theft-detection-v1.0.zip

# 2. Install
pip install -r requirements.txt

# 3. Run
python app.py

# 4. Access
# Open: http://localhost:8105
# Login: admin / password
```

---

**Last Updated**: October 27, 2025  
**Version**: 1.0  
**Size**: 37 MB (13 MB zipped)
