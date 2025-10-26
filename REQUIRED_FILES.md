# 📋 Required Files to Run This Project

**Last Updated**: October 21, 2025

---

## ✅ ESSENTIAL FILES (Must Have)

### 1. **Core Application**
```
✅ app.py                    # Main Flask application (REQUIRED)
✅ config.py                 # Configuration settings
✅ requirements.txt          # Python dependencies
```

### 2. **Source Code Modules**
```
✅ src/
   ├── data_preprocessing.py    # Data processing
   ├── models.py                # ML model definitions
   ├── intrusion_detection.py   # Detection logic
   ├── feature_engineering.py   # Feature creation
   └── utils.py                 # Utility functions
```

### 3. **Web Interface**
```
✅ templates/
   └── index_with_timeline.html  # Main web interface
```

### 4. **Trained Models**
```
✅ models/
   ├── sgcc_random_forest_model.pkl  # Main model (22 MB)
   ├── sgcc_scaler.pkl               # Data scaler (102 KB)
   └── minimal_rf_model.pkl          # Backup model (361 KB)
```

### 5. **Dataset**
```
✅ data/
   ├── raw/
   │   └── Electricity_Theft_Data.csv  # Original Kaggle data (14 MB)
   └── processed/
       └── sgcc_extended_2014_2025.csv  # Extended dataset (2.4 GB)
```

### 6. **Results & Metrics**
```
✅ results/
   ├── training_results.json           # Model metrics
   ├── MODEL_PERFORMANCE_SUMMARY.md    # Performance summary
   └── plots/
       ├── sgcc_random_forest_confusion_matrix.png
       ├── sgcc_random_forest_feature_importance.png
       ├── minimal_confusion_matrix.png
       └── minimal_feature_importance.png
```

---

## 📚 DOCUMENTATION FILES (Recommended)

### Essential Docs:
```
✅ START_HERE.md             # Navigation guide
✅ README.md                 # Main documentation
✅ QUICKSTART.md             # Quick setup
✅ PROJECT_SUMMARY.md        # Project overview
✅ PRESENTATION_GUIDE.md     # Demo guide
```

---

## ❌ NOT REQUIRED (Optional/Utility)

### Training Scripts (Already Trained):
```
❌ train_*.py               # Model training scripts (models already trained)
❌ import_*.py              # Data import scripts (data already imported)
❌ extend_to_2025.py        # Data extension script (already done)
❌ download_sgcc.py         # Data download script (data already present)
```

### Testing/Debug Scripts:
```
❌ test_*.py                # Testing scripts
❌ debug_api.py             # Debug utility
❌ view_2025_data.py        # Data viewer
❌ check_*.sh               # Shell scripts
❌ wait_for_training.sh     # Training monitor
```

### Old/Archived Files:
```
❌ main.py                  # Old main file (use app.py instead)
❌ old_apps/                # Archived Flask apps (7 old versions)
❌ docs/archive/            # Archived documentation (13 files)
❌ PROJECT_EXPLANATION.txt  # Duplicate documentation
❌ Power_Theft_Detection_Project_Explanation.pdf  # PDF version
```

---

## 🎯 Minimum Files to Run

**Absolute minimum to run the web app**:

```
power-theft-detection/
├── app.py                                    ✅ REQUIRED
├── config.py                                 ✅ REQUIRED
├── requirements.txt                          ✅ REQUIRED
├── src/                                      ✅ REQUIRED (all 5 files)
├── templates/index_with_timeline.html        ✅ REQUIRED
├── models/                                   ✅ REQUIRED (3 model files)
└── data/processed/sgcc_extended_2014_2025.csv ✅ REQUIRED
```

**Total**: ~2.5 GB (mostly the dataset)

---

## 📊 File Size Breakdown

### Large Files:
```
2.4 GB - data/processed/sgcc_extended_2014_2025.csv  (Dataset)
202 MB - data/processed/sgcc_theft_data.csv          (Optional)
156 MB - data/raw/dataset.csv                        (Optional)
244 MB - data/processed/only_2025_data.csv           (Optional)
22 MB  - models/sgcc_random_forest_model.pkl         (Model)
14 MB  - data/raw/Electricity_Theft_Data.csv         (Original)
```

### Small Files:
```
28 KB  - app.py                    (Main app)
14 KB  - README.md                 (Documentation)
~50 KB - src/ (all modules)        (Source code)
~300 KB - results/plots/           (Visualizations)
```

---

## 🚀 What You Need to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies** (from requirements.txt):
- Flask
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- tensorflow (optional, for deep learning models)

### Step 2: Verify Required Files Exist
```bash
# Check main app
ls app.py

# Check models
ls models/*.pkl

# Check dataset
ls data/processed/sgcc_extended_2014_2025.csv

# Check templates
ls templates/index_with_timeline.html
```

### Step 3: Run
```bash
python app.py
```

### Step 4: Access
```
http://localhost:8105
```

---

## ✅ Quick Verification Checklist

Run this to check if you have everything:

```bash
# Essential files check
[ -f app.py ] && echo "✅ app.py" || echo "❌ app.py MISSING"
[ -f config.py ] && echo "✅ config.py" || echo "❌ config.py MISSING"
[ -f requirements.txt ] && echo "✅ requirements.txt" || echo "❌ requirements.txt MISSING"
[ -d src ] && echo "✅ src/" || echo "❌ src/ MISSING"
[ -d templates ] && echo "✅ templates/" || echo "❌ templates/ MISSING"
[ -d models ] && echo "✅ models/" || echo "❌ models/ MISSING"
[ -f data/processed/sgcc_extended_2014_2025.csv ] && echo "✅ Dataset" || echo "❌ Dataset MISSING"
```

---

## 🎯 Summary

### To Run the Project You Need:

**Core Files** (Required):
- ✅ `app.py` - Main application
- ✅ `config.py` - Configuration
- ✅ `requirements.txt` - Dependencies
- ✅ `src/` - Source code (5 modules)
- ✅ `templates/` - Web interface
- ✅ `models/` - Trained models (3 files)
- ✅ `data/processed/sgcc_extended_2014_2025.csv` - Dataset

**Documentation** (Recommended):
- ✅ `START_HERE.md` - Navigation
- ✅ `README.md` - Full docs
- ✅ `QUICKSTART.md` - Quick start

**Results** (For Presentation):
- ✅ `results/` - Metrics and plots

### You DON'T Need:

- ❌ Training scripts (models already trained)
- ❌ Import scripts (data already imported)
- ❌ Test scripts (optional)
- ❌ Old apps (archived)
- ❌ Shell scripts (utilities)
- ❌ Debug scripts (development only)

---

## 💾 If You Want to Share/Deploy

### Minimum Package (for running only):
```
app.py
config.py
requirements.txt
src/ (folder)
templates/ (folder)
models/ (folder)
data/processed/sgcc_extended_2014_2025.csv
```

**Size**: ~2.5 GB (mostly dataset)

### With Documentation:
Add:
```
START_HERE.md
README.md
QUICKSTART.md
PROJECT_SUMMARY.md
PRESENTATION_GUIDE.md
results/ (folder)
```

**Size**: ~2.5 GB + 1 MB

---

## 🎉 Bottom Line

**You have everything you need!**

All essential files are present:
- ✅ Application code
- ✅ Trained models
- ✅ Dataset
- ✅ Web interface
- ✅ Documentation
- ✅ Results

**Just run**: `python app.py` and you're good to go! 🚀

---

*Last Verified: October 21, 2025*
