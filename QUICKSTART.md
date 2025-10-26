# Quick Start Guide - Power Theft Detection System

## ✅ Installation Complete!

All dependencies are installed. You're ready to use the system.

## 🚀 Running the Project

### Option 1: Full Training Pipeline

Train all models and generate results:

```bash
python3 main.py
```

**What this does:**
- Generates 10,000 sample smart meter readings
- Preprocesses data with feature engineering
- Trains Neural Network, Random Forest, SVM, and other models
- Evaluates all models
- Creates visualizations (ROC curves, confusion matrices, etc.)
- Saves models to `models/` directory
- Generates report in `results/`

**Expected time:** 5-10 minutes

---

### Option 2: Quick Test (Recommended First)

Test the installation with a quick demo:

```bash
python3 test_installation.py
```

**What this does:**
- Verifies all imports work
- Generates 100 sample records
- Trains a small model (2 epochs)
- Tests the IDS system
- Takes only 1-2 minutes

---

### Option 3: Web Dashboard

Launch the interactive web interface:

```bash
python3 app.py
```

Then open your browser to: **http://localhost:5000**

**Features:**
- Real-time monitoring dashboard
- Detection statistics
- Alert management
- Simulation mode
- Export functionality

---

## 📝 Important Commands

### Use `python3` not `python`

On your Mac, Python 3 is accessed via `python3`:

```bash
# ✅ Correct
python3 main.py
python3 app.py
python3 test_installation.py

# ❌ Wrong (will give "command not found")
python main.py
```

### Check Python Version

```bash
python3 --version
# Should show: Python 3.13.x
```

### Reinstall Dependencies (if needed)

```bash
pip3 install -r requirements.txt
```

---

## 📂 Project Structure

After running `main.py`, you'll have:

```
power-theft-detection/
├── data/
│   ├── raw/
│   │   └── sample_data.csv          # Generated sample data
│   └── processed/
│       ├── X_train.npy              # Preprocessed training features
│       ├── X_test.npy               # Preprocessed test features
│       ├── y_train.npy              # Training labels
│       └── y_test.npy               # Test labels
├── models/
│   ├── neural_network_model.h5      # Trained neural network
│   ├── random_forest_model.pkl      # Trained random forest
│   ├── gradient_boosting_model.pkl  # Trained gradient boosting
│   └── svm_model.pkl                # Trained SVM
├── results/
│   ├── plots/
│   │   ├── training_history.png     # Training curves
│   │   ├── confusion_matrix_*.png   # Confusion matrices
│   │   ├── roc_curve_*.png          # ROC curves
│   │   └── pr_curve_*.png           # Precision-Recall curves
│   └── report_*.txt                 # Performance report
└── logs/                            # Training logs
```

---

## 🎯 What to Expect

### When Running `main.py`:

```
======================================================================
               POWER THEFT DETECTION SYSTEM
          AI-Based Intrusion Detection for Smart Grids
======================================================================

✓ Directories created

============================================================
DATA LOADING
============================================================
Generating sample smart meter data...
Generated 10000 samples with 1000 theft cases (10.0%)
Dataset shape: (10000, 4)

============================================================
DATA PREPROCESSING
============================================================
Missing values before: 0
Missing values after: 0
Extracted time features. New shape: (10000, 18)
Extracted consumption features. New shape: (10000, 45)
...

============================================================
TRAINING NEURAL NETWORK
============================================================
Model: "functional"
...
Epoch 1/100
...

============================================================
FINAL REPORT
============================================================
MODEL PERFORMANCE SUMMARY:
------------------------------------------------------------
Neural Network       | Accuracy: 0.9500 | AUC: 0.9800
Random Forest        | Accuracy: 0.9400 | AUC: 0.9750
...
```

---

## 🔧 Troubleshooting

### Issue: "command not found: python"

**Solution:** Use `python3` instead of `python`

### Issue: "No module named 'src'"

**Solution:** Make sure you're in the `power-theft-detection` directory:
```bash
cd /Users/saigirish050704/CascadeProjects/personal-website/power-theft-detection
python3 main.py
```

### Issue: Training is slow

**Solution:** 
- Your Mac has Apple Silicon (M-series chip) which TensorFlow can use
- First run might be slower as it compiles
- Subsequent runs will be faster
- For quick testing, use `test_installation.py` instead

### Issue: Out of memory

**Solution:** Reduce the sample size in `main.py`:
```python
# Change this line
df = load_or_generate_data(use_sample=True)

# In the function, modify:
df = generate_sample_data(n_samples=1000, theft_ratio=0.1)  # Reduced from 10000
```

---

## 📊 Viewing Results

### 1. Check the Report

```bash
cat results/report_*.txt
```

### 2. View Plots

Open the `results/plots/` folder in Finder:
```bash
open results/plots/
```

### 3. Check Saved Models

```bash
ls -lh models/
```

---

## 🌐 Using the Web Dashboard

1. **Start the server:**
   ```bash
   python3 app.py
   ```

2. **Open browser:** http://localhost:5000

3. **Initialize system:** Click "Initialize System" button

4. **Test detection:** Click "Simulate Detection"

5. **View alerts:** Scroll down to see detected theft cases

6. **Export data:** Use the API endpoints to download results

---

## 📚 Next Steps

1. ✅ **Run quick test:** `python3 test_installation.py`
2. ✅ **Train models:** `python3 main.py`
3. ✅ **Launch dashboard:** `python3 app.py`
4. 📖 **Read documentation:** See `README.md` and `USAGE_GUIDE.md`
5. 🔧 **Customize:** Edit `config.py` for your needs
6. 📊 **Use your data:** Replace sample data with real smart meter data

---

## 💡 Tips

- **Start small:** Use `test_installation.py` first to verify everything works
- **Monitor progress:** Training shows progress bars and epoch information
- **Check logs:** Look in `logs/` directory for detailed information
- **Save results:** All outputs are automatically saved
- **Experiment:** Try different model parameters in `config.py`

---

## 🆘 Need Help?

- **Documentation:** `README.md` - Complete project overview
- **Usage Guide:** `USAGE_GUIDE.md` - Detailed usage instructions
- **Code Examples:** See `main.py` for complete pipeline example
- **API Reference:** Check individual files in `src/` directory

---

**Ready to start? Run this command:**

```bash
python3 test_installation.py
```

**Good luck with your project! ⚡🎓**
