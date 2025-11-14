# **Day 18 — Customer Segmentation using K-Means**  
Unsupervised Machine Learning Project

## 🚀 Overview  
This project focuses on **customer segmentation** using the **K-Means clustering algorithm**.  
By grouping customers based on spending patterns, we enable businesses to:

- Personalize marketing campaigns  
- Understand customer behavior  
- Improve targeting and retention  
- Identify high-value segments  

This project also includes visualizations, insights, and a reproducible ML workflow.

---

## 📂 Project Structure
```
Day-18-Customer-Segmentation/
│── run_kmeans.py              # Main script (data loading → clustering → insights)
│── requirements.txt           # Dependencies
│── clustered_customers.csv    # Output: Customers + assigned cluster labels
│── cluster_insights.csv       # Output: Summary insights per cluster
│── elbow.png                  # Elbow method curve (optimal cluster estimation)
│── clusters_pca.png           # PCA 2D visualization of clusters
│── README.md                  # Project documentation
│── USAGE.md                   # How to run the script
│── NOTES.md                   # Developer notes
│── DATA_INFO.md               # Dataset details
│── PROJECT_GOALS.md           # Project objectives
│── ARCHITECTURE.md            # Architectural overview
│── FUTURE_WORK.md             # Planned improvements
│── CHANGELOG.md               # Versioning history
│── SAMPLE_OUTPUT.md           # Quick look at project outputs
│── DATA_SCHEMA.md             # Features used for clustering
│── RUN_HISTORY.md             # Execution logs
└── .gitignore
```

---

## 🧪 How It Works

### **1️⃣ Data Loading**
If no dataset is provided, the script automatically generates a **synthetic dataset** with:  
- Age  
- Annual Income  
- Spending Score  

### **2️⃣ Preprocessing**
- Standardization using **StandardScaler**
- Outlier-friendly scaling
- Data validation checks

### **3️⃣ Finding Optimal K**
Uses **Elbow Method** → generates `elbow.png`.

### **4️⃣ Apply K-Means Clustering**
- Trains the model  
- Assigns each customer to a segment  
- Saves results to `clustered_customers.csv`

### **5️⃣ PCA Visualization**
- Reduces data to 2D  
- Produces an interpretable plot → `clusters_pca.png`

### **6️⃣ Cluster Insights**
Outputs:  
- Mean values per cluster  
- Behavioral patterns  
- Summary written to `cluster_insights.csv`

---

## 📊 Visual Outputs  
### **Elbow Plot**
Shows distortion score for K=1→10 to find the optimal cluster count.

### **PCA Cluster Plot**
2D visualization showing how distinct each customer segment is.

---

## ▶️ How to Run  
See full instructions in **USAGE.md**.  
Quick version:

```
pip install -r requirements.txt
python3 run_kmeans.py --clusters 4
```

---

## 🧠 Key Learnings  
- Unsupervised learning & clustering  
- K-Means algorithm workflow  
- Dimensionality reduction (PCA)  
- Business-oriented data segmentation  
- Clean ML pipeline design  
- Insight generation from numerical patterns  

---

## 📌 Future Improvements  
See **FUTURE_WORK.md**, but highlights include:  
- Adding DBSCAN, Hierarchical Clustering  
- Better synthetic data generation  
- Web dashboard using Streamlit or FastAPI  
- Auto-report generation (PDF/HTML)

---

## 👤 Author  
**Abhineet Singh**  
Part of the **64-Day AI Challenge** series.

---

## 📄 License  
MIT License — free to use and modify.