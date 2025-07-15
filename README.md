# D212 - Data Mining II - Task 2: Principal Component Analysis (PCA) on Customer Data

This project applies Principal Component Analysis (PCA) to telecom customer data to identify which features explain the most variance in customer behavior. The goal is to reduce dimensionality while maintaining insight into key influencing factors—ultimately helping the organization better segment and understand its customer base.

## 🧠 Objective

**Business Question:**  
Which factors are most significant in determining customer behavior patterns?

**Goal:**  
Identify the key dimensions driving variance in customer data to enable meaningful segmentation and improve business decision-making.

## 📁 Files

- `D212_Task2.py`: Full Python script that loads, preprocesses, and analyzes the data using PCA.
- `cleaned_data.csv`: Standardized dataset used for PCA, exported after preprocessing.
- `Jeremy_Dorrough_D212_Task_2_Performance_Assessment.docx`: Detailed write-up of the methodology, results, and business implications.

## 📊 Features Used

The following continuous variables were included in the PCA analysis:

- `Zip`
- `Lat`
- `Lng`
- `Population`
- `Children`
- `Age`
- `Income`
- `Outage_sec_perweek`
- `Email`
- `Contacts`
- `Yearly_equip_failure`
- `Tenure`
- `MonthlyCharge`
- `Bandwidth_GB_Year`

## ⚙️ Tools and Libraries

- **Python 3.x**
- `pandas` – Data manipulation
- `numpy` – Numerical operations
- `matplotlib` – Visualization
- `scikit-learn` – PCA, StandardScaler

## 🔍 Process Overview

1. **Load Data**  
   Read in the churn dataset and select relevant columns.

2. **Standardize Variables**  
   Apply `StandardScaler` to normalize features before PCA.

3. **Initial PCA**  
   Performed with all features to assess explained variance.

4. **Scree Plot (Elbow Method)**  
   Identified optimal number of components to retain.

5. **Final PCA with 12 Components**  
   Retained components that collectively explained over **99%** of the variance.

6. **Interpretation**  
   - `Tenure` and `Bandwidth_GB_Year` dominated PC1, indicating customer lifespan and usage behavior are closely linked.
   - PCs 2 and 3 highlighted geographic impact via variables like `Zip`, `Lng`, `Lat`, and `Population`.

## 📈 Results

**Explained Variance (Top 12 PCs):**

- PC1: 14.25%  
- PC2: 13.63%  
- PC3: 8.79%  
- PC4–PC12: Gradual decline, totaling **99.35%** cumulative explained variance.

**Insights:**

- Only slight dimensionality reduction was achieved (from 14 to 12), suggesting highly distributed variance.
- PCA still revealed strong relationships between customer location, tenure, and bandwidth use—informing future segmentation.

## 👤 Author

Jeremy Dorrough  
Western Governors University  

## 📚 References

- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  
- Brownee, J. (2020). *Machine Learning Mastery*  
- Kumar, S. (2020). *Towards Data Science*  
- Lever et al. (2017). *Nature Methods*

