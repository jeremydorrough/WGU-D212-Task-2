import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

# Import data
df = pd.read_csv('~/Desktop/D212_Task_2/churn_clean.csv', usecols=[
    'Zip', 'Lat', 'Lng', 'Population', 'Children', 'Age', 'Income',
    'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure',
    'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year'
])

# Standardize continuous variables
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
df = pd.DataFrame(scaled_data, columns=df.columns)

# Export cleaned data
df.to_csv("~/Desktop/D212_Task_2/cleaned_data.csv", index=False)

# Perform initial PCA
pca = PCA()
principal_components = pca.fit_transform(df)

# Get loadings matrix
loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings,
                           columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                           index=df.columns)

# Scree plot (elbow method)
explained_variance = pca.explained_variance_ratio_
plt.figure()
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', markersize=8)
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
for i, value in enumerate(explained_variance):
    plt.text(i + 1, value, f"{i+1}")
plt.show()

# Choose number of principal components
num_principal_components = 12

# Perform PCA with selected number of components
pca = PCA(n_components=num_principal_components)
principal_components = pca.fit_transform(df)

# Create DataFrame for selected components
principal_components_df = pd.DataFrame(
    data=principal_components,
    columns=[f'PC{i+1}' for i in range(num_principal_components)],
    index=df.index
)

# Variance explained
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_percentages = explained_variance_ratio * 100
total_variance_explained = np.sum(explained_variance_percentages)

# Loadings matrix for selected components
loadings = pca.components_.T
loadings_df = pd.DataFrame(
    loadings,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=df.columns
)

# Print variance explained
print("Percentage of Variance Explained by Each Principal Component:")
for i, percentage in enumerate(explained_variance_percentages, start=1):
    print(f"PC{i}: {percentage:.2f}%")
print(f"\nTotal Variance Explained by the {num_principal_components} Principal Components: {total_variance_explained:.2f}%")
