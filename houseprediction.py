import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Excel file
data_frame = pd.read_excel("HousePricePrediction.xlsx")

# Display the first few rows
print("Preview of dataset:")
print(data_frame.head())

# Identify columns by data type
categorical_features = data_frame.select_dtypes(include='object').columns
integer_features = data_frame.select_dtypes(include='int64').columns
float_features = data_frame.select_dtypes(include='float64').columns

# Print counts of each type
print(f"Number of categorical features: {len(categorical_features)}")
print(f"Number of integer features: {len(integer_features)}")
print(f"Number of float features: {len(float_features)}")

# Select only numerical columns
numeric_data = data_frame.select_dtypes(include=['number'])

# Compute correlation matrix
corr_matrix = numeric_data.corr()

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)

plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()

# Save the figure
output_file = "correlation_heatmap.png"
plt.savefig(output_file)

print(f"Heatmap successfully saved as {output_file}")

# Count unique values in each categorical column
unique_counts = [
    data_frame[col].nunique() for col in categorical_features
]

# Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(x=categorical_features, y=unique_counts)

plt.title("Number of Unique Values in Categorical Features")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Unique Count")
plt.xlabel("Categorical Features")

plt.tight_layout()
plt.show()

# Plot distribution of categorical features
num_cols = len(categorical_features)
cols_per_row = 4
rows = (num_cols + cols_per_row - 1) // cols_per_row  # dynamic row count

plt.figure(figsize=(18, 5 * rows))

for idx, col in enumerate(categorical_features, start=1):
    value_counts = data_frame[col].value_counts()

    plt.subplot(rows, cols_per_row, idx)
    sns.barplot(x=value_counts.index, y=value_counts.values)

    plt.title(col)
    plt.xticks(rotation=45, ha='right')

plt.suptitle("Distribution of Categorical Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

# Drop 'Id' column if it exists
if 'Id' in data_frame.columns:
    data_frame = data_frame.drop(columns=['Id'])
    
# Fill numerical columns with mean
num_cols = data_frame.select_dtypes(include='number').columns
data_frame[num_cols] = data_frame[num_cols].fillna(data_frame[num_cols].mean())

# Fill categorical columns with mode
cat_cols = data_frame.select_dtypes(include='object').columns
for col in cat_cols:
    data_frame[col].fillna(data_frame[col].mode()[0], inplace=True)

# Verify no missing values remain
print(data_frame.isnull().sum())

from sklearn.preprocessing import OneHotEncoder

# Identify categorical columns
categorical_features = data_frame.select_dtypes(include='object').columns.tolist()

# Display results
print("Categorical features:")
print(categorical_features)

print(f"Total number of categorical features: {len(categorical_features)}")

# Show categorical features with their unique counts
for col in categorical_features:
    print(f"{col}: {data_frame[col].nunique()} unique values")
    
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Separate features and target
X = data_frame.drop(columns=['SalePrice'])
y = data_frame['SalePrice']

# Identify categorical columns
categorical_features = X.select_dtypes(include='object').columns

# Initialize encoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Apply encoding
encoded_array = encoder.fit_transform(X[categorical_features])

# Convert to DataFrame
encoded_df = pd.DataFrame(
    encoded_array,
    index=X.index,
    columns=encoder.get_feature_names_out(categorical_features)
)

# Drop original categorical columns and combine
X_numeric = X.drop(columns=categorical_features)
X_processed = pd.concat([X_numeric, encoded_df], axis=1)

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_processed, y,
    test_size=0.2,
    random_state=42
)

#
# 1. SVM - Support vector Machine

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# Scale features (important for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Initialize and train model
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = svr_model.predict(X_valid_scaled)

# Evaluation
mape = mean_absolute_percentage_error(y_valid, y_pred)
print(f"MAPE: {mape:.4f}")

#2. Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Initialize model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_valid)

# Evaluate
mape = mean_absolute_percentage_error(y_valid, y_pred)
print(f"Random Forest MAPE: {mape:.4f}")

# 3. Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# Initialize model
lr_model = LinearRegression()

# Train model
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_valid)

# Evaluation
mape = mean_absolute_percentage_error(y_valid, y_pred)
print(f"Linear Regression MAPE: {mape:.4f}")

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

results = {
    "Linear Regression": mean_absolute_percentage_error(y_valid, lr_model.predict(X_valid)),
    "Random Forest": mean_absolute_percentage_error(y_valid, rf_model.predict(X_valid)),
    "SVR": mean_absolute_percentage_error(y_valid, svr_model.predict(X_valid_scaled))
}

for model, score in results.items():
    print(f"{model}: {score:.4f}")
    
    

