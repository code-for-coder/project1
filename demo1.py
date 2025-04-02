import pandas as pd
import sweetviz as sv
from ydata_profiling import ProfileReport
import streamlit as st
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import  LabelEncoder, StandardScaler
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore  # Also needed for anomaly detection
import plotly.graph_objects as go  # Ensure plotly is imported
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import hashlib

# Database setup
def create_users_table():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)")
    conn.commit()
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
    except sqlite3.IntegrityError:
        st.error("Username already exists!")
    conn.close()

def validate_login(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

def update_password(username, new_password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password=? WHERE username=?", (new_password, username))
    conn.commit()
    conn.close()
    st.success("Password updated successfully!")


# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = "username"
    


# Function to handle missing values
def handle_missing_values(df):
    st.write("## Handle Missing Values")
    st.write(df.isnull().sum())

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            st.write(f"### Column: {col}")

            # Convert all object-type columns to string
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)  # Ensure consistent string format
                mode_value = df[col].mode()[0]  # Get the most frequent value
                st.write(f"⚠️ Column `{col}` is categorical. Filling missing values with mode: `{mode_value}`")
                df[col] = df[col].fillna(mode_value)  # Fill missing with mode

            # Handle datetime columns
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                st.write(f"⚠️ Column `{col}` is a datetime column. Converting to string format.")
                df[col] = df[col].astype(str)  # Convert datetime to string

            # Handle categorical numeric columns separately
            elif df[col].nunique() <= 10:  # If it's categorical numeric
                st.write("*Suggested Method:* KNN Imputation")
                imputer = KNNImputer(n_neighbors=2)
                df[col] = imputer.fit_transform(df[[col]])

            # Handle continuous numeric columns
            else:
                st.write("*Suggested Method:* Mean/Median Imputation")
                imputer = SimpleImputer(strategy='mean')  # Default to mean, can be changed to median
                df[col] = imputer.fit_transform(df[[col]])

    return df


def handle_outliers(df):
    st.write("## Outlier Detection & Treatment")

    iforest = IsolationForest(contamination=0.1, random_state=42)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() > 10:  # Apply only to continuous variables
            st.write(f"### Checking Outliers in: {col}")

            df['outlier'] = iforest.fit_predict(df[[col]])
            st.write("Outliers detected:", df[df['outlier'] == -1][col])

            option = st.radio(f"How to handle outliers in {col}?", ['Replace with Median', 'Cap at Percentiles', 'Remove Outliers'], key=col)

            if option == 'Replace with Median':
                median_value = df[col].median()
                df.loc[df['outlier'] == -1, col] = median_value

            elif option == 'Cap at Percentiles':
                lower, upper = np.percentile(df[col], [5, 95])
                df[col] = np.clip(df[col], lower, upper)

            elif option == 'Remove Outliers':
                df = df[df['outlier'] != -1]

            df.drop(columns=['outlier'], inplace=True)

    return df



def feature_engineering_suggestions(df):
    st.write("## Feature Engineering Suggestions")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        st.write("No categorical columns found for encoding.")
        return df
    
    selected_cols = st.multiselect("Select categorical columns for encoding:", categorical_cols)
    
    if selected_cols:
        encoding_type = st.radio("Choose encoding :", ["Label Encoding"], key="encoding_type")
    
    apply_changes = st.button("Apply Feature Engineering")
    
    if apply_changes and selected_cols:
        df = df.copy()  # Avoid modifying original dataframe
        
        
        if encoding_type == "Label Encoding":
            label_encoder = LabelEncoder()
            for col in selected_cols:
                df[col] = label_encoder.fit_transform(df[col])

        # Standard Scaling only for numeric columns (without adding extra ones)
        num_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        st.write("### After Feature Engineering", df)

        

    return df





def main_app():
    # Streamlit File Uploader
    st.title("📊 AI-Powered Automated Data Insights")
    st.markdown("---")


    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.write(df.dtypes)
        # Generate Pandas Profiling Report
        if st.button("Generate Pandas Profiling Report"):
            profile = ProfileReport(df, explorative=True)
            profile.to_file("eda_report.html")

            # Read the HTML file as binary
            with open("eda_report.html", "rb") as file:
                st.download_button(
                    label="Download Pandas Profiling Report",
                    data=file,
                    file_name="eda_report.html",
                    mime="text/html"
                )

        # Generate SweetViz Report
        if st.button("Generate SweetViz Report"):
            report = sv.analyze(df)
            report.show_html("sweetviz_report.html", open_browser=False)  # Prevents automatic opening

            # Read the HTML file as binary
            with open("sweetviz_report.html", "rb") as file:
                st.download_button(
                    label="Download SweetViz Report",
                    data=file,
                    file_name="sweetviz_report.html",
                    mime="text/html"
                )

        


        # Load Free NLP Model (Hugging Face)
        st.subheader("🔍 AI-Generated Insights")

        # Generate dataset summary
        summary = df.describe().transpose()  # Transpose for better readability
        st.dataframe(summary)  # Display as an interactive table

        df = handle_missing_values(df)
        st.write("### After Handling Missing Values", df)

        df = handle_outliers(df)
        st.write("### After Handling Outliers", df)

        df = feature_engineering_suggestions(df)
        # Let the user select the target column
        target_column = st.selectbox("🎯 Select the Target Column:", df.columns)

        if target_column:
            # Convert categorical features into numbers
            df_encoded = df.copy()
            label_encoders = {}  # Dictionary to store encoders

            for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
                df_encoded[col] = df_encoded[col].astype(str)  # Ensure all are strings
                label_encoders[col] = LabelEncoder()
                df_encoded[col] = label_encoders[col].fit_transform(df_encoded[col])
        # if st.button("🚀 Split Data"):
        # Splitting the data
        X = df_encoded.drop(columns=[target_column])  # Features
        y = df_encoded[target_column]  # Target column
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # TPOT AutoML
        st.subheader("🔬 AutoML Model Selection")
        if st.button("Run TPOT AutoML"):
            tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
            tpot.fit(X_train, y_train)
            best_model = tpot.fitted_pipeline_
            st.success(f"Best TPOT Model: {best_model}")

        # Train Models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store metrics
            results[name] = {
                "R² Score": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
            }

        # Show Model Comparisons
        st.write("### Model Performance Comparison")
        st.write(pd.DataFrame(results).T)

        

        
        

        # Select time series column
        time_col = st.selectbox("Select Date/Time Column", df.columns,index=0)
        value_col = st.selectbox("Select Value Column", df.columns)

        # Ensure the selected columns exist
        if time_col and value_col:
            # Convert to datetime (Ensure correct format)
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')  
            df = df.dropna(subset=[time_col, value_col])  # Drop rows where date or value is missing
            df = df.sort_values(by=time_col)  # Ensure time series order

            # Ensure value column is numeric
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

            # Convert datetime to numeric format (Unix timestamp)
            df["timestamp"] = df[time_col].astype('int64') // 10**9

            # Ensure enough data points
            if len(df) > 12:
                decomposition = seasonal_decompose(df[value_col], period=12, model='additive', extrapolate_trend='freq')

                # Remove NaN values in decomposition results
                df["trend"] = decomposition.trend
                df["seasonal"] = decomposition.seasonal
                df["resid"] = decomposition.resid
                df.dropna(subset=["trend", "seasonal", "resid"], inplace=True)

                # Detect Anomalies using Z-score
                df["Z-Score"] = np.abs(zscore(df[value_col]))
                df["Anomaly"] = df["Z-Score"] > 2.5  # Mark as anomaly if Z-score > 2.5

                # Plot Time Series with Trend and Anomalies
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=df[time_col], y=df[value_col], mode='lines', name='Original Data'))
                fig.add_trace(go.Scatter(x=df[time_col], y=df["trend"], mode='lines', name='Trend', line=dict(color='orange')))

                # Mark anomalies
                anomalies = df[df["Anomaly"]]
                fig.add_trace(go.Scatter(x=anomalies[time_col], y=anomalies[value_col], mode='markers',
                                    name='Anomalies', marker=dict(color='red', size=8)))

                # Update layout
                fig.update_layout(title="Trend & Anomaly Detection", xaxis_title="Time", yaxis_title="Value")

                # Display the plot
                st.plotly_chart(fig)

                # Show Decomposition Components
                st.write("### Seasonal Component")
                st.line_chart(df["seasonal"])

                st.write("### Residual (Noise)")
                st.line_chart(df["resid"])
            else:
                st.warning("Not enough data points for seasonal decomposition. At least 12 observations are required.")

        

        st.write("## SQL Query Builder")

        if uploaded_file:
            # Convert dataframe to SQLite database
            conn = sqlite3.connect(":memory:")
            df.to_sql("data", conn, index=False, if_exists="replace")

            # User input for SQL query
            query = st.text_area("Write your SQL query (e.g., SELECT * FROM data WHERE column > 100)")

            if st.button("Run Query"):
                try:
                    result_df = pd.read_sql_query(query, conn)
                    st.write("### Query Results", result_df)
                except Exception as e:
                    st.error(f"Error: {e}")

        st.write("## KPI Dashboard")

        if uploaded_file:
            kpi_metrics = {
            "Total Rows": df.shape[0],
            "Total Columns": df.shape[1],
            "Total Sales (if applicable)": df["Sales"].sum() if "Sales" in df.columns else "N/A",
            "Average Value per Column": df.describe().mean().to_dict()
            }

            st.json(kpi_metrics)  # Show KPIs in JSON format

        
        st.write("## Data Alerts & Notifications")

        if uploaded_file:
            alert_column = st.selectbox("Select column for alert", df.columns)
            alert_condition = st.selectbox("Condition", [">", "<", "=", ">=", "<="])
            alert_value = st.number_input("Threshold Value")

            # Check condition
            if alert_condition == ">":
                alert_triggered = df[alert_column].max() > alert_value
            elif alert_condition == "<":
                alert_triggered = df[alert_column].min() < alert_value
            elif alert_condition == "=":
                alert_triggered = (df[alert_column] == alert_value).any()
            elif alert_condition == ">=":
                alert_triggered = df[alert_column].max() >= alert_value
            elif alert_condition == "<=":
                alert_triggered = df[alert_column].min() <= alert_value

            if alert_triggered:
                st.warning(f"🚨 Alert! Condition met: {alert_column} {alert_condition} {alert_value}")
            else:
                st.success("No alerts triggered")


        numeric_df = df.select_dtypes(include=[np.number])  # Select only numerical columns
        
        if numeric_df.shape[1] > 1:  # Ensure there are at least two numerical columns
            st.write("## 🔥 Correlation Heatmap (Numerical Columns Only)")
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            st.pyplot(plt)
        else:
            st.write("⚠️ Not enough numerical columns to create a correlation heatmap.")



create_users_table()

menu = ["Login", "Register", "Forgot Password"]
if st.session_state["logged_in"]:
    menu.append("Logout")
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.subheader("Create an Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        register_user(username, password)
        st.success("Account created successfully! Please login.")

elif choice == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if validate_login(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Logged in successfully!")
            #main_app()
        else:
            st.error("Invalid username or password")

elif choice == "Forgot Password":
    st.subheader("Reset Your Password")
    username = st.text_input("Enter your username")
    new_password = st.text_input("Enter new password", type="password")
    confirm_password = st.text_input("Confirm new password", type="password")
    if st.button("Reset Password"):
        if new_password == confirm_password:
            update_password(username, new_password)
        else:
            st.error("Passwords do not match!")  

elif choice == "Logout":
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.success("Logged out successfully!")
    st.experimental_rerun() 

if st.session_state["logged_in"]:
    main_app()