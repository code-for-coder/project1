import pandas as pd
import sweetviz as sv
from ydata_profiling import ProfileReport
import streamlit as st
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import  LabelEncoder, StandardScaler, OrdinalEncoder
import category_encoders as ce
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
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.exceptions import NotFittedError
import traceback
import io



# Database setup
# def create_users_table():
#     conn = sqlite3.connect("users.db")
#     cursor = conn.cursor()
#     cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)")
#     conn.commit()
#     conn.close()

# def register_user(username, password):
#     conn = sqlite3.connect("users.db")
#     cursor = conn.cursor()
#     try:
#         cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
#         conn.commit()
#     except sqlite3.IntegrityError:
#         st.error("Username already exists!")
#     conn.close()

# def user_exists(username):
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM users WHERE username=?", (username,))
#     result = c.fetchone()
#     conn.close()
#     return result is not None

# def validate_login(username, password):
#     conn = sqlite3.connect("users.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
#     user = cursor.fetchone()
#     conn.close()
#     return user is not None

# def update_password(username, new_password):
#     conn = sqlite3.connect("users.db")
#     cursor = conn.cursor()
#     cursor.execute("UPDATE users SET password=? WHERE username=?", (new_password, username))
#     conn.commit()
#     conn.close()
#     st.success("Password updated successfully!")

# def user_exists(username):
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM users WHERE username = ?", (username,))
#     result = c.fetchone()
#     conn.close()
#     return result is not None

# def is_same_password(username, new_password):
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute("SELECT password FROM users WHERE username = ?", (username,))
#     result = c.fetchone()
#     conn.close()
#     return result and result[0] == new_password

# def update_password(username, new_password):
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute("UPDATE users SET password = ? WHERE username = ?", (new_password, username))
#     conn.commit()
#     conn.close()


# Initialize session state
# if "logged_in" not in st.session_state:
#     st.session_state["logged_in"] = False
# if "username" not in st.session_state:
#     st.session_state["username"] = "username"
    

def regression_impute(df, target_col):
    df_copy = df.copy()
    not_null = df_copy[df_copy[target_col].notnull()]
    is_null = df_copy[df_copy[target_col].isnull()]

    # Drop non-numeric columns for regression
    X = not_null.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    y = not_null[target_col]

    if X.empty:
        st.warning(f"No suitable numeric predictors found for regression imputation of `{target_col}`")
        return df

    model = LinearRegression()
    model.fit(X, y)

    X_pred = is_null[X.columns]
    y_pred = model.predict(X_pred)

    df_copy.loc[df_copy[target_col].isnull(), target_col] = y_pred
    return df_copy

# Function to handle missing values
def handle_missing_values(df):
    st.write("## Handle Missing Values")
    st.write("### Missing Value Summary")
    st.write(df.isnull().sum())

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            st.write(f"---\n### Column: `{col}`")
            conclusion = ""

            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = df[col].astype(str)
                valid_values = df[col][~df[col].isin(['nan', 'NaN'])]

                if not valid_values.empty:
                    mode_value = valid_values.mode()[0]
                    st.info(f"Categorical column → Using Mode Imputation → `{mode_value}`")
                    conclusion = (
                        "This column has text or category values. So, we filled the empty spots with the most common value."
                    )
                    df[col] = df[col].replace(['nan', 'NaN'], np.nan)
                    df[col] = df[col].fillna(mode_value)
                else:
                    st.warning(f"No valid values found. Filling `{col}` with 'Unknown'")
                    conclusion = (
                        "There were no good values to use, so we filled the blanks with 'Unknown' to keep the column usable."
                    )
                    df[col] = df[col].fillna('Unknown')

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                st.info("Datetime column → Using Mode Imputation")
                conclusion = (
                    "This column has dates or times. So, we used the most common date/time to fill the empty cells."
                )

            elif np.issubdtype(df[col].dtype, np.number):
                missing_ratio = df[col].isnull().mean()
                skewness = df[col].skew()
                unique_vals = df[col].nunique()

                st.write(f"Missing Ratio: `{missing_ratio:.2f}`, Skewness: `{skewness:.2f}`, Unique: `{unique_vals}`")

                if abs(skewness) > 1:
                    imputer = SimpleImputer(strategy='median')
                    df[col] = imputer.fit_transform(df[[col]])
                    st.info("Highly skewed numeric column → Using Median Imputation")
                    conclusion = (
                        "This column has numbers with some big outliers. So, we used the middle value (median) instead of average."
                    )

                elif unique_vals <= 10:
                    imputer = KNNImputer(n_neighbors=2)
                    df[col] = imputer.fit_transform(df[[col]])
                    st.info("Categorical numeric column → Using KNN Imputation")
                    conclusion = (
                        "This number column looks like categories (like ratings). We filled missing values using similar rows."
                    )

                elif 0.1 < missing_ratio < 0.5:
                    st.info("Moderate missing ratio detected → Using Regression Imputation")
                    df = regression_impute(df, col)
                    conclusion = (
                        "This column had some missing numbers. So, we guessed the missing ones using other related columns."
                    )

                elif missing_ratio < 0.3:
                    imputer = IterativeImputer(random_state=0)
                    df[[col]] = imputer.fit_transform(df[[col]])
                    st.info("Low missing ratio → Using MICE (Multiple Column Estimation)")
                    conclusion = (
                        "Only a few values are missing. We looked at other columns together to make better guesses."
                    )

                else:
                    imputer = SimpleImputer(strategy='mean')
                    df[col] = imputer.fit_transform(df[[col]])
                    st.info("Defaulting to Mean Imputation")
                    conclusion = (
                        "This column has numbers, and nothing special is going on. So, we filled blanks using the average."
                    )

            st.markdown(f"**🧾 Easy Explanation:** {conclusion}")

    return df





def handle_outliers(df):
    st.write("## 🚨 Outlier Detection & Easy Fixes")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    detection_results = {}

    for col in numeric_cols:
        if df[col].nunique() > 10:
            st.write(f"---\n### 🔍 Checking Column: `{col}`")

            col_data = df[[col]].dropna()
            scaler = StandardScaler()
            scaled_col = scaler.fit_transform(col_data)

            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = col_data[z_scores > 3]
            detection_results.setdefault(col, {})['Z-Score'] = z_outliers.index.tolist()

            # Modified Z-score method
            median_val = np.median(col_data)
            mad = np.median(np.abs(col_data - median_val))
            modified_z = 0.6745 * (col_data - median_val) / (mad if mad else 1)
            mod_z_outliers = col_data[np.abs(modified_z) > 3.5]
            detection_results[col]['Modified Z-Score'] = mod_z_outliers.index.tolist()

            # Isolation Forest
            iso = IsolationForest(contamination=0.1, random_state=42)
            iso_preds = iso.fit_predict(col_data)
            iso_outliers = col_data[iso_preds == -1]
            detection_results[col]['Isolation Forest'] = iso_outliers.index.tolist()

            # LOF
            lof = LocalOutlierFactor(n_neighbors=20)
            lof_preds = lof.fit_predict(scaled_col)
            lof_outliers = col_data[lof_preds == -1]
            detection_results[col]['LOF'] = lof_outliers.index.tolist()

            # DBSCAN
            db = DBSCAN(eps=0.5, min_samples=5)
            db_preds = db.fit_predict(scaled_col)
            db_outliers = col_data[db_preds == -1]
            detection_results[col]['DBSCAN'] = db_outliers.index.tolist()

            # Best method selection
            outlier_counts = {method: len(outliers) for method, outliers in detection_results[col].items()}
            best_method = min(outlier_counts, key=outlier_counts.get)
            best_outliers = detection_results[col][best_method]

            # Box plot with outliers overlay
            st.write("#### 📊 Outlier Visualization (Box Plot with Red Dots)")
            fig = px.box(df, y=col, title=f"Box Plot for {col}")
            if best_outliers:
                outlier_vals = df.loc[best_outliers, col]
                fig.add_scatter(y=outlier_vals, mode="markers", marker=dict(color="red", size=8), name="Detected Outliers")
            st.plotly_chart(fig)

            # Mode and Median values
            mode_val = df[col].mode()[0]
            median_val = df[col].median()

            st.write(f"---\n### ✅ **Best Method for `{col}`:** `{best_method}`")

            # Apply and explain in simple terms
            if best_method == 'Z-Score':
                df[col] = df[col].apply(lambda x: median_val if df.index[df[col] == x].tolist()[0] in best_outliers else x)
                st.write("📘 **Why this method?** The values in this column are evenly spread. We removed any value that was very far from the average.")
                st.write("🔧 **Fix Used**: Replaced far-away values with the middle value (median).")

            elif best_method == 'Modified Z-Score':
                df[col] = df[col].apply(lambda x: median_val if df.index[df[col] == x].tolist()[0] in best_outliers else x)
                st.write("📘 **Why this method?** Some values are extremely high or low. This method spots them better than normal Z-Score.")
                st.write("🔧 **Fix Used**: Replaced extreme values with the middle value (median).")

            elif best_method == 'Isolation Forest':
                df[col] = df[col].apply(lambda x: mode_val if df.index[df[col] == x].tolist()[0] in best_outliers else x)
                st.write("📘 **Why this method?** This column has tricky or uneven values. Isolation Forest is good for that.")
                st.write("🔧 **Fix Used**: Replaced odd values with the most common one (mode).")

            elif best_method == 'LOF':
                df[col] = df[col].apply(lambda x: mode_val if df.index[df[col] == x].tolist()[0] in best_outliers else x)
                st.write("📘 **Why this method?** Some values are very different from their neighbors. LOF catches them.")
                st.write("🔧 **Fix Used**: Replaced different values with the most common one (mode).")

            elif best_method == 'DBSCAN':
                df[col] = df[col].apply(lambda x: mode_val if df.index[df[col] == x].tolist()[0] in best_outliers else x)
                st.write("📘 **Why this method?** This column has clear groups. DBSCAN found lonely points far from any group.")
                st.write("🔧 **Fix Used**: Replaced those with the most common value (mode).")

    return df




def feature_engineering_suggestions(df, target_column=None):
    st.write("## 🛠 Feature Engineering Suggestions")

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        st.write("No categorical columns found for encoding.")
        return df

    selected_cols = st.multiselect("🔘 Select categorical columns for encoding:", categorical_cols)

    encoding_choices = {}  # To store chosen encoding per column

    if selected_cols:
        for col in selected_cols:
            unique_values = df[col].nunique()
            st.markdown(f"---\n### ✏️ Column: `{col}` ({unique_values} unique values)")

            # Suggest encoding type based on unique values
            if unique_values <= 10:
                st.info("📘 Few unique values: One-Hot or Ordinal Encoding is usually best.")
            else:
                st.info("📘 Many unique values: Label, Target, Frequency, or Binary Encoding are better.")

            encoding_type = st.radio(
                f"Choose encoding method for `{col}`:",
                ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding", "Target Encoding", "Frequency Encoding", "Binary Encoding"],
                key=f"encoding_{col}"
            )
            encoding_choices[col] = encoding_type

    apply_changes = st.button("✅ Apply Feature Engineering")

    if apply_changes and selected_cols:
        df = df.copy()  # Work on a copy

        if target_column:
            y = df[target_column]

        for col, method in encoding_choices.items():
            if method == "Label Encoding":
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
                st.write(f"✅ `{col}`: Label Encoding applied.")

            elif method == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=[col], prefix=[col])
                st.write(f"✅ `{col}`: One-Hot Encoding applied.")

            elif method == "Ordinal Encoding":
                ordinal_encoder = OrdinalEncoder()
                df[col] = ordinal_encoder.fit_transform(df[[col]])
                st.write(f"✅ `{col}`: Ordinal Encoding applied.")

            elif method == "Target Encoding":
                if target_column:
                    encoder = ce.TargetEncoder(cols=[col])
                    df[col] = encoder.fit_transform(df[col], y)
                    st.write(f"✅ `{col}`: Target Encoding applied.")
                else:
                    st.warning(f"⚠️ `{col}`: Target Encoding skipped because target_column is not defined.")

            elif method == "Frequency Encoding":
                freq_map = df[col].value_counts() / len(df)
                df[col] = df[col].map(freq_map)
                st.write(f"✅ `{col}`: Frequency Encoding applied.")

            elif method == "Binary Encoding":
                encoder = ce.BinaryEncoder(cols=[col])
                df = encoder.fit_transform(df)
                st.write(f"✅ `{col}`: Binary Encoding applied.")

        # Scale numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            st.write("📏 Standard Scaling applied to numeric columns.")

        st.write("### 🔍 Final Dataset after Feature Engineering:")
        st.dataframe(df)

    return df


def main_app(df):
    # Streamlit File Uploader
    st.title("📊 AI-Powered Automated Data Insights")
    st.markdown("---")


    # uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    # if uploaded_file:
    #     df = pd.read_csv(uploaded_file, encoding="latin1") if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    #     st.write(df.dtypes)
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
    #if len(df_encoded)<5:
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
        if len(df_encoded) < 5:
            st.warning("⚠️ Not enough data to train models. Please upload at least 5 rows.")
        else:
            # Splitting the data
            X = df_encoded.drop(columns=[target_column])
            y = df_encoded[target_column]
            if y.nunique() < 2:
                st.error("❌ The target column must contain at least 2 unique classes for classification.")
            else:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
            # Proceed with AutoML or manual model training here


                    # TPOT AutoML
                    st.subheader("🔬 AutoML Model Selection")
                    if st.button("Run TPOT AutoML"):
                        if len(X_train) < 2:
                            st.warning("⚠️ Not enough training data after split. Try using more records.")
                        else:
                            tpot = tpot = TPOTClassifier(
                                generations=5,
                                population_size=20,
                                verbosity=2,
                                random_state=42,
                                config_dict='TPOT light'  # Simpler models, avoids many such conflicts
                            )
                            tpot.fit(X_train, y_train)

                            if hasattr(tpot, 'fitted_pipeline_'):
                                best_model = tpot.fitted_pipeline_
                                st.success(f"✅ Best TPOT Model:\n\n{best_model}")
                                test_score = best_model.score(X_test, y_test)

                                st.markdown("### 🤖 Why This Model?")
                                st.info("TPOT tried many models and chose the one that gave the best results on your data. "
                                        "It uses techniques like Decision Trees, SVMs, or Ensemble models and tunes them automatically.")
                                st.success("✅ TPOT successfully found the best model!")
                                st.markdown(f"### 🧠 Best TPOT Model:\n```python\n{best_model}\n```")
                                st.markdown(f"### 📊 Test Accuracy: **{test_score:.4f}**")

                            else:
                                st.error("❌ TPOT did not complete model optimization. Try again with more data or simpler input.")
                except ValueError as ve:
                    st.error(f"🚨 Data splitting error: {ve}")
                except Exception as e:
                    st.error(f"🚨 Unexpected error: {e}")
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
            st.write("### 📊 Model Performance Comparison")
            st.write(pd.DataFrame(results).T)


    # Add "No column selected" option
    time_col = st.selectbox("Select Date Column", ["No column selected"] + list(df.columns), index=0)
    value_col = st.selectbox("Select Value Column", ["No column selected"] + list(df.columns), index=0)

    # Proceed only if valid columns are selected
    if time_col != "No column selected" and value_col != "No column selected":
        # Convert to datetime (Ensure correct format)
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')  
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')  # Ensure value is numeric

        
        # Drop rows where date or value is missing
        df_clean = df.dropna(subset=[time_col, value_col])
        df_clean = df_clean.sort_values(by=time_col)  # Ensure time series order

        st.write(f"Available observations after cleaning: {len(df_clean)}")
        st.dataframe(df_clean[[time_col, value_col]])

        # Convert datetime to numeric format (Unix timestamp)
        df_clean["timestamp"] = df_clean[time_col].astype('int64') // 10**9

        # Ensure enough data points
        if len(df_clean) >= 12:
            decomposition = seasonal_decompose(df_clean[value_col], period=12, model='additive', extrapolate_trend='freq')

            # Add decomposition results
            df_clean["trend"] = decomposition.trend
            df_clean["seasonal"] = decomposition.seasonal
            df_clean["resid"] = decomposition.resid

            # Detect Anomalies using Z-score
            df_clean["Z-Score"] = np.abs(zscore(df_clean[value_col]))
            df_clean["Anomaly"] = df_clean["Z-Score"] > 2.5

            # Plot Time Series with Trend and Anomalies
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_clean[time_col], y=df_clean[value_col], mode='lines', name='Original Data'))
            fig.add_trace(go.Scatter(x=df_clean[time_col], y=df_clean["trend"], mode='lines', name='Trend', line=dict(color='orange')))

            # Mark anomalies
            anomalies = df_clean[df_clean["Anomaly"]]
            fig.add_trace(go.Scatter(x=anomalies[time_col], y=anomalies[value_col], mode='markers',
                                    name='Anomalies', marker=dict(color='red', size=8)))

            # Update layout
            #fig.update_layout(title="Trend & Anomaly Detection", xaxis_title="Time", yaxis_title="Value")

            # Display the plot
            #st.plotly_chart(fig)
            # Trend
            st.write("### Trend Component")
            trend_smoothed = df_clean.set_index(time_col)["trend"].resample('W').mean()
            st.line_chart(trend_smoothed)

            # Trend Analysis based on the smoothed trend line
            start_trend = trend_smoothed.iloc[0]
            end_trend = trend_smoothed.iloc[-1]

            if end_trend > start_trend * 1.05:
                trend_conclusion = "The trend shows an overall **increasing pattern** over time."
            elif end_trend < start_trend * 0.95:
                trend_conclusion = "The trend shows an overall **decreasing pattern** over time."
            else:
                trend_conclusion = "The trend has remained **relatively stable** over the observed period."

            # Display description and conclusion
            st.write("""
            **What it shows:**  
            The trend component represents the long-term progression or direction in the data over time, ignoring short-term fluctuations and seasonality.
            """)

            st.write(f"""
            **Conclusion:**  
            {trend_conclusion}
            """)


            # Show Decomposition Components
            st.write("### Seasonal Component")
            # Resample or average by week to reduce clutter
            seasonal_smoothed = df_clean.set_index(time_col)["seasonal"].resample('W').mean()

            # Plot using line_chart
            st.line_chart(seasonal_smoothed)

            # Seasonal analysis (check how strong or fluctuating the seasonal values are)
            seasonal_range = seasonal_smoothed.max() - seasonal_smoothed.min()

            if seasonal_range > df_clean[value_col].std() * 0.5:
                season_conclusion = "There is a **strong seasonal pattern** in the data, with clear periodic fluctuations."
            elif seasonal_range > df_clean[value_col].std() * 0.2:
                season_conclusion = "The data shows a **moderate seasonal pattern** — periodic variations are present but not very strong."
            else:
                season_conclusion = "There is **little to no clear seasonal pattern** detected in the data."

            # Display description and conclusion
            st.write("""
            **What it shows:**  
            The seasonal component captures recurring patterns or cycles within the data, like monthly sales peaks or seasonal demands.
            """)

            st.write(f"""
            **Conclusion:**  
            {season_conclusion}
            """)


            
            st.write("### Residual (Noise)")
            # Resample or average by week to reduce clutter
            residual_smoothed = df_clean.set_index(time_col)["resid"].resample('W').mean()

            # Plot using line_chart
            st.line_chart(residual_smoothed)

            # Residual analysis (check how noisy/volatile the residual is)
            residual_std = residual_smoothed.std()

            if residual_std > df_clean[value_col].std() * 0.5:
                residual_conclusion = "The data contains **high residual noise**, indicating significant random fluctuations and possible anomalies."
            elif residual_std > df_clean[value_col].std() * 0.2:
                residual_conclusion = "The residual component shows **moderate noise** — some fluctuations remain unexplained."
            else:
                residual_conclusion = "The residual noise is **low**, meaning the trend and seasonal components explain most of the variability."

            # Display description and conclusion
            st.write("""
            **What it shows:**  
            The residual (noise) component captures irregular, unpredictable variations that are not part of the trend or seasonal pattern.
            """)

            st.write(f"""
            **Conclusion:**  
            {residual_conclusion}
            """)


        else:
            st.warning(f"Not enough data points for seasonal decomposition. At least 12 observations are required.")
            st.write(f"Available observations: {len(df_clean)}")
            st.write("Here’s the available data:")
            st.dataframe(df_clean[[time_col, value_col]])

    else:
        st.info("Please select valid Date/Time and Value columns to continue.")


    st.write("## SQL Query Builder")

    if df is not None and not df.empty:
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

    if df is not None and not df.empty:
        kpi_metrics = {
        "Total Rows": df.shape[0],
        "Total Columns": df.shape[1],
        "Total Sales (if applicable)": df["Sales"].sum() if "Sales" in df.columns else "N/A",
        #"Average Value per Column": df.select_dtypes(include=[np.number]).describe().mean().to_dict()
        }

        st.json(kpi_metrics)  # Show KPIs in JSON format

    
    st.write("## Data Alerts & Notifications")

    if df is not None and not df.empty:
        alert_column = st.selectbox("Select column for alert", df.columns)
        alert_condition = st.selectbox("Condition", [">", "<", "=", ">=", "<="])
        alert_value = st.number_input("Threshold Value")

        # Convert column to numeric, handling NaT/NaN values
        df[alert_column] = pd.to_numeric(df[alert_column], errors='coerce')

        # Drop NaN values before comparison
        filtered_df = df[alert_column].dropna()

        alert_triggered = False  # Default
        # Check condition
        if not filtered_df.empty:  # Ensure column has valid data
            if alert_condition == ">":
                alert_triggered = filtered_df.max() > alert_value
            elif alert_condition == "<":
                alert_triggered = filtered_df.min() < alert_value
            elif alert_condition == "=":
                alert_triggered = (filtered_df == alert_value).any()
            elif alert_condition == ">=":
                alert_triggered = filtered_df.max() >= alert_value
            elif alert_condition == "<=":
                alert_triggered = filtered_df.min() <= alert_value

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

    # Get column names from the DataFrame
    columns = df.columns.tolist()

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Let user select X-Axis and Y-Axis
    x_axis = st.selectbox("Select X-Axis", options=columns) # Use all columns
    y_axis = st.selectbox("Select Y-Axis", options=columns) # Use all columns

   

# Assuming df, x_axis, y_axis, numeric_columns, and categorical_columns are already defined

    # Let user select chart type
    st.sidebar.subheader("Choose Chart Type")
    all_chart_types = ["Scatter Plot", "Line Chart", "Area Chart", "Bar Chart", "Box Plot", "Violin Plot", "Pie Chart", "Treemap", "Sunburst Chart"]
    chart_type = st.sidebar.selectbox("Select Chart Type", all_chart_types)

    # Generate Chart based on selected type
    if chart_type == "Scatter Plot" and x_axis in numeric_columns and y_axis in numeric_columns:
        st.subheader(f"Scatter Plot of {y_axis} vs {x_axis}")
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart" and x_axis in numeric_columns and y_axis in numeric_columns:
        st.subheader(f"Line Chart of {y_axis} vs {x_axis}")
        fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Area Chart" and x_axis in numeric_columns and y_axis in numeric_columns:
        st.subheader(f"Area Chart of {y_axis} vs {x_axis}")
        fig = px.area(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar Chart":
        st.subheader(f"Bar Chart of {y_axis} vs {x_axis}")
        fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot" and y_axis in numeric_columns and x_axis in categorical_columns:
        st.subheader(f"Box Plot of {y_axis} by {x_axis}")
        fig = px.box(df, x=x_axis, y=y_axis, title=f"Box plot of {y_axis} by {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Violin Plot" and y_axis in numeric_columns and x_axis in categorical_columns:
        st.subheader(f"Violin Plot of {y_axis} by {x_axis}")
        fig = px.violin(df, x=x_axis, y=y_axis, box=True, points="all", title=f"Violin plot of {y_axis} by {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart" and x_axis in categorical_columns and y_axis in categorical_columns:
        st.subheader(f"Pie Chart of {y_axis} by {x_axis}")
        fig = px.pie(df, names=x_axis, values=y_axis, title=f"{x_axis} vs {y_axis} Distribution")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Treemap" and x_axis in categorical_columns and y_axis in numeric_columns:
        st.subheader(f"Treemap of {y_axis} by {x_axis}")
        fig = px.treemap(df, path=[x_axis], values=y_axis, title=f"Treemap of {y_axis} by {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Sunburst Chart" and x_axis in categorical_columns and y_axis in numeric_columns:
        st.subheader(f"Sunburst Chart of {y_axis} by {x_axis}")
        fig = px.sunburst(df, path=[x_axis], values=y_axis, title=f"Sunburst chart of {y_axis} by {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Please select a valid chart type.")

    if 'cleaned_df' in st.session_state:
        st.write("### Download Cleaned Dataset")
    
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        st.session_state['cleaned_df'].to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Download button
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )





# #create_users_table()

# menu = ["Login", "Register", "Forgot Password"]
# if st.session_state["logged_in"]:
#     menu.append("Logout")
# choice = st.sidebar.selectbox("Menu", menu)

# if choice == "Register":
#     st.subheader("Create an Account")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")

#     if st.button("Register"):
#         # Check if the user already exists
#         if user_exists(username):  
#             st.warning("Username already exists! Please choose a different one.")
#         else:
#             register_user(username, password)
#             st.success("Account created successfully! Please login.")

# elif choice == "Login":
#     st.subheader("Login to Your Account")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         if validate_login(username, password):
#             st.session_state["logged_in"] = True
#             st.session_state["username"] = username
#             st.success("Logged in successfully!")
#             #main_app()
#         else:
#             st.error("Invalid username or password")

# elif choice == "Forgot Password":
#     st.subheader("Reset Your Password")
#     username = st.text_input("Enter your username")
#     new_password = st.text_input("Enter new password", type="password")
#     confirm_password = st.text_input("Confirm new password", type="password")

    # if st.button("Reset Password"):
    #     if not user_exists(username):  # Check if the user exists
    #         st.error("User not found! Please check your username.")
    #     elif new_password != confirm_password:
    #         st.error("Passwords do not match!")
    #     elif is_same_password(username, new_password):  # Check if the new password is same as old
    #         st.warning("New password cannot be the same as the current password.")
    #     else:
    #         update_password(username, new_password)
    #         st.success("Password reset successful! You can now login.") 

# elif choice == "Logout":
#     st.session_state["logged_in"] = False
#     st.session_state["username"] = ""
#     st.success("Logged out successfully!")
#     st.experimental_rerun() 

# if st.session_state["logged_in"]:
    #main_app(df)