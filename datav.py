import streamlit as st
import sqlite3
import pandas as pd
import os

# Path to your uploaded DB file
DB_PATH = "users.db"

# Admin credentials
ADMIN_USERNAME = "admin123"
ADMIN_PASSWORD = "admin@123"

# Function to check login
def check_login(username, password):
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

# Function to get all tables and display them
def show_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if not tables:
        st.warning("No tables found in the database.")
        return

    for table in tables:
        table_name = table[0]
        st.subheader(f"📄 Table: {table_name}")
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        st.dataframe(df)

    conn.close()

# ------------------ Streamlit UI ------------------ #
def log_main():
    #st.set_page_config(page_title="Admin Login", layout="centered")
    st.title("🔐 Admin Login")

    # Login input
    username = st.text_input("Username",key="admin_login")
    password = st.text_input("Password", type="password",key="admin_pass")
    login_btn = st.button("Login")

    # Session state to store login status
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Handle login
    if login_btn:
        if check_login(username, password):
            st.session_state.logged_in = True
            st.success("Login successful! ✅")
        else:
            st.error("Invalid credentials. Please try again.")

    # If logged in, show database
    if st.session_state.logged_in:
        if os.path.exists(DB_PATH):
            st.markdown("---")
            st.header("📊 Database Contents")
            show_database(DB_PATH)
        else:
            st.error("Database file not found.")
