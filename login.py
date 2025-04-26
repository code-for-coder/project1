import sqlite3
import streamlit as st
import hashlib
import analysis
import data_entry
import pandas as pd
import datav

# Database Path
DB_PATH = "users.db"
DATA_ENTRY_DB_PATH = "data_entry.db" #Added Data entry DB path

# --- Database Interaction Functions ---
def get_connection(db_path=DB_PATH):
    """Establishes and returns a database connection. Creates the database if it doesn't exist."""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        st.error(f"Database Connection Error: {e}")
        return None

def create_users_table():
    """Creates the user table if it doesn't exist."""
    conn = get_connection()
    if conn is None:
        return
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
            """
        )
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error creating table: {e}")
        conn.rollback()
    finally:
        conn.close()

def register_user(username, password):
    """Registers a new user with a hashed password."""
    conn = get_connection()
    if conn is None:
        return False, "Failed to connect to the database."
    cursor = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password),
        )
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        conn.rollback()
        return False, "Username already exists."
    except sqlite3.Error as e:
        conn.rollback()
        return False, f"Error during registration: {e}"
    finally:
        conn.close()

def get_user_data(username):
    """Retrieves user data (including hashed password)"""
    conn = get_connection()
    if conn is None:
        return None
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        return cursor.fetchone()
    except sqlite3.Error as e:
        st.error(f"Error fetching user data: {e}")
        return None
    finally:
        conn.close()

def validate_login(username, password):
    """Validates the username and password against the stored hashed password."""
    user_data = get_user_data(username)
    if user_data is None:
        return False, "User not found."
    stored_password_hash = user_data[0]
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    if hashed_password == stored_password_hash:
        return True, None
    else:
        return False, "Invalid password."

def update_password(username, new_password):
    """Updates the user's password with a new hashed password."""
    conn = get_connection()
    if conn is None:
        return False, "Failed to connect to the database."
    cursor = conn.cursor()
    hashed_new_password = hashlib.sha256(new_password.encode()).hexdigest()
    try:
        cursor.execute(
            "UPDATE users SET password = ? WHERE username = ?",
            (hashed_new_password, username),
        )
        conn.commit()
        return True, None
    except sqlite3.Error as e:
        conn.rollback()
        return False, f"Error updating password: {e}"
    finally:
        conn.close()

def user_exists(username):
    """Checks if a user exists in the database."""
    conn = get_connection()
    if conn is None:
        return False
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        st.error(f"Error checking user existence: {e}")
        return False
    finally:
        conn.close()

def is_same_password(username, new_password):
    """Checks if the new password is the same as the old password."""
    conn = get_connection()
    if conn is None:
        return False
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if result is None:
            return False
        stored_password_hash = result[0]
        hashed_new_password = hashlib.sha256(new_password.encode()).hexdigest()
        return stored_password_hash == hashed_new_password
    except sqlite3.Error as e:
        st.error(f"Error checking password: {e}")
        return False
    finally:
        conn.close()

def login_page():
    """Displays the login page."""
    if st.button('Admin'):
        datav.log_main()
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    login_button = st.sidebar.button("Login")

    register_option = st.sidebar.checkbox("Register New User")
    forgot_password_option = st.sidebar.checkbox("Forgot Password")

    if login_button:
        success, message = validate_login(username, password)
        if success:
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = "main_menu" #set the page to main_menu
            st.experimental_rerun()
        else:
            st.error(message)

    if register_option:
        st.sidebar.subheader("Register")
        new_username = st.sidebar.text_input("New Username", key="register_username")
        new_password = st.sidebar.text_input("New Password", type="password", key="register_password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="register_confirm_password")
        if st.sidebar.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif user_exists(new_username):
                st.error("Username already exists.")
            else:
                success, message = register_user(new_username, new_password)
                if success:
                    st.success("Registration successful! Please log in.")
                else:
                    st.error(f"Registration failed: {message}")

    if forgot_password_option:
        st.sidebar.subheader("Forgot Password")
        forgot_username = st.sidebar.text_input("Username", key="forgot_username")
        new_password = st.sidebar.text_input("New Password", type="password", key="forgot_new_password")
        confirm_password = st.sidebar.text_input("Confirm New Password", type="password", key="forgot_confirm_password")
        if st.sidebar.button("Reset Password"):
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            elif not user_exists(forgot_username):
                st.error("Username not found.")
            elif is_same_password(forgot_username, new_password):
                st.warning("New password cannot be same as old password")
            else:
                success, message = update_password(forgot_username, new_password)
                if success:
                    st.success("Password reset successful! Please log in with new password")
                else:
                    st.error(f"Password reset failed: {message}")



def main():
    st.link_button("Go to admin","")
    """Main Streamlit application function."""
    create_users_table()  # Ensure the user table exists
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "page" not in st.session_state:  # Initialize 'page' here
        st.session_state.page = "login"

    if st.session_state.page == "login":
        login_page()  # Show the login page
    else:
        #moved show main menu here.
        show_main_menu()

def show_main_menu():
    """Displays the main menu options after successful login."""
    st.sidebar.title("Main Menu")
    menu_options = ["Data Entry", "Analysis", "Logout"]
    choice = st.sidebar.selectbox("Select an option:", menu_options)

    if choice == "Data Entry":
        st.session_state.page = "data_entry"

    elif choice == "Analysis":
        st.session_state.page = "analysis"

    elif choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.page = "login"  # Reset to login page

    #These functions are called everytime
    if st.session_state.page == "data_entry":
        show_data_entry()
    elif st.session_state.page == "analysis":
        show_analysis_page()
    elif st.session_state.page == "login":
        login_page() #show login page again
    # There is no else condition, so if the page is "main_menu", nothing will be shown
    # other than the sidebar.

def show_data_entry():
    """Displays the data entry page"""
    st.subheader("Data Entry Page")
    data_entry.main()  # Call data_entry.main()

def show_analysis_page():
    """Handles the analysis page functionality."""
    st.subheader("Analysis Page")
    with st.sidebar:
        st.subheader("Analysis Options")
        analysis_options = ["Upload File", "Perform analysis on present data"]
        st.session_state.analysis_option = st.radio(
            "Select Data Source:", analysis_options
        )
    if st.session_state.analysis_option == "Upload File":
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.write("Data from uploaded file:")
                st.dataframe(df)
                analysis.main_app(df)
            except UnicodeDecodeError as e:
                st.error(f"Error reading file: {e}.  Attempting to read with 'latin1' encoding.")
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, encoding='latin1')
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.write("Data from uploaded file:")
                    st.dataframe(df)
                    analysis.main_app(df)
                except Exception as e2:
                    st.error(f"Error reading file with latin1 encoding: {e2}.  Please ensure the file is a valid CSV or Excel file.")
            except Exception as e:
                st.error(f"Error reading file: {e}.  Please ensure the file is a valid CSV or Excel file.")

    elif st.session_state.analysis_option == "Perform analysis on present data":
        conn = get_connection(DATA_ENTRY_DB_PATH)
        table_list, error = data_entry.get_table_list(conn)  # Get table list from data_entry

        if error:
            st.error(f"Error fetching table list: {error}")
        elif table_list:  # Check if the table list is not empty
            table_name = st.selectbox("Select a table from data_entry.db", table_list)
            if table_name:
                df, fetch_error = data_entry.fetch_table_data(conn, table_name)
                if fetch_error:
                    st.error(f"Error fetching data from table: {fetch_error}")
                else:
                    analysis.main_app(df)
        else:
            st.warning("No tables found in data_entry.db to analyze.")
            # Perform analysis even if no table is shown (you might have a default behavior)
            st.info("Attempting analysis with no specific table selected...")
            analysis.main_app(pd.DataFrame())  # Example: Passing an empty DataFrame



if __name__ == "__main__":
    main()
