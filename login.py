import sqlite3
import streamlit as st
import hashlib  # Import the hashlib library for password hashing

# Database Path
DB_PATH = "users.db"

# --- Database Interaction Functions ---
def get_connection():
    """Establishes and returns a database connection.  Creates the database if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except sqlite3.Error as e:
        st.error(f"Database Connection Error: {e}")
        return None  # Important:  Return None on failure

def create_users_table():
    """Creates the user table if it doesn't exist."""
    conn = get_connection()
    if conn is None:
        return  # Exit if connection failed
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
    # Hash the password before storing it
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
        return False, "User not found."  # User doesn't exist

    stored_password_hash = user_data[0]  # Get the stored hashed password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()  # Hash the provided password

    if hashed_password == stored_password_hash:
        return True, None  # Login successful
    else:
        return False, "Invalid password."  # Login failed



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
            return False  # User not found
        stored_password_hash = result[0]
        hashed_new_password = hashlib.sha256(new_password.encode()).hexdigest()
        return stored_password_hash == hashed_new_password
    except sqlite3.Error as e:
        st.error(f"Error checking password: {e}")
        return False
    finally:
        conn.close()



def main():
    """Main Streamlit application function."""
    st.title("Login Page")
    create_users_table()  # Ensure the user table exists

    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    menu = ["Login", "Register", "Forgot Password", "Logout"] #added logout to menu
    if st.session_state["logged_in"]:
        menu.append("Logout")
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register":
        st.subheader("Create an Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Register"):
            # Check if the user already exists
            if user_exists(username):
                st.warning("Username already exists! Please choose a different one.")
            else:
                success, reg_message = register_user(username, password) # Capture the return values
                if success:
                    st.success(reg_message + " Please login.")
                else:
                    st.error(reg_message)

    elif choice == "Login":
        st.subheader("Login to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, message = validate_login(username, password) # Capture the return values
            if success:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(message)
                st.experimental_rerun()
            else:
                st.error(message)

    elif choice == "Forgot Password":
        st.subheader("Reset Your Password")
        username = st.text_input("Enter your username")
        new_password = st.text_input("Enter new password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")

        if st.button("Reset Password"):
            if not user_exists(username):  # Check if the user exists
                st.error("User not found! Please check your username.")
            elif new_password != confirm_password:
                st.error("Passwords do not match!")
            elif is_same_password(username, new_password):  # Check if the new password is same as old
                st.warning("New password cannot be the same as the current password.")
            else:
                success, message = update_password(username, new_password) # Capture the return values
                if success:
                    st.success(message)
                else:
                    st.error(message)

    elif choice == "Logout":
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.success("Logged out successfully!")
        st.experimental_rerun()

if __name__ == "__main__":
    main()
