import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

import os

DB_PATH = "data_entry.db"  # Path for the database file


def get_connection():
    """Establishes and returns a database connection. Creates the database if not exists."""
    if not os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sqlite_master (id INTEGER PRIMARY KEY)"
            )  # Dummy table to ensure DB creation
    return sqlite3.connect(DB_PATH)


# --- Database Interaction Functions ---


def get_connection():
    """Establishes and returns a database connection."""
    return sqlite3.connect(DB_PATH)


def create_table(conn, table_name, columns):
    """Creates a table with the specified name and columns."""
    col_defs = ", ".join([f"{name} {dtype}" for name, dtype in columns])
    query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, {col_defs})"
    try:
        conn.execute(query)
        conn.commit()
        return True, None  # Success
    except Exception as e:
        return False, str(e)  # Failure with error message


def insert_record(conn, table_name, data):
    """Inserts a record into the specified table."""
    columns = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    values = list(data.values())
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    try:
        conn.execute(query, values)
        conn.commit()
        return True, None
    except Exception as e:
        return False, str(e)


def fetch_table_data(conn, table_name):
    """Fetches all data from the specified table."""
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        return df, None
    except Exception as e:
        return None, str(e)


def get_table_list(conn):
    """Fetches a list of table names in the database."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        return tables, None
    except Exception as e:
        return [], str(e)


def get_column_names(conn, table_name):
    """Fetches column names and types for a given table."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        # Return a list of (name, type) tuples
        return [(info[1], info[2]) for info in columns_info], None
    except Exception as e:
        return [], str(e)


def delete_table(conn, table_name):
    """Deletes the specified table."""
    try:
        query = f"DROP TABLE IF EXISTS {table_name}"
        conn.execute(query)
        conn.commit()
        return True, None
    except Exception as e:
        return False, str(e)


def delete_record(conn, table_name, record_id):
    """Deletes a record from the specified table based on its ID."""
    try:
        query = f"DELETE FROM {table_name} WHERE id = ?"
        conn.execute(query, (record_id,))
        conn.commit()
        return True, None
    except Exception as e:
        return False, str(e)


# --- Streamlit UI ---


def main():
    """Main Streamlit application function."""
    st.title("Dynamic Data Entry Dashboard")
    conn = get_connection()  # Establish connection here

    # --- Sidebar for Table Selection/Creation ---
    with st.sidebar:
        st.header("Table Management")
        table_options = ["Create New Table", "Select Existing Table", "Delete Table"]
        selected_option = st.radio("Choose Action:", table_options)

        if selected_option == "Create New Table":
            new_table_name = st.text_input("New Table Name:")
            num_columns = st.number_input("Number of Columns:", min_value=1, step=1)
            new_columns = []
            for i in range(num_columns):
                col_name = st.text_input(f"Column {i+1} Name:")
                col_type = st.selectbox(
                    f"Column {i+1} Type:", ["TEXT", "INTEGER", "REAL", "DATE", "TIME"]
                )
                new_columns.append((col_name, col_type))

            if st.button("Create Table"):
                if not new_table_name:
                    st.error("Table name cannot be empty.")
                elif not all(col_name for col_name, _ in new_columns):
                    st.error("Column names cannot be empty.")
                else:
                    success, error = create_table(conn, new_table_name, new_columns)
                    if success:
                        st.success(f"Table '{new_table_name}' created successfully!")
                        st.session_state.selected_table = (
                            new_table_name
                        )  # Auto-select the new table
                    else:
                        st.error(f"Error creating table: {error}")

        elif selected_option == "Select Existing Table":
            tables, error = get_table_list(conn)
            if error:
                st.error(f"Error fetching tables: {error}")
                tables = []
            if not tables:
                st.info("No tables found. Create a new table first.")
                tables = []  # Ensure tables is an empty list
            selected_table = st.selectbox("Select a Table:", tables)
            if selected_table:
                st.session_state.selected_table = selected_table  # persist table selection

        elif selected_option == "Delete Table":
            tables, error = get_table_list(conn)
            if error:
                st.error(f"Error fetching tables: {error}")
                tables = []
            if not tables:
                st.info("No tables found.  Create a new table first.")
                tables = []
            table_to_delete = st.selectbox("Select a Table to Delete:", tables)
            if st.button("Delete Selected Table"):
                if not table_to_delete:
                    st.error("Please select a table to delete.")
                else:
                    success, error = delete_table(conn, table_to_delete)
                    if success:
                        st.success(f"Table '{table_to_delete}' deleted successfully!")
                        if (
                            "selected_table" in st.session_state
                            and st.session_state.selected_table == table_to_delete
                        ):
                            del st.session_state.selected_table  # clear deleted table
                    else:
                        st.error(f"Error deleting table: {error}")

    # --- Main Area for Data Entry and Display ---
    if "selected_table" in st.session_state:  # Only show if a table is selected
        st.header(f"Data Entry for Table: {st.session_state.selected_table}")

        columns, error = get_column_names(conn, st.session_state.selected_table)
        if error:
            st.error(f"Error fetching column names: {error}")
            columns = []

        if columns:  # only show data entry if columns exist
            with st.form("data_entry_form"):
                data = {}
                for col_name, col_type in columns:
                    if col_type == "INTEGER":
                        data[col_name] = st.number_input(
                            f"Enter {col_name} ({col_type}):", step=1
                        )
                    elif col_type == "REAL":
                        data[col_name] = st.number_input(
                            f"Enter {col_name} ({col_type}):", format="%.2f"
                        )
                    elif col_type == "DATE":
                        date_str = st.text_input(
                            f"Enter {col_name} ({col_type} YYYY-MM-DD):"
                        )
                        try:
                            data[col_name] = (
                                datetime.strptime(date_str, "%Y-%m-%d").strftime(
                                    "%Y-%m-%d"
                                )
                                if date_str
                                else None
                            )
                        except ValueError:
                            st.error(
                                f"Invalid date format for {col_name}. Please use YYYY-MM-DD."
                            )
                            data[col_name] = (
                                None
                            )  # Set to None to avoid inserting invalid data
                    elif col_type == "TIME":
                        time_obj = st.time_input(f"Enter {col_name} ({col_type}):")
                        data[col_name] = (
                            time_obj.strftime("%H:%M:%S") if time_obj else None
                        )
                    else:
                        data[col_name] = st.text_input(
                            f"Enter {col_name} ({col_type}):"
                        )

                submit_button = st.form_submit_button("Add Record")

            if submit_button:
                # Check if all data values are not None (for required fields)
                if all(value is not None for value in data.values()):
                    success, error = insert_record(
                        conn, st.session_state.selected_table, data
                    )
                    if success:
                        st.success("Record added successfully!")
                    else:
                        st.error(f"Error adding record: {error}")
                else:
                    st.warning("Please fill in all fields with valid values.")

        # --- Display Table Data ---
        st.subheader(f"Data in Table: {st.session_state.selected_table}")
        df, error = fetch_table_data(conn, st.session_state.selected_table)
        if error:
            st.error(f"Error fetching table data: {error}")
        elif (
            df is not None and not df.empty
        ):  #  Show dataframe if it is not None and not empty
            st.dataframe(df)

            # --- Delete Record Option ---
            st.subheader("Delete Record")
            record_id_to_delete = st.number_input(
                "Enter ID of record to delete:", min_value=1, step=1
            )
            if st.button("Delete Record"):
                success, error = delete_record(
                    conn, st.session_state.selected_table, record_id_to_delete
                )
                if success:
                    st.success("Record deleted successfully!")
                    # Refresh the table display after deletion
                    df, error = fetch_table_data(
                        conn, st.session_state.selected_table
                    )
                    if error:
                        st.error(f"Error fetching updated table data: {error}")
                    elif df is not None and not df.empty:
                        st.dataframe(df)
                    else:
                        st.info("No data available in this table.")
                else:
                    st.error(f"Error deleting record: {error}")

        elif df is not None and df.empty:
            st.info("No data available in this table.")

        else:
            st.info("Please select or create a table to start.")

    # Close connection at the end
    conn.close()


if __name__ == "__main__":
    main()

