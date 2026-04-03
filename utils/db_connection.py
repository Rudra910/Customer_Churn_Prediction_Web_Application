import mysql.connector
from mysql.connector import Error
import pandas as pd

def create_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='',  # Change to your MySQL username
            password='',  # Change to your MySQL password
            database='churn_prediction_db'
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def create_database_and_tables():
    """Create database and tables if they don't exist"""
    connection = None
    cursor = None
    
    try:
        # Connect without database
        connection = mysql.connector.connect(
            host='localhost',
            user='',
            password=''  # Add your MySQL password here
        )
        cursor = connection.cursor()
        
        # Create database
        cursor.execute("CREATE DATABASE IF NOT EXISTS churn_prediction_db")
        cursor.execute("USE churn_prediction_db")
        
        # Create input_data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS input_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                Customer_ID VARCHAR(50),
                Age INT,
                Gender VARCHAR(10),
                Total_Spend FLOAT,
                Average_Order_Value FLOAT,
                Purchase_Frequency INT,
                Last_Purchase_Days INT,
                Customer_Rating FLOAT,
                Complaint_Raised VARCHAR(10),
                Return_Count INT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create prediction_results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                Customer_ID VARCHAR(50),
                Age INT,
                Gender VARCHAR(10),
                Total_Spend FLOAT,
                Average_Order_Value FLOAT,
                Purchase_Frequency INT,
                Last_Purchase_Days INT,
                Customer_Rating FLOAT,
                Complaint_Raised VARCHAR(10),
                Return_Count INT,
                churn_probability FLOAT,
                churn VARCHAR(10),
                retention_strategy TEXT,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        connection.commit()
        print("✅ Database and tables created successfully")
        
    except Error as e:
        print(f"❌ Error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
            print("🔒 MySQL connection closed")

def delete_old_data():
    """Delete all existing data from both tables"""
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Delete all data from prediction_results first (due to foreign key if any)
            cursor.execute("DELETE FROM prediction_results")
            deleted_predictions = cursor.rowcount
            print(f"✅ Deleted {deleted_predictions} old records from prediction_results")
            
            # Delete all data from input_data
            cursor.execute("DELETE FROM input_data")
            deleted_input = cursor.rowcount
            print(f"✅ Deleted {deleted_input} old records from input_data")
            
            # Reset auto-increment counters (optional)
            cursor.execute("ALTER TABLE input_data AUTO_INCREMENT = 1")
            cursor.execute("ALTER TABLE prediction_results AUTO_INCREMENT = 1")
            
            connection.commit()
            print("✅ All old data cleared successfully")
            
        except Error as e:
            print(f"❌ Error deleting old data: {e}")
            connection.rollback()
        finally:
            if cursor:
                cursor.close()
            if connection.is_connected():
                connection.close()
    else:
        print("❌ Failed to connect to database")

def insert_input_data(df):
    """Delete old data and insert new uploaded data into input_data table"""
    # First delete all old data
    delete_old_data()
    
    # Then insert new data
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO input_data 
                    (Customer_ID, Age, Gender, Total_Spend, Average_Order_Value, 
                     Purchase_Frequency, Last_Purchase_Days, Customer_Rating, 
                     Complaint_Raised, Return_Count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    row['Customer_ID'], row['Age'], row['Gender'], 
                    row['Total_Spend'], row['Average_Order_Value'],
                    row['Purchase_Frequency'], row['Last_Purchase_Days'],
                    row['Customer_Rating'], row['Complaint_Raised'], row['Return_Count']
                ))
            
            connection.commit()
            print(f"✅ Inserted {len(df)} new rows into input_data")
            
        except Error as e:
            print(f"❌ Error inserting data: {e}")
            connection.rollback()
        finally:
            if cursor:
                cursor.close()
            if connection.is_connected():
                connection.close()
    else:
        print("❌ Failed to connect to database")

def insert_prediction_results(df):
    """Delete old data and insert new prediction results"""
    # Note: delete_old_data() is already called in insert_input_data
    # So we don't need to delete again, just insert
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO prediction_results 
                    (Customer_ID, Age, Gender, Total_Spend, Average_Order_Value, 
                     Purchase_Frequency, Last_Purchase_Days, Customer_Rating, 
                     Complaint_Raised, Return_Count, churn_probability, churn, retention_strategy)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    row['Customer_ID'], row['Age'], row['Gender'], 
                    row['Total_Spend'], row['Average_Order_Value'],
                    row['Purchase_Frequency'], row['Last_Purchase_Days'],
                    row['Customer_Rating'], row['Complaint_Raised'], row['Return_Count'],
                    float(row['churn_probability']), row['churn'], row['retention_strategy']
                ))
            
            connection.commit()
            print(f"✅ Inserted {len(df)} new rows into prediction_results")
            
        except Error as e:
            print(f"❌ Error inserting data: {e}")
            connection.rollback()
        finally:
            if cursor:
                cursor.close()
            if connection.is_connected():
                connection.close()
    else:
        print("❌ Failed to connect to database")

def get_all_predictions():
    """Get all prediction results for Power BI"""
    connection = create_connection()
    if connection:
        try:
            query = "SELECT * FROM prediction_results"
            df = pd.read_sql(query, connection)
            return df
        except Error as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()
        finally:
            if connection.is_connected():
                connection.close()
    return pd.DataFrame()

def get_table_counts():
    """Helper function to check current record counts"""
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM input_data")
            input_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM prediction_results")
            prediction_count = cursor.fetchone()[0]
            
            print(f"📊 Current records - input_data: {input_count}, prediction_results: {prediction_count}")
            
            cursor.close()
            connection.close()
            
            return input_count, prediction_count
            
        except Error as e:
            print(f"❌ Error getting counts: {e}")
            return 0, 0
    return 0, 0