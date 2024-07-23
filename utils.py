import psycopg2
import pandas as pd
# from prediction import connect_to_db
#import eyed3
import numpy as np
from dotenv import load_dotenv
import os
import random

load_dotenv()


def check_existing_rating(features, cur, table_name):
    #select_query = f"SELECT duration_ms, danceability, energy, loudness, popularity FROM {table_name}"
    select_query = f"SELECT spectral_rolloff,zero_crossing_rate,spectral_bandwidth,spectral_centroid,spectral_contrast,mfcc,chroma,popularity FROM {table_name}"

    cur.execute(select_query)
    rows = cur.fetchall()
    tolerance = 1e-5
    label = None

    # Extract the values from the input features for the relevant columns
    #relevant_columns = ['duration_ms', 'danceability', 'energy', 'loudness']
    relevant_columns = ['spectral_rolloff','zero_crossing_rate','spectral_bandwidth','spectral_centroid','spectral_contrast','mfcc','chroma']

    features_values = features[relevant_columns].values[0]

    for row in rows:
        # Extract the relevant values from the row
        row_values = row[:-1]  
        row_label = row[-1]

        try:
            # Compare the values using np.isclose for numeric comparison
            if np.all(np.isclose(features_values, row_values, atol=tolerance)):
                label = row_label
                break
        except Exception as e:
            print(f"Error during comparison: {e}")
            print(f"Features Values: {features_values}")
            print(f"Row Values: {row_values}")
            continue

    return label


def check_rating(features):
    cur, conn = connect_to_db()

    alpha=check_existing_rating(features,cur,'song_feedback')
    conn.commit()
    cur.close()
    conn.close()
    return alpha


def create_databases():
    cur, conn =connect_to_db()
    if cur is not None and conn is not None:

        try:
            create_big_data(cur,conn)
            feedback_database(cur,conn)
            conn.commit
            cur.close
            conn.close
        
        except Exception as e:
            print(f"An error occurred: {e}")

    else:
        return 0


def create_big_data(cur, conn):
    # Check if the table already exists
    cur.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'Big_data')")
    table_exists = cur.fetchone()[0]
    
    if not table_exists:
        create_table_sql = """
        CREATE TABLE Big_data (
            spectral_rolloff FLOAT,
            zero_crossing_rate FLOAT,
            spectral_bandwidth FLOAT,
            spectral_centroid FLOAT,
            spectral_contrast FLOAT,
            mfcc FLOAT,
            chroma FLOAT,
            popularity INTEGER
        )
        """
        cur.execute(create_table_sql)
        print("Table created successfully.")

        # Prepare CSV file
        csv_file_path = 'dataset.csv'
        df = pd.read_csv(csv_file_path)
        df.drop(columns=['track_id', 'tempo', 'rms'], inplace=True)
        db_columns = [
            'spectral_rolloff', 'zero_crossing_rate', 'spectral_bandwidth', 'spectral_centroid', 'spectral_contrast', 'mfcc',
            'chroma', 'popularity'
        ]
        df = df.reindex(columns=db_columns)
        adjusted_csv_file_path = 'adjusted_dataset.csv'
        df.to_csv(adjusted_csv_file_path, index=False)
        print("CSV file created successfully.")

        # SQL statement to copy data from the new CSV file to the table
        copy_sql = """
        COPY Big_data (spectral_rolloff, zero_crossing_rate, spectral_bandwidth, spectral_centroid, spectral_contrast, mfcc, chroma, popularity)
        FROM stdin
        WITH CSV HEADER
        DELIMITER ','
        """

        # Execute the copy command
        with open(adjusted_csv_file_path, 'r') as f:
            cur.copy_expert(copy_sql, f)

        # Commit the transaction
        conn.commit()
        print("Data imported successfully.")
    else:
        print("Table already exists. Skipping table creation and data import.")
    return


# Function to adjust the values in top100 and bot36 to random values between 85-99, 3-15
def adjust_csv_values():
    df = pd.read_csv('features_top100_bot37.csv')
    copy_df = df.copy()
    
    for i, row in df.iterrows():
        if row['popularity'] == 0:
            copy_df.loc[i, 'popularity'] = random.randint(3,15)
        elif row['popularity'] == 100:
            copy_df.loc[i, 'popularity'] = random.randint(90,99)
    
    copy_df.to_csv('features_top100_bot37.csv')
    
    return
    


# Function to populate the song_feedback table with top 200 and bot 36 songs
def populate_feedback_table():
    cur, conn = connect_to_db()

    # SQL statement to copy data from the new CSV file to the table
    copy_sql = """
    COPY song_feedback (spectral_rolloff, zero_crossing_rate, spectral_bandwidth, spectral_centroid, spectral_contrast, mfcc, chroma, popularity)
    FROM stdin
    WITH CSV HEADER
    DELIMITER ','
    """
    
    csv_path = 'features_top100_bot37.csv'
    # Execute the copy command
    with open(csv_path, 'r') as f:
        cur.copy_expert(copy_sql, f)

    # Commit the transaction
    conn.commit()
    print("Data imported into song_feedback table.")
    
    return


def feedback_database(cur,conn):
    # Connect to the PostgreSQL database
    try:
        #conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        #cur = conn.cursor()

        # Create the tables if they don't already exist
        print("in feedback data base")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS song_feedback (
                spectral_rolloff FLOAT,
                zero_crossing_rate FLOAT,
                spectral_bandwidth FLOAT,
                spectral_centroid FLOAT,
                spectral_contrast FLOAT,
                mfcc FLOAT,
                chroma FLOAT,
                popularity INTEGER
            )
        """)
        conn.commit()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS retrain_db (
                spectral_rolloff FLOAT,
                zero_crossing_rate FLOAT,
                spectral_bandwidth FLOAT,
                spectral_centroid FLOAT,
                spectral_contrast FLOAT,
                mfcc FLOAT,
                chroma FLOAT,
                popularity INTEGER
            )
        """)
        conn.commit()

        print("Database setup complete.")
        return conn, cur  # Return the connection and cursor for further use
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return


def store_feedback_db(feedback, features):
    cur, conn = connect_to_db()
    
    # Get existing ratings for both tables
    existing_rating_feedback = check_existing_rating(features, cur, "song_feedback")
    print("existing rating for feedback table  : ",existing_rating_feedback)

    existing_rating_retrain = check_existing_rating(features, cur, "retrain_db")
    print("existing rating for retrain table : ",existing_rating_retrain)

    def update_or_insert(table_name, existing_rating, features, feedback):
        if existing_rating is not None:
            # Update the user rating for the existing record in the specified table
            update_query = f"""
            UPDATE {table_name}
            SET popularity = %s
            WHERE 
                spectral_rolloff = %s AND zero_crossing_rate = %s AND spectral_bandwidth = %s AND spectral_centroid = %s AND
                spectral_contrast = %s AND mfcc = %s AND chroma = %s
            """
            cur.execute(update_query, [feedback] + list(features.values[0]))  # Convert DataFrame row to list
        else:
            # Insert a new record if no existing features match in the specified table
            insert_query = f"""
            INSERT INTO {table_name} (spectral_rolloff, zero_crossing_rate, spectral_bandwidth, spectral_centroid, spectral_contrast, mfcc, chroma, popularity)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            cur.execute(insert_query, list(features.values[0]) + [feedback])  # Convert DataFrame row to list

    # Update song_feedback table
    update_or_insert("song_feedback", existing_rating_feedback,features,feedback)
    # Update retrain_db table
    update_or_insert("retrain_db", existing_rating_retrain,features,feedback)
    #check_and_retrain(cur,conn)
    conn.commit()

    cur.close()
    conn.close()
    
    return


def store_feedback(features1, feedback1):
    store_feedback_db(feedback1, features1)
    return


def get_big_data_as_dataframe():
    cur, conn = connect_to_db()

    query = "SELECT * FROM big_data;"
    df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    return df


# Function to concat retrain_db into big_data table and truncating the retrain_db
def concat_tables_to_retrain():
    RETRAIN_COUNT = 50
    
    columns = [
        'spectral_rolloff', 'zero_crossing_rate', 'spectral_bandwidth', 'spectral_centroid',
        'spectral_contrast', 'mfcc', 'chroma', 'popularity'
    ]
    
    is_concated = False
    
    try:
        cur, conn = connect_to_db()
        
        # Check if retrain_db has exactly 10 rows
        cur.execute("SELECT COUNT(*) FROM retrain_db;")
        count = cur.fetchone()[0]
        if count < RETRAIN_COUNT:
            print("The retrain_db does not have enough rows.")
            return

        # Construct the INSERT INTO ... SELECT query
        insert_query = "INSERT INTO big_data ({}) SELECT {} FROM retrain_db;".format(
            ", ".join(columns),
            ", ".join(columns)
        )

        cur.execute(insert_query)

        cur.execute("TRUNCATE TABLE retrain_db;")
        conn.commit()
        is_concated = True
        print("Tables concated successfully")
    except Exception as e:
        print("Error:", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        
        return is_concated


def connect_to_db():
    dbname = os.getenv("dbname")
    user = os.getenv("user")
    password = os.getenv("password")
    host = os.getenv("host")
    port = os.getenv("port")

    # Connect to the PostgreSQL database
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        cur = conn.cursor()
    except:
        print("Could not connect to DB.")
        return None,None

    return cur, conn

# create_databases()