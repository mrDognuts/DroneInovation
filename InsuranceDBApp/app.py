from flask import Flask, render_template, jsonify
import sqlite3
import subprocess
import os

app = Flask(__name__)

DATABASE_PATH = 'InsuranceDB.db'
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create tblClients table if it does not exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tblClients (
        ClientID INT PRIMARY KEY,
        First_Name VARCHAR NOT NULL,
        Last_Name VARCHAR NOT NULL,
        DOB DATE NOT NULL,
        Gender CHAR(1) NOT NULL,
        Email VARCHAR NOT NULL
    );
    ''')
    
    # Insert some data into tblClients if it's empty
    cursor.execute("SELECT COUNT(*) FROM tblClients")
    count = cursor.fetchone()[0]
    if count == 0:  # Insert if the table is empty
        cursor.executemany('''
        INSERT INTO tblClients (ClientID, First_Name, Last_Name, DOB, Gender, Email)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', [
            (625340, 'Alice', 'Smith', '1990-04-23', 'F', 'alice.smith@example.com'),
            (625341, 'Bob', 'Johnson', '1985-11-12', 'M', 'bob.johnson@example.com'),
            (625342, 'Carol', 'Williams', '1992-07-08', 'F', 'carol.williams@example.com'),
            (625343, 'David', 'Brown', '1988-02-19', 'M', 'david.brown@example.com'),
            (625344, 'Eve', 'Davis', '1995-09-30', 'F', 'eve.davis@example.com')
        ])
    
    conn.commit()
    conn.close()

# Initialize the database
init_db()

EXTERNAL_SCRIPT_PATH = os.path.abspath(r"C:\Users\tumel\OneDrive\Desktop\DroneInovation\Depreciated\Drone\main.py")




@app.route("/")
def home():
    return render_template("dashboard.html")  # Serve the main HTML page

@app.route('/login', methods=['POST'])
def target_page():
    # You can add logic here to check login details or perform other actions
    return render_template('login.html')  # Render the target page after login

@app.route('/drones')
def drones():
    return render_template('drones.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/clients')
def clients():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tblClients")
    clients_data = cursor.fetchall()
    conn.close()

    return render_template('clients.html',clients=clients_data)

@app.route('/run-script')
def run_script():
    try:
        # Run the external Python script
        subprocess.run(['python', EXTERNAL_SCRIPT_PATH], check=True)
        return jsonify({"status": "Script executed successfully"})
    except subprocess.CalledProcessError as e:
        # Return an error message in case of failure
        return jsonify({"status": "An error occurred", "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
