from flask import Flask, render_template

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)
