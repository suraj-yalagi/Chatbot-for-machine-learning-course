from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask server is working!"

if __name__ == "__main__":
    print("Starting test server on http://localhost:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
        print("Server started successfully")
    except Exception as e:
        print(f"Error starting server: {e}") 