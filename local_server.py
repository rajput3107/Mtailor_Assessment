# local_server.py
from flask import Flask, request, jsonify
import json
import base64
from PIL import Image
import io

# Import your main module
from main import run

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <html>
        <body>
            <h1>Image Classifier API</h1>
            <h2>Available Endpoints:</h2>
            <ul>
                <li>GET /health - Health check</li>
                <li>POST /classify - Classify an image</li>
                <li>GET /test - Test page with image upload</li>
            </ul>
        </body>
    </html>
    '''

@app.route('/health', methods=['GET'])
def health():
    result = run()  # Call with no parameters for health check
    return jsonify(result)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json() or {}
    
    # Extract image from request
    image_base64 = data.get('image')
    
    if not image_base64:
        return jsonify({"error": "No image provided", "status_code": 400}), 400
    
    # Call the run function with image
    result = run(image=image_base64)
    return jsonify(result)

@app.route('/test', methods=['GET'])
def test_page():
    return '''
    <html>
        <head>
            <title>Image Classifier Test</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .result { margin-top: 20px; padding: 10px; background: #f0f0f0; }
                img { max-width: 300px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Image Classifier Test</h1>
            <input type="file" id="imageFile" accept="image/*">
            <button onclick="classifyImage()">Classify</button>
            <div id="preview"></div>
            <div id="result"></div>
            
            <script>
                document.getElementById('imageFile').addEventListener('change', function(e) {
                    const file = e.target.files[0];
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            document.getElementById('preview').innerHTML = '<img src="' + e.target.result + '">';
                        }
                        reader.readAsDataURL(file);
                    }
                });
                
                function classifyImage() {
                    const fileInput = document.getElementById('imageFile');
                    const file = fileInput.files[0];
                    
                    if (!file) {
                        alert('Please select an image first');
                        return;
                    }
                    
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const base64 = e.target.result.split(',')[1];
                        
                        fetch('/classify', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({image: base64})
                        })
                        .then(response => response.json())
                        .then(data => {
                            let html = '<h3>Results:</h3>';
                            if (data.predictions) {
                                html += '<ol>';
                                data.predictions.forEach(pred => {
                                    html += '<li>' + pred.class + ': ' + (pred.confidence * 100).toFixed(2) + '%</li>';
                                });
                                html += '</ol>';
                            } else {
                                html += '<p>Error: ' + (data.error || 'Unknown error') + '</p>';
                            }
                            document.getElementById('result').innerHTML = html;
                        })
                        .catch(error => {
                            document.getElementById('result').innerHTML = '<p>Error: ' + error + '</p>';
                        });
                    };
                    reader.readAsDataURL(file);
                }
            </script>
        </body>
    </html>
    '''

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Image Classifier Local Server")
    print("=" * 60)
    print("Available endpoints:")
    print("  - Home:     http://127.0.0.1:5000/")
    print("  - Health:   http://127.0.0.1:5000/health")
    print("  - Classify: http://127.0.0.1:5000/classify (POST)")
    print("  - Test UI:  http://127.0.0.1:5000/test")
    print("=" * 60)
    app.run(debug=True, port=5000)
    