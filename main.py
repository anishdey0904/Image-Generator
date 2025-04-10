# In[3]:

import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
# import base64git 
import io
import os
from PIL import Image
from image_generator import get_image
from datetime import datetime


# In[12]:

app = Flask(__name__)

UPLOAD_FOLDER = r'D:\Image\predict'
DOWNLOAD_FOLDER = r'D:\Image\output'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

@app.route('/generated-image')
def get_generated_image():
    file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'generated_image.png')
    if os.path.exists(file_path):
        return send_from_directory(app.config['DOWNLOAD_FOLDER'], 'generated_image.png')
    else:
        return jsonify({'error': 'Generated image not found'}), 404


@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        # Get the base64-encoded image from the request
        data = request.get_json()
        image_data = data['image']
        
        # Decode the base64 image
        image_data = image_data.split(",")[1]  # Remove base64 header
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Save the received image on the server
        received_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'received_image.png')
        img.save(received_image_path)
        print(f"Image saved successfully at {received_image_path}")
        
        # Process the image (your existing get_image function)
        get_image()
        
        # Return the URL to the generated image
        return jsonify({
            'message': 'Image received and processed successfully',
            'image_url': '/generated-image'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    print("Server running at: http://127.0.0.1:5000")
    app.run(debug=False)

# In[ ]:





# In[ ]:




