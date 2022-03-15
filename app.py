from flask import Flask,jsonify,request
import io
import json
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'jpg','jpeg'}

model = models.googlenet(pretrained=True)
imagenet_index = json.load(open('./imagenet_class_index.json'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def image_transformation(image_bytes):
    img_transformations = transforms.Compose([transforms.Resize(255),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.485,0.456,0.406),
                                                (0.229,0.224,0.225)
                                            )])
    uploaded_image = Image.open(io.BytesIO(image_bytes))
    return img_transformations(uploaded_image).unsqueeze(0)

def prediction(image_bytes):
    tensor = image_transformation(image_bytes)
    model_output = model.forward(tensor)
    predicted_value = model_output.max(1)[1]
    return imagenet_index[str(predicted_value.item())]
    
@app.route('/', methods=['GET','POST'])
def index():
    return ("Welcome to the Deploying a Pytorch model with Flask app")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            image_bytes = file.read()
            class_id, class_name = prediction(image_bytes)
            return jsonify({'class id': class_id, 'class name': class_name})
    return "Couldn't predict string"

if __name__ == '__main__':
    app.run()
