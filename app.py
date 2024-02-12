import json
from flask import Flask, request, jsonify
from prediction_pipeline import pred_and_plot
import torch
from model import TinyVGG

device  = "cuda" if torch.cuda.is_available() else "cpu"

state_dict = torch.load("/Users/shoumik20/Shoumik_work/Repos/Pothole_Hackathon/final_model.pth", map_location=torch.device('cpu'))

input_shape = 3
hidden_units = 10
output_shape = 2  
model = TinyVGG(input_shape, hidden_units, output_shape)

model.load_state_dict(state_dict)

model.eval()
application=Flask(__name__)

app = application

@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data)
    custom_image_path = data['image']
    print(custom_image_path)
    pred=pred_and_plot(model=model,image_path = custom_image_path)
    return jsonify({"predictions": pred})

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True,port=5001)
