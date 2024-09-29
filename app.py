# from flask import Flask, request, render_template
# import numpy as np
# import pickle

# # Importing the model
# model = pickle.load(open('model.pkl', 'rb'))
# # If crop.pkl is required later, you can load it here as needed.
# # crop = pickle.load(open('crop.pkl', 'rb'))

# # Creating the Flask app
# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=['POST'])
# def predict():
#     try:
#         # Change the request keys to match your HTML input names
#         N = float(request.form['N_SOIL'])        # Changed from 'Nitrogen' to 'N_SOIL'
#         P = float(request.form['P_SOIL'])        # Changed from 'Phosphorus' to 'P_SOIL'
#         K = float(request.form['K_SOIL'])        # Changed from 'Potassium' to 'K_SOIL'
#         temp = float(request.form['Temperature'])
#         humidity = float(request.form['Humidity'])
#         ph = float(request.form['Ph'])
#         rainfall = float(request.form['Rainfall'])

#         # Prepare feature list for prediction
#         feature_list = [N, P, K, temp, humidity, ph, rainfall]
#         single_pred = np.array(feature_list).reshape(1, -1)  # Ensure the data type is float

#         # Make the prediction
#         prediction = model.predict(single_pred)

#         crop_dict = {
#             1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
#             6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
#             11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
#             15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
#             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
#         }

#         # Determine the crop based on prediction
#         crop = crop_dict.get(prediction[0], "Sorry, we could not determine the best crop to be cultivated with the provided data.")
#         result = "{} is the best crop to be cultivated right there.".format(crop) if prediction[0] in crop_dict else crop

#     except ValueError:
#         result = "Please enter valid numeric values for all fields."
#     except Exception as e:
#         result = "An error occurred: {}".format(str(e))
    
#     return render_template('index.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, render_template
# import numpy as np
# import pickle

# # Importing the model
# model = pickle.load(open('model.pkl', 'rb'))

# # Creating the Flask app
# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=['POST'])
# def predict():
#     # Change the request keys to match your HTML input names
#     N = request.form['N_SOIL']        # Nitrogen
#     P = request.form['P_SOIL']        # Phosphorus
#     K = request.form['K_SOIL']        # Potassium
#     temp = request.form['Temperature'] # Temperature
#     humidity = request.form['Humidity'] # Humidity
#     ph = request.form['Ph']            # pH
#     rainfall = request.form['Rainfall'] # Rainfall
#     state = request.form['STATE']      # State
#     crop_price = request.form['CROP_PRICE'] # Crop Price

#     # Prepare feature list for prediction
#     feature_list = [N, P, K, temp, humidity, ph, rainfall, state, crop_price]
#     single_pred = np.array(feature_list, dtype=float).reshape(1, -1)  # Ensure the data type is float

#     # Make the prediction without scaling (using the model directly)
#     prediction = model.predict(single_pred)

#     crop_dict = {
#         1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
#         6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
#         11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
#         15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
#         19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
#     }

#     # Determine the crop based on prediction
#     if prediction[0] in crop_dict:
#         crop = crop_dict[prediction[0]]
#         result = "{} is the best crop to be cultivated right there.".format(crop)
#     else:
#         result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    
#     return render_template('index.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your model (assuming it's saved as 'model.pkl')
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Welcome to the Crop Recommendation API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Extract parameters from the request
    N_SOIL = float(data['N_SOIL'])
    P_SOIL = float(data['P_SOIL'])
    K_SOIL = float(data['K_SOIL'])
    Temperature = float(data['Temperature'])
    Humidity = float(data['Humidity'])
    Ph = float(data['Ph'])
    Rainfall = float(data['Rainfall'])
    State = data['State']  # New field for State
    CropPrice = float(data['CropPrice'])  # New field for Crop Price

    # Prepare the input data for prediction
    input_data = np.array([[N_SOIL, P_SOIL, K_SOIL, Temperature, Humidity, Ph, Rainfall, CropPrice]])
    
    # Predict the crop using your model
    prediction = model.predict(input_data)

    # Assuming the model outputs a numerical value representing the crop
    crop_recommendation = str(prediction[0])  # Convert prediction to string if needed

    # Optional: Additional logic based on State can be implemented here
    if State:  # Check if State is provided
        # You can add any additional filtering or processing based on State here
        pass

    return jsonify({'crop_recommendation': crop_recommendation})

if __name__ == '__main__':
    app.run(debug=True)











# from flask import Flask, request, jsonify
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load your model (assuming it's saved as 'model.pkl')
# with open('model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# @app.route('/')
# def home():
#     return "Welcome to the Crop Recommendation API!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get JSON data from the request
#     data = request.get_json()

#     # Extract parameters from the request
#     N_SOIL = float(data['N_SOIL'])
#     P_SOIL = float(data['P_SOIL'])
#     K_SOIL = float(data['K_SOIL'])
#     Temperature = float(data['Temperature'])
#     Humidity = float(data['Humidity'])
#     Ph = float(data['Ph'])
#     Rainfall = float(data['Rainfall'])
#     State = data['State']  # New field for State
#     CropPrice = float(data['CropPrice'])  # New field for Crop Price

#     # Prepare the input data for prediction
#     input_data = np.array([[N_SOIL, P_SOIL, K_SOIL, Temperature, Humidity, Ph, Rainfall, CropPrice]])
    
#     # Predict the crop using your model
#     prediction = model.predict(input_data)

#     # Assuming the model outputs a numerical value representing the crop
#     crop_recommendation = str(prediction[0])  # Convert prediction to string if needed

#     # Optional: Additional logic based on State can be implemented here
#     if State:  # Check if State is provided
#         # You can add any additional filtering or processing based on State here
#         pass

#     return jsonify({'crop_recommendation': crop_recommendation})

# if __name__ == '__main__':
#     app.run(debug=True)
