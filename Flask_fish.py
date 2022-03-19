import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
pickle_in = open("fish_linearRegression.pkl", "rb")
regressor = pickle.load(pickle_in)

@app.route("/")
def home():
    # print("render home page successfully!")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [x for x in request.form.values()] # height=request.args.get("height") # width=request.args.get("width")
    # print(float_features[0:2])
    # print(type(float_features))
    final_features = [np.array(float_features[0:2])]
    prediction = regressor.predict(final_features)  # prediction=regressor.predict([[height, width]])
    return render_template("index.html", 
    prediction_text = f"The weight of this fish should be {str(np.round(prediction,2))}gram.")

if __name__=="__main__":
    app.debug = True
    app.run()