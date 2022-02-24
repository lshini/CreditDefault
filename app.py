#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
from flask import Flask
app = Flask(__name__)

from flask import request, render_template
from keras.models import load_model
@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        Age = request.form.get("Age")
        Income = request.form.get("Income")
        Loan = request.form.get("Loan")
        print(Age, Income, Loan)
        model =  joblib.load("NN_CreditDefault")
        pred = model.predict([[float(Income), float(Age), float(Loan)]])
        pred = pred[0]
        s = "The predicted credit default risk is: " + str(pred)
        return(render_template("index.html", result=s))
    else:
        return(render_template("index.html", result="2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




