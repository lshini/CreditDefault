#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
app = Flask(__name__)


# In[2]:


from flask import request, render_template
import joblib
import numpy as np
from scipy import stats
import pandas as pd

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        df = pd.read_csv("Credit Card Default II (balance).csv")

        df_income = list(df.iloc[:, 0])
        df_age = list(df.iloc[:, 1])
        df_loan = list(df.iloc[:, 2])
        Income = request.form.get("Income")
        Age = request.form.get("Age")
        Loan = request.form.get("Loan")
        print(Income, Age, Loan)
        
        df_income.append(float(Income))
        df_age.append(float(Age))
        df_loan.append(float(Loan))
        zincome = stats.zscore(df_income)
        zage = stats.zscore(df_age)
        zloan = stats.zscore(df_loan)
        
        model =  joblib.load("NN_CreditDefault")
        pred = model.predict([[float(zincome[-1]), float(zage[-1]), float(zloan[-1])]])
        pred = pred[0]
        s = "The predicted credit default risk is: " + str(pred)
        return(render_template("index.html", result=s))
    else:
        return(render_template("index.html", result=""))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




