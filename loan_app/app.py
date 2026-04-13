import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History',
    'Property_Area_Semiurban', 'Property_Area_Urban', 'Debt_Income_Ratio'
]

def encode_input(form):
    gender         = 1 if form['gender'] == 'Male' else 0
    married        = 1 if form['married'] == 'Yes' else 0
    dependents     = int(form['dependents'].replace('3+', '3'))
    education      = 1 if form['education'] == 'Not Graduate' else 0
    self_employed  = 1 if form['self_employed'] == 'Yes' else 0
    applicant_inc  = float(form['applicant_income'])
    coapplicant_inc= float(form['coapplicant_income'])
    loan_amount    = float(form['loan_amount'])
    loan_term      = float(form['loan_amount_term'])
    credit_history = float(form['credit_history'])
    area           = form['property_area']
    area_rural      = 1 if area == 'Rural' else 0
    area_semi      = 1 if area == 'Semiurban' else 0
    area_urban     = 1 if area == 'Urban' else 0
    dept_income_ratio = loan_amount / (applicant_inc + coapplicant_inc + 1)

    return np.array([[
        gender, married, dependents, education, self_employed,
        applicant_inc, coapplicant_inc, loan_amount, loan_term,
        credit_history, area_rural, area_semi, area_urban, dept_income_ratio 
    ]])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    inputs = None
    if request.method == 'POST':
        X = encode_input(request.form)
        pred = model.predict(X)[0]
        result = 'Aprovado' if pred == 1 else 'Reprovado'
        inputs = dict(request.form)
    return render_template('index.html', result=result, inputs=inputs)

if __name__ == '__main__':
    app.run(debug=True)