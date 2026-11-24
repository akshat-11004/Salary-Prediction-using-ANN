# Salary Prediction using Artificial Neural Network (ANN)

## Topic
**Deep Learning | Artificial Neural Networks | Regression | Salary Prediction**

## Short Description
This project uses an **Artificial Neural Network (ANN)** to predict a bank customer's estimated annual salary from demographic and banking information, such as age, geography, credit score, account balance, and tenure. The model is trained with TensorFlow/Keras on the `Churn_Modelling.csv` dataset. A **Streamlit** web application allows users to enter customer details and view predicted salary values.

## Files and Folders

| File / Folder | Description |
| --- | --- |
| `Churn_Modelling.csv` | Dataset used to train the salary prediction model. |
| `salary_prediction.ipynb` | Jupyter notebook for preprocessing, training, and evaluating the ANN regression model. |
| `regression_app.py` | Streamlit interface for entering customer details and predicting salary. |
| `regression_model.h5` | Saved trained ANN regression model. |
| `label_encoder_gender.pkl` | Saved label encoder for the gender feature. |
| `onehot_encoder_geo.pkl` | Saved one-hot encoder for the geography feature. |
| `scaler.pkl` | Saved feature scaler used during training and prediction. |
| `requirements.txt` | Python dependencies required for the project. |
| `.devcontainer/devcontainer.json` | Development container configuration. |
| `.github/workflows/jira-ci.yml` | GitHub Actions workflow for Jira ticket checks, code validation, and pull-request feedback. |
| `scripts/validate_jira.py` | Script to validate Jira ticket references and ticket details. |
| `LICENSE` | Repository license information. |
| `README.md` | Project documentation. |

## Run the Application

```bash
git clone https://github.com/akshat-11004/Salary-Prediction-using-ANN.git
cd Salary-Prediction-using-ANN
pip install -r requirements.txt
streamlit run regression_app.py
```

## Technologies Used
Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Streamlit, and Jupyter Notebook.

> **Note:** The current Jira CI workflow references `calculator.py` and `tests/`, which are not present in the repository. Those checks will need to be updated or the referenced files added before the workflow can pass.
