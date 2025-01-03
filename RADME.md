# Insurance Premium Prediction using Regression

## Project Overview

This project focuses on predicting medical insurance charges based on various factors such as age, sex, BMI, number of children, smoking status, and region. The primary objective is to build a predictive model using regression techniques to estimate the insurance charges a person would have to pay. Several machine learning models, including support vector machines (SVM), XGBoost, CatBoost, and random forests, have been employed and tuned for optimal performance.

### Key Features of the Project:
- **Data Preprocessing:** Handles missing values, performs feature scaling, and applies encoding techniques for categorical data.
- **Multiple Models:** Trained multiple regression models and compared their performance using R² score.
- **Hyperparameter Tuning:** Applied hyperparameter optimization techniques such as Grid Search to improve model performance.
- **Streamlit App:** A simple web interface to input personal data manually or upload a CSV file to predict insurance charges.

---

## Dataset

The dataset used for this project contains various attributes related to health and demographics that influence the medical insurance charges. It includes the following columns:
- `age`: Age of the person.
- `sex`: Gender of the person (male or female).
- `bmi`: Body Mass Index (BMI) of the person.
- `children`: Number of children/dependents covered by the insurance.
- `smoker`: Whether the person is a smoker (yes or no).
- `region`: The region of the person (southeast, southwest, northwest, northeast).
- `charges`: The medical insurance charges, which is the target variable.

The dataset can be found and downloaded from the following Kaggle link:
[Medical Insurance Cost with Linear Regression - Dataset](https://www.kaggle.com/code/mariapushkareva/medical-insurance-cost-with-linear-regression/input)

---

## How to Run the Project

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/jadhavgaurav/Insurance-premium-prediction-using-Regression.git
    cd Insurance-premium-prediction-using-Regression
    ```

2. **Set up Virtual Environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

5. **Input Method**:
    - **Manual Input:** Enter personal information directly into the app to get the predicted insurance charges.
    - **Upload CSV:** Upload a CSV file containing multiple rows of personal information to get predictions for all entries.

---

## Contributing

Feel free to open an issue or submit a pull request if you'd like to contribute to this project. Suggestions for model improvements or other regression techniques are welcome!

---

## License

This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for more information.
