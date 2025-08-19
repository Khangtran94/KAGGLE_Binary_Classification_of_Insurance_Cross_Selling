# EDA and Classification of Insurance Cross-Selling

## Overview
This project focuses on performing Exploratory Data Analysis (EDA) and building classification models to predict whether customers will respond to an insurance cross-selling offer. Using a dataset from Kaggle's "Binary Classification of Insurance Cross Selling" competition, the project leverages Python to clean data, visualize key patterns, and apply machine learning techniques to classify customer responses with an accuracy of approximately 88%. The primary goal is to identify factors influencing customer decisions and build predictive models for insurance cross-selling.

## Dataset
The dataset is sourced from Kaggle's Binary Classification of Insurance Cross Selling competition. It includes:
- **Customer Information**: Features such as age, gender, driving license status, region code, and more.
- **Insurance Details**: Metrics like premium amounts, vehicle age, vehicle damage history, and previous insurance status.
- **Target Variable**: A binary column (`Response`) indicating whether a customer is interested (1) or not (0) in the cross-selling offer.

The dataset is used for EDA and training classification models.

## Features
- **Data Cleaning**: Handling missing values, encoding categorical variables, and preprocessing data for analysis using `pandas`.
- **Exploratory Data Analysis (EDA)**: Visualizing relationships between features (e.g., age, premium, vehicle damage) and the target variable using `matplotlib` and `seaborn`.
- **Data Visualization**: Creating plots such as histograms, bar charts, and correlation heatmaps to identify patterns and feature importance.
- **Feature Engineering**: Transforming and selecting relevant features to improve model performance.
- **Machine Learning Models**:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Gradient Boosting (e.g., XGBoost or LightGBM, if used)
- **Model Evaluation**: Assessing model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC, achieving up to 88% accuracy.
- **Hyperparameter Tuning**: Optimizing model parameters to enhance predictive performance.

## Requirements
To run the notebook, you need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost` (optional, if gradient boosting is used)

Install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Usage
1. **Access the Notebook**:
   - The notebook is hosted on Kaggle: [EDA & Classification Insurance](https://www.kaggle.com/code/khangtran94vn/eda-classification-insurance).
   - You can run it directly on Kaggle or download it to your local machine.

2. **Download the Dataset**:
   - Obtain the dataset from Kaggle's Binary Classification of Insurance Cross Selling competition.
   - Ensure the dataset is placed in the correct directory if running locally.

3. **Run the Notebook**:
   - Open `eda-classification-insurance.ipynb` in a Kaggle kernel, Jupyter Notebook, or a compatible environment.
   - Follow the steps to preprocess data, perform EDA, train models, and evaluate results.

4. **View Results**:
   - The notebook includes visualizations (e.g., feature distributions, correlation matrices) and model performance metrics (e.g., accuracy, confusion matrix).

## Results
The project delivers the following insights:
- **EDA Insights**: Identification of key factors influencing customer responses, such as age, vehicle damage history, and premium amounts.
- **Model Performance**: Classification models achieve up to 88% accuracy, with detailed metrics like precision, recall, and ROC-AUC provided.
- **Visualizations**: Plots highlight feature distributions, correlations, and predictive patterns, aiding in understanding customer behavior.

## Contributing
Contributions are welcome! To contribute:
1. Fork the Kaggle notebook or repository (if hosted externally).
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Share your updates via a Kaggle comment or pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details (if applicable).

## Acknowledgments
- Kaggle for hosting the dataset and competition.
- The open-source community for tools like `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`.

## Citation
This project is based on the Kaggle notebook: [EDA & Classification Insurance](https://www.kaggle.com/code/khangtran94vn/eda-classification-insurance).[](https://www.kaggle.com/code/ivanergiev01/car-insurance-classification-models-eda-88)