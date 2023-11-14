# Abalone Age Prediction


> Predicting the age of abalones based on physical measurements using PySpark and Gradient Descent variations.

---

## Table of Contents

- [Description](#description)
- [Objective](#objective)
- [Dataset Information](#dataset-information)
- [Methodology](#methodology)
- [Results and Impact](#results-and-impact)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Description

Welcome to the Abalone Age Prediction project! This repository hosts the code and documentation for a machine learning project focused on predicting the age of abalones using PySpark and various Gradient Descent variations.

---

## Objective

The primary objective of this project is to develop a robust predictive model capable of accurately estimating the age of abalones based on their physical measurements. The model's application extends to conservation, sustainable resource management, and marine biology research.

---

## Dataset Information

**Dataset Name:** Abalone Data Set  
**Source:** [Abalone Data Set on Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) by Rodolfo Mendes  
**Format:** Tabular (CSV)

---

## Methodology

### Data Cleansing and EDA

- Rigorous examination for missing values and outliers.
- Visualizations and summary statistics for data distribution and correlations.
- Identified relationships between features and age (Rings).

### Data Preprocessing

- Categorical variable encoding (One-Hot Encoding for 'Sex').
- Data scaling and encoding for machine learning models.
- Dataset split into training and testing sets.

### Gradient Descent Variations

Explored various gradient descent variations for optimization:

1. **Bold Driver Approach**
2. **Full Batch Gradient Descent**
3. **Stochastic Gradient Descent**
4. **Mini-Batch Gradient Descent**
5. **Adagrad Approach**
6. **RMSprop Approach**
7. **Adam Optimization**

Evaluated each method's performance using MSE, RMSE, and R-squared scores.

### Model Evaluation

- Comprehensive comparison table for optimization methods.
- Visualization of model predictions against actual values.
- Identified Mini-Batch Gradient Descent as the standout performer.

---

## Results and Impact

This project has achieved:

- Accurate age prediction of abalones.
- Robust model adaptability to variations.
- Comprehensive comparison of Gradient Descent optimization methods.

The impact of this work extends to marine conservation, resource management, and research in marine biology.

---

## Project Structure

The repository structure is organized as follows:

- `src/`: Contains the source code for data preprocessing, model training, and evaluation.
- `data/`: Stores the dataset used in the project.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development.
- `images/`: Image files related to the project.

Feel free to explore each directory for detailed information.

---

## Usage

To run the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/abalone-age-prediction.git`
2. Navigate to the project directory: `cd abalone-age-prediction`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the main script: `python src/main.py`

Make sure to replace "your-username" with your GitHub username.

---

## Contributing

If you'd like to contribute to this project, please follow the guidelines outlined in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

This project is licensed under the [MIT License](LICENSE).

