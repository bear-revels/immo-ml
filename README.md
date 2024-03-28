# ImmoWeb Property Price Prediction Model

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## 📒 Description

The ImmoWeb Property Price Prediction Model is a Python application designed to predict property prices based on data scraped from ImmoWeb, Belgium's leading real estate website. The application consists of a set of Python scripts that preprocess the scraped data and train three different regression models: Linear Regression, LightGBM, and Random Forest. These models allow users to make accurate predictions on housing prices, leveraging the most up-to-date property listing data.

![Property Prices Illustration](https://media.istockphoto.com/id/932743856/vector/property-prices-illustration.jpg?s=612x612&w=0&k=20&c=MpCykgUFuTxQje0JksjDApv9u5ywb5nkJE0brZ-4GiA=)

## 📦 Repo Structure

```
.
├── data/
│ └── external_data/
│ └── performance_png/
│ └── raw_data.csv
├── models/
├── src/
│ ├── main.py
│ ├── models.py
│ └── utils.py
├── MODELSCARD.md
├── README.md
└── requirements.txt
```

## 🎮 Usage

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the `main.py` file to execute the model training process.
4. Select a model (Linear Regression, LightGBM or Random Forest) and specify whether to refresh the data.
5. Upon completion, the model evaluation metrics will be displayed in the terminal, and the trained model will be saved to the models folder

## ⏱️ Timeline

The development of this project took 5 days for completion.

## 📌 Team Members

This project was completed solo as part of the AI Bootcamp at BeCode.org by [Bear Revels](https://www.linkedin.com/in/bear-revels/).
