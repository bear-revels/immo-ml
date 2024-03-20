# ImmoWeb Property Price Prediction Model

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## 📒 Description

The ImmoWeb Property Price Prediction Model is a Python application designed to predict property prices based on data scraped from ImmoWeb, Belgium's leading real estate website. The application consists of a set of Python scripts that preprocess the scraped data and train three different regression models: Linear Regression, Gradient Boosted Decision Tree, and Random Forest. These models allow users to make accurate predictions on housing prices, leveraging the most up-to-date property listing data.

![Property Prices Illustration](https://media.istockphoto.com/id/932743856/vector/property-prices-illustration.jpg?s=612x612&w=0&k=20&c=MpCykgUFuTxQje0JksjDApv9u5ywb5nkJE0brZ-4GiA=)

## 🧩 Data Assumptions

It's important to understand the preprocessing steps taken and of course you're welcome to look through these steps in the utils.py file. For a quick glance, the manipulations and assumptions are listed herein:
1. dropped rows if null in `Price`, `LivingArea`, `Latitude`, or `Longitude`
2. removed duplicates by `ID` and all rows concatenates less `ID`
3. replaced null for 0 in binanary features `Furnished`, `Fireplace`, `TerraceArea`, `GardenArea`, `SwimmingPool`, `BidStylePricing`, `ViewCount`, `bookmarkCount`
4. filter to `SaleType` == `residential_sale` & `BidStylePricing` == 0
5. corrected text formatting
6. type casting columns to int/float where possible
7. adjusted `BedroomCount` + 1, -`EnergyConsumptionPerSqm` to 0, and `ConstructionYear` > current year +10 to null
8. removed outliers in `PricePerSqm` and `SqmPerRoom` when grouped by `PostalCode` and `PropertySubType`
9. normalized the severely right-skewed distribution of `Price`, `LivingArea`, `GardenArea`, and `BedroomCount` using log10

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
4. Select a model (Linear Regression, Gradient Boosted Decision Tree, or Random Forest) and specify whether to refresh the data.
5. Upon completion, the model evaluation metrics will be displayed in the terminal, and the trained model will be saved to the models folder

## ⏱️ Timeline

The development of this project took 5 days for completion.

## 📌 Team Members

This project was completed as part of the AI Bootcamp at BeCode.org by Team Python Pricers:

1. [Bear Revels](https://www.linkedin.com/in/bear-revels/)
2. [Caroline Van Hoeke](https://www.linkedin.com/in/caroline-van-hoeke-8a3b87123/)
3. [Geraldine Nadela](https://www.linkedin.com/in/geraldine-nadela-60827a11)
4. [Viktor Cosaert](https://www.linkedin.com/in/viktor-cosaert/)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
