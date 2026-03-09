# DataVision 2025 - Air Quality Analysis

This project analyzes air quality data from major Indian cities and develops a machine learning model to predict the Air Quality Index (AQI). The project combines exploratory data analysis, visualization, and machine learning techniques to identify pollution trends and estimate AQI values.

## Live Application
[Open Streamlit App](https://datavision-2025-airquality-jorutpyl2r8ptkyc3glyjq.streamlit.app/)

## Project Overview

Air pollution is a major environmental and public health challenge in many urban regions. This project investigates air quality patterns across multiple cities and identifies key pollutants that influence AQI levels.

The project includes an interactive web application built using Streamlit that allows users to explore pollution data and generate AQI predictions using a trained machine learning model.

## Objectives

- Analyze air quality trends across multiple cities  
- Identify dominant pollutants influencing AQI  
- Study seasonal patterns in pollution levels  
- Build a machine learning model to predict AQI  
- Deploy the model through a web-based interface  

## Key Findings

Analysis of the dataset revealed several important observations:

- Ahmedabad recorded the highest AQI value of 397, followed by Delhi with 262.  
- Winter shows the highest average AQI level of 167, indicating more severe pollution during colder months.  
- PM2.5 was identified as the dominant pollutant contributing to AQI variations.  
- A Random Forest regression model achieved the best performance with an R² score of 0.83.

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  
Seaborn  
Streamlit  

## Machine Learning Model

The project uses a Random Forest regression model to predict AQI values based on pollutant concentration features.

Model Performance:

R² Score: 0.83

## Project Workflow

1. Data collection and preprocessing  
2. Exploratory data analysis and visualization  
3. Feature selection and preparation  
4. Training machine learning models  
5. Model evaluation and performance comparison  
6. Deployment using Streamlit

## Project Structure

DataVision-2025-AirQuality
│
├── app.py
├── model.pkl
├── dataset.csv
├── requirements.txt
└── README.md

## Running the Project Locally

Clone the repository

git clone https://github.com/your-username/repository-name.git

Navigate to the project directory

cd repository-name

Install required dependencies

pip install -r requirements.txt

Run the Streamlit application

streamlit run app.py

## Dataset

The dataset contains air quality measurements including pollutant concentrations such as PM2.5, PM10, NO2, SO2, CO, and O3 collected from multiple Indian cities.

## Future Improvements

- Improve model performance using additional machine learning techniques  
- Integrate real-time air quality monitoring data  
- Expand the dashboard with more detailed visualizations  

## Author

DataVision 2025 Project
