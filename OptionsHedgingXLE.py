# Initial imports
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import math

# Fetch data function
def fetch_data():
    # Energy_Sector_ETF, Crude_Oil, Natural_Gas, Coal and Currencies of Energy-importing countries to USD.
    # Define the list of tickers and currency pairs.
    currencies = ['CAD', 'MXN', 'BRL', 'SAR', 'IQD', 'COP']
    tickers = ['XLE', 'CL=F', 'NG=F', 'MTF=F'] + [f'USD{currency}=X' for currency in currencies]
    
    # Fetch close prices for tickers and currency pairs for the last 365 days and drop NaN
    energy = yf.download(tickers, period="1y")['Close'].dropna()
    
    # Return DataFrame
    return energy

# Linear Regression function
def linear_regression_prediction(energy, prediction_days):
    # Create list to store predicted prices and DataFrame to store predicted prices
    predicted_prices = []
    predicted_prices_df = pd.DataFrame(columns=['Date', 'Predicted Price'])
    
    # Extract last date from original DataFrame
    last_date = energy.index[-1]
    
    # Loop
    for i in range(1, prediction_days + 1):
        
        # Define Features and Target
        X = energy.drop(columns='XLE').shift(i).dropna()
        y = energy['XLE'][i:]
        
        # Initialize the StandardScaler, Fit and Transform the training data
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        # Create the selected model and Fit the model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Get today's features (all except XLE), Scale today's features
        today_features = energy.drop(columns='XLE').iloc[-1]
        today_features_scaled = scaler.transform([today_features])
        
        # Predict and Append predicted price to the list
        predicted_price = model.predict(today_features_scaled)[0]
        predicted_prices.append(predicted_price)
        
    # Generate dates for the predicted prices and Assign the predicted prices and new dates to the DataFrame 
    new_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
    predicted_prices_df['Date'] = new_dates
    predicted_prices_df['Predicted Price'] = predicted_prices
    
    # Return predicted DataFrame
    return predicted_prices_df

# Random Forest function
def random_forest_prediction(energy, prediction_days):
    
    # Create list to store predicted prices and DataFrame to store predicted prices
    predicted_prices = []
    predicted_prices_df = pd.DataFrame(columns=['Date', 'Predicted Price'])
    
    # Extract last date from original DataFrame
    last_date = energy.index[-1]
    
    # Loop 
    for i in range(1, prediction_days + 1):
        
        # Define Features and Target
        X = energy.drop(columns='XLE').shift(i).dropna()
        y = energy['XLE'][i:]
        
        # Initialize the StandardScaler, Fit and Transform the training data
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        # Create the selected model and Fit the model
        model = RandomForestRegressor()
        model.fit(X_scaled, y)
        
        # Get today's features (all except XLE), Scale today's features
        today_features = energy.drop(columns='XLE').iloc[-1]
        today_features_scaled = scaler.transform([today_features])
        
        # Predict and Append predicted price to the list
        predicted_price = model.predict(today_features_scaled)[0]
        predicted_prices.append(predicted_price)
        
    # Generate dates for the predicted prices and Assign the predicted prices and new dates to the DataFrame
    new_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
    predicted_prices_df['Date'] = new_dates
    predicted_prices_df['Predicted Price'] = predicted_prices
    
    # Return predicted DataFrame
    return predicted_prices_df

# Neural Network function
def neural_network_prediction(energy, prediction_days):
    
    # Create list to store predicted prices and DataFrame to store predicted prices
    predicted_prices = []
    predicted_prices_df = pd.DataFrame(columns=['Date', 'Predicted Price'])
    
    # Extract last date from original DataFrame
    last_date = energy.index[-1]
    
    # Loop
    for i in range(1, prediction_days + 1):
        
        # Define Features and Target
        X = energy.drop(columns='XLE').shift(i).dropna()
        y = energy['XLE'][i:]
        
        # Initialize the StandardScaler, Fit and Transform the training data
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        # Create the selected model and Fit the model
        model = MLPRegressor(hidden_layer_sizes=(100, 50),
                             activation='relu',
                             solver='adam',
                             random_state=1)
        model.fit(X_scaled, y)
        
        # Get today's features (all except XLE), Scale today's features
        today_features = energy.drop(columns='XLE').iloc[-1]
        today_features_scaled = scaler.transform([today_features])
        
        # Predict and Append predicted price to the list
        predicted_price = model.predict(today_features_scaled)[0]
        predicted_prices.append(predicted_price)
        
    # Generate dates for the predicted prices and Assign the predicted prices and new dates to the DataFrame
    new_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
    predicted_prices_df['Date'] = new_dates
    predicted_prices_df['Predicted Price'] = predicted_prices
    
    # Return predicted DataFrame
    return predicted_prices_df

# Average of all functin
def average_of_all_predictions(energy, prediction_days):
     
    # Run all 3 predictive models and store the results in variables
    linear_pred = linear_regression_prediction(energy, prediction_days)['Predicted Price']
    rf_pred = random_forest_prediction(energy, prediction_days)['Predicted Price']
    nn_pred = neural_network_prediction(energy, prediction_days)['Predicted Price']
    
    # Calculate average of predictions
    avg_pred = (linear_pred + rf_pred + nn_pred) / 3
    
    # Create DataFrame to store the average predictions
    predicted_prices_df = pd.DataFrame(columns=['Date', 'Predicted Price'])
    
    # Extract last date from original DataFrame
    last_date = energy.index[-1]
    
    # Generate dates for the predicted prices and Assign the predicted prices and new dates to the DataFrame
    new_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
    predicted_prices_df['Date'] = new_dates
    predicted_prices_df['Predicted Price'] = avg_pred
    
    # Return predicted DataFrame
    return predicted_prices_df


# Streamlit app
def main():
    st.title("Energy Sector ETF Prediction and Risk Analysis")

    # Fetch data
    energy = fetch_data()

    # Sidebar for user input
    st.sidebar.header("User Input")
    shares = st.sidebar.number_input("Enter the number of shares:", value=100, step=100)
    
    # Run prediction based on user's choice
    model_choice = st.sidebar.selectbox("Choose a prediction model", ["Linear Regression", 
                                                                      "Random Forest", 
                                                                      "Neural Network", 
                                                                      "Average of All Predictions"])
    # Slider for predictions
    prediction_days = st.sidebar.slider("Number of days for prediction", 1, 365, 90)

    #predicted_prices_df = pd.DataFrame(columns=['Date', 'Predicted Price'])
    
    # Create if function to select models to predict 
    if model_choice == "Linear Regression":
        predicted_prices_df = linear_regression_prediction(energy, prediction_days)
    elif model_choice == "Random Forest":
        predicted_prices_df = random_forest_prediction(energy, prediction_days)
    elif model_choice == "Neural Network":
        predicted_prices_df = neural_network_prediction(energy, prediction_days)
    elif model_choice == "Average of All Predictions":
        predicted_prices_df = average_of_all_predictions(energy, prediction_days)
        
    # Calculate current price and portfolio value
    current_day = energy.index[-1].strftime("%Y-%m-%d")
    current_price = energy.loc[current_day, 'XLE']
    portfolio_value = shares * current_price
    
    # Display current price and predicted prices
    st.subheader("Portfolio Information")
    st.write(f"The current price of the asset is: ${current_price:.2f}")
    st.write(f"Predicted price in {prediction_days} days is: ${round(predicted_prices_df['Predicted Price'].iloc[-1], 2)}")
        
    # Plot predicted prices
    st.subheader(f"Predicted Prices ({model_choice} Model)")
    st.line_chart(predicted_prices_df.set_index('Date'))
    
    
    # Options section
    expiration_target = st.sidebar.date_input("Enter the expiration target date:", format = 'YYYY-MM-DD')
    strike_target = st.sidebar.slider("Enter percentage to hedge", .05, .1, .07)

    # Stock Price
    S = energy['XLE'][249]
    # Time to Expiration
    T = prediction_days
    # Risk free return
    r = .045
    # Strike price for hedging
    K = math.floor(S*(1-strike_target))
    #volatility
    sigma = energy['XLE'][150:249].std()

    # Calculate Black Scholes target option price
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    price = (K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    # Calculate portfolio values using the user-input shares
    portfolio = shares * S
    
    # EMV Assessment
    predictive_power = 0.85
    predicted_change = portfolio - (predicted_prices_df['Predicted Price'].iloc[-1] * shares)
    EMV = predicted_change * predictive_power
    opt_price = price * shares
    hedge = EMV - opt_price

   
    last_price = 0
    
    # Options Section
    st.subheader("Options Analysis")
    st.write(f"Fair market price for the contract should be close to ${round(price, 2)} per contract")
    st.write(f"Target Hedge Strike Price is ${K}")
    
    if st.sidebar.button("Display Options Chain"):
        xle = yf.Ticker("XLE")
        opt = xle.option_chain(date= (f"{expiration_target}"))
        chain = opt.puts
        st.write(chain[10:30])
        last_price = chain[chain['strike'] == K]['lastPrice']
        
        
    # Display portfolio analysis
    st.subheader("Portfolio Analysis")
    st.write(f"The current portfolio value is: ${portfolio_value:.2f}")
    st.write(f"Expected Change in Portfolio Value: ${-predicted_change:.2f}")
    st.write(f"Expected Monetary Value (EMV): ${-EMV:.2f}")
    st.write(f"Option Price: ${float(last_price)*100}")
    st.write(f"Hedge: ${hedge:.2f}")


        

if __name__ == "__main__":
    main()
