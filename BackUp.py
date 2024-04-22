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

# Prediction function
def predict_prices(energy, prediction_days, model_choice):
    # List to store predicted prices
    predicted_prices = []

    # Create DataFrame to store predicted prices
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

        # Create the selected model
        if model_choice == 'Linear Regression':
            model = LinearRegression()
        elif model_choice == 'Random Forest':
            model = RandomForestRegressor()
        elif model_choice == 'Neural Network':
            model = MLPRegressor(hidden_layer_sizes=(50, 25),
                                 activation='relu',
                                 solver='adam',
                                 random_state=1) # Longer Processing time

        # Fit the model
        model.fit(X_scaled, y)

        # Get today's features (all except XLE), Scale today's features
        today_features = energy.drop(columns='XLE').iloc[-1]
        today_features_scaled = scaler.transform([today_features])

        # Predict
        predicted_price = model.predict(today_features_scaled)[0]

        # Append predicted price to the list
        predicted_prices.append(predicted_price)

    # Generate dates for the predicted prices
    new_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]

    # Assign the predicted prices and new dates to the DataFrame
    predicted_prices_df['Date'] = new_dates
    predicted_prices_df['Predicted Price'] = predicted_prices

    # Return predicted DataFrame
    return predicted_prices_df

# Calculate accuracy metrics
def calculate_accuracy(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return mae, mse, rmse


# Streamlit app
def main():
    st.title("Energy Sector ETF Prediction and Risk Analysis")

    # Fetch data
    energy = fetch_data()

    # Sidebar for user input
    st.sidebar.header("User Input")
    shares = st.sidebar.number_input("Enter the number of shares:", value=100, step=100)
    
    # Dropdown for model choice
    model_choice = st.sidebar.selectbox("Choose a prediction model", ["Linear Regression", 
                                                                      "Random Forest", 
                                                                      "Neural Network",]) # Longer Processing time
    # Slider for predictions
    prediction_days = st.sidebar.slider("Number of days for prediction", 1, 365, 90)

    # Predict prices based on user's choice of model
    predicted_prices_df = predict_prices(energy, prediction_days, model_choice)

    # Calculate current price and portfolio value
    current_day = energy.index[-1].strftime("%Y-%m-%d")
    current_price = energy.loc[current_day, 'XLE']
    portfolio_value = shares * current_price

    # Display current price and predicted prices
    st.subheader("Portfolio Information")
    st.write(f"The current price of the asset is: ${current_price:.2f}")
    st.write(f"Predicted price in {prediction_days} days is: ${round(predicted_prices_df['Predicted Price'].iloc[-1], 2)}")
    
    # Plot predicted prices
    st.subheader("Predicted Prices")
    st.line_chart(predicted_prices_df.set_index('Date'))
    

    # Calculate accuracy metrics
    #y_true = energy['XLE'].values[-prediction_days:]
    #y_pred = predicted_prices_df['Predicted Price'].values
    #mae, mse, rmse = calculate_accuracy(y_true, y_pred)

    #st.subheader("Accuracy Metrics")
    #st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    #st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    #st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

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

