import streamlit as st
import numpy as np  
import pandas as pd
import plotly.express as px

def generate_house_data(n_sample=100):
    np.random.seed(50)
    size = np.random.normal(1400, 50, n_sample)
    price = size * 50 + np.random.normal(0, 50, n_sample)
    return pd.DataFrame({'size': size, 'price': price})


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def train_model():
    df=generate_house_data(n_sample=100)
    x=df[['size']]
    y=df[['price']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    return model

def main():
    st.write("# House Price Prediction APP")
    
    st.write("This is a simple app to predict house prices based on size.")
    
    model=train_model()
    size = st.number_input('house size', min_value=500, max_value=2000, value=1500)
    
    if st.button('Predict'):
        predicted_price = model.predict([[size]])
        predicted_price_value = predicted_price.item()
        st.success(f'The predicted price for a house of size is : ${predicted_price_value:,.2f}')
            
        # st.write("The model was trained on synthetic data and may not reflect real-world prices.")      
        # st.write("Note: This is a simple linear regression model and may not reflect real-world prices.")
        
        df = generate_house_data()
        
        fig = px.scatter(df, x='size', y='price', title=' Size vs House Price')
        fig.add_scatter(x=[size], y=[predicted_price_value], 
                mode='markers',
                marker=dict(color='red', size=10),
                name='Predicted Price')
        
        st.plotly_chart(fig)
        
if __name__ == "__main__":
    main() 