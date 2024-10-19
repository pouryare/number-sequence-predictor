import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple

# Set page config
st.set_page_config(page_title="Number Sequence Predictor", layout="wide")

# Function to split sequence (cached for performance)
@st.cache_data
def split_sequence(seq: List[int], n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a univariate sequence into samples for supervised learning.

    Args:
        seq (List[int]): The input sequence.
        n_steps (int): The number of time steps to use as input.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (input) and y (output) arrays.
    """
    X, y = [], []
    for i in range(len(seq)):
        end_ix = i + n_steps
        if end_ix > len(seq) - 1:
            break
        seq_x, seq_y = seq[i:end_ix], seq[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to create and train model (cached for performance)
@st.cache_resource
def create_and_train_model(X: np.ndarray, y: np.ndarray, n_steps: int, n_features: int) -> keras.Model:
    """
    Create and train the LSTM model.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Target data.
        n_steps (int): Number of time steps.
        n_features (int): Number of features.

    Returns:
        keras.Model: Trained LSTM model.
    """
    model = keras.Sequential([
        keras.layers.LSTM(250, activation='relu', input_shape=(n_steps, n_features)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError())
    model.fit(X, y, epochs=300, verbose=0, validation_split=0.2)
    return model

# Function to generate future predictions
def generate_future_predictions(model: keras.Model, last_sequence: np.ndarray, num_predictions: int, n_steps: int, n_features: int) -> List[float]:
    """
    Generate future predictions using the trained model.

    Args:
        model (keras.Model): The trained LSTM model.
        last_sequence (np.ndarray): The last known sequence of values.
        num_predictions (int): The number of future predictions to make.
        n_steps (int): Number of time steps used in the model.
        n_features (int): Number of features used in the model.

    Returns:
        List[float]: A list of predicted future values.
    """
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(num_predictions):
        x_input = current_sequence.reshape((1, n_steps, n_features))
        predicted = model.predict(x_input, verbose=0)[0][0]
        future_predictions.append(predicted)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = predicted

    return future_predictions

# Streamlit app
def main():
    st.title("Number Sequence Predictor")
    st.write("This app predicts future numbers based on a given sequence using LSTM.")

    # Input form
    with st.form("sequence_input_form"):
        sequence_input = st.text_input("Enter a comma-separated sequence of numbers:", "10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150")
        n_steps = st.slider("Select the number of time steps:", min_value=2, max_value=10, value=2)
        num_predictions = st.slider("Select the number of future predictions:", min_value=1, max_value=20, value=10)
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Process input
        try:
            data = [int(x.strip()) for x in sequence_input.split(',')]
            if len(data) < n_steps + 1:
                st.error(f"Please enter at least {n_steps + 1} numbers in the sequence.")
                return
        except ValueError:
            st.error("Invalid input. Please enter a comma-separated sequence of numbers.")
            return

        # Prepare data
        X, y = split_sequence(data, n_steps)
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        # Create and train model
        with st.spinner("Training the model..."):
            model = create_and_train_model(X, y, n_steps, n_features)

        # Generate future predictions
        last_known_sequence = np.array(data[-n_steps:])
        future_predictions = generate_future_predictions(model, last_known_sequence, num_predictions, n_steps, n_features)

        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Input Sequence:")
            st.line_chart(data)

        with col2:
            st.write("Future Predictions:")
            st.line_chart(future_predictions)

        # Visualize the original sequence and future predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(data)), data, label='Original Sequence', marker='o')
        ax.plot(range(len(data), len(data) + len(future_predictions)), future_predictions, label='Future Predictions', marker='x', linestyle='--')
        ax.set_title('Original Sequence and Future Predictions')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
        st.pyplot(fig)

        st.write("Future predictions:", future_predictions)

if __name__ == "__main__":
    main()