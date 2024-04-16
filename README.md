# Machine Learning Model Parameter Tuning Web App

## Overview
This Flask web application is designed to assist users in determining the best parameters for machine learning models (LSTM, GRU, and CNN) based on a provided dataset dimension (`X_train` shape). The app utilizes TensorFlow for model training and evaluation.

## Features
- Input `X_train` shape and select a model (LSTM, GRU, or CNN).
- Automatically determine the best parameters (number of layers, number of units, activation functions) for the chosen model based on accuracy metrics.
- Display the best parameters and accuracy results to the user.

## Installation
1. Clone the repository to your local machine:
2. Navigate to the project directory:
3. Create and activate a virtual environment (optional but recommended):
4. Install the required packages:

## Usage
1. Start the Flask app:
2. Open a web browser and go to `http://localhost:5000`.
3. Input your `X_train` shape and choose a machine learning model.
4. Click "Submit" to see the best parameters and accuracy results.

## Dependencies
- Flask
- TensorFlow
- Other dependencies are listed in `requirements.txt`.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Contact
For any inquiries or issues, please contact [Saish Shinde](saish.shinde.jb@gmail.com).
