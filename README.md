# Machine Learning Model Parameter Tuning Web App

## Overview
This Flask web application is designed to assist users in determining the best parameters for machine learning models (LSTM, GRU, and CNN) based on a provided dataset dimension (`X_train` shape). The app utilizes TensorFlow for model training and evaluation.

## Purpose and Impact:
This project aims to streamline the process of machine learning model development by providing an intuitive interface for parameter tuning. By automating the optimization process and utilizing custom-made models, users can achieve better accuracy and efficiency in their machine learning projects without the limitations of standard approaches like grid search.

## Why I Didn't Use Grid Search:
While grid search is a popular method for hyperparameter tuning, I opted not to use it in this project due to several limitations and drawbacks:

Computational Complexity: Grid search can become computationally intensive, especially with a large number of hyperparameters and their possible values. This can lead to long training times and resource constraints.

Limited Exploration: Grid search explores a predefined grid of hyperparameter values, which may not cover the entire search space effectively. This can result in suboptimal parameter combinations being overlooked.

No Adaptability: Grid search does not adapt or learn from previous iterations, leading to a lack of flexibility in finding the best parameter settings dynamically.
Manual Configuration: Setting up a grid search requires manual configuration of hyperparameter ranges, which can be time-consuming and prone to human error.

Scalability Issues: Grid search may struggle to scale efficiently to high-dimensional parameter spaces or when dealing with large datasets, limiting its applicability in complex machine learning tasks.

By developing custom-made models and leveraging TensorFlow's capabilities for automated parameter tuning, I aimed to overcome these limitations and provide a more efficient and adaptable solution for optimizing machine learning models.

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
