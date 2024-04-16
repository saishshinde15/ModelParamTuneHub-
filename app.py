from flask import Flask, render_template, request
import tensorflow as tf
from itertools import product
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data
        x_train_shape = request.form['x_train_shape']
        model_choice = request.form['model_choice']

        # Convert x_train_shape to tuple
        x_train_shape_tuple = tuple(map(int, x_train_shape.split(',')))

        # Run model selection based on user inputs
        best_params, best_accuracy = run_model_selection(x_train_shape_tuple, model_choice)

        return render_template('result.html', best_params=best_params, best_accuracy=best_accuracy)
    else:
        return render_template('index.html')

def run_model_selection(x_train_shape, model_choice):
    # Load and preprocess your data as X and y here
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model parameters based on model_choice
    if model_choice == 'LSTM':
        layers_list = [1, 2, 3]
        units_list = [64, 128, 256]
        activation_list = ['relu', 'sigmoid', 'tanh']
        model_class = tf.keras.layers.LSTM
    elif model_choice == 'GRU':
        layers_list = [1, 2, 3]
        units_list = [64, 128, 256]
        activation_list = ['relu', 'sigmoid', 'tanh']
        model_class = tf.keras.layers.GRU
    elif model_choice == 'CNN':
        filters_list = [32, 64, 128]
        kernel_sizes_list = [(3, 3), (5, 5), (7, 7)]
        units_list = [64, 128, 256]
        activation_list = ['relu', 'sigmoid', 'tanh']
        model_class = tf.keras.layers.Conv2D

    best_accuracy = 0.0
    best_params = {}

    for params in product(layers_list, units_list, activation_list):
        model = tf.keras.models.Sequential()
        for layer in range(params[0]):
            model.add(model_class(units=params[1], activation=params[2], return_sequences=True if layer < params[0] - 1 else False, input_shape=x_train_shape))
            model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

        history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stopping])

        val_accuracy = max(history.history['val_accuracy'])

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = {'layers': params[0], 'units': params[1], 'activation': params[2]}

    return best_params, best_accuracy

if __name__ == '__main__':
    app.run(debug=True)
