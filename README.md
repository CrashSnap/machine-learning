# Machine Learning

## Table of Contents
- [YoloV8 Car Damage Detection](#yolov8-car-damage-detection)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Model Training](#model-training)
    - [Prediction](#prediction)

- [Repair Cost Regression Model](#repair-cost-regression-model)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Data Generation](#data-generation)
    - [Preprocessing](#preprocessing)
    - [Model Training](#model-training)
    - [Prediction](#prediction)
    - [Saving the Model](#saving-the-model)

- [Contributing](#contributing)
- [License](#license)

## YOLOv8 Car Damage Detection

This is a project for detecting and classifying car damages using the YOLOv8 model. The project is designed to help users quickly and accurately identify various types of car damages through image analysis.

### Introduction

Car damage detection is a crucial task for ensuring the safety and reliability of vehicles. This project leverages the power of YOLOv8, a state-of-the-art object detection model, to identify and classify different types of car damages from images. The model can detect the following classes:
- Bumper Scratch
- Door Scratch
- Door Dent
- Quarter Panel Scratch
- Tire Flat
- Headlight Damage
- Crack
- Quarter Panel Dent
- Front Windscreen Damage
- Bumper Dent
- Taillight Damage
- Rear Windscreen Damage
- Car Window Damage
- Bonnet Dent
- Trunk Door Dent

### Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/ylov8_car_damage_detection.git

cd ylov8_car_damage_detection

pip install -r requirements.txt
```

Ensure you have the necessary hardware (e.g., GPU) and software (e.g., CUDA) configured for optimal performance.

### Usage
#### Data Preparation
Prepare your dataset in the required format. The dataset should include annotated images for training the YOLOv8 model. Use tools like Roboflow for annotation.

#### Model Training
To train the YOLOv8 model, follow the steps in the Jupyter notebook provided in the repository. The training script is included in the `ylov8_car_damage_detection.ipynb` notebook.

#### Prediction
After training, use the trained model to make predictions on new images. The prediction script is included in the `ylov8_car_damage_detection.ipynb` notebook.

```python
# Example of using the trained model for prediction
from yolov8 import YOLOv8
model = YOLOv8('path/to/trained/model')
results = model.predict('path/to/image.jpg')
print(results)
```

## Repair Cost Regression Model
This project generates dummy data for car damage costs, preprocesses it, trains a neural network model to predict the total cost of repairs, and makes predictions on new data.

### Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/car-damage-cost-estimation.git
cd car-damage-cost-estimation
pip install -r requirements.txt
```

### Usage
#### Data Generation
Generate dummy data for car damages and calculate the total repair cost ranges. Save the generated data to a CSV file.

```python
# Generate dummy data
data = {
    'image_id': [f'{i:03d}' for i in range(1, n_samples + 1)],
    'bonnet_dent': np.random.randint(0, 3, n_samples),
    ...
}

# Calculate total cost
total_cost_min = np.zeros(n_samples)
total_cost_max = np.zeros(n_samples)
for damage, (min_cost, max_cost) in damage_cost_ranges.items():
    total_cost_min += np.array(data[damage]) * min_cost
    total_cost_max += np.array(data[damage]) * max_cost

data['total_cost_min'] = total_cost_min
data['total_cost_max'] = total_cost_max

# Create dataframe and save to CSV
df = pd.DataFrame(data)
df.to_csv('repair_cost_dummy_data.csv', index=False)
```

#### Preprocessing
1. Load Data: Load the generated CSV file.
   ```python
   df = pd.read_csv('repair_cost_dummy_data.csv')
   ```
2. Feature Selection: Define the features (damage types) and the target (repair cost ranges).
   ```python
   X = df.drop(['image_id', 'total_cost_min', 'total_cost_max'], axis=1)
   y = np.column_stack((df['total_cost_min'], df['total_cost_max']))
   ```
3. Scaling: Normalize the features using StandardScaler and scale the target variable using `MinMaxScaler`.
   ```python
   scaler = StandardScaler()
   X = scaler.fit_transform(X)

   y_scaler = MinMaxScaler()
   y = y_scaler.fit_transform(y)
   ```
4. Train-Test Split: Split the data into training and testing sets.
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
#### Model Training
1. Model Definition: Define a neural network model using TensorFlow with dense layers.
   ```python
   model = tf.keras.Sequential([
     tf.keras.layers.Dense(16, input_shape=[X_train.shape[1]]),
     tf.keras.layers.Dense(2)])


2. Compilation: Compile the model with mean squared error loss and Adam optimizer.
   ```python
   model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
   ```
3. Training: Train the model on the training data with a validation split.
   ```python
   history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=32)
   ```

#### Prediction
1. New Data Prediction: Standardize new data and use the trained model to predict the repair cost ranges.
   ```python
   new_data = pd.DataFrame({
    'bonnet_dent': [0, 0],
    'bumper_dent': [0, 0],
    ...})
   new_data_scaled = scaler.transform(new_data)
   predicted_costs_scaled = model.predict(new_data_scaled)
   ```
2. Inverse Transform: Convert the scaled predictions back to the original cost range.
   ```python
   predicted_costs = y_scaler.inverse_transform(predicted_costs_scaled)
   for min_cost, max_cost in predicted_costs:
    print(f'Predicted Cost Range: Rp{int(min_cost)} - Rp{int(max_cost)}')
   ```
### Saving the Model
Save the trained model in the `SavedModel` format and convert it to TensorFlow.js format for use in web applications.
```python
model.save('my_saved_model.h5')

!tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  my_saved_model \
  my_tfjs_model
```

### Contributing
Feel free to open issues or submit pull requests for improvements.

### License
This project is licensed under the MIT License.
