# BadhKavach: Flood Prediction Model

This project, BadhKavach, is a flood prediction model built using various geospatial and machine learning techniques. The model uses terrain data, height maps, and segmentation maps to simulate flood scenarios and predict flood-prone areas.

## Features

1. **Slope Calculation**: Computes the slope of the terrain using height maps.
2. **Segmentation**: Extracts labels from segmentation maps to identify different terrain types such as water, grassland, forest, etc.
3. **Flood Simulation**: Simulates flooding scenarios based on terrain data and water rise amounts.
4. **CNN Model**: A Convolutional Neural Network (CNN) model for further analysis and prediction.

## Installation

To run this project, you'll need to install the following dependencies:

```bash
!pip install rasterio
!pip install tensorflow
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install seaborn
!pip install scikit-learn
!pip install pillow
!pip install opencv-python
```

## Usage

1. **Loading Data**: Use the provided functions to load height maps, segmentation maps, and terrain maps.

2. **Calculate Slope**: Use the `calculate_slope()` function to compute the slope of the terrain.

3. **Extract Segmentation Labels**: Use the `extract_segmentation_labels()` function to convert segmentation maps into numerical labels.

4. **Flood Simulation**: Use the `flood_simulation()` function to simulate flood scenarios based on the height map.

5. **Visualize Flood Map**: Use the `visualize_flood_map()` function to visualize the flood map overlay on the height map.

6. **Build and Train CNN Model**: Use the `build_cnn_model()` function to build the CNN model and then train it with your data.

## Example

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
import cv2

# Load data
height_map_path = "/content/0001_h.png"
segmentation_map_path = "/content/0001_i2.png"
terrain_map_path = "/content/0001_t.png"

height_data = loadimg(height_map_path)
segmentation_data = loadimg(segmentation_map_path)
terrain_data = loadimgopencv(terrain_map_path)

# Calculate slope
resolution = 400.0
slope = calculate_slope(height_data, resolution)

# Extract segmentation labels
color_map = {
    1: (17, 141, 215),  # Water
    2: (225, 227, 155),  # Grassland
    3: (127, 173, 123),  # Forest
    4: (185, 122, 87),   # Hills
    5: (230, 200, 181),  # Desert
    6: (150, 150, 150),  # Mountain
    7: (193, 190, 175)   # Tundra
}
numerical_segmentation = extract_segmentation_labels(segmentation_data, color_map)

# Simulate flood
flood_map = flood_simulation(height_data)

# Visualize flood map
visualize_flood_map(flood_map)

# Build and train CNN model
input_shape = (512, 512, 3)
model = build_cnn_model(input_shape)
model.fit(X_train, y_train.reshape(y_train.shape[0], output_size), epochs=10, validation_split=0.1)
loss, accuracy = model.evaluate(X_test, y_test.reshape(y_test.shape[0], output_size))
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all the contributors and open-source projects that made this possible.
- Special thanks to the HackMIT team for providing the platform to develop this project.

## Author

- Siddhartha Dikshit, Madhoor Deo

## Repository

You can find the project repository [here](https://github.com/divyansh-pathak129/BadhKavach).
