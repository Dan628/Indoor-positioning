# knn_indoor_position

KnnLocalizer is a python implementation of a K Nearest Neighbors indoor positioning algorithm used for radio map based indoor positioning in multi-floor environments. 

The localization algorithm requires a three-dimensional label space of coordinates in the form: longitude, latitude, and floor. Each coordinate is predicted using a KNN regression and every sample prediction is produced as an array containing coordinates eg [longitude, latitude, floor] where floor is rounded to the nearest floor.

KnnLocalizer was designed to provide as a source of ground truth for rss based indoor positioning problems in large-scale environments and to serve as a convenient instrument to estimate the discriminative variability of RSS radio-map data after performing dimensionality reduction and feature extraction techniques.

## Installation

pip install knn_indoor_position

```bash
pip install knn_indoor_position
```

## Usage

```python
import knn_indoor_position
localizer = KnnLocalizer(X_train, y_train, 10) # Initialize
new_predictions = localizer.fit_predict(X_test) # Predict
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
