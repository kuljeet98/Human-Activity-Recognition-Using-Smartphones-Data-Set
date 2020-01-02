## HAR using Classical Machine Learning models and LSTM models.
### Dataset Link :https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones<br>
#### Models used in Human Activity Recognition:
```
(1) LSTM Model
(2) Logistic regression with GridSearch
(3) Linear SVM with GridSearch
(4) Kernal SVM with GridSearch
(5) Random Forest with GridSearch
```
#### Some Basic Libraries
``` python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
%matplotlib inline


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, Confusion_matrix, Classification_report
from sklearn.manifold import TSNE

from keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

```
####

