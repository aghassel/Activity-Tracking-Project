import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, recall_score
from sklearn.decomposition import PCA

# Load and label the data
def load_and_label_data(data_files):
    labeled_data = []
    for person, activities in data_files.items():
        for activity, positions in activities.items():
            for position, file_path in positions.items():
                data = pd.read_csv(file_path)
                data["Activity"] = activity_codes[activity]
                labeled_data.append(data)
    return labeled_data

# Preprocessing and feature extraction functions
def remove_five_seconds_rolling(df, window_size):
    start_time = df["Time (s)"].min()
    end_time = df["Time (s)"].max()
    df = df[(df["Time (s)"] > start_time + 5) & (df["Time (s)"] < end_time - 5)]

    df = df.rolling(window_size).mean()
    df = df.dropna()
    return df

def segment_data(data, window_size=5, sample_rate=100):
    samples = len(data)
    window_samples = window_size * sample_rate
    segments = []
    for i in range(0, samples, window_samples):
        segment = data[i:i+window_samples]
        if len(segment) == window_samples:
            segments.append(segment)
    np.random.shuffle(segments)
    
    return segments

def extract_features(segment):
    features = []
    for column in segment.columns[:-1]:  # Exclude the 'Activity' column
        column_data = segment[column]
        mean = column_data.mean()
        std = column_data.std()
        minimum = column_data.min()
        maximum = column_data.max()
        skewness = column_data.skew()
        kurtosis = column_data.kurtosis()
        energy = np.sum(column_data ** 2)
        features.extend([mean, std, minimum, maximum, skewness, kurtosis, energy])
    return features

# Load the data and preprocess
data_files = {
        "abdellah": {
            "jumping": {
                "backright": "data/jumping/ajbrp_data.csv",
                "frontleft": "data/jumping/ajflp_data.csv",
            },
            "walking": {
                "backright": "data/walking/awbrp_data.csv",
                "frontleft": "data/walking/awflp_data.csv",
            },
    },
  }  # Your data files dictionary with activity and positions

activity_codes = {"jumping": 0, "walking": 1}

labeled_data = load_and_label_data(data_files)
preprocessed_data = [remove_five_seconds_rolling(df, 100) for df in labeled_data]

segmented_data = []
for data in preprocessed_data:
    segmented_data.extend(segment_data(data))

# Extract features and labels
feature_matrix = []
label_vector = []

for segment in segmented_data:
    features = extract_features(segment)
    label = segment.iloc[0, -1]
    feature_matrix.append(features)
    label_vector.append(label)

feature_matrix = np.array(feature_matrix)
label_vector = np.array(label_vector)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(feature_matrix, label_vector, test_size=0.1, random_state=42)

# define Standard Scaler to normalize inputs
scaler = StandardScaler()
l_reg = LogisticRegression(max_iter=10000)
pca = PCA(n_components=2)

# defining classifier and pupeline
l_reg = LogisticRegression(max_iter=10000)
pca_pipe = make_pipeline(StandardScaler(), pca)
clf = make_pipeline(l_reg)

X_train_pca = pca_pipe.fit_transform(X_train)
X_test_pca = pca_pipe.fit_transform(X_test)

clf.fit(X_train_pca, Y_train)

# train
# pca_pipe.fit(X_train, Y_train)

y_pred_clf = clf.predict(X_test_pca)
#y_clf_prob = clf.predict_proba(X_test)
print("y_pred_clf is: ", y_pred_clf)
#print("y_clf_prob is: ", y_clf_prob)

acc = accuracy_score(Y_test, y_pred_clf)
print("Accuracy: ", acc)
      
recall = recall_score(Y_test, y_pred_clf)
print("recall is : ", recall)

