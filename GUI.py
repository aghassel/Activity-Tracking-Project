## GUI for the activity tracking project
## Group 44
## Abdellah Ghassel (20230384), Kyle Deverell (20217339), Liam O'Regan (20233042)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Preprocessing functions
activity_codes = {"jumping": 0, "walking": 1}
activity_labels = {0: "Jumping", 1: "Walking"}

def removeFiveSecRollFunc(df, window_size):
    start_time = df["Time (s)"].min()
    end_time = df["Time (s)"].max()
    df = df[(df["Time (s)"] > start_time + 5) & (df["Time (s)"] < end_time - 5)]

    df = df.rolling(window_size).mean().dropna()
    return df

def featuresFunc(segment):
    features = []
    columnData = segment['Absolute acceleration (m/s^2)']
    
    mean = columnData.mean()
    std = columnData.std()
    minimum = columnData.min()
    maximum = columnData.max()
    skewness = columnData.skew()
    mean_squared = np.mean(columnData**2)
    kurtosis = columnData.kurtosis()
    variance = columnData.var()
    median = columnData.median()
    q1 = columnData.quantile(0.25)
    q3 = columnData.quantile(0.75)
    iqr = q3 - q1
    range = maximum - minimum
    
    features.extend([mean, std, minimum, maximum, skewness, mean_squared, kurtosis, variance, median, q1, q3, iqr, range])
    return features

def segmentFunc(data):
    samples = len(data)
    windowSize = 5
    sampleRate = 100
    windowSamples = windowSize * sampleRate
    segments = []
    for i in range(0, samples, windowSamples):
        segment = data[i:i+windowSamples]
        if len(segment) == windowSamples:
            segments.append(segment)
    np.random.shuffle(segments)
    
    return segments

#modified normalization function since there's no activity column
def normalizeFunc(segment):
    segment_copy = segment.copy()
    scaler = MinMaxScaler()
    segment_copy.iloc[:, 1:] = scaler.fit_transform(segment_copy.iloc[:, 1:])  # Exclude the 'Time (s)' column
    return segment_copy

# new functions

def loadModel(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def browseFile():
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if path:
        pathVar.set(path)
        predictButton.config(state=tk.NORMAL)

def saveFile(predictions):
    path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if path:
        df = pd.DataFrame(predictions, columns=["Activity"])
        df["Activity"] = df["Activity"].map(activity_labels)
        df.to_csv(path, index=False)
        messagebox.showinfo("Success", "File saved! Enjoy! :)")

def predictFunc(path, model):
    try:
        data = pd.read_csv(path)
        preprocessedData = removeFiveSecRollFunc(data, 5)
        normalizedData = normalizeFunc(preprocessedData)
        segmentedData = segmentFunc(normalizedData)
        features = []
        for segment in segmentedData:
            features.append(featuresFunc(segment))
        predictions = model.predict(features)
        return data, preprocessedData, normalizedData, segmentedData, predictions
    except Exception as e:
        messagebox.showerror("Error", f"Error processing file: {e}")
        return None

def predictionAndPlots():
    global predictions
    path = pathVar.get()
    results = predictFunc(path, model)
    if results is not None:
        data, preprocessedData, normalizedData, segmentedData, predictions = results
        segmentedData = segmentFunc(normalizedData)
        predictedCode = int(stats.mode(predictions)[0][0])  # Get the mode of predictions
        predictedActivity = activity_labels[predictedCode]
        resultVar.set(predictedActivity)
        resultLabel.config(text="Activity: ")

        saveButton.config(state=tk.NORMAL)
        plotData(data, preprocessedData, normalizedData, segmentedData, predictions)
    else:
        saveButton.config(state=tk.DISABLED)

def plotData(data, preprocessedData, normalizedData, segmentedData, predictions):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    # raw data
    axes[0, 0].plot(data["Time (s)"], data["Absolute acceleration (m/s^2)"], 'o', markersize=2, color='red')
    axes[0, 0].set_title('Raw Data')

    # preprocessed data
    axes[0, 1].plot(preprocessedData["Time (s)"], preprocessedData["Absolute acceleration (m/s^2)"], 'o', markersize=2, color='blue')
    axes[0, 1].set_title('Preprocessed Data')

    # normalized data
    axes[1, 0].plot(normalizedData["Time (s)"], normalizedData["Absolute acceleration (m/s^2)"], 'o', markersize=2, color='green')
    axes[1, 0].set_title('Normalized Data')

    # segmented outputs
    for i, segment in enumerate(segmentedData):
        color = 'orange' if predictions[i] == 0 else 'purple'
        axes[1,1].plot(segment["Time (s)"], segment["Absolute acceleration (m/s^2)"], 'o', markersize=2, color=color)
    axes[1,1].set_title('Segmented Predicted Outputs')

    jumpingLegend = plt.Line2D([0], [0], color='orange', label='jumping')
    walkingLegend = plt.Line2D([0], [0], color='purple', label='walking')
    axes[1,1].legend(handles=[jumpingLegend, walkingLegend])

    for ax in axes.flatten():
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Absolute acceleration (m/s^2)')

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=4, column=0, columnspan=2, padx=10, pady=10)

root = tk.Tk()
root.title("Activity Tracking Project")
root.geometry("1000x650")

model = loadModel("model.pkl")

pathVar = tk.StringVar()
resultVar = tk.StringVar()

browseButton = tk.Button(root, text="Browse", command=browseFile, font=("Roboto", 10, "bold"))
browseButton.grid(row=0, column=0, padx=10, pady=10)

pathLabel = tk.Label(root, textvariable=pathVar, font=("Roboto", 10))
pathLabel.grid(row=1, column=0, padx=10, pady=10)

predictButton = tk.Button(root, text="Predict Activity", state=tk.DISABLED, command=predictionAndPlots, font=("Roboto", 10, "bold"))
predictButton.grid(row=0, column=1, padx=10, pady=10)

resultLabel = tk.Label(root, text="", font=("Roboto", 12))
resultLabel.grid(row=3, column=0, padx=10, pady=10)

resultValLabels = tk.Label(root, textvariable=resultVar, font=("Roboto", 12, "bold"), fg="blue")
resultValLabels.grid(row=3, column=1, padx=10, pady=10)

saveButton = tk.Button(root, text="Save Predictions", state=tk.DISABLED, command=lambda: saveFile(predictions), font=("Roboto", 10, "bold"))
saveButton.grid(row=0, column=2, padx=10, pady=10)

root.mainloop()