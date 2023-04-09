import matplotlib.pyplot as plt
import pandas as pd
import h5py 

# Graph Acceleration vs Time in x direction for walking/jumping combined data 
data1 = pd.read_csv("data/combined/Combined_dataset.csv")
xacceleration = data1['Linear Acceleration x (m/s^2)']
time = data1['Time (s)']
plt.plot(time, xacceleration)
plt.title('Combined Linear Acceleration in x vs Time')
plt.ylabel('Linear Acceleration x (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Acceleration vs Time in x direction for walking data 
data2 = pd.read_csv("data/combined/walking_combined.csv")
xacceleration = data2['Linear Acceleration x (m/s^2)']
time = data2['Time (s)']
plt.plot(time, xacceleration)
plt.title('Walking Linear Acceleration in x vs Time')
plt.ylabel('Linear Acceleration y (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Acceleration vs Time in x direction for jumping data 
data3 = pd.read_csv("data/combined/jumping_combined.csv")
xacceleration = data3['Linear Acceleration x (m/s^2)']
time = data3['Time (s)']
plt.plot(time, xacceleration)
plt.title('Jumping Linear Acceleration in x vs Time')
plt.ylabel('Linear Acceleration x (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Acceleration vs Time in y direction for walking/jumping combined data 
data4 = pd.read_csv("data/combined/Combined_dataset.csv")
yacceleration = data4['Linear Acceleration y (m/s^2)']
time = data4['Time (s)']
plt.plot(time, yacceleration)
plt.title('Combined Linear Acceleration in y vs Time')
plt.ylabel('Linear Acceleration y (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Acceleration vs Time in y direction for walking data 
data5 = pd.read_csv("data/combined/walking_combined.csv")
yacceleration = data5['Linear Acceleration y (m/s^2)']
time = data5['Time (s)']
plt.plot(time, yacceleration)
plt.title('Walking Linear Acceleration in y vs Time')
plt.ylabel('Linear Acceleration y (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Acceleration vs Time in y direction for jumping data 
data6 = pd.read_csv("data/combined/jumping_combined.csv")
yacceleration = data6['Linear Acceleration y (m/s^2)']
time = data6['Time (s)']
plt.plot(time, yacceleration)
plt.title('Jumping Linear Acceleration in y vs Time')
plt.ylabel('Linear Acceleration y (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Acceleration vs Time in z direction for walking/jumping combined data 
data7 = pd.read_csv("data/combined/Combined_dataset.csv")
zacceleration = data4['Linear Acceleration z (m/s^2)']
time = data7['Time (s)']
plt.plot(time, zacceleration)
plt.title('Combined Linear Acceleration in z vs Time')
plt.ylabel('Linear Acceleration z (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Acceleration vs Time in z direction for walking data 
data8 = pd.read_csv("data/combined/walking_combined.csv")
zacceleration = data8['Linear Acceleration y (m/s^2)']
time = data8['Time (s)']
plt.plot(time, zacceleration)
plt.title('Walking Linear Acceleration in z vs Time')
plt.ylabel('Linear Acceleration z (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Acceleration vs Time in z direction for jumping data 
data9 = pd.read_csv("data/combined/jumping_combined.csv")
zacceleration = data9['Linear Acceleration y (m/s^2)']
time = data9['Time (s)']
plt.plot(time, zacceleration)
plt.title('Jumping Linear Acceleration in z vs Time')
plt.ylabel('Linear Acceleration z (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Absolute Acceleration vs Time for walking/jumping combined data 
data10 = pd.read_csv("data/combined/Combined_dataset.csv")
cacceleration = data10['Absolute acceleration (m/s^2)']
time = data10['Time (s)']
plt.plot(time, cacceleration)
plt.title('Absolute Acceleration vs Time')
plt.ylabel('Absolute acceleration (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Absolute Acceleration vs Time for walking data 
data11 = pd.read_csv("data/combined/walking_combined.csv")
cacceleration = data11['Absolute acceleration (m/s^2)']
time = data11['Time (s)']
plt.plot(time, cacceleration)
plt.title('Walking Absolute Acceleration vs Time')
plt.ylabel('Absolute acceleration (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# Graph Absolute Acceleration vs Time for jumping data 
data12 = pd.read_csv("data/combined/jumping_combined.csv")
cacceleration = data12['Absolute acceleration (m/s^2)']
time = data12['Time (s)']
plt.plot(time, cacceleration)
plt.title('Jumping Absolute Acceleration vs Time')
plt.ylabel('Absolute acceleration (m/s^2)')
plt.xlabel('Time (s)')
plt.show()

# # Walking vs jumping on the same plot for x Acceleration 
# walk_data = pd.read_csv('data/combined/walking_combined.csv')
# jump_data = pd.read_csv('data/combined/jumping_combined.csv')

# # Extract data for the two datasets
# x = walk_data['Time (s)']
# y1 = walk_data['Linear Acceleration x (m/s^2)']
# y2 = jump_data['Linear Acceleration y (m/s^2)']

# fig, ax = plt.subplots()
# ax.plot(x, y1, label='Walking Data')
# ax.plot(x, y2, label='Jumping Data')
# ax.legend()
# plt.title('Walking and Jumping Acceleration in x vs Time')
# plt.xlabel('Linear Acceleration in x (m/s^2)')
# plt.ylabel('Time (s)')
# plt.show()