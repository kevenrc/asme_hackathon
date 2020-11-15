import matplotlib.pyplot as plt
import pandas as pd
from preprocess_bridgeport import DataLoader
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

filename = 'bridgeport1week1-train.csv'

data = DataLoader(filename)

x = data.data_labels
x = x.dropna()

y_radial = x['Machines > Bridgeport Mill 1 > Spindle > Y-Radial > Damage Accumulation']
x_axial = x['Machines > Bridgeport Mill 1 > Spindle > X-Axial > Damage Accumulation']

y_radial['datetime'] = pd.to_datetime(y_radial['Time'])
x_axial['datetime'] = pd.to_datetime(x_axial['Time'])

plt.figure(0)
plt.plot(y_radial.datetime, y_radial.Value, '.')
plt.yticks([0, 55])

plt.figure(1)
plt.plot(x_axial.datetime, x_axial.Value, '.')
plt.yticks([0, 55])

plt.figure(2)
plt.plot(x_axial.datetime, x_axial.Value.cumsum(), '.')
plt.yticks([])

plt.figure(3)
plt.plot(pd.to_datetime(x['Machines > Bridgeport Mill 1 > Spindle > Y-Radial > Damage Accumulation']['Time']), x['Machines > Bridgeport Mill 1 > Spindle > Y-Radial > Damage Accumulation']['Value'])
plt.show()
