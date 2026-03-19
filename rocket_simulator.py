import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# -------------------------------
# Physics Simulation
# -------------------------------
def simulate_flight(thrust, fuel):
    g, mass_dry, dt = 9.81, 50.0, 0.1
    v, h, m_f = 0, 0, fuel
    max_h = 0

    while h >= 0:
        curr_m = mass_dry + m_f
        a = (thrust / curr_m) - g if m_f > 0 else -g
        v += a * dt
        h += v * dt

        m_f = max(0, m_f - 1.0)
        max_h = max(max_h, h)

        if h > 100000:
            break

    return max_h

# -------------------------------
# Dataset Generation
# -------------------------------
data = []
for _ in range(1000):
    t = np.random.uniform(1500, 5000)
    f = np.random.uniform(50, 200)
    apogee = simulate_flight(t, f)
    data.append([t, f, apogee])

df = pd.DataFrame(data, columns=['Thrust', 'Fuel', 'Apogee'])

# -------------------------------
# ML Model
# -------------------------------
X = df[['Thrust', 'Fuel']]
y = df['Apogee']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict([[3000, 100]])
print(f"Predicted Apogee: {prediction[0]:.2f} m")
