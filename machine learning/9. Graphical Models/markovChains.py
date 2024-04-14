# Code for weather forecasting using Markov Chains

# Define transition matrix (e.g., sunny to rainy, rainy to sunny)
transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])

# Define initial state distribution (e.g., sunny, rainy)
initial_state = np.array([0.6, 0.4])

# Generate weather forecast for the next 5 days
forecast = np.random.choice(['Sunny', 'Rainy'], size=5, p=initial_state)
for i in range(1, len(forecast)):
    forecast[i] = np.random.choice(['Sunny', 'Rainy'], p=transition_matrix[forecast[i - 1] == 'Sunny'])

print("Weather Forecast:", forecast)
