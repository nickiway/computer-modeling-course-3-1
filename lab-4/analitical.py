import math
import pandas as pd

# Initial values
total_customers = 5
min_cashiers_required = 1

service_speed = 1

# Calculate the probability of 'k' number of customers arriving with lambda
def calculate_poisson_probability(lmbda, k):
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)

# Function to compute probabilities for each number of customers
def calculate_probabilities(lmbda):
    probabilities = []
    
    for num_clients in range(total_customers + 1):
        probability_rate = calculate_poisson_probability(lmbda, num_clients)
        probabilities.append(probability_rate)
        
    return probabilities

# Calculate the average number of clients in the system
def calculate_average_clients(probabilities):
    average_clients = 0
    
    for k, probability in enumerate(probabilities):
        average_clients += k * probability
        
    return average_clients

# Initial parameters
arrival_rate_per_cashier = 0.4  
lambda_value = total_customers * arrival_rate_per_cashier  

# Compute probabilities and expected clients
probabilities = calculate_probabilities(lambda_value)
expected_clients = calculate_average_clients(probabilities)

# Prepare dataset for DataFrame
data = {
    'Number of Clients': list(range(total_customers + 1)),
    'Probability of Cashier Usage': probabilities
}
 
df = pd.DataFrame(data).drop(0)

print(df)
print("Average number of clients in the system:", round(expected_clients, 2))
