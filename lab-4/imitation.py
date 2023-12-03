import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CashierSimulation:
    def __init__(self, total_simulated_time, time_interval, arrival_rate_hours, service_time_constant, payment_time, max_cashiers, max_queue_length) -> None:
        self.total_simulated_time = total_simulated_time
        self.time_interval = time_interval
        self.arrival_rate_hours = arrival_rate_hours
        self.service_time_constant = service_time_constant
        self.payment_time = payment_time
        self.max_cashiers = max_cashiers
        self.max_queue_length = max_queue_length
        
        self.current_time = 0
        self.customers_waiting = 0
        self.customer_wait_times = []
        self.queue_length = 0
        self.active_cashiers = 1
        
        self.simulation_data = {
            'Time': [],
            'Customers Arrived': [],
            'Service Times': [],
            'Customers Served': [],
            'Customers Left Queue': [],
            'Active Cashiers': []
        }
        
    def simulate(self):
        while self.current_time < self.total_simulated_time:
            hour = self.current_time // 60
            lambda_hour = self.arrival_rate_hours[hour // 6]
            arriving_customers = np.random.poisson(lambda_hour) + self.customers_waiting
            
            self.customer_wait_times = []
            self.customers_waiting = 0
            service_times = []
            service_times.extend(self.customer_wait_times)
            
            if arriving_customers > self.max_queue_length and self.active_cashiers < self.max_cashiers:
                self.active_cashiers += 1
            elif arriving_customers < 2 and self.active_cashiers > 1:
                self.active_cashiers -= 1
            
            time_left = self.time_interval
            served_customers = 0
            
            for _ in range(arriving_customers):
                goods_quantity = np.random.binomial(6, 0.6)
                service_time = int(self.service_time_constant * goods_quantity * 0.5 + self.payment_time)
                service_times.append(service_time)
 
            for service_time in service_times:
                if service_time <= time_left:
                    served_customers += 1
                else:
                    self.customers_waiting += 1
                    if self.customers_waiting == 1:
                        self.customer_wait_times.append(service_time - time_left)
                    else:
                        self.customer_wait_times.append(service_time)
                    
                time_left -= service_time
            
            customers_left_queue = self.customers_waiting
            
            self.simulation_data['Time'].append(self.current_time)
            self.simulation_data['Customers Arrived'].append(arriving_customers)
            self.simulation_data['Service Times'].append(service_times)
            self.simulation_data['Customers Served'].append(served_customers) 
            self.simulation_data['Customers Left Queue'].append(customers_left_queue)
            self.simulation_data['Active Cashiers'].append(self.active_cashiers)

            self.queue_length = max(0, self.queue_length - self.active_cashiers)
            self.current_time += self.time_interval

    def format_simulation_results(self):
        self.simulation_data['Time'] = ['{:02}:{:02}'.format(time // 60, time % 60) for time in self.simulation_data['Time']]
        return pd.DataFrame(self.simulation_data) 

total_simulation_time = 24 * 60
time_step_interval = 20
arrival_rates = [1, 3, 5, 2]
service_time_multiplier = 1
payment_processing_time = 2

simulation = CashierSimulation(total_simulation_time, time_step_interval, arrival_rates, service_time_multiplier, payment_processing_time, 5, 7)
simulation.simulate()

pd.set_option('display.max_rows', None) 
simulation_results = simulation.format_simulation_results()

print("SIMULATION RESULTS:")
print(simulation_results)

# Function to plot histograms
def plot_histogram(average_values, x_label, y_label, title):
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(average_values)), average_values, color='darkgreen')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    x_labels = [f'{i:02}:00-{(i + 1) % 24:02}:00' for i in range(len(average_values))]
    plt.xticks(range(len(average_values)), x_labels, rotation=90, fontsize=10, ha='center')

    plt.tight_layout()
    plt.show()


# Plotting histogram for average number of clients
clients = simulation_results['Customers Arrived'].tolist()
average_clients = [sum(clients[i:i+3]) / 3 for i in range(0, len(clients), 3)]
plot_histogram(average_clients, 'Time', 'Average number of clients', 'Histogram of clients per hour')


# Plotting histogram for average number of cash registers open
cash_registers = simulation_results['Active Cashiers'].tolist()
average_cash_registers = [sum(cash_registers[i:i+3]) / 3 for i in range(0, len(cash_registers), 3)]
plot_histogram(average_cash_registers, 'Time', 'Average number of open cash registers', 'Histogram of open cash registers per hour')