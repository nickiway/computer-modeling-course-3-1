import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime 

step = 20
total_step = 24 * 60


class Customer(object): 
    def __init__(self) -> None:
        self.products = 0
        self.duration = 0
    
    def generate(self):
        self.products = np.random.binomial(20, 0.6)
        self.duration = self.products * 0.3 + 0.5 # ax + b
    
    
class Statistics(object):
    def __init__(self) -> None:
        self.df = pd.DataFrame(data = [[0, 0, 0]], columns=['products', 'duration',  'hours'])
        
    def update(self, prodcuts, duration, time) -> None:
        new_info = pd.DataFrame(data=[[prodcuts, duration, time]], columns=['products', 'duration', 'hours'])
        self.df = pd.concat([self.df, new_info], axis=0, ignore_index=True)
    
    
class Fund(object):
    def __init__(self, current_time) -> None:
        self.statistics = Statistics()
        self.customer = Customer()
        
        self.queue = 0
        
        self.current_time = current_time
        self.customers_count = 0 


        
    def display_info(self):
        print('-----', self.current_time, '-----')
        print('The number of customers arrived: ', self.customers_count)
        print('time gona be seized from next sess: %.2f' % self.customer.duration)    
        print('current queue:', self.queue)    


    def increase_time(self):
        self.current_time += datetime.timedelta(minutes=step)
        
        
    def get_mathematically_incline(self): 
        if self.current_time.hour > 8 and self.current_time.hour <= 10:
            return 3
        elif self.current_time.hour > 10 and self.current_time.hour <= 17:
            return 2
        elif self.current_time.hour > 17 and self.current_time.hour <= 23:
            return 5
        else:
            return 1
      
      
    def customer_arrived(self): 
        self.customers_count = np.random.poisson(self.get_mathematically_incline())
        self.queue += self.customers_count
    

    def customer_service(self):
        time = step

        while self.queue > 0 and time > 0:
            if self.customer.duration == 0:
                self.customer.generate()
                self.queue -= 1
            
                self.statistics.update(self.customer.products, self.customer.duration,  self.current_time.hour)

                
            time -= self.customer.duration
            if time >= 0:
                self.customer.duration = 0
            else:
                self.customer.duration = -time
                time = 0
                

Fund_instance = Fund(datetime.datetime(2022, 9, 1, 0))
Fund_instance.display_info()


def simulation(n_step):
    iterator = 0
    
    while iterator < n_step:
        Fund_instance.increase_time()
        Fund_instance.customer_arrived()
        Fund_instance.customer_service()
        Fund_instance.display_info()
        
        iterator += step
        
# run simulation
simulation(total_step)

print('--------------------')
print('Log of customers')
print(Fund_instance.statistics.df)

print('\nThe total sold goods')
print(round(Fund_instance.statistics.df.select_dtypes(include=np.number)['products'].sum(),2), "goods")  # Seleproducts)


# Output the graph of the most common number of products (hist)
def output_hist_graph():
    plt.hist(Fund_instance.statistics.df.iloc[1:, 0], bins = 30, density=True)
    plt.xlabel("products, item")
    
    plt.title('The graphic of the most common number of products')
    plt.show()


# Group by 'hours' and calculate total price for each hour
def output_bar_graph():
    hourly_price = Fund_instance.statistics.df.groupby('hours')['products'].sum()
    
    plt.figure(figsize=(8, 6))
    plt.bar(hourly_price.index, hourly_price.values, align='center', alpha=0.7)
    plt.xlabel('Hour')

    plt.ylabel('Total Amount')
    plt.title('Total Amount Earned for Each Hour')

    plt.grid(True)
    plt.show()
    
    
output_hist_graph()
output_bar_graph()