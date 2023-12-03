import numpy as np
import matplotlib.pyplot as plt


# function to calculate temperature at a given time
def get_temperature_at_time(t_env, t_start, coefficient, time):
    return t_env + (t_start - t_env) * np.exp(-coefficient * time)


# initial data
t_env = [30, 15]
t_start = 90
time = np.linspace(0, 5, 1000)
coefficient = 0.05
t_milk = [20, 15]


# getting girl's coffee temp
def calculate_data_girl(t_start, t_milk, coefficient, time, t_env):
    t_start_girl = 0.7 * t_start + 0.3 * t_milk
    data_girl = get_temperature_at_time(t_env, t_start_girl, coefficient, time)
    return data_girl


# getting boy's coffee temp
def calculate_data_boy(t_start, t_milk, coefficient, time, t_env):
    data_boy = get_temperature_at_time(t_env, t_start, coefficient, time)
    data_boy_after_return = 0.7 * data_boy[-1] + 0.3 * t_milk
   
    data_boy[-1] = data_boy_after_return
    return data_boy


# drawing graphs with given args
def draw_graphs(time, data, t_env_value, t_milk_value):
    plt.title(f'The graph of temp of drinks (according to time)  \nEnvironment T: {t_env_value}(°C), Milk T: {t_milk_value}(°C),')
    for item in data:
        plt.plot(time, item["data"], item["color"], label=item["label"])

    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    
    plt.grid(True)
    plt.show()


# function to draw the table
def draw_table(time, data, t_env_value, t_milk_value):    
    fig, ax = plt.subplots()

    # Select every 200th data point
    time_values = time[::200]
    girl_data = data[0]["data"][::200]
    boy_data = data[1]["data"][::200]
    
    # include the last element of the collections
    time_values = np.append(time_values, time[-1])
    girl_data = np.append(girl_data, data[0]["data"][-1])
    boy_data = np.append(boy_data, data[1]["data"][-1])

    # creating a table
    table_data = [['Time (minutes)', 'Girl Temperature (°C)', 'Boy Temperature (°C)']]
    for i in range(len(time_values)):
        table_data.append([f'{time_values[i]:.2f}', f'{girl_data[i]:.2f}', f'{boy_data[i]:.2f}'])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.1, 0.2, 0.2])

    # set up table design
    table.auto_set_font_size(False)
    table.scale(2.5, 2)
    table.set_fontsize(10)    

    ax.axis('off')

    plt.title(f'Table of coffee temperature \n Environment T: {t_env_value}(°C) Milk T: {t_milk_value}(°C)')
    plt.show()

    
# setting some different situations of milk condition
def app():
    for t_milk_value in t_milk:
        for t_env_value in t_env:
            data = [
                {
                    "data": calculate_data_girl(t_start, t_milk_value, coefficient, time, t_env_value),
                    "color": 'red',
                    "label": 'Girl'
                },
                {
                    "data": calculate_data_boy(t_start, t_milk_value, coefficient, time, t_env_value),
                    "color": 'blue',
                    "label": 'Boy'
                },
            ]
            
            draw_graphs(time, data, t_env_value, t_milk_value)
            draw_table(time, data, t_env_value, t_milk_value)


# calling our app
app()