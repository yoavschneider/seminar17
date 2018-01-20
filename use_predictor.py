from refugee_predictor import train_model,save_plot, load_data, load_model, create_model

lookback = 6
forecast = 3
model_path = './models/new/model'
data_path = "ALL_DATA_2_1.csv"
plot_start = 2850
plot_end = 2904
number_of_epochs = 10
steps = 10
input_dimension = 5 + lookback * 5
train_amount = 300
values_per_country = 444


def create_predicator():
    model = create_model(input_dimension,lookback,forecast);

def train_predicator():
    # here we enter the country we want to plot

    # Get the data
    x, y, country_encoder, disaster_encoder = load_data(data_path,lookback,forecast,values_per_country)

    # Train the model
    model = load_model(model_path + '.h5')
    model = train_model(model,x,y,lookback,forecast,plot_start,plot_end,number_of_epochs,steps,input_dimension,train_amount,values_per_country)
    # Save model
    model.save(model_path + '.h5')
    print("Saved model to: " + model_path + '.h5')

def get_countryData(name):
    #To finish!
     #   get countries from loaddata,
    # save data in new array, leave sorted
     #return data array
    pass
    #give that array to the train fct so it trains, plots ,... with right data

#create_predicator()
train_predicator()



