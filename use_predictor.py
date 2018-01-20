from refugee_predictor import train_model,save_plot, load_data, load_model, create_model, split_data, get_scalers, get_country_data

lookback = 6
forecast = 6
model_path = './models/new/model'
data_path = 'ALL_DATA_2_1.csv'
plot_path = './models/new/plots/'
plot_start = 2850
plot_end = 2904
number_of_epochs = 100
steps = 10
input_dimension = 5 + lookback * 5
train_amount = 300
values_per_country = 444


def create_predicator():
    model = create_model(input_dimension,lookback,forecast);
    model.save(model_path + '.h5')

def train_predicator():
    # here we enter the country we want to plot

    # Get the data
    model = load_predicator()
    x, y, _, _ = load_data(data_path,lookback,forecast,values_per_country)

    # Train the model
    model = train_model(model,x,y,lookback,forecast,plot_start,plot_end,number_of_epochs,steps,input_dimension,train_amount,values_per_country)

    # Save model
    model.save(model_path + '.h5')
    print("Saved model to: " + model_path + '.h5')

def load_predicator():
    model = load_model(model_path + '.h5')
    return model

def predict_and_plot(country):
    model = load_predicator()
    x, y, country_encoder, disaster_encoder = load_data(data_path,lookback,forecast,values_per_country)
    x_train, y_train, _, _ = split_data(x, y, lookback, forecast, train_amount,values_per_country)
    scaler_x, scaler_y = get_scalers(x_train, y_train)

    x, y = get_country_data(x,y,country_encoder,country)

    save_plot(model, x, y, scaler_x, scaler_y, plot_path + country, forecast)


#create_predicator()
#train_predicator()
predict_and_plot("Afghanistan")
predict_and_plot("Namibia")
predict_and_plot("Nepal")
predict_and_plot("Swaziland")
predict_and_plot("Thailand")