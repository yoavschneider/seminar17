from refugee_predictor import train_model,save_plot, load_model, load_data, create_model, split_data, get_scalers, get_country_data
from datetime import datetime, timedelta

lookback = 6
forecast = 6

model_path = './models/no_weather/model'
data_path = './models/no_weather/CLEAN_DATA_2_1_NO_WEATHER.csv'
plot_path = './models/no_weather/SVG/'

number_of_epochs = 100
steps = 1
input_dimension = 5 + lookback * 5
train_amount = 180
values_per_country = 216

def create_predictor():
    model = create_model(input_dimension,lookback,forecast);
    model.save(model_path + '.h5')


def train_predictor(model, x, y):
    # Train the model
    model = train_model(model,x,y,lookback,forecast,number_of_epochs,steps,input_dimension,train_amount,values_per_country)

    # Save model
    model.save(model_path + '.h5')
    print("Saved model to: " + model_path + '.h5')

def load_predictor_and_data():
    predicator = load_model(model_path + '.h5')
    x, y, _, _ = load_data(data_path,lookback,forecast,values_per_country)
    return predicator,x,y

def predict_and_plot_all_countries():
    model,_,_ = load_predictor_and_data()
    x, y, country_encoder, disaster_encoder = load_data(data_path,lookback,forecast,values_per_country)
    x_train, y_train, _, _ = split_data(x, y, lookback, forecast, train_amount,values_per_country)
    scaler_x, scaler_y = get_scalers(x_train, y_train)

    for country in country_encoder:
        x_country, y_country = get_country_data(x,y,country_encoder,country)
        save_plot(model, x_country, y_country, scaler_x, scaler_y, disaster_encoder, plot_path + country, forecast, values_per_country)


#start_time = datetime.now()
#end_time = start_time + timedelta(hours=12)

predict_and_plot_all_countries()