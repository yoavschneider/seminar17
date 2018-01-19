
from refugee_predictor import train_predictor,save_plot, load_data, load_model

def use_predictor():
    lookback = 6
    forecast = 3
    plot_start = 2850
    plot_end = 2904
    number_of_epochs = 500
    steps = 50
    input_dimension = 5 + lookback * 5
    train_amount = 300
    values_per_country = 444

    #here we enter the country we want to plot
    x_train,y_train,x,y,scX,scY = train_predictor(lookback,forecast,plot_start,plot_end,number_of_epochs,steps,input_dimension,train_amount,values_per_country)

    # PATH
    path = './models/new/model'

    # Or load model
    model = load_model(path + '.h5')
    print(model.summary())

    # Train and plot results
    for i in range(50, steps + 50):
        model.fit(x_train, y_train, epochs=number_of_epochs, batch_size=train_amount, verbose=2)

    # Plot results
    save_plot(model,input_dimension, x, y, scX, scY, path, forecast, plot_start, plot_end)

    # Save model
    model.save(path + '.h5')
    print("Saved model to: " + path + '.h5')

def get_countryData(name):
    #To finish!
     #   get countries from loaddata,
    # save data in new array, leave sorted
 #return data array

 #give that array to the train fct so it trains, plots ,... with right data


use_predictor()



