import re
import time
import datetime
import math
import sys
import csv

# Stations File Column Indexes
STATION_ID = 0
COUNTRY = 8

# Data File Column Indexes
STATION_ID = 0
DATE = 2
TEMPERATURE = 3

##
start_year = 1980
end_year = 2017

test_limit = -1


# For example, in monthly data: 
# 
# 	January 2005 = 2005 + (1 - 0.5) / 12 = 2005.042
# 	June 2008 = 2008 + (6 - 0.5) / 12 = 2008.458
def convert_date(date):
    value = math.modf(float(date))
    year = int(value[1])
    month = int(round(value[0] * 12 + 0.5))

    return (year, month)

def import_stations_from_file(location):
    countries = set()
    station_countries = dict()

    with open(location) as stations:
        for line in stations:
            # Skip preamble
            if (line[0] != '%'):
                splitted = re.split(r'\t+', line)
                station_countries[int(splitted[STATION_ID])] = splitted[COUNTRY].strip()
                countries.add(splitted[COUNTRY].strip())

    return (countries, station_countries)

# Load Temperature data (AVG, MIN, MAX) and ....
def import_from_file(location, countries, station_countries):
    global test_limit
    
    with open(location) as data:
        monthly_values_all_stations = dict()

        print ("Reading file: " + location)

        # Get values as tuples (year,month,value) in a dictionary of countries
        for line in data:
            test_limit -= 1
            if (line[0] != '%'):
                splitted = re.split(r'\t+', line)
                country = station_countries[int(splitted[STATION_ID])]
                year, month = convert_date(splitted[DATE])
                temp = splitted[TEMPERATURE]
                
                if (year > start_year):
                    try:
                        monthly_values_all_stations[country].append((year, month, temp))
                    except KeyError:
                        monthly_values_all_stations[country] = [(year,month,temp)]

            if (test_limit == 0):
                break

        country_monthly_average_values = dict()

        # Get average temperature from all stations for each month
        print ("Aggregating...")
        i = 0
        for country in countries:
            i += 1
            sys.stdout.write('\r' + str(i) + ' out of ' + str(len(countries)) + ' countries')
            sys.stdout.flush()

            try:
                values = monthly_values_all_stations[country]
                monthly_values = dict()
                # All temperature values from all stations for a month
                for value in values:
                    year = value[0]
                    month = value[1]
                    temp = value[2]
                    try:
                        monthly_values[(year,month)].append(float(temp))
                    except KeyError:
                        monthly_values[(year,month)] = [float(temp)]
                
                for item in monthly_values.items():
                    year_month_tuple = item[0]
                    temperature_values = item[1]

                    try:
                        country_monthly_average_values[country][year_month_tuple]= sum(temperature_values) / len(temperature_values)
                    except KeyError:
                        country_monthly_average_values[country] = dict()
                        country_monthly_average_values[country][year_month_tuple]= sum(temperature_values) / len(temperature_values)
            except KeyError:
                pass

        print ('\n')

    return country_monthly_average_values

ts = time.time()
print ("Start: " + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

all_countries = set()

countries, station_countries = import_stations_from_file('../tavg/site_detail.txt')
for country in countries:
    all_countries.add(country)
country_monthly_average_values = import_from_file('../tavg/data.txt', countries, station_countries)
countries, station_countries = import_stations_from_file('../tmax/site_detail.txt')
for country in countries:
    all_countries.add(country)
country_monthly_max_values = import_from_file('../tmax/data.txt', countries, station_countries)
countries, station_countries = import_stations_from_file('../tmin/site_detail.txt')
for country in countries:
    all_countries.add(country)
country_monthly_min_values = import_from_file('../tmin/data.txt', countries, station_countries)

with open('weather_data.csv', 'w', newline='') as csvfile:
    for country in all_countries:
        try:
            average = country_monthly_average_values[country]
            maximum = country_monthly_min_values[country]
            minimum = country_monthly_max_values[country]

            for year in range(start_year, end_year):
                for month in range(1,12):
                    try:
                        monthly_average = average[(year, month)]
                    except KeyError:
                        monthly_average = "*"
                    try:
                        monthly_max = maximum[(year, month)]
                    except KeyError:
                        monthly_max = "*"
                    try:
                        monthly_min = minimum[(year, month)]
                    except KeyError:
                        monthly_min = "*"
                    
                    csvfile.write(country + ";")
                    csvfile.write(str(year) + ";")
                    csvfile.write(str(month) + ";")
                    csvfile.write(str(monthly_average) + ";")
                    csvfile.write(str(monthly_max) + ";")
                    csvfile.write(str(monthly_min) + ";")
                    csvfile.write("\n")

        except KeyError:
            print("no data found for " + country)
            continue

ts = time.time()
print ("Finish: " + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
