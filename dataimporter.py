import re
import time
import datetime
import math
import sys

# Stations File Column Indexes
STATION_ID = 0
COUNTRY = 8

# Data File Column Indexes
STATION_ID = 0
DATE = 2
TEMPERATURE = 3

##
start_year = 1980

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
            if (line[0] <> '%'):
                splitted = re.split(r'\t+', line)
                station_countries[int(splitted[STATION_ID])] = splitted[COUNTRY].strip()
                countries.add(splitted[COUNTRY].strip())

    return (countries, station_countries)

# Load Temperature data (AVG, MIN, MAX) and ....
def import_from_file(location, countries, station_countries):
    with open(location) as data:
        test_limit = 1000

        country_monthly_values_all_stations = dict()

        print "Reading file: " + location

        # Get values as tuples (year,month,value) in a dictionary of countries
        for line in data:
            test_limit -= 1
            if (line[0] <> '%'):
                splitted = re.split(r'\t+', line)
                country = station_countries[int(splitted[STATION_ID])]
                year, month = convert_date(splitted[DATE])
                temp = splitted[TEMPERATURE]
                
                if (year > start_year):
                    try:
                        country_monthly_values_all_stations[country].append((year, month, temp))
                    except KeyError:
                        country_monthly_values_all_stations[country] = [(year,month,temp)]

            if (test_limit == 0):
                break

        country_monthly_average_values = dict()

        # Get average temperature from all stations for each month
        print "Aggregating..."
        i = 0
        for country in countries:
            i += 1
            sys.stdout.write('\r' + str(i) + ' out of ' + str(len(countries)) + ' countries')
            sys.stdout.flush()

            try:
                values = country_monthly_values_all_stations[country]
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
                        # Average temperature value for a month
                
                average_monthly_values = dict()
                for item in monthly_values.iteritems():
                    year_month_tuple = item[0]
                    temperature_values = item[1]

                    try:
                        country_monthly_average_values[country].append((year_month_tuple, (sum(temperature_values) / len(temperature_values))))
                    except KeyError:
                        country_monthly_average_values[country] = [(year_month_tuple, sum(temperature_values) / len(temperature_values))]
            except:
                pass

        print '\n'

    return average_monthly_values

ts = time.time()
print "Start: " + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

countries, station_countries = import_stations_from_file('../tavg/site_detail.txt')
country_monthly_average_values = import_from_file('../tavg/data.txt', countries, station_countries)
country_monthly_max_values = import_from_file('../tmax/data.txt', countries, station_countries)
country_monthly_min_values = import_from_file('../tmin/data.txt', countries, station_countries)

# for country in countries:
#     try:
#         print country + ": " + str(len(country_monthly_average_values[country])) + " months"
#     except KeyError:
#         print country + ": No data"

#print country_monthly_average_values['Somalia']

ts = time.time()
print "Finish: " + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
