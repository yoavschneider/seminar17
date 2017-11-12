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
countries = set()

# For example, in monthly data: 
# 
# 	January 2005 = 2005 + (1 - 0.5) / 12 = 2005.042
# 	June 2008 = 2008 + (6 - 0.5) / 12 = 2008.458
def convert_date(date):
    value = math.modf(float(date))
    year = int(value[1])
    month = int(round(value[0] * 12 + 0.5))

    return (year, month)

# Dictionary containing station_id to country mapping
station_countries = dict()

ts = time.time()
print "Start: " + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

with open('../avg/site_detail.txt') as stations:
    for line in stations:
        # Skip preamble
        if (line[0] <> '%'):
            splitted = re.split(r'\t+', line)
            station_countries[int(splitted[STATION_ID])] = splitted[COUNTRY].strip()
            countries.add(splitted[COUNTRY].strip())

# Load Temperature data (AVG, MIN, MAX) and ....
with open('../avg/data.txt') as data:
    test_limit = 10000000

    country_monthly_values_all_stations = dict()
    country_monthly_average = dict()

    print "Reading file..."

    # Get values as tuples (year,month,value) in a dictionary of countries
    for line in data:
        test_limit -= 1
        if (line[0] <> '%'):
            splitted = re.split(r'\t+', line)
            country = station_countries[int(splitted[STATION_ID])]
            year, month = convert_date(splitted[DATE])
            temp = splitted[TEMPERATURE]
            
            try:
                country_monthly_values_all_stations[country].append((year, month, temp))
            except KeyError:
                country_monthly_values_all_stations[country] = [(year,month,temp)]

        if (test_limit == 0):
            break

    country_monthly_average_values = dict()

    # Get average temperature from all stations
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


    print "\n"


print country_monthly_average_values['Somalia']

ts = time.time()
print "Finish: " + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
