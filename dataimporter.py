import re

STATION_ID = 0
COUNTRY = 8

with open('../avg/site_detail.txt') as stations:
    for line in stations:
        if (line[0] == '%'):
            # skip preamble
            1==1
        else:
            splitted = re.split(r'\t+', line)
            print splitted[STATION_ID] + ", " + splitted[COUNTRY]

print 'file opened successfully'



