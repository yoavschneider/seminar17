import math
import sys
import csv
import re
import string

#we have dataset: (residence,origin,year,month,value)
#it was easier in txt for me

#contains all data from txt

residence=[]
origin=[]
year=[]
month=[]
value=[]
data = []
inputFile = open("rmtext.txt", "r") 

#c=0

#counter= 0
for line in csv.reader(inputFile):

	data.append(line)
	#debugs the stupid wrong
	if len(line)<5:
		#print(len(line))
		#print("damn son")
		x = line[0].split(',')
		line = x
		#print(line)
		#c+=1

	for l in range(0,len(line)):
		if l==0:
			residence.append(line[l])
			#works fine
		elif l==1:
			origin.append(line[l])
		elif l==2:
			year.append(line[l])
		elif l==3:
			month.append(line[l])
		elif l==4:
			value.append(line[l])
			#print(line[l])
			
#here we create a dictionary which gives us the data, we have to type the r,o,y,m and get val
dictionary = {}
DictVal = zip(residence,origin,year,month)
#print(len(DictVal))
for i in range(0,len(DictVal)):
	#value[i] = value[i][:-2]
	if value[i] == "":
		value[i]= "0"
	dictionary[DictVal[i]]= value[i]

#Thats how you find value of res, origin,year, month
print(dictionary['Italy','Israel','2016','February'])
#wait ~5 seconds to get result

