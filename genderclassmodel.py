
import numpy as np
import sklearn, scipy, csv

#data = np.loadtxt('/Users/jcauteru/Desktop/Kaggle/train.csv', delimiter = ',', skiprows= 1, usecols= (1,2))


infile = csv.reader(open('/Users/jcauteru/Desktop/Kaggle/train.csv', 'rb')) #Load in the csv file
print infile
header = infile.next()
data=[]
for row in infile:
    data.append(row)
data = np.array(data)
print header
print data[:,:]

vars = ['survived', 'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'room_count']

for record in data[:,3]:
    if record == 'male': record = 1
    else: record = 2

rooms = []
for record in data[:,9]:
    hold = [record.count(' ')]
    rooms.append(hold)

room_var = np.array(rooms)
array2 = np.hstack((data,room_var))

for record in array2[:,:]:
    print record

#test_file_obect = csv.reader(open('../csv/test.csv', 'rb'))
#open_file_object = csv.writer(open("../csv/genderclasspricebasedmodelpy.csv", "wb"))

#header = test_file_obect.next()

#First thing to do is bin up the price file



