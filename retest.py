from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from re import *

# read in the data from emails.csv
with open('emails.csv','r') as f: #email data from kaggle, stores in f, r is read mode
    lines = [] # create an empty array to keep track of lines
    for line in f.readlines():
        lines.append(line.strip())

header = lines[0]
header = ','.join(header.split(',')[1:])
data = lines[1:]
data = [','.join(line.split(',')[1:]) for line in data]

training, testing = train_test_split(data, test_size=0.2)

rf = RandomForestClassifier(n_estimators=500, max_depth=50)
x_train, y_train = [],[]
for line in training:
  line = line.split(',')
  d=line[:-1]
  c=line[-1]
  x_train.append(d)
  y_train.append(c)

rf.fit(x_train, y_train) 

x_test,y_test = [],[]

for line in testing:
  line = line.split(',')
  d=line[:-1]
  c=line[-1]
  x_test.append(d)
  y_test.append(c)

print(rf.score(x_test,y_test))  #Determining if its Spam or Ham







  