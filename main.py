from sklearn.ensemble import RandomForestClassifier #pip install scikit learn #rf is like funnel
from sklearn.model_selection import train_test_split 
from re import *
import pickle

# read in the data from emails.csv
with open('emails.csv','r') as f: #email data from kaggle, stores in f, r is read mode
    lines = [] # create an empty array to keep track of lines
    for line in f.readlines():
        lines.append(line.strip())
    print(len(lines))

header = lines[0]
header = header.split(',')[1:]
data = lines[1:]
data = [','.join(line.split(',')[1:]) for line in data]


train, test = train_test_split(data, test_size=0.2) #20% for test
rf = RandomForestClassifier(n_estimators=500, max_depth=50)


X_train, y_train = [], []
for line in train:
    line = line.split(',')
    X_train.append(line[:-1])
    y_train.append(line[-1])

rf.fit(X_train, y_train)

with open('jar.pkl', 'wb') as jar:
    pickle.dump(rf, jar)



def get_word_counts(text):
    text = text.lower() # convert to lowercase
    counts = [] # create blank array to store counts
    for word in header[:-1]:
        counts.append(len(findall(word, text))) # find all instances of word in text and store in counts
    return counts


teststr = """looking for medication ? we ` re the best source .
it is difficult to make our material condition better by the best law , but it is easy enough to ruin it by bad laws .
excuse me . . . : ) you just found the
best and simpliest site for
medication on the net . no perscription , easy
delivery .
private , secure , and easy .
better see rightly on a pound a week than squint on a million .
we ` ve got
anything that you will ever want .
erection treatment pills , anti - depressant pills , weight loss , and
more ! http : / / splicings . bombahakcx . com / 3 /
knowledge and human power are synonymous .
only high - quality stuff for low rates !
100 % moneyback guarantee !
there is no god , nature sufficeth unto herself in no wise hath she need of an author ."""

"""attached is a brief memo outline some of the transtion issues with hpl to aep
this is the first draft .
the itilized items currently require some more action .
please add any items and forward back to me . i will update
thanks
bob"""

spamcounts = get_word_counts(teststr)
print(rf.predict([spamcounts])) # Determining if its Spam or Ham



# get accuracy
# X_test, y_test = [], []
# for line in test:
#     line = line.split(',')
#     X_test.append(line[:-1])
#     y_test.append(line[-1])



# print("accuracy", rf.score(X_test, y_test))
