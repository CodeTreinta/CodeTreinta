# based on practice from https://www.youtube.com/watch?v=XdJAF_InNGA
# first try on machine learning

from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn import svm

wine=datasets.load_wine()


features = wine.data
labels = wine.target

train_feats, test_feats, train_labels, test_labels = tts(features, labels, test_size=0.2)

clf = svm.SVC(kernel='linear')

# train
clf.fit(train_feats, train_labels)


# predict
predictions=clf.predict(test_feats)
print(predictions)

score=0
for i in range(len(predictions)):
    if predictions[i] == test_labels[i]:
        score+=1
print(int(score/len(predictions)*100))

# first used svm.SVC(), then tree,
# then RandomForestClassifier, lastly svm.SVC(kernel='linear')
# gotta try understand the math behind those hahahahahahaha
