import extractor as ex
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import itertools

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

def scale_data(inX):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(inX)
    joblib.dump(scaler,"scaler_file_stand.pkl")
    X_scaled = scaler.transform(inX)
    #If we want to make a prediction on a given X_test, we can use the same “scaler” to scale X_test as follows:
    #X_test_scaled = scaler.transform(X_test)
    return X_scaled

def custom_dump_svmlight_file(X,y,scaled_file):
    #X = X.tolist()
    featinds = [" " + str(i) + ":" for i in range(1, len(X[0])+1)]
    with open(scaled_file, 'w') as f:
        for ind, row in enumerate(X):
            f.write(str(y[ind]) + " " \
            + "".join([x for x in itertools.chain.from_iterable(zip(featinds,map(str,row))) if x]) + "\n")

def train():
    label_mass = ex.extraction('train.csv')
    y = label_mass[0]
    X = label_mass[1]
    #print(y); print(X)
    X_scaled = scale_data(X)
    #print(np.array(X).shape)
    custom_dump_svmlight_file(X_scaled, y, "libsvm_train_file3")

    clf = SVC(C=1,gamma=0.125)
    #clf = SVC(gamma=0.125)
    #clf = SVC(C=2097152, gamma=0.00006103515625)
    #clf = SVC()
    clf.fit(X_scaled,y)
    print (clf.get_params)
    joblib.dump(clf, 'round4_slim_model.pkl')

def test():
    clf = joblib.load('round4_slim_model.pkl')
    label_mass = ex.extraction('test.csv')

    scaler = joblib.load('scaler_file_stand.pkl')

    y = label_mass[0]
    X = label_mass[1]
    #print(np.array(X).shape)

    X_scaled = scaler.transform(X)
    preArr = clf.predict(X_scaled)

    print("Accuracy score: ", accuracy_score(y, preArr)*100,"%\n")
    print("precision scores")
    print("Macro: ",precision_score(y, preArr, average='macro')*100,"%")
    print("recall scores")
    print("Macro: ",recall_score(y, preArr, average='macro')*100,"%")
    print("f1 scores")
    print("Macro: ",f1_score(y, preArr, average='macro')*100,"%")
train()
test()
