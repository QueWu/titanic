import pandas as pd
import numpy as np

def patches(inFrame):
    #mean age to fill nan
    avg = int(sum(inFrame['Age'].fillna(0).tolist())/int(inFrame['Age'].count())*100)/100
    #avg = int(sum(inFrame['Age'].fillna(0).tolist())//int(inFrame['Age'].count()))
    inFrame['Age'] = inFrame['Age'].fillna(avg)
    #male->0 female->1
    inFrame['Sex'] = inFrame['Sex'].replace('male',0)
    inFrame['Sex'] = inFrame['Sex'].replace('female',1)
    #fares nan->0
    inFrame['Fare'] = inFrame['Fare'].fillna(0)
    #C->1 Q->2 S->3
    inFrame['Embarked'] = inFrame['Embarked'].replace('C',1)
    inFrame['Embarked'] = inFrame['Embarked'].replace('Q',2)
    inFrame['Embarked'] = inFrame['Embarked'].replace('S',3)
    return inFrame

def extraction(inFile):
    frame = patches(pd.DataFrame(pd.read_csv(inFile,delimiter=',',index_col ="PassengerId")))
    mass_list = []
    ylabel = frame['Survived'].tolist()
    for index, rows in frame.iterrows():
        elements = [rows.Pclass, rows.Sex, \
                    rows.Age, rows.SibSp, rows.Parch]
        '''
        elements = [rows.Pclass, rows.Sex, \
                    rows.Age, rows.SibSp, rows.Parch, \
                    rows.Fare, rows.Embarked]
        '''
        mass_list.append(elements)
    #for i in mass_list:print(i)
    return (ylabel,mass_list)
#extraction('train.csv')
