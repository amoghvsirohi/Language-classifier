import numpy as np
from pprint import pprint
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

def importdata():
    """
    This function helps read our data from a text file and store as a list of list
    :return: list of list, of features and label
    """
    f = open('C:\\Users\\Admin\\Desktop\\dat.dat', 'r')

    rows = []
    simp = []

    for line in f:
        store, row = line[:2], line[3:]
        store = store.split()
        row = row.split()
        store[0] = str(store[0])
        row[0] = str(store[0])
        simp.append(store)
        rows.append((row))

    return rows, simp

def listToString(s):
    """
    helper function to convert list to string
    :param s: the list to be converted
    :return: a string of the contents of the list
    """
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

def check(string, sub_str):
    """
    helper function to check if a substring is present
    :param string: The target string which is to be checked
    :param sub_str: Which substring to check on
    :return: returns a list containing true or false value for the presence of a string
    """
    dummy = []
    if (string.find(sub_str) == -1):
        dummy.append("False")
    else:
        dummy.append("True")
    return dummy

def processing(df):
    # process the given data and check if or not the given substring is present
    final = []
    sample1 = []
    sample2 = []
    sample3 = []
    sample4 = []
    sample5 = []
    sample6 = []
    sample7 = []
    sample8 = []
    sample9 = []
    sample10 = []

    for i in range (len(df)):
        store1 = df[i]
        str = listToString(store1)

        help1 = (check(str, "the"))
        help2 = (check(str, "of"))
        help3 = (check(str, "an"))
        help4 = (check(str, "to"))
        help5 = (check(str, "his"))
        help6 = (check(str, "van"))
        help7 = (check(str, "het"))
        help8 = (check(str, "oo"))
        help9 = (check(str, "ik"))
        help10 = (check(str, "ij"))
        sample1.append(listToString(help1))
        sample2.append(listToString(help2))
        sample3.append(listToString(help3))
        sample4.append(listToString(help4))
        sample5.append(listToString(help5))
        sample6.append(listToString(help6))
        sample7.append(listToString(help7))
        sample8.append(listToString(help8))
        sample9.append(listToString(help9))
        sample10.append(listToString(help10))
    final.append(sample1)
    final.append(sample2)
    final.append(sample3)
    final.append(sample4)
    final.append(sample5)
    final.append(sample6)
    final.append(sample7)
    final.append(sample8)
    final.append(sample9)
    final.append(sample10)
    return final

def entropy_target(target_attribute):
    """
    We find entropy of the target attribute variable by storing unique values in an np array -pi*log(pi) looping it
    over the size of target attribute its a helper function for information gain
    :param target_attribute: the target feature for the dataset
    :return: integer value of entropy
    """
    attributes, values = np.unique(target_attribute, return_counts=True)
    entropy = np.sum([(-values[i] / np.sum(values)) * np.log2(values[i] / np.sum(values)) for i in range(len(attributes))])
    return entropy

def informationgain(data, split_attribute_name, store_attribute_name="label"):
    """
    We find entropy of the required split attribute variable by storing unique values in an np array -pi*log(pi) looping
    it over the size of split attribute this is a helper fucntion for our decsion tree
    :param data: the dataset
    :param split_attribute_name: one of the features
    :param store_attribute_name: the target feature of the dataset
    :return: integer value of information gain
    """
    totalEntropyTarget = entropy_target(data[store_attribute_name])
    values, counts = np.unique(data[split_attribute_name], return_counts=True)
    attributeEntropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy_target(data.where(data[split_attribute_name] == values[i]).dropna()[store_attribute_name])
         for i in range(len(values))])

    informationGain = totalEntropyTarget - attributeEntropy
    return informationGain

def traintree(train_data, originaldata, features, store_attribute_name="label", parent_node=None):
    """
    We train the tree and find the the best possible split for each node and build the tree recursively
    :param train_data: the data for training
    :param originaldata: the original dataset
    :param features: the total features in the dataset
    :param store_attribute_name: the target feature
    :param parent_node: the node with the largest value
    :return: returns dictionary of dictionary to obtain a tree structure
    """
    if len(np.unique(train_data[store_attribute_name])) <= 1:
        return np.unique(train_data[store_attribute_name])[0]

    elif len(train_data) == 0:
        return np.unique(originaldata[store_attribute_name])[
            np.argmax(np.unique(originaldata[store_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node

    else:
        parent_node = np.unique(train_data[store_attribute_name])[np.argmax(np.unique(train_data[store_attribute_name], return_counts=True)[1])]

        item_values = [informationgain(train_data, feature, store_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]

        for value in np.unique(train_data[best_feature]):
            value = value
            sub_data = train_data.where(train_data[best_feature] == value).dropna()
            subtree = traintree(sub_data, dataset, features, store_attribute_name, parent_node)
            tree[best_feature][value] = subtree

        return (tree)

def test(data, tree):
    """
    To test our tree and check its accuracy
    :param data: the test dataset
    :param tree: the tree returned from training
    :return: the predicted values
    """
    dict = data.iloc[:, :-1].to_dict(orient="records")

    predicted = pd.DataFrame(columns=["predicted_values"])

    for i in range(len(data)):
        predicted.loc[i, "predicted_values"] = predicting(dict[i], tree, 1.0)
    print('The prediction accuracy is: ', (np.sum(predicted["predicted_values"] == data["label"]) / len(data)) * 100, '%')

    return predicted

def predicting(query, tree, default=1.0):
    """
    helper function for testing
    :param query: dictionary value of the converted dataframe
    :param tree: the trained tree
    :param default: to handle exception of unknown query for a feature
    :return: tree
    """
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predicting(query, result)
            else:
                return result

def train_test_split(dataset):
    """
    function to split dataset into test and train data
    :param dataset: the data set
    :return: training and testing data
    """
    training_data = dataset.iloc[:100].reset_index(drop=True)    #First 100 columns for training with 50 english and 50 dutch words
    testing_data = dataset.iloc[100:].reset_index(drop=True)     #Last column-100 columns for testing with 50 english and 50 dutch words
    return training_data, testing_data

def adaboost(dataset):
    X = dataset.drop(["label"], axis=1)
    Y = dataset["label"]

    model = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    AdaBoost = AdaBoostClassifier(base_estimator=model, n_estimators=400, learning_rate=1)

    AdaBoost.fit(X, Y)

    prediction = AdaBoost.score(X, Y)

    print('The accuracy is: ', prediction * 100, '%')

x,y = importdata()
help = processing(x)
data1 = pd.DataFrame(help)
store = data1.T
data2 = pd.DataFrame(y)
store.insert(10, "label", data2, True)
#store = shuffle(store)
#print(store)
store.to_csv("C:\\Users\\Admin\\Desktop\\final.csv", index=False, header=False) #the dataset is stored in this csv file

#input for the dataset
dataset = pd.read_csv("C:\\Users\\Admin\\Desktop\\final.csv", names=["the","of","an","to","his","van","het","oo","ik","ij","label"])
# print(dataset)
# print(informationgain(dataset,"the"))
# print(informationgain(dataset,"of"))
# print(informationgain(dataset,"an"))
# print(informationgain(dataset,"to"))
# print(informationgain(dataset,"his"))
# print(informationgain(dataset,"van"))
# print(informationgain(dataset,"het"))
# print(informationgain(dataset,"oo"))
# print(informationgain(dataset,"ik"))
# print(informationgain(dataset,"ij"))

training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]

print("----------------------Decision Tree------------------------------")
tree = traintree(training_data, training_data, training_data.columns[:-1])
pprint(tree)
print("-------------------------Accuracy--------------------------------")
final = test(testing_data, tree)
final = final.to_string(index=True)
file = open("C:\\Users\\Admin\\Desktop\\predict.txt","w")
file.write(final)
file.close()
print("-------------------------AdaBoost---------------------------------")
adaboost(dataset)