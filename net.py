from scipy.io import arff

import collections
import math
import random
import sys


def create_instances_from_arff_data(train_data):
    train_instances = []
    for instance in train_data:
        instance = list(instance)
        attrs = instance[:-1]
        attributes = []
        for attribute in attrs:
            attributes.append(attribute)
        class_label = bytes.decode(instance[-1])
        instance = Instance(attributes, class_label)
        train_instances.append(instance)
        # print(instance)
    return train_instances


class Instance(object):
    ''' An instance of the dataset. '''

    def __init__(self, attributes, label):
        self.attributes = attributes
        self.label = label

    def __repr__(self):
        sb = ''
        for attribute in self.attributes:
            sb += str(attribute) + ','
        sb += str(self.label)
        return sb


class Net(object):
    ''' An abstract net. '''

    def __init__(self, labels, attr_names, attr_values):
        self.labels = labels
        self.attr_names = attr_names
        self.attr_values = attr_values


class NeuralNet(Net):
    ''' '''

    def __init__(self, labels, attr_names, attr_values):
        Net.__init__(self, labels, attr_names, attr_values)


class Logistic(Net):
    ''' '''

    def __init__(self, labels, attr_names, attr_values):
        Net.__init__(self, labels, attr_names, attr_values)



if __name__ == '__main__':

    # Load training ARFF data and create instances
    training_set_path = str(sys.argv[3])
    train_data, train_meta = arff.loadarff(training_set_path)
    train_instances = create_instances_from_arff_data(train_data)
    # Load testing ARFF data and create instances
    testing_set_path = str(sys.argv[4])
    test_data, test_meta = arff.loadarff(testing_set_path)
    test_instances = create_instances_from_arff_data(test_data)

    # Create reference data structures for labels and attributes
    labels = list(train_meta['class'][1])
    attr_names = train_meta.names()[:-1]
    attr_values = collections.OrderedDict()
    for attr in train_meta:
        if attr != 'class':
            if train_meta[attr][0] == 'numeric':
                attr_values[attr] = 'numeric'
            else:
                attr_values[attr] = train_meta[attr][-1]
    # print(labels)
    # print(attr_names)
    # print(attr_values)


    # Logistic Regression
    if sys.argv[5] == 'l':
        lr = Logistic(labels, attr_names, attr_values)
















    # Neural Net
    elif sys.argv[5] == 'n':
        nn = NeuralNet(labels, attr_names, attr_values)







































