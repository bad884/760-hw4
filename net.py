from scipy.io import arff

import collections
import math
import random
import sys


def create_instances_from_arff_data(train_data, attr_names, attr_types, attr_values):
    instances = []
    for instance in train_data:
        instance = list(instance)
        attrs = instance[:-1]
        attributes = []
        for attr_value in attrs:
            if attr_types[attr_names[attrs.index(attr_value)]] == 'numeric':
                attributes.append(float(attr_value))
            else:
                attributes.append(attr_value)
        class_label = bytes.decode(instance[-1])
        instance = Instance(attributes, class_label)
        instances.append(instance)
        # print(instance)
    return instances

def standardize_instances(instances, attr_names, attr_types, attr_values):
    attr_sums = collections.OrderedDict()
    attr_sum_sq_diffs = collections.OrderedDict()
    for attr_name, _type in attr_types.items():
        if _type == 'numeric':
            attr_sums[attr_name] = 0.0
            attr_sum_sq_diffs[attr_name] = 0.0

    # Sum up values
    for instance in instances:
        for attr_value in instance.attributes:
            if attr_names[instance.attributes.index(attr_value)] in attr_sums.keys():
                attr_sums[attr_names[instance.attributes.index(attr_value)]] += attr_value
    # print()
    # for attr_name, _sum in attr_sums.items():
    #     print('name: {}\tsum = {}'.format(attr_name, _sum))

    # Calculate means
    attr_means = collections.OrderedDict()
    for attr_name, _sum in attr_sums.items():
        attr_means[attr_name] = float(attr_sums[attr_name] / len(instances))
    # print()
    # for attr_name, _mean in attr_means.items():
    #     print('name: {}\tmean = {}'.format(attr_name, _mean))

    # Find the sum of the squared differences
    for instance in instances:
        for attr_value in instance.attributes:
            if attr_names[instance.attributes.index(attr_value)] in attr_means.keys():
                attr_sum_sq_diffs[attr_names[instance.attributes.index(attr_value)]] += \
                        (attr_value - attr_means[attr_names[instance.attributes.index(attr_value)]])**2
    # print()
    # for attr_name, _sum_sq_diff in attr_sum_sq_diffs.items():
    #     print('name: {}\tsum_sq_diff = {}'.format(attr_name, _sum_sq_diff))

    # Calculate standard deviation
    attr_std_devs = collections.OrderedDict()
    for attr_name, _sum in attr_sum_sq_diffs.items():
        attr_std_devs[attr_name] = math.sqrt( float(attr_sum_sq_diffs[attr_name] / len(instances)) )
    # print()
    # for attr_name, _std_dev in attr_std_devs.items():
    #     print('name: {}\tstd_dev = {}'.format(attr_name, _std_dev))

    # Update instances with standardized values
    std_instances = []
    for instance in instances:
        std_attr_values = []
        for attr_value in instance.attributes:
            if attr_names[instance.attributes.index(attr_value)] in attr_std_devs.keys():
                new_attr_value = (attr_value - attr_means[attr_names[instance.attributes.index(attr_value)]]) / \
                                        attr_std_devs[attr_names[instance.attributes.index(attr_value)]]
                std_attr_values.append(new_attr_value)
            else:
                std_attr_values.append(attr_value)
        std_instance = Instance(std_attr_values, instance.label)
        std_instances.append(instance)
    return std_instances


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

    def __init__(self, labels, attr_names, attr_values, train_instances):
        self.labels = labels
        self.attr_names = attr_names
        self.attr_values = attr_values
        self.train_instances = train_instances


class NeuralNet(Net):
    ''' '''

    def __init__(self, labels, attr_names, attr_values, train_instances):
        Net.__init__(self, labels, attr_names, attr_values, train_instances)


class Logistic(Net):
    ''' '''

    def __init__(self, labels, attr_names, attr_values, train_instances):
        Net.__init__(self, labels, attr_names, attr_values, train_instances)



if __name__ == '__main__':

    num_epochs = int(sys.argv[2])
    learning_rate = float(sys.argv[1])

    # Load training ARFF data
    training_set_path = str(sys.argv[3])
    train_data, train_meta = arff.loadarff(training_set_path)

    # Load testing ARFF data
    testing_set_path = str(sys.argv[4])
    test_data, test_meta = arff.loadarff(testing_set_path)

    # Create reference data structures for labels and attributes
    labels = list(train_meta['class'][1])
    attr_names = train_meta.names()[:-1]

    attr_types = collections.OrderedDict()
    for attr in train_meta:
        if attr != 'class':
            if train_meta[attr][0] == 'numeric':
                attr_types[attr] = 'numeric'
            else:
                attr_types[attr] = 'nominal'

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

    # Create and standardize training instances
    train_instances = create_instances_from_arff_data(train_data, attr_names, attr_types, attr_values)
    std_train_instances = standardize_instances(train_instances, attr_names, attr_types, attr_values)

    # for instance in std_train_instances:
        # print(instance)

    # Randomize training instances


    # Create and standardize testing instances
    # test_instances = create_instances_from_arff_data(test_data, attr_types)



    # Logistic Regression
    if sys.argv[5] == 'l':
        lr = Logistic(labels, attr_names, attr_values, std_train_instances)
















    # Neural Net
    elif sys.argv[5] == 'n':
        nn = NeuralNet(labels, attr_names, attr_values, std_train_instances)


