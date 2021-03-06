from scipy.io import arff
from enum import Enum

import collections
import math
import numpy as np
import random
import sys


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def create_instances_from_arff_data(data, attr_names, attr_types, attr_values):
    instances = []
    for instance in data:
        instance = list(instance)
        attrs = instance[:-1]
        attributes = []
        for attr_value in attrs:
            if attr_types[attr_names[attrs.index(attr_value)]] == 'numeric':
                attributes.append(float(attr_value))
            else:
                vals_for_attr = list(attr_values[attr_names[attrs.index(attr_value)]])
                idx = vals_for_attr.index(attr_value)
                attributes.append(float(idx))
        class_label = bytes.decode(instance[-1])
        instance = Instance(attributes, class_label)
        instances.append(instance)
        # print(instance)
    return instances

def standardize_instances(train_instances, test_instances, attr_names, attr_types, attr_values):
    attr_sums = collections.OrderedDict()
    attr_sum_sq_diffs = collections.OrderedDict()
    for attr_name, _type in attr_types.items():
        if _type == 'numeric':
            attr_sums[attr_name] = 0.0
            attr_sum_sq_diffs[attr_name] = 0.0
    # Sum up values
    for instance in train_instances:
        for attr_value in instance.attributes:
            if attr_names[instance.attributes.index(attr_value)] in attr_sums.keys():
                attr_sums[attr_names[instance.attributes.index(attr_value)]] += attr_value
    # Calculate means
    attr_means = collections.OrderedDict()
    for attr_name, _sum in attr_sums.items():
        attr_means[attr_name] = float(attr_sums[attr_name] / len(train_instances))
    # Find the sum of the squared differences
    for instance in train_instances:
        for attr_value in instance.attributes:
            if attr_names[instance.attributes.index(attr_value)] in attr_means.keys():
                attr_sum_sq_diffs[attr_names[instance.attributes.index(attr_value)]] += \
                        (attr_value - attr_means[attr_names[instance.attributes.index(attr_value)]])**2
    # Calculate standard deviation
    attr_std_devs = collections.OrderedDict()
    for attr_name, _sum in attr_sum_sq_diffs.items():
        attr_std_devs[attr_name] = math.sqrt( float(attr_sum_sq_diffs[attr_name] / len(train_instances)) )
    # Update train_instances with standardized values
    std_train_instances = []
    for instance in train_instances:
        std_attr_values = []
        for attr_value in instance.attributes:
            if attr_names[instance.attributes.index(attr_value)] in attr_std_devs.keys():
                new_attr_value = (attr_value - attr_means[attr_names[instance.attributes.index(attr_value)]]) / \
                                        attr_std_devs[attr_names[instance.attributes.index(attr_value)]]
                std_attr_values.append(new_attr_value)
            else:
                std_attr_values.append(attr_value)
        std_instance = Instance(std_attr_values, instance.label)
        std_train_instances.append(std_instance)
    # Update test_instances with standardized values
    std_test_instances = []
    for instance in test_instances:
        std_attr_values = []
        for attr_value in instance.attributes:
            if attr_names[instance.attributes.index(attr_value)] in attr_std_devs.keys():
                new_attr_value = (attr_value - attr_means[attr_names[instance.attributes.index(attr_value)]]) / \
                                        attr_std_devs[attr_names[instance.attributes.index(attr_value)]]
                std_attr_values.append(new_attr_value)
            else:
                std_attr_values.append(attr_value)
        std_instance = Instance(std_attr_values, instance.label)
        std_test_instances.append(std_instance)
    return std_train_instances, std_test_instances


class Instance(object):
    def __init__(self, attributes, label):
        self.attributes = attributes
        self.label = label

    def __repr__(self):
        sb = ''
        for attribute in self.attributes:
            sb += str(attribute) + ','
        sb += str(self.label)
        return sb


class NodeType(Enum):
    INPUT = 0
    BIAS_ = 1
    HIDDEN = 2
    OUTPUT = 3


class Node(object):
    def __init__(self, node_type):
        self.node_type = node_type
        self.output_value = 0.0
        self.parent_nwps = []
        self.delta_j = 0.0

    def get_output(self):
        if self.node_type == NodeType.INPUT:
            return self.output_value
        elif self.node_type == NodeType.BIAS_:
            return 1.0
        elif self.node_type == NodeType.HIDDEN or self.node_type == NodeType.OUTPUT:
            self.output_value = 0.0
            for parent_nwp in self.parent_nwps:
                self.output_value += parent_nwp.parent_node.get_output() * parent_nwp.weight
            self.output_value = sigmoid(self.output_value)
            return self.output_value

    def __repr__(self):
        return '{}:\t\toutput = {}\t\tdelta_j = {}'.format(self.node_type.name, self.get_output(), self.delta_j)


class NodeWeightPair(object):
    def __init__(self, parent_node, child_node, weight):
        self.parent_node = parent_node
        self.child_node = child_node
        self.weight = weight
        self.delta_w = 0.0

    def update_weight(self):
        self.weight = self.weight + self.delta_w

    def __repr__(self):
        return '{} to {}:\t{}\t\t{}\t\t{}'.format(self.parent_node.node_type.name, self.child_node.node_type.name, self.parent_node.get_output(), self.weight, self.delta_w)


class Net(object):
    def __init__(self, labels, attr_names, attr_values, train_instances, learning_rate, num_epochs):
        self.labels = labels
        self.attr_names = attr_names
        self.attr_values = attr_values
        self.train_instances = train_instances
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def train(self):
        current_epoch = 1
        if DEBUG:
            self.print_net()
        while current_epoch <= self.num_epochs:
            if DEBUG:
                print('\n=====================================================================')
                print('epoch number:\t' + str(current_epoch))
            sum_cross_entropy_error = 0.0
            num_correct = num_wrong = 0

            if DEBUG:
                self.train_instances = self.train_instances[:10]
            else:
                random.shuffle(self.train_instances)

            for instance in self.train_instances:
                net_output_value = self.forward_pass(instance)
                sum_cross_entropy_error += self.calc_cross_entropy_error(instance, net_output_value)
                self.output_node.delta_j = self.calc_delta_j_output(instance, net_output_value)
                if DEBUG:
                    print('------------------------------------------------------------------------')
                    print('actual_class: {}\t\toutput: {}\toutput_node.delta_j: {}'.format(str(instance.label), str(net_output_value), str(self.output_node.delta_j)))
                self.backward_pass()
                self.update_weights()
                if DEBUG:
                    self.print_net()
                if ((net_output_value < 0.5) and (instance.label == self.labels[0])) or ((net_output_value > 0.5) and (instance.label == self.labels[1])):
                    num_correct += 1
                else:
                    num_wrong += 1
            print('{}\t{}\t{}\t{}'.format(current_epoch, sum_cross_entropy_error, num_correct, num_wrong))
            current_epoch += 1

    def calc_delta_j_output(self, instance, net_output_value):
        o = net_output_value
        y = 0.0
        if instance.label == self.labels[1]:
            y = 1.0
        return y - o

    def calc_cross_entropy_error(self, instance, net_output_value):
        o = net_output_value
        y = 0.0
        if instance.label == self.labels[1]:
            y = 1.0
        return ( (-y * math.log(o)) - ((1.0 - y) * math.log(1.0 - o)) )


class NeuralNet(Net):
    def __init__(self, labels, attr_names, attr_values, train_instances, learning_rate, num_epochs, num_hidden):
        Net.__init__(self, labels, attr_names, attr_values, train_instances, learning_rate, num_epochs)
	self.num_hidden = num_hidden

        # Create empty input nodes, update values in forward pass for each instance
        self.input_nodes = []
        bias_to_hidden_node = Node(NodeType.BIAS_)
        self.input_nodes.append(bias_to_hidden_node)
        for _ in range(len(attr_names)):
            input_node = Node(NodeType.INPUT)
            self.input_nodes.append(input_node)

        # Create hidden nodes and link it to all the input nodes with random weights
        self.hidden_nodes = []
        bias_to_output_node = Node(NodeType.BIAS_)
        self.hidden_nodes.append(bias_to_output_node)
        for _ in range(self.num_hidden):
            hidden_node = Node(NodeType.HIDDEN)
            for input_node in self.input_nodes:
                nwp = NodeWeightPair(input_node, hidden_node, random.choice([-0.01,0.01]))
                hidden_node.parent_nwps.append(nwp)
            self.hidden_nodes.append(hidden_node)

        # Create output node and link it to all the hidden nodes with random weights
        self.output_node = Node(NodeType.OUTPUT)
        for hidden_node in self.hidden_nodes:
            nwp = NodeWeightPair(hidden_node, self.output_node, random.choice([-0.01,0.01]))
            self.output_node.parent_nwps.append(nwp)

    def forward_pass(self, instance):
        for i, attr_value in enumerate(instance.attributes):
            self.input_nodes[i + 1].output_value = attr_value
        for hidden_node in self.hidden_nodes:
            hidden_node.get_output()
        return self.output_node.get_output()

    def backward_pass(self):
        for nwp in self.output_node.parent_nwps:
            hidden_node = nwp.parent_node
            nwp.delta_w = self.learning_rate * hidden_node.get_output() * self.output_node.delta_j
            hidden_node.delta_j = ( hidden_node.get_output() * (1 - hidden_node.get_output()) ) * (self.output_node.delta_j * nwp.weight)
        for hidden_node in self.hidden_nodes:
            for nwp in hidden_node.parent_nwps:
                nwp.delta_w = self.learning_rate * nwp.parent_node.get_output() * hidden_node.delta_j

    def update_weights(self):
        for nwp in self.output_node.parent_nwps:
            nwp.update_weight()
        for hidden_node in self.hidden_nodes:
            for nwp in hidden_node.parent_nwps:
                nwp.update_weight()

    def print_net(self):
        # for hidden_node in self.hidden_nodes[1:]:
        #     print
        #     for nwp in hidden_node.parent_nwps:
        #         print(nwp)
        print
        for nwp in self.output_node.parent_nwps:
            print(nwp)
        print
        print(self.output_node)


class Logistic(Net):
    def __init__(self, labels, attr_names, attr_values, train_instances, learning_rate, num_epochs):
        Net.__init__(self, labels, attr_names, attr_values, train_instances, learning_rate, num_epochs)

        # Create input nodes
        self.input_nodes = []
        bias_to_output_node = Node(NodeType.BIAS_)
        self.input_nodes.append(bias_to_output_node)
        for _ in range(len(attr_names)):
            input_node = Node(NodeType.INPUT)
            self.input_nodes.append(input_node)

        # Create output node and link it to all the input nodes with random weights
        self.output_node = Node(NodeType.OUTPUT)
        for input_node in self.input_nodes:
            nwp = NodeWeightPair(input_node, self.output_node, random.choice([-0.01,0.01]))
            self.output_node.parent_nwps.append(nwp)

    def forward_pass(self, instance):
        for i, attr_value in enumerate(instance.attributes):
            self.input_nodes[i + 1].output_value = attr_value
        net_output_value = self.output_node.get_output()
        return net_output_value

    def backward_pass(self):
        for nwp in self.output_node.parent_nwps:
            nwp.delta_w = self.learning_rate * nwp.parent_node.get_output() * self.output_node.delta_j

    def update_weights(self):
        for nwp in self.output_node.parent_nwps:
            nwp.update_weight()


if __name__ == '__main__':
    # 0      1   2     3    4    5      6
    # net.py l/n train test rate epochs num_hidden

    DEBUG = False

    num_epochs = int(sys.argv[5])
    learning_rate = float(sys.argv[4])

    # Load training ARFF data
    training_set_path = str(sys.argv[2])
    train_data, train_meta = arff.loadarff(training_set_path)

    # Load testing ARFF data
    testing_set_path = str(sys.argv[3])
    test_data, test_meta = arff.loadarff(testing_set_path)

    # Create reference data structures for labels and attributes
    labels = list(train_meta['class'][1])
    attr_names = train_meta.names()[:-1]

    attr_types = attr_values = collections.OrderedDict()
    for attr in train_meta:
        if attr != 'class':
            if train_meta[attr][0] == 'numeric':
                attr_types[attr] = 'numeric'
                attr_values[attr] = 'numeric'
            else:
                attr_types[attr] = 'nominal'
                attr_values[attr] = train_meta[attr][-1]

    # print(labels)
    # print(attr_names)
    # print(attr_values)

    # Create, standardize, and randomize training and testing instances
    train_instances = create_instances_from_arff_data(train_data, attr_names, attr_types, attr_values)
    test_instances = create_instances_from_arff_data(test_data, attr_names, attr_types, attr_values)

    std_train_instances, std_test_instances = standardize_instances(train_instances, test_instances, attr_names, attr_types, attr_values)
    if not DEBUG:
        random.shuffle(std_train_instances)
        random.shuffle(std_test_instances)

    # Logistic Regression
    if sys.argv[1] == 'l':
        model = Logistic(labels, attr_names, attr_values, std_train_instances, learning_rate, num_epochs)

    # Neural Net
    elif sys.argv[1] == 'n':
	num_hidden = int(sys.argv[6])
        model = NeuralNet(labels, attr_names, attr_values, std_train_instances, learning_rate, num_epochs, num_hidden)

    model.train()

    # Run each test instance through model
    TEST = True
    if TEST and not DEBUG:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        num_correct = 0
        num_wrong = 0
        for instance in std_test_instances:
            net_output_value = model.forward_pass(instance)
            # print('net_output_value: {}'.format(net_output_value))
            predicted_label = 0
            if net_output_value > 0.5:
                predicted_label = 1
            actual_label = 0
            if instance.label == labels[1]:
                actual_label = 1
            if predicted_label == 1:
                if actual_label == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if actual_label == 0:
                    TN += 1
                else:
                    FN += 1
            if predicted_label == actual_label:
                num_correct += 1
            else:
                num_wrong += 1
            print('{:.9f}\t{}\t{}'.format(net_output_value, predicted_label, actual_label))

        print('TP: {}\tTN: {}\tFP: {}\tFN: {}'.format(TP, TN, FP, FN))
        print('{}\t{}'.format(num_correct, num_wrong))
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        print('precision: {}\trecall: {}'.format(precision, recall))
        f1 = 2 * ( (precision * recall) / (precision + recall) )
        print(f1)




