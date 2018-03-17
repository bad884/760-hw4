from scipy.io import arff

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
            attributes.append(bytes.decode(attribute))
        class_label = bytes.decode(instance[-1])
        instance = Instance(attributes, class_label)
        train_instances.append(instance)
        # print(instance)
    return train_instances


class Instance(object):
    ''' An instance. '''

    def __init__(self, attributes, label):
        self.attributes = attributes
        self.label = label

    def __repr__(self):
        sb = ''
        for attribute in self.attributes:
            sb += str(attribute) + ','
        sb += str(self.label)
        return sb


if __name__ == '__main__':

    # Load training ARFF data and create instances
    training_set_path = str(sys.argv[1])
    train_data, train_meta = arff.loadarff(training_set_path)
    train_instances = create_instances_from_arff_data(train_data)

    attr_values = []
    for attr in train_meta:
        attr_values.append(len(train_meta[attr][-1]))
    attr_names = train_meta.names()[:-1]
    labels = list(train_meta['class'][1])

    CV = False
    if CV:
        test_instances = []
        for i in range(14):
            random.shuffle(train_instances)
            test_instances.append(train_instances.pop(i))
    else:
        testing_set_path = str(sys.argv[2])
        test_data, test_meta = arff.loadarff(testing_set_path)
        test_instances = create_instances_from_arff_data(test_data)

    # Use bayes or TAN?
    bayes = str(sys.argv[3])
    tan = False
    if bayes is 't':
        tan = True

    if tan:
        tan = TAN(train_instances, attr_names, attr_values, labels)
        # tan.print_attrs_label_counts()
        # tan.print_label_counts()
        # tan.print_cmi_matrix()

        if not CV:
            # Print parents
            for idx, attr_name in enumerate(tan.attr_names):
                if idx != 0:
                    print(attr_name + ' ' + tan.get_parent_attr(attr_name) + ' class')
                else:
                    print(attr_name + ' class')
            print('')

        # Classify each test case
        num_correctly_classified = 0
        # for instance in test_instances[:1]:
        for instance in test_instances:
            classification = tan.classify(instance)
            if not CV:
                print(classification.label + ' ' + instance.label + ' ' + str(classification.prob))
            if classification.label == instance.label:
                num_correctly_classified += 1

        if not CV:
            print('\n' + str(num_correctly_classified))
        else:
            print(str(num_correctly_classified))

    else:
        nb = NaiveBayes(train_instances, attr_names, attr_values, labels)
        # nb.print_attrs_label_counts()
        # nb.print_label_counts()

        if not CV:
            # Print parents
            for attr_name in nb.attr_names:
                print(attr_name + ' class')
            print('')

        # Classify each test case
        num_correctly_classified = 0
        for instance in test_instances:
            classification = nb.classify(instance)
            if not CV:
                print(classification.label + ' ' + instance.label + ' ' + str(format(classification.prob, '.12f')))
            if classification.label == instance.label:
                num_correctly_classified += 1

        if not CV:
            print('\n' + str(num_correctly_classified))
        else:
            print(str(num_correctly_classified))






