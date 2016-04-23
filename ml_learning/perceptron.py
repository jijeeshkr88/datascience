import numpy as np

training_data = np.array([[0, 0,  0], [0, 1,  1], [1, 0,  1], [1, 1,  1]])

test_data = [[1, 3], [-1, 1], [-1, -1]]


def feature_add(training_data):
    feature_row = [-1 if i == 0 else i for i in training_data[:, -1]]
    training_data[:, -1] = feature_row
    return training_data


def perceptron_training(training_data, init_weight=[0, 0, 0]):
    count_thresh = training_data.shape[0]
    weight_array = init_weight
    temp_count = 0
    # print count_thresh, temp_count
    while count_thresh > temp_count:
        for row in training_data:
            result = np.dot(row, weight_array)
            # print "point = ", row
            # print "w = ", weight_array
            # print "result = ", result
            # print "unit return", unit_step(result)
            if result > 0:
                temp_count += 1
            else:
                weight_array = weight_array + row
                temp_count = 0
            # print "count ", temp_count
            # raw_input()

    return weight_array


def perceptron_testing(test_data, weight_array):
    test_data_modified = test_data
    labels = []
    for i, v in enumerate(test_data):
        test_data_modified[i].extend([1])

    for row in test_data_modified:
        if np.dot(row, weight_array) > 0:
            labels.append(1)
        else:
            labels.append(0)
    return labels


new_data = feature_add(training_data)
weight_array = perceptron_training(new_data)
print "test_result", perceptron_testing(test_data, weight_array)
