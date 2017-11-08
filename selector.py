import re

def find_validation_error():
    with open('experiments/validation_error.txt', 'w+') as recordf:
        largest_accuracy = 0
        corresponding_file = ''
        for d in range(1, 5):
            index_list = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]
            for k in index_list:
                filename = 'experiments/d=' + str(d) + '_k=' + str(k) + '_out.txt'
                with open(filename, 'r') as f:
                    for line in f:
                        if "Cross Validation Accuracy" in line:
                            accuracy = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                            if float(accuracy[0]) > largest_accuracy:
                                largest_accuracy = float(accuracy[0])
                                corresponding_file = filename
                            message = filename + ':\n' + line + '\n'
                            recordf.write(message)
        recordf.write('The best accuracy is ' + str(largest_accuracy) + ' at ' + corresponding_file)


if __name__ == '__main__':
    find_validation_error()
