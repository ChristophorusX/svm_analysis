
def parse_file(filename):
    with open(filename, 'r') as f:
        with open(filename + '_parsed', 'w+') as parsedf:
            for line in f:
                feature_list = line.split(',')
                label = 2 * int(feature_list[-1]) - 1
                new_line = str(label).rstrip()
                for i in range(0, len(feature_list) - 1):
                    concat = ' ' + str(i + 1) + ':' + str(feature_list[i])
                    new_line += concat
                parsedf.write(new_line + '\n')


if __name__ == '__main__':
    parse_file('spambase_test')
    parse_file('spambase_train')
