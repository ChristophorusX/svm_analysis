import numpy as np
from sklearn import preprocessing
import sklearn.svm as sksvm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# def fetch_training_data(filename):
#     with open(filename, 'r') as f:
#         features = []
#         labels = []
#         for line in f:
#             container = line.rstrip().split(' ')
#             label = int(container[0])
#             del container[0]
#             pattern = re.compile(r"[-+]?\d+:([-+]?\d*\.\d+|[-+]?\d+)")
#             feature = []
#             for phrase in container:
#                 target = re.findall(pattern, phrase)
#                 feature.append(float(target[0]))
#             if len(feature) == 57:
#                 features.append(feature)
#                 labels.append(label)
#         features = np.array(features)
#         labels = np.array(labels)
#         return features, labels


# def fetch_testing_data(filename):
#     return fetch_training_data(filename)


# def train_svm(features, labels, c, d, ker='poly'):
#     clf = sksvm.SVC(C=c, degree=d, kernel=ker)
#     clf.fit(features, labels)
#     return clf


# def error_rate(true_labels, class_labels):
#     sample_size = class_labels.shape
#     error = np.sum(np.abs(true_labels - class_labels)) / 2
#     rate = error / sample_size
#     return rate


def fetch_data_from_raw(filename):
    data = np.loadtxt(filename, delimiter=',')
    data_train_feature = data[:3000, :57]
    data_train_label = data[:3000, 57]
    data_test_feature = data[3000:, :57]
    data_test_label = data[3000:, 57]
    scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    scaler.fit_transform(data_train_feature)
    data_train_feature = scaler.transform(data_train_feature)
    data_test_feature = scaler.transform(data_test_feature)
    return data_train_feature, data_train_label, data_test_feature, data_test_label


def cross_validation_and_test(features, labels, c, d, testing_features, true_labels, ker):
    print('Generating svc...')
    clf = sksvm.SVC(kernel=ker, C=c, degree=d)
    print('Behaving cross validation...')
    errors = np.ones(10) - cross_val_score(clf, features, labels, cv=10)
    print('Fitting the model...')
    clf.fit(features, labels)
    support_vecs = clf.support_vectors_
    print('Computing test error...')
    error = 1 - clf.score(testing_features, true_labels)
    return errors, error, np.array(support_vecs).shape[0]


def validation_with_parameters(features, labels, k_range, testing_features, true_labels, ker='poly'):
    cross_val_array = []
    test_error_array = []
    num_sup_vecs_array = []
    for d in range(1, 5):
        for k in k_range:
            c = 2**k
            print('I am about to work...')
            errors, test_error, num_sup_vecs = cross_validation_and_test(
                features, labels, c, d, testing_features, true_labels, ker)
            print('I am done working this round...')
            test_error_array.append([test_error, k, d])
            num_sup_vecs_array.append(num_sup_vecs)
            mean = errors.mean()
            deviation = errors.std()
            data = np.array([mean, mean + deviation, mean - deviation, k, d])
            cross_val_array.append(data)
            # model = train_svm(features, labels, c, d)
            # class_labels = np.array(model.predict(testing_features))
            # print(class_labels)
            # print("Error rate: ", error_rate(true_labels, class_labels))
    cross_val_array = np.array(cross_val_array)
    test_error_array = np.array(test_error_array)
    num_sup_vecs_array = np.array(num_sup_vecs_array)
    if ker == 'poly':
        np.save('cross_val_array_data_new', cross_val_array)
        np.save('test_error_array_data_new', test_error_array)
        np.save('support_vectors_data_new', num_sup_vecs_array)
    else:
        np.save('cross_val_array_kernel_data_new', cross_val_array)
        np.save('test_error_array_kernel_data_new', test_error_array)
        np.save('support_vectors_kernel_data_new', num_sup_vecs_array)


def plot_validation(filename):
    cross_val_array = np.load(filename)
    plt.style.use('ggplot')
    f, axarr = plt.subplots(4, figsize=(10, 20))
    for d in range(1, 5):
        matrix = cross_val_array[(d - 1) * 11:d * 11, :]
        axarr[d - 1].plot(matrix[:, -2], matrix[:, 0],
                      label='Average cross validation error', marker='D')
        axarr[d - 1].plot(matrix[:, -2], matrix[:, 1],
                      label='Plus standard deviation', marker='^', linestyle='-.')
        axarr[d - 1].plot(matrix[:, -2], matrix[:, 2],
                      label='Minus standard deviation', marker='v', linestyle='-.')
        axarr[d - 1].legend()
        axarr[d - 1].set_title(
            'Cross Validation Result on Polynomial Kernels with Degree ' + str(d))
        axarr[d - 1].grid(True, which='both')
    plt.tight_layout()
    if filename == 'cross_val_array_data_new.npy':
        plt.savefig('validation_result.png', dpi=200)
    else:
        plt.savefig('validation_result_kernel.png', dpi=200)
    plt.show()


def plot_against_d(validation_filename, test_filename):
    cross_val_array = np.load(validation_filename)
    test_error_array = np.load(test_filename)
    plt.style.use('ggplot')
    for d in range(1, 5):
        xaxis = [1, 2, 3, 4]
        starting_pt = 10
        arr = cross_val_array[starting_pt::11, 0]
        test_arr = test_error_array[starting_pt::11, 0]
        vali, = plt.plot(xaxis, arr, marker='D')
        tes, = plt.plot(xaxis, test_arr, marker='D')
        plt.legend([vali, tes], [
                   'Average cross validation error for k=9', 'Test error for k=9'])
        plt.title(
            'Cross Validation and Test Error Result on Polynomial Kernels')
        plt.grid(True, which='both')
    if validation_filename == 'cross_val_array_data_new.npy':
        plt.savefig('validation_result_d.png', dpi=200)
    else:
        plt.savefig('validation_result_d_kernel.png', dpi=200)
    plt.show()


def plot_sv(filename):
    num_sup_vecs = np.load(filename)
    plt.style.use('ggplot')
    num_vec = np.mean(num_sup_vecs.reshape(-1, 11), axis=1)
    xaxis = [1, 2, 3, 4]
    nump = plt.plot(xaxis, num_vec, marker='D', color='C5')
    plt.title('Average number of support vectors against degree d')
    plt.grid(True, which='both')
    plt.legend(nump, "n")
    if filename == 'support_vectors_data_new.npy':
        plt.savefig('support_vector_result.png', dpi=200)
    else:
        plt.savefig('support_vector_result_kernel.png', dpi=200)
    plt.show()


def kernel_generator(i, j):
    return lambda x1, x2: (x1.dot(x2.T))**i * (x1.dot(x2.T))**j


def kernel_G4():
    poly_kernel = [kernel_generator(i, j)
                   for i in range(1, 5) for j in range(i, 5)]

    def compute_ker_val(x1, x2):
        sum = 0
        for func in poly_kernel:
            sum += func(x1, x2)
        return sum
    return compute_ker_val


if __name__ == '__main__':
    # plt.xkcd()
    features, labels, testing_features, true_labels = fetch_data_from_raw('spambase')
    k_range = [-9, -8, -4, -2, -1, 0, 1, 2, 4, 8, 9]
    validation_with_parameters(features, labels, k_range, testing_features,
    true_labels)
    # G4_kernel = kernel_G4()
    # validation_with_parameters(
    #     features, labels, k_range, testing_features, true_labels, ker=G4_kernel)
    plot_validation('cross_val_array_data_new.npy')
    # plot_validation('cross_val_array_kernel_data_new.npy')
    # cross_val_array = np.load('cross_val_array_data_new.npy')
    # print(cross_val_array)
    # minimum = np.min(cross_val_array[:, 0])
    # print(minimum)
    # for i in range(cross_val_array.shape[0]):
    #     if cross_val_array[i, 0] == minimum:
    #         print(i + 1)
    # The best performance is at d=2, k=16
    plot_against_d('cross_val_array_data_new.npy',
                   'test_error_array_data_new.npy')
    # plot_against_d('cross_val_array_kernel_data_new.npy',
    #    'test_error_array_kernel_data_new.npy')
    plot_sv('support_vectors_data_new.npy')
    # plot_sv('support_vectors_kernel_data_new.npy')
