# -----------------sigmoid---------------- #
# ---------------- MSE ---------------- #
from mnist import MNIST
import numpy as np
# import matplotlib.pyplot as plt
from scipy.special import expit

mndata = MNIST('sample')
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()


def feature_standardization(data, data_size):

    mean = np.mean(data, axis=0)
    standard_deviation = np.std(data, axis=0)
    data = data - np.dot(np.ones(data_size).reshape(data_size, 1), mean.reshape(1, 784))

    for i in range(784):
        if standard_deviation[i] != 0:
            sd = np.multiply(np.ones(data_size), standard_deviation[i])
            data[:, i] = np.divide(data[:, i], sd)

    return data


def init_params(num_input_features, num_hidden_units, num_output_class):

    u = np.random.uniform(-0.5, 0.5, size=num_hidden_units * num_input_features
                          ).reshape(num_hidden_units, num_input_features)
    v = np.random.uniform(-0.5, 0.5, size=num_output_class * num_hidden_units
                          ).reshape(num_output_class, num_hidden_units)
    bias = np.random.uniform(-0.25, 0.25, size=num_hidden_units).reshape(num_hidden_units, 1)

    return u, v, bias


def encode_labels(y, num_labels):

    onehot = np.zeros(num_labels * len(y)).reshape(len(y), num_labels)

    for i in range(len(y)):
        onehot[i, y[i]] = 1.0

    return onehot


def forward(data, u, v, bias, num_hidden_units):

    hidden_layer_input = np.dot(data, np.transpose(u)).reshape(num_hidden_units, 1) + bias
    hidden_layer_output = expit(hidden_layer_input)
    score = np.dot(v, hidden_layer_output)

    return hidden_layer_output, score


def gradient(data, label, hidden_layer_output, score, v, num_input_features, num_hidden_units, num_output_class):

    delta = score - label.reshape(10, 1)
    v_grad = np.dot(delta.reshape(num_output_class, 1), hidden_layer_output.reshape(1, num_hidden_units))
    x1 = np.dot(np.transpose(v), delta.reshape(num_output_class, 1))
    x2 = np.multiply(hidden_layer_output, 1 - hidden_layer_output).reshape(num_hidden_units, 1)
    bias_grad = np.multiply(x1, x2)
    u_grad = np.dot(bias_grad, data.reshape(1, num_input_features))

    return v_grad, u_grad, bias_grad


def update_params(u, v, bias, u_grad, v_grad, bias_grad, learning_rate):

    u = u - learning_rate * u_grad
    v = v - learning_rate * v_grad
    bias = bias - learning_rate * bias_grad

    return u, v, bias


def cost(score, label):

    loss = .5 * np.sum(np.square(score - label.reshape(10, 1)))

    return loss


def error(data, label, u, v, bias, data_size, num_hidden_units):

    misclassification = 0

    for i in range(data_size):
        _, score = forward(data[i], u, v, bias, num_hidden_units)
        predicted_label = np.argmax(score)
        if predicted_label != label[i]:
            misclassification += 1

    return misclassification / data_size


def train(data, label, integer_label, test_data, test_label, integer_test_label, batch_size, epoch, learning_rate, num_input_features, num_hidden_units, data_size):

    u, v, bias = init_params(num_input_features, num_hidden_units, 10)
    empirical_loss = np.zeros(epoch)
    train_error = np.zeros(epoch)
    test_error = np.zeros(epoch)
    empirical_batch_loss = np.zeros(data_size // batch_size)

    for i in range(epoch):
        for j in range(data_size // batch_size):
            batch_loss = 0
            v_grad = 0
            bias_grad = 0
            u_grad = 0
            for k in range(batch_size):
                h, score = forward(data[j * batch_size + k], u, v, bias, num_hidden_units)
                batch_loss += cost(score, label[j * batch_size + k])
                v_delta, u_delta, bias_delta = gradient(data[j * batch_size + k], label[j * batch_size + k], h, score, v, num_input_features,num_hidden_units, 10)
                v_grad += v_delta
                bias_grad += bias_delta
                u_grad += u_delta
            empirical_batch_loss[j] = batch_loss / batch_size
            print(np.max(v_grad/batch_size))
            print(np.max(u_grad / batch_size))
            print(np.max(bias_grad / batch_size))
            input()
            u, v, bias = update_params(u, v, bias, u_grad / batch_size, v_grad / batch_size, bias_grad / batch_size, learning_rate)
        empirical_loss[i] = np.mean(empirical_batch_loss)
        train_error[i] = error(data, integer_label, u, v, bias, 60000, num_hidden_units)
        test_error[i] = error(test_data, integer_test_label, u, v, bias, 10000, num_hidden_units)
        # learning_rate = learning_rate * 0.99
        permutation = np.random.permutation(data_size)
        data = [data[i] for i in permutation]
        label = [label[i] for i in permutation]

    return u, v, bias, empirical_loss, test_error, train_error


train_data = feature_standardization(images, 60000)
test_data = feature_standardization(test_images, 10000)
encoded_labels = encode_labels(labels, 10)
encoded_test_labels = encode_labels(test_labels, 10)
u, v, bias, loss, test_error, train_error = train(train_data, encoded_labels, labels, test_data, encoded_test_labels, test_labels, 100, 100, 0.5, 784, 60, 60000)
print("min error on train:", np.amin(train_error))
print("min error on test:", np.amin(test_error))
np.save("loss_1_2_MSE.npy", loss)
np.save("train_error_1_2_MSE.npy", train_error)
np.save("test_error_1_2_MSE", test_error)