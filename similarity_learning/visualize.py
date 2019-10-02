import matplotlib.pyplot as plt


def plot_keras_history(history):
    """

    :param history:
    :return:
    """
    # the history object gives the metrics keys.
    # we will store the metrics keys that are from the training sesion.
    metrics_names = [key for key in history.history.keys() if
                     not key.startswith('val_')]

    for i, metric in enumerate(metrics_names):

        # getting the training values
        metric_train_values = history.history.get(metric, [])

        # getting the validation values
        metric_val_values = history.history.get("val_{}".format(metric), [])

        # As loss always exists as a metric we use it to find the
        epochs = range(1, len(metric_train_values) + 1)

        # leaving extra spaces to allign with the validation text
        training_text = "   Training {}: {:.5f}".format(metric,
                                                        metric_train_values[
                                                            -1])

        # metric
        plt.figure(i, figsize=(12, 6))

        plt.plot(epochs,
                 metric_train_values,
                 'b',
                 label=training_text)

        # if we validation metric exists, then plot that as well
        if metric_val_values:
            validation_text = "Validation {}: {:.5f}".format(metric,
                                                             metric_val_values[
                                                                 -1])

            plt.plot(epochs,
                     metric_val_values,
                     'g',
                     label=validation_text)

        # add title, xlabel, ylabel, and legend
        plt.title('Model Metric: {}'.format(metric))
        plt.xlabel('Epochs')
        plt.ylabel(metric.title())
        plt.legend()

    plt.show()
