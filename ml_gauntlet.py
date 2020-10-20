import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb

from wandb.keras import WandbCallback
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn import svm

config = {
  "dropout": 0.2,
  "batch_size": 10,
  "epochs": 5,
  "kernel": 4
}

wandb.init(project="reservoir-computing", config=config)

def create_data(nprint_file, open_time, close_time):
    packets = []
    labels = []
    time = 0 #Total time elapsed
    with open(nprint_file) as npr_fl:
        reader = csv.reader(npr_fl)
        next(reader)
        for row in reader:
                for i in range(1,len(row)):
                    row[i] = int(row[i])
                packets.append(row[1:])
                time += row[1]
                if (time > open_time and time < close_time):
                    labels.append(1)
                else:
                    labels.append(0)
    return packets, labels


def train_model_kfold(x, y, model_constructor, **kwargs):
    scaler = StandardScaler()
    y_predictions = []
    x_tests, y_tests = [], []
    models = []

    for train, test in KFold().split(x):
        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
        x_tests.append(x_test)
        y_tests.append(y_test)
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        model = model_constructor()
        model.fit(x_train, y_train, **kwargs)
        models.append(model)
        y_predict = model.predict(x_test)
        y_predictions.append(y_predict)

    return y_predictions, models, x_tests, y_tests

def plot_custom_conf_matrix(y_predictions, y_truth, name):
    for predict in y_predictions:
        predict[0] = round(predict[0])  #Predictions to binary value
    cm = confusion_matrix(y_truth, y_predictions)
    plt.figure()
    plt.title("Confusion matrix")
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([0,1], [0,1])
    plt.yticks([0,1], [0,1])
    plt.colorbar()
    plt.savefig("./graphs/" + name + "_cm.png")

def plot_custom_pr_curve(y_predictions, y_truth, name):
    precision, recall, thresholds = precision_recall_curve(y_truth, y_predictions)
    plt.figure()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.scatter(recall, precision)
    plt.savefig("./graphs/" + name + "_prc.png")

def plot_custom_roc_curve(y_predictions, y_truth, name):
    fpr, tpr, threshold = roc_curve(y_truth, y_predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
    plt.legend(loc = "lower right")
    plt.plot([0, 1], [0, 1],"r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig("./graphs/" + name + "_roc.png")

def nn_plot_scores(y_predictions, y_truths, name):
    if (name == "LSTM" or name == "1D_CNN"):   #LSTM & 1D_CNN data is not currently going through KFOLD
        plot_custom_conf_matrix(y_predictions, y_truths, name)
        plot_custom_pr_curve(y_predictions, y_truths, name)
        plot_custom_roc_curve(y_predictions, y_truths, name)
        return
    for y_prediction, y_truth in zip(y_predictions, y_truths):
        plot_custom_conf_matrix(y_prediction, y_truth, name)
        plot_custom_pr_curve(y_prediction, y_truth, name)
        plot_custom_roc_curve(y_prediction, y_truth, name)

def plot_scores(scores, models, x_tests, y_tests, name, num_to_plot=1): #Allows flexibility for future KFold comparisons
    for i in range(0, min(len(models), num_to_plot)):
        plot_confusion_matrix(models[i], x_tests[i], y_tests[i])
        plt.savefig("./graphs/" + name + "_cm.png")
        plot_precision_recall_curve(models[i], x_tests[i], y_tests[i])
        plt.savefig("./graphs/" + name + "_prc.png")
        plot_roc_curve(models[i], x_tests[i], y_tests[i])
        plt.savefig("./graphs/" + name + "_roc.png")


def train_RFC(x, y):
    results = train_model_kfold(x, y, RandomForestClassifier)
    plot_scores(*results, "RFC")
    """
    #Uncomment to create Weights and Biases visual (available to team on WandB.com)
    data = []
    feature_labels = []
    with open("data/nprints/open_lower_fridge/open_lower_fridge_1.npt") as npr_fl:
        reader = csv.reader(npr_fl)
        feature_labels = reader.__next__()
    features_importance = results[1][0].feature_importances_
    for i in range(0,len(features_importance)):
        data.append([i, features_importance[i]])
    table = wandb.Table(data=data, columns=["feature", "height"])
    #find name of nprint field to label x axis
    line = wandb.plot.line(table, x="feature", y="height", title="Feature Importance Line")
    scatter = wandb.plot.scatter(table, x="feature", y="height", title="Feature Importance Scatter")
    wandb.log({"scatter": scatter, 
                "line": line})
    """

def train_SVM(x, y):
    results = train_model_kfold(x, y, svm.SVC)
    plot_scores(*results, "SVM")


def train_FF(x, y, batch_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(2048, input_shape=(batch_size, x.shape[1])),
            tf.keras.layers.Dense(2048),
            tf.keras.layers.Dense(2048),
            tf.keras.layers.Dropout(config["dropout"]),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    y_predictions, models, x_tests, y_tests = train_model_kfold(x, y, lambda : model, epochs=3) #callbacks=[WandbCallback()]) to store run on WandB
    nn_plot_scores(y_predictions, y_tests, "FF")

def train_CNN(x, y):
    kernel_length = config["kernel"]
    x = x.reshape(x.shape[0], x.shape[1], 1) #1DCNN data: batch_size, features, timesteps

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(16,kernel_length,input_shape=(x.shape[1], 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
    model.fit(x_train,y_train, epochs=config["epochs"])

    y_predict = model.predict(x_test)
    nn_plot_scores(y_predict, y_test, "1D_CNN")

def train_LSTM(x, y):
    x = x.reshape(x.shape[0], x.shape[1], 1) #LSTM data: batch_size, features, timesteps

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(32,input_shape=(x.shape[1], 1),dropout=config["dropout"]),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
    model.fit(x_train,y_train, epochs=config["epochs"])

    y_predict = model.predict(x_test)
    nn_plot_scores(y_predict, y_test, "LSTM")


def main():
    packets = []
    labels = []
    usable_captures = {"capture1","capture2","capture3","capture4","capture5"} #Captures with available labels
    rootdir = "./data/raw"
    OFFSET = len(rootdir)+1
    """
    for subdir, dirs, files in os.walk(rootdir):
        if (directory[OFFSET:] in usable_captures):
            with open(directory + "/timesheet.csv") as timesheet:
                reader = csv.reader(timesheet)
                for row in reader:
                    start, stop = int(reader[1]), int(reader[3]) #deltatime microseconds
                    nprint_file = directory + "/" + reader[0] + ".npt"
                    for packet_set, label_set in create_data("fridge_0.npt", start, stop):
                        packets.append(packet_set)
                        labels.append(label_set)
    """
    packets, labels = create_data("fridge_0.npt", 8000, 18000) #Currently tester data
    x = np.array(packets)
    y = np.array(labels)
    batch_size = len(labels)//3
    train_RFC(x, y)
    train_SVM(x, y)
    train_FF(x, y, batch_size)
    train_CNN(x, y)
    train_LSTM(x, y)
    #Echo State Network (Reservoir Computing) developed in another file, further comparisons to be made
    return



if __name__ == "__main__":
    main()