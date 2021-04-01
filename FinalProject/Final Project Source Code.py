import numpy as np
import pandas as pd
import seaborn as sns
from function import admission_decision, func_confusion_matrix
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
import sklearn.svm as svm
import pylab as pl
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Data Processing
# Load Data
df = pd.read_csv("Admission_Predict_Ver1.1.csv")
# Clean up Data
df.columns = ["Number", "GRE", "TOEFL", "University Rating", "SOP", "LOR", "CGPA", "Research", "Chance of Admit"]
X = df[["GRE", "TOEFL", "University Rating", "SOP", "LOR", "CGPA", "Research"]]
X = np.array(X)
Y = df["Chance of Admit"]
Y = np.array(Y)
# Rescale X for Regression Problem
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
Xrs = min_max_scaler.fit_transform(X)
# Data Splitting for Regression Problem
Xrs_train, Xrs_test, Y_train, Y_test = train_test_split(Xrs, Y, test_size=0.5, random_state=425)
Xrs_train, Xrs_valid, Y_train, Y_valid = train_test_split(Xrs_train, Y_train, test_size=0.2, random_state=425)
# Data Splitting for Classification Problem
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=425)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=425)
# Convert to Classification Problem
Y_train_bi = admission_decision(Y_train).ravel()
Y_test_bi = admission_decision(Y_test).ravel()
Y_valid_bi = admission_decision(Y_valid).ravel()

# Data Description
# summary statistics
df.describe().round(2)
# correlation plot
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
# pair plot
sns.pairplot(data=df.iloc[:, 1:], kind='scatter')
# transform Y into binary label
Y_bi = admission_decision(Y)
df_bi = pd.DataFrame(np.hstack((df, Y_bi)))
# separate the categorical features and numeric features
df_bi.columns = ["Number", "GRE", "TOEFL", "University Rating", "SOP", "LOR", "CGPA", "Research", "Chance of Admit",
                 "Admission Decision"]
cat_columns = ['University Rating', 'SOP', 'LOR', 'Research']
num_columns = ["GRE", "TOEFL", "CGPA"]
# strip plot for categorical features
fig, ax = plt.subplots(1, 4, figsize=(6*4, 6))
for i, col_name in enumerate(cat_columns):
    plt.subplot(ax[i])
    sns.stripplot(data=df_bi, x=col_name, y='Chance of Admit', hue="Admission Decision", jitter=True)
plt.show()
# scatter plots for numeric features
for i, col_name in enumerate(num_columns):
    sns.lmplot(data=df_bi, x=col_name, y='Chance of Admit', hue="Admission Decision")
plt.show()

# Regression Problem
# Linear Regression
lr = LinearRegression().fit(Xrs_train, Y_train)
print("Validation set score: {:.2f}".format(lr.score(Xrs_valid, Y_valid)))
print("Test set score: {:.2f}".format(lr.score(Xrs_test, Y_test)))

# Ridge Regression
ridge = Ridge().fit(Xrs_train, Y_train)
# 5-fold Cross Validation for choosing alpha
r_alpha = [10, 1, 0.1, 0.01, 0.001]
score = np.zeros(5)
rcv_score = pd.DataFrame([score, score, score, score, score])
for i in range(0, len(r_alpha)):
    ridge = Ridge(alpha=r_alpha[i]).fit(Xrs_train, Y_train)
    rcv_score[i] = cross_val_score(ridge, Xrs_valid, Y_valid, cv=5, scoring='r2')
rcv_score.columns = ["alpha = 10", "alpha = 1", "alpha = 0.1", "alpha = 0.01", "alpha = 0.001"]
rcv_score.index = ["fold 1 score", "fold 2 score", "fold 3 score", "fold 4 score", "fold 5 score"]
mean_valid_r2 = [np.mean(rcv_score["alpha = 10"]), np.mean(rcv_score["alpha = 1"]), np.mean(rcv_score["alpha = 0.1"]),
                 np.mean(rcv_score["alpha = 0.01"]), np.mean(rcv_score["alpha = 0.001"])]
rcv_score.loc["mean validation r2"] = mean_valid_r2
best_ra = r_alpha[mean_valid_r2.index(max(mean_valid_r2))]
print("The optimal alpha for Ridge Regression is:", best_ra)
ridge = Ridge(alpha=best_ra).fit(Xrs_train, Y_train)
print("Validation set score: {:.2f}".format(ridge.score(Xrs_valid, Y_valid)))
print("Test set score: {:.2f}".format(ridge.score(Xrs_test, Y_test)))

# Lasso Regression
lasso = Lasso().fit(Xrs_train, Y_train)
# 5-fold Cross Validation for choosing alpha
l_alpha = [0.01, 0.001, 0.0001]
score = np.zeros(3)
lcv_score = pd.DataFrame([score, score, score, score, score])
for j in range(0, len(l_alpha)):
    ridge = Ridge(alpha=l_alpha[j]).fit(Xrs_train, Y_train)
    lcv_score[j] = cross_val_score(ridge, Xrs_valid, Y_valid, cv=5, scoring='r2')
lcv_score.columns = ["alpha = 0.01", "alpha = 0.001", "alpha = 0.0001"]
lcv_score.index = ["fold 1 score", "fold 2 score", "fold 3 score", "fold 4 score", "fold 5 score"]
mean_valid_r2 = [np.mean(lcv_score["alpha = 0.01"]), np.mean(lcv_score["alpha = 0.001"]),
                 np.mean(lcv_score["alpha = 0.0001"])]
lcv_score.loc["mean validation r2"] = mean_valid_r2
best_la = l_alpha[mean_valid_r2.index(max(mean_valid_r2))]
print("The optimal alpha for Lasso Regression is:", best_la)
lasso = Lasso(alpha=best_la, max_iter=10000).fit(Xrs_train, Y_train)
print("Validation set score: {:.2f}".format(lasso.score(Xrs_valid, Y_valid)))
print("Number of features used:", np.sum(lasso.coef_ != 0))
print("Test set score: {:.2f}".format(lasso.score(Xrs_test, Y_test)))

# Elastic Net
# Find Optimal k
kfold = [3, 5, 7, 9, 10, 50, 100]
score = np.zeros(3)
encv_score = pd.DataFrame([score, score, score, score, score, score, score])
encv_score.columns = ["k", "Validation set score", "Number of features used"]
for k in range(0, len(kfold)):
    ENet = ElasticNetCV(cv=kfold[k], random_state=425)
    ENet.fit(Xrs_train, Y_train)
    encv_score["k"][k] = kfold[k]
    encv_score["Validation set score"][k] = ENet.score(Xrs_valid, Y_valid)
    encv_score["Number of features used"][k] = np.sum(ENet.coef_ != 0)
best_k = kfold[encv_score.set_index("Validation set score").index.get_loc(max(encv_score["Validation set score"]))]
ENet = ElasticNetCV(cv=best_k, random_state=425)
ENet.fit(Xrs_train, Y_train)
print("Validation set score: {:.2f}".format(ENet.score(Xrs_valid, Y_valid)))
print("Number of features used:", np.sum(ENet.coef_ != 0))
print("Test set score: {:.2f}".format(ENet.score(Xrs_test, Y_test)))

# FeedForward Neural Network
# fnn model 1
# classification with model fnn1
input1 = tf.keras.layers.InputLayer(input_shape=(7,))
hidden1_1 = tf.keras.layers.Dense(50, activation="elu")
output1 = tf.keras.layers.Dense(1, activation="sigmoid")
model1 = tf.keras.Sequential([input1, hidden1_1, output1])
opt = keras.optimizers.Adam(learning_rate=0.01)
model1.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model1.fit(X_train, Y_train, epochs=100)
# classification accuracy in validation set
y_pred1 = model1.predict(X_valid)
y_pred_trans1 = admission_decision(y_pred1).ravel()
conf_matrix1, accuracy1, recall_array1, precision_array1 = func_confusion_matrix(Y_valid_bi, y_pred_trans1)
print("Accuracy of model fnn1 in validation set is", accuracy1)
# classification accuracy in test set
y_pred2 = model1.predict(X_test)
y_pred_trans2 = admission_decision(y_pred2).ravel()
conf_matrix2, accuracy2, recall_array2, precision_array2 = func_confusion_matrix(Y_test_bi, y_pred_trans2)
print("Accuracy of model fnn1 in test set is", accuracy2)

# regression with model fnn1
output2 = tf.keras.layers.Dense(1)
model2 = tf.keras.Sequential([input1, hidden1_1, output2])
opt = keras.optimizers.Adam(learning_rate=0.01)
model2.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt, metrics=["accuracy"])
model2.fit(Xrs_train, Y_train, epochs=100)

# regression accuracy in validation set
y_pred_3 = model2.predict(Xrs_valid)
r2_1 = r2_score(Y_valid, y_pred_3)
print("R^2 of model fnn1(linear regression) in validation set is", r2_1)

# regression accuracy in test set
y_pred_4 = model2.predict(Xrs_test)
r2_2 = r2_score(Y_test, y_pred_4)
print("R^2 of model fnn1(linear regression) in test set is", r2_2)

# fnn model 2
# classification with model fnn2
input3 = tf.keras.layers.InputLayer(input_shape=(7,))
hidden3_1 = tf.keras.layers.Dense(10, activation="relu")
hidden3_2 = tf.keras.layers.Dense(80, activation="relu")
output3 = tf.keras.layers.Dense(1, activation="sigmoid")
model3 = tf.keras.Sequential([input3, hidden3_1, hidden3_2, output3])
opt = keras.optimizers.Adam(learning_rate=0.01)
model3.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model3.fit(X_train, Y_train, epochs=100)

# classification accuracy in validation set
y_pred5 = model3.predict(X_valid)
y_pred_trans3 = admission_decision(y_pred5).ravel()
conf_matrix3, accuracy3, recall_array3, precision_array3 = func_confusion_matrix(Y_valid_bi, y_pred_trans3)
print("Accuracy of model fnn2 in validation set is", accuracy3)

# classification accuracy in test set
y_pred6 = model3.predict(X_test)
y_pred_trans4 = admission_decision(y_pred6).ravel()
conf_matrix4, accuracy4, recall_array4, precision_array4 = func_confusion_matrix(Y_test_bi, y_pred_trans4)
print("Accuracy of model fnn2 in test set is", accuracy4)

# regression with model fnn2
output4 = tf.keras.layers.Dense(1)
model4 = tf.keras.Sequential([input3, hidden3_1, hidden3_2, output4])
opt = keras.optimizers.Adam(learning_rate=0.01)
model4.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt, metrics=["accuracy"])

model4.fit(Xrs_train, Y_train, epochs=100)

# regression accuracy in validation set
y_pred_7 = model4.predict(Xrs_valid)
r2_3 = r2_score(Y_valid, y_pred_7)
print("R^2 of model fnn2(linear regression) in validation set is", r2_3)

# regression accuracy in test set
y_pred_8 = model4.predict(Xrs_test)
r2_4 = r2_score(Y_test, y_pred_8)
print("R^2 of model fnn2(linear regression) in test set is", r2_4)

# fnn model 3
# classification with model fnn3
input5 = tf.keras.layers.InputLayer(input_shape=(7,))
hidden5_1 = tf.keras.layers.Dense(5, activation="relu")
hidden5_2 = tf.keras.layers.Dense(10, activation="elu")
output5 = tf.keras.layers.Dense(1, activation="sigmoid")
model5 = tf.keras.Sequential([input5, hidden5_1, hidden5_2, output5])
opt = keras.optimizers.Adam(learning_rate=0.01)
model5.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model5.fit(X_train, Y_train, epochs=100)

# classification accuracy in validation set
y_pred9 = model5.predict(X_valid)
y_pred_trans9 = admission_decision(y_pred9).ravel()
conf_matrix5, accuracy5, recall_array5, precision_array5 = func_confusion_matrix(Y_valid_bi, y_pred_trans9)
print("Accuracy of model fnn3 in validation set is", accuracy5)

# classification accuracy in test set
y_pred10 = model5.predict(X_test)
y_pred_trans10 = admission_decision(y_pred10).ravel()
conf_matrix6, accuracy6, recall_array6, precision_array6 = func_confusion_matrix(Y_test_bi, y_pred_trans10)
print("Accuracy of model fnn3 in test set is", accuracy6)

# regression with model fnn3
output6 = tf.keras.layers.Dense(1)
model6 = tf.keras.Sequential([input5, hidden5_1, hidden5_2, output6])
opt = keras.optimizers.Adam(learning_rate=0.01)
model6.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt, metrics=["accuracy"])

model6.fit(Xrs_train, Y_train, epochs=100)

# regression accuracy in validation set
y_pred_11 = model6.predict(Xrs_valid)
r2_5 = r2_score(Y_valid, y_pred_11)
print("R^2 of model fnn3(linear regression) in validation set is", r2_5)

# regression accuracy in test set
y_pred_12 = model6.predict(Xrs_test)
r2_6 = r2_score(Y_test, y_pred_12)
print("R^2 of model fnn3(linear regression) in test set is", r2_6)

# confusion matrix
# The optimal model is fnn1, the confusion matrix is following
print("Confusion Matrix: ")
print(conf_matrix2)
print("Average Accuracy: {}".format(accuracy2))
print("Per-Class Precision: {}".format(precision_array2))
print("Per-Class Recall: {}".format(recall_array2))

# bootstrap
def fun_calaccuracy(predY, trueY):
    matrix = np.zeros((2, 2))
    for i in range(len(predY)):
        if predY[i] == 0:
            if trueY[i] == 0:
                matrix[1][1] += 1
            else:
                matrix[0][1] += 1
        else:
            if trueY[i] == 0:
                matrix[1][0] += 1
            else:
                matrix[0][0] += 1

    accu = ((matrix[0][0]) + ((matrix[1][1]))) / (len(predY))
    return accu

def optimal_fnn(data):
    X = np.array(data.iloc[:, 1:8])
    Y = np.array(data.iloc[:, 8]).ravel()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=425)

    Y_test_bi = admission_decision(Y_test).ravel()

    input7 = tf.keras.layers.InputLayer(input_shape=(7,))
    hidden7_1 = tf.keras.layers.Dense(50, activation="elu")
    output7 = tf.keras.layers.Dense(1, activation="sigmoid")
    model7 = tf.keras.Sequential([input7, hidden7_1, output7])
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model7.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    model7.fit(x=X_train, y=Y_train)

    y_pred_13 = model7.predict(X_test)
    y_pred_trans13 = admission_decision(y_pred_13).ravel()
    score = fun_calaccuracy(y_pred_trans13, Y_test_bi)
    return score
# define the bootstrap function
def bootstrap_1(data):
    sample_index = np.random.choice(len(data), len(data), replace=True)
    data_boot = data.iloc[sample_index, :]
    return optimal_fnn(data_boot)

def bootstrap_fnn(data, size):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_1(data)
    return bs_replicates

# for classification problem optimal model: fnn1
# get bootstrap mean and std of accuracy
score_boot_fnn = bootstrap_fnn(df, 100)
mu_boot_fnn = np.mean(score_boot_fnn)
mu_boot_fnn
std_boot_fnn = np.std(score_boot_fnn)


def optimal_rs(data):
    X = np.array(data.iloc[:, 1:8])
    Y = np.array(data.iloc[:, 8]).ravel()
    Xrs = min_max_scaler.fit_transform(X)
    Xrs_train, Xrs_test, Yrs_train, Yrs_test = train_test_split(Xrs, Y, test_size=0.2,
                                                                random_state=425)

    input8 = tf.keras.layers.InputLayer(input_shape=(7,))
    hidden8_1 = tf.keras.layers.Dense(5, activation="relu")
    hidden8_2 = tf.keras.layers.Dense(10, activation="elu")
    output8 = tf.keras.layers.Dense(1)
    model8 = tf.keras.Sequential([input8, hidden8_1, hidden8_2, output8])
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model8.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt, metrics=["accuracy"])

    model8.fit(x=Xrs_train, y=Yrs_train)

    y_pred_14 = model8.predict(Xrs_test)
    score = r2_score(Yrs_test, y_pred_14)

    return score

# define the bootstrap function
def bootstrap_1_rs(data):
    sample_index = np.random.choice(len(data), len(data), replace=True)
    data_boot = data.iloc[sample_index, :]
    return optimal_rs(data_boot)

def bootstrap_rs(data, size):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_1_rs(data)
    return bs_replicates

# for regression problem optimal model: fnn3
# get bootstrap mean and std of accuracy
score_boot_rs = bootstrap_rs(df, 100)
mu_boot_rs = np.mean(score_boot_rs)
mu_boot_rs
std_boot_rs = np.std(score_boot_rs)

# Compare R2 of Regression Models over test set
r2_reg = pd.DataFrame({"R square of Regression Model": [lr.score(Xrs_test, Y_test), ridge.score(Xrs_test, Y_test),
                                                        lasso.score(Xrs_test, Y_test), ENet.score(Xrs_test, Y_test),
                                                        r2_6]})
r2_reg.index = ["Linear Regression", "Ridge Regression", "LASSO", "Elastic Net", "FeedForward Neural Network"]

# Classification Problem
# Logistic Regression
clf = LogisticRegression(max_iter=100000).fit(Xrs_train, Y_train_bi)
# 5-fold Cross Validation for choosing C
cvalues = [0.01, 0.1, 1, 10, 100]
score = np.zeros(5)
lrcv_score = pd.DataFrame([score, score, score, score, score])
for c in range(0, len(cvalues)):
    clf = LogisticRegression(C=cvalues[c], max_iter=10000).fit(Xrs_train, Y_train_bi)
    lrcv_score[c] = cross_val_score(clf, Xrs_valid, Y_valid_bi, cv=5, scoring='accuracy')
lrcv_score.columns = ["C = 0.01", "C = 0.1", "C = 1", "C = 10", "C = 100"]
lrcv_score.index = ["fold 1 accuracy", "fold 2 accuracy", "fold 3 accuracy", "fold 4 accuracy", "fold 5 accuracy"]
mean_valid_accuracy = [np.mean(lrcv_score["C = 0.01"]), np.mean(lrcv_score["C = 0.1"]), np.mean(lrcv_score["C = 1"]),
                       np.mean(lrcv_score["C = 10"]), np.mean(lrcv_score["C = 100"])]
lrcv_score.loc["mean validation accuracy"] = mean_valid_accuracy
best_c = cvalues[mean_valid_accuracy.index(max(mean_valid_accuracy))]
print("The optimal C for Logistic Regression is:", best_c)
clf = LogisticRegression(C=best_c, max_iter=10000).fit(Xrs_train, Y_train_bi)
Y_valid_pred = admission_decision(clf.predict_proba(Xrs_valid)[:, 1]).ravel()
Y_pred = admission_decision(clf.predict_proba(Xrs_test)[:, 1]).ravel()
print("Validation set accuracy: {:.2f}".format(accuracy_score(Y_valid_bi, Y_valid_pred)))
print("Test set accuracy: {:.2f}".format(accuracy_score(Y_test_bi, Y_pred)))

# Confusion Metrics for Logistic Regression
conf_matrix, accuracy_lr, recall_array, precision_array = func_confusion_matrix(Y_test_bi, Y_pred)
print("Confusion Matrix for Logistic Regression Model: ")
print(conf_matrix)
print("Average Accuracy: {}".format(accuracy_lr))
print("Per-Class Precision: {}".format(precision_array))
print("Per-Class Recall: {}".format(recall_array))

# ROC Curve and AUC Value
Y_score = clf.predict_proba(Xrs_test)[:, 1]
n_classes = 1
fpr = dict()
tpr = dict()
roc_auc = dict()
print('total number of classes:', n_classes)
# Class 0 ROC&AUC
fpr[0], tpr[0], _ = roc_curve(Y_test_bi, Y_score)
roc_auc[0] = auc(fpr[0], tpr[0])
n = 0
plt.figure()
lw = 2
plt.plot(fpr[n], tpr[n], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[n])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print("The AUC value of Class 0 using Logistic Regression is", roc_auc_score(Y_test_bi, Y_score))

# SVM
# Model selection over validation set
c_range = [0.1, 1, 2, 4, 6, 10, 20]
kernel_types = ['linear', 'poly', 'rbf']
svm_error = []
xlab = []
for c_value in c_range:
    for kernel_value in kernel_types:
        model = svm.SVC(kernel=kernel_value, C=c_value)
        model.fit(X=X_train, y=Y_train_bi)
        error = 1. - model.score(X_valid, Y_valid_bi)
        svm_error.append(error)
        xlab.append(str(c_value) + "*" + kernel_value)

plt.figure()
plt.plot(xlab, svm_error)
plt.title('SVM by C and Kernels')
plt.xlabel('C and Kernels')
plt.ylabel('error')
plt.xticks(xlab)
pl.xticks(rotation=90)
plt.show()

# Select the best model and apply it over the testing subset
model = svm.SVC(kernel="linear", C=1)
model.fit(X=X_train, y=Y_train_bi)
print("Accuracy of SVM: {}".format(model.score(X_test, Y_test_bi)))

# Confusion matrix
Y_pred = model.predict(X_test)
conf_matrix, accuracy_svm, recall_array, precision_array = func_confusion_matrix(Y_test_bi, Y_pred)

print("Confusion Matrix: ")
print(conf_matrix)
print("Average Accuracy: {}".format(accuracy_svm))
print("Per-Class Precision: {}".format(precision_array))
print("Per-Class Recall: {}".format(recall_array))

# Bootstrap
# Define the optimal svm function
def optimal_svm(data, best_c, best_kernel):
    X = np.array(data.iloc[:, 1:8])
    Y = np.array(data.iloc[:, 8]).ravel()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=425)

    Y_train_bi = admission_decision(Y_train).ravel()
    Y_test_bi = admission_decision(Y_test).ravel()

    model = svm.SVC(kernel="linear", C=1)
    model.fit(X=X_train, y=Y_train_bi)
    score = model.score(X_test, Y_test_bi)
    return score


# Define the bootstrap function
def bootstrap_1(data):
    sample_index = np.random.choice(len(data), len(data), replace=True)
    data_boot = data.iloc[sample_index, :]
    return optimal_svm(data_boot, 1, "linear")


def bootstrap(data, size):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_1(data)
    return bs_replicates

# get bootstrap mean and std of accuracy
score_boot = bootstrap(df, 1000)
mu_boot = np.mean(score_boot)
std_boot = np.std(score_boot)
print("Mean of Bootstrap Accuracy:{}".format(mu_boot))
print("Standard Error of Bootstrap Accuracy:{}".format(std_boot))


# Decision tree
boosted_dt = AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=425)
boosted_dt.fit(X_train, Y_train_bi)
boosted_yHat = boosted_dt.predict(X_test)
boosted_accuracy = np.mean(1*(boosted_yHat == Y_test_bi))
print("Accuracy of Decision Tree Model is {}".format(boosted_accuracy))

# Random Forest
boosted_rf = RandomForestClassifier(n_estimators=100, random_state=425)
boosted_rf.fit(X_train, Y_train_bi)
boosted_yHat1 = boosted_rf.predict(X_test)
boosted_accuracy1 = np.mean(1*(boosted_yHat1 == Y_test_bi))
print("Accuracy of Random Forest Model is {}".format(boosted_accuracy1))

# Compare accuracy of Classification Models over test set
accuracy_clf = pd.DataFrame({"Accuracy of Classification Model": [accuracy_lr, accuracy2, accuracy_svm,
                                                                  boosted_accuracy, boosted_accuracy1]})
accuracy_clf.index = ["Logistic Regression", "FeedForward Neural Network", "Kernel SVM", "Decision Tree",
                      "Random Forest"]
