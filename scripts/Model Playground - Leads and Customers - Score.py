# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,scripts//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, random

import sklearn
from sklearn import ensemble, model_selection, preprocessing, tree, linear_model, neighbors
from sklearn.metrics import confusion_matrix
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix


# -

# ### I'm using  plot_confusion_matrix  function taken from kernel found on Kaggle
# ####  Credit to [George Fisher](https://www.kaggle.com/grfiv4/plot-a-confusion-matrix/notebook) who has licensed it under an [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)  open source license

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# ### A function for pickling the classifiers (not necessary)

# +
def pickle_classifier(filepath, classifier):
    """Function to automate the pickling of a classifier
    """
    with open(filepath, 'wb') as f:
        pickle.dump(classifier, f)

def load_pickled_classifier(filepath):
    """Function to load a pickled classifier
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# +
df = pd.read_json('..\\data\\cleaned_df_numerical_columns_only.json')
df = df.astype(float)
# moving converted to last column (was thinking of using random.shuffle)
df_score = df['score']
df = df.drop(['score'], axis=1)
df['score'] = df_score

redundant_columns = [
    'score_normalized',
    'days_since_signup_normalized'
]
df.drop(redundant_columns, axis=1, inplace=True)
# -

df.head()

# ### Random Forest

X = df.drop(columns=['score'])
X = preprocessing.scale(X)
y = df.score

# +
# for i in range(1, 11):
#     X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=.25, random_state=None)

#     n_trees = 500
#     clf = ensemble.RandomForestClassifier(max_features='auto', n_estimators=n_trees, random_state=None, n_jobs=-1)
#     clf.fit(X_train, y_train)
#     accuracy = clf.score(X_test, y_test)
#     print('Test {0} Accuracy:\t{1}'.format(i, accuracy))
    
# pickle_classifier('..\\classifiers\\random_forest_score.pickle', clf)

# +
# y_pred = clf.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# cm

# +
# plot_confusion_matrix(cm           = cm, 
#                       normalize    = False,
#                       target_names = ['No', 'Yes'],
#                       title        = "Confusion Matrix -- Random Forest")

# +
# print(clf.decision_path(X))
# print(clf.feature_importances_)
# -

# ##### ROC Curve

# +
# fig, ax = plt.subplots(figsize=(10,10))
# roc_viz = ROCAUC(clf)
# roc_viz.score(X_test,y_test)

# roc_viz.poof()
# -

# ## Logistic Regression

X = df.drop(columns=['score'])
X = preprocessing.scale(X)
y = df.score

# +
# for i in range(1, 11):
#     X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=.25, random_state=None)

#     clf = linear_model.LogisticRegression(random_state=None, solver='lbfgs', n_jobs=-1)
#     clf.fit(X_train, y_train)
#     accuracy = clf.score(X_test, y_test)
#     print('Test {0} Accuracy:\t{1}'.format(i, accuracy))
    
# pickle_classifier('..\\classifiers\\logistic_regression_score.pickle', clf)

# +
# y_pred = clf.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# cm

# +
# print(clf.decision_function(X))
# print(clf.decision_function(X).mean())
# pd.DataFrame(clf.decision_function(X)>0).describe()

# +
# plot_confusion_matrix(cm           = cm, 
#                       normalize    = False,
#                       target_names = ['No', 'Yes'],
#                       title        = "Confusion Matrix -- Logistic Regression")
# -

# ## Linear Regression

X = df.drop(columns=['score'])
X = preprocessing.scale(X)
y = df.score

# +
for i in range(1, 11):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=.25, random_state=None)

    clf = linear_model.LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('Test {0} Accuracy:\t{1}'.format(i, accuracy))
    
pickle_classifier('..\\classifiers\\linear_regression_score.pickle', clf)
# -

y_pred = clf.predict(X_test)
# y_pred = [0 if c <= 150 else 1 for c in y_pred]
y_pred
# cm = confusion_matrix(y_test, y_pred)
# cm

# +
# plot_confusion_matrix(cm           = cm, 
#                       normalize    = False,
#                       target_names = ['No', 'Yes'],
#                       title        = "Confusion Matrix -- Linear Regression")
# -

# ## K Nearest Neighbor (KNN)

X = df.drop(columns=['score'])
X = preprocessing.scale(X)
y = df.score

# #### KNN Regressor

# +
for i in range(1, 2):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=.25, random_state=None)

    clf = neighbors.KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('Test {0} Accuracy:\t{1}'.format(i, accuracy))
    
pickle_classifier('..\\classifiers\\knn_regressor_score.pickle', clf)
# -

y_pred = clf.predict(X_test)
# y_pred = [0 if c <= 0.5 else 1 for c in y_pred]
# cm = confusion_matrix(y_test, y_pred)
# cm

y_pred









plot_confusion_matrix(cm           = cm, 
                      normalize    = False,
                      target_names = ['No', 'Yes'],
                      title        = "Confusion Matrix -- KNN Regressor")

# #### KNN Classifier

# +
for i in range(1, 11):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=.25, random_state=None)

    clf = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('Test {0} Accuracy:\t{1}'.format(i, accuracy))
    
pickle_classifier('..\\classifiers\\knn_classifier.pickle', clf)
# -

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

plot_confusion_matrix(cm           = cm, 
                      normalize    = False,
                      target_names = ['No', 'Yes'],
                      title        = "Confusion Matrix -- KNN Classifier")


