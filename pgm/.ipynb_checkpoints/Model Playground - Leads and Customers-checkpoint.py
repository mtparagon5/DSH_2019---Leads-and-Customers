# ---
# jupyter:
#   jupytext:
#     formats: ipynb,pgm//py
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

import sklearn
from sklearn import ensemble, model_selection, preprocessing, tree, linear_model
from sklearn.metrics import confusion_matrix
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix

# -

df = pd.read_json('data\\cleaned_df_numerical_columns_only.json')

df

# +
# columns = np.array(df.columns)

# fig, axs = plt.subplots(len(columns),1,figsize=(15,150))

# # df.T.iloc[0].name
# N=100000

# for i in range(len(columns)):
#     xs = df.T.iloc[i]
#     ys = df.converted
#     colors = np.random.rand(N)
#     axs[i].scatter(xs, ys, c=colors)
#     axs[i].set_xlabel(df.T.iloc[i].name)
#     axs[i].set_title(df.T.iloc[i].name)
#     plt.tight_layout()

# +
# df_rel_vals = df.drop(columns=['score', 'days_since_signup'])

# columns = np.array(df_rel_vals.columns)

# fig, axs = plt.subplots(1,1,figsize=(15,15))

# # df.T.iloc[0].name
# N=1000
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# xs, ys = [], []

# for i in range(len(columns)):
#     xs = df_rel_vals.T.iloc[i]
#     ys = df_rel_vals.T.loc['score_normalized']
    
#     plt.scatter(xs, ys, s=area, alpha=0.5)
#     plt.tight_layout()
# # xs

# -

# ### Random Forest

X = df.drop(columns=['converted'])
y = df.converted

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=.25, random_state=42)

# +
rf1 = ensemble.RandomForestClassifier(random_state=42)
rf1.fit(X_train, y_train)
rf1.score(X_test, y_test)

cm = confusion_matrix(y_test, y_pred)
# -

cm

plot_confusion_matrix(cm           = cm, 
                      normalize    = False,
                      target_names = ['No', 'Yes'],
                      title        = "Confusion Matrix -- Random Forest")

# ##### ROC Curve

# +
# fig, ax = plt.subplots(figsize=(10,10))
# roc_viz = ROCAUC(rf1)
# roc_viz.score(X_test,y_test)

# roc_viz.poof()
# -

# ## Logistic Regression

# +
X = df.drop(columns=['converted'])
y = df.converted

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=.25, random_state=0)
# -

classifier = linear_model.LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

classifier.score(X_test, y_test)

cm = confusion_matrix(y_test, y_pred)

cm

# +
import numpy as np


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

# -

plot_confusion_matrix(cm           = cm, 
                      normalize    = False,
                      target_names = ['No', 'Yes'],
                      title        = "Confusion Matrix -- Logistic Regression")


