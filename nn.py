import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from datasets_func import get_data, split_dataset


def calc_error(xcv, ycv):
    x_cv_scaled = scaler.transform(xcv)
    y_p = clf.predict(x_cv_scaled)
    return 1 - np.array(y_p == ycv).astype(int).mean()


X, y = get_data('full_set')
X, y, X_cv, y_cv = split_dataset(X, y)
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='logistic', hidden_layer_sizes=(50,),
                    random_state=1, max_iter=2000)
train_sizes, train_scores, valid_scores = learning_curve(clf, X_scaled, y, shuffle=True)
plt.plot(train_sizes, train_scores.mean(axis=1))
plt.plot(train_sizes, valid_scores.mean(axis=1))
plt.show()

clf.fit(X_scaled, y)

print(f'nn score: {clf.score(X_scaled, y)}')
print(f'error: {calc_error(X_cv, y_cv)}')
