from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets_func import get_data

X, y = get_data('full_set')

X_embedded = TSNE(n_components=2).fit_transform(X)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab20')
plt.show()
