__author__ = 'sean'

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

from cross_val import get_x_y_scorer_folds
from mut_group_pred_pack import get_full_mut_list, MutationGroup, \
    stratified_tt_split



X, y, scorer, folds = get_x_y_scorer_folds(
    MutationGroup(get_full_mut_list()), 2
)








svc = SVC(kernel='rbf', cache_size=2000, class_weight='auto')

clf = GridSearchCV(svc, param_grid=param_grid, scoring=scorer, cv=3)

X = PCA(n_components=2).fit_transform(X, y=y)

X_train, y_train, X_test, y_test = stratified_tt_split(X, y, test_size=0.2)



x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

mesh_grid = np.c_[xx.ravel(), yy.ravel()]

meshPredict = clf.fit(X, y).predict(mesh_grid)

meshPredict = meshPredict.reshape(xx.shape)

plt.figure(figsize=(9, 4))

plt.subplot(1, 2, 1)

levels = [-.5, .5, 1.5, 2.5, 3.5]
contour = plt.contourf(xx, yy, meshPredict, levels,
                       colors=get_contour_colors(2))
line = plt.contour(xx, yy, meshPredict, levels, colors=('k'))


zero_class = np.where(y_test == 0)
one_class = np.where(y_test == 1)

plt.scatter(X_test[zero_class, 0], X_test[zero_class, 1], c='#027878',
            label='Null')
plt.scatter(X_test[one_class, 0], X_test[one_class, 1], c='#c22326',
            label='Pathogenic')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks()
plt.yticks()

plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.legend(loc="lower right")


#plt.title(titles[i])
"""
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('multipage.pdf')
pp.savefig()
pp.close()
"""

plt.show()

def get_contour_colors(num_cats):
    color = {
        2: ('#5abdbd', '#e05254'),
        22: ('#5abdbd', '#e05254'),
        3: ('#5abdbd', '#f69568', '#e05254'),
        33: ('#5abdbd', '#f69568', '#e05254'),
        4: ('#5abdbd', '#f69568', '#fdc865', '#e05254'),
    }[num_cats]
    return color

def get_scatter_colors_labels(num_cats):
    colors = {
        2: ('#027878', '#c22326'),
        22: ('#027878', '#c22326'),
        3: ('#027878', '#f69568', '#e05254'),
        33: ('#027878', '#f69568', '#e05254'),
        4: ('#027878', '#f69568', '#fdc865', '#e05254'),
    }[num_cats]
    labels = {
        2: ('Null', 'Pathogenic'),
        22: ('Null+Autism', 'Pathogenic'),
        3: ('Null', 'Autism', 'Pathogenic'),
        33: ('Null', 'Autism+Somatic', 'PHTS'),
        4: ('Null', 'Autism', 'Somatic', 'PHTS'),
    }[num_cats]
    return colors, labels

# #027878 blue light ##5abdbd
# #f37338 orange light #f69568
# #fdb632 yellow light #fdc865
# #c22326 red light #e05254

def plot_contour(num_cats, meshPredict):
    levels = [-.5, .5, 1.5, 2.5, 3.5]
    plt.contourf(xx, yy, meshPredict, levels, colors=get_contour_colors(
        num_cats))
    plt.contour(xx, yy, meshPredict, levels, colors=('k'))

def plot_scatter(num_cats, X, y):
    colors, labels = get_scatter_colors_labels(num_cats)
    for i in range(len(colors)):
        indices = np.where(y == i)
        plt.scatter(X[indices, 0], X[indices, 1], c=colors[i],
            label=labels[i])

def plot_subfigure(mut_group, num_cats, subplot):
    X, y, scorer, folds = get_x_y_scorer_folds(mut_group, num_cats)

    X = PCA(n_components=2).fit_transform(X, y=y)
    X_train, y_train, X_test, y_test = stratified_tt_split(X, y, test_size=0.2)

    param_grid = [
    {'C': [.1, 1, 10, 100, 1000 ],
     'gamma': [0.1, 0.01, 0.001, 0.0001]},
    ]

    svc = SVC(kernel='rbf', cache_size=2000, class_weight='auto')
    clf = GridSearchCV(svc, param_grid=param_grid, scoring=scorer, cv=folds)

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    mesh_grid = np.c_[xx.ravel(), yy.ravel()]
    meshPredict = clf.fit(X, y).predict(mesh_grid)
    meshPredict = meshPredict.reshape(xx.shape)

    plt.subplot(1, 2, subplot)

    plot_contour(num_cats, meshPredict)

    plot_scatter(num_cats, X_test, y_test)





if __name__ == "__main__":

    classes = [2, 22]

    plt.figure(9, 5)

    pten_mutations = MutationGroup(get_full_mut_list())

    for i, num_cats in enumerate(classes):

        plot_subfigure(pten_mutations, num_cats, i)
