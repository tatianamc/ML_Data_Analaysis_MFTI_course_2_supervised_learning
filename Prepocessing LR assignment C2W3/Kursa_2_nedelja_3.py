from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score


def plot_scores(optimizer):
    scores = [[item[0]['C'], 
               item[1], 
               (np.sum((item[2]-item[1])**2)/(item[2].size-1))**0.5] for item in optimizer.grid_scores_]
    scores = np.array(scores)
    plt.semilogx(scores[:,0], scores[:,1])
    plt.fill_between(scores[:,0], scores[:,1]-scores[:,2], 
                                  scores[:,1]+scores[:,2], alpha=0.3)
    plt.show()
    
def write_answer_1(auc_1, auc_2):
    auc = (auc_1 + auc_2)/2
    with open("preprocessing_lr_answer1.txt", "w") as fout:
        fout.write(str(auc))
        
param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
cv = 3

# place your code here
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search 
# объединякс данные после модификации
X_test_zeros = np.hstack((X_test_real_zeros, X_test_cat_oh))
X_train_zeros = np.hstack((X_train_real_zeros, X_train_cat_oh))
X_test_mean = np.hstack((X_test_real_mean, X_test_cat_oh))
X_train_mean = np.hstack((X_train_real_mean, X_train_cat_oh))
#  стратегия кросс-валидации
from sklearn.cross_validation import  StratifiedKFold
cv = StratifiedKFold(y_train, cv)
#  задаем классификатор
classifier = LogisticRegression()
grid_cv_zeros = GridSearchCV(classifier, param_grid, cv=cv)
grid_cv_mean = GridSearchCV(classifier, param_grid, cv=cv)
grid_cv_zeros.fit(X_train_zeros, y_train)
plot_scores(grid_cv_zeros)
print grid_cv_zeros.best_score_
print grid_cv_zeros.best_params_
grid_cv_mean.fit(X_train_mean, y_train)
plot_scores(grid_cv_mean)
print grid_cv_mean.best_score_
print grid_cv_mean.best_params_
from sklearn.metrics import roc_auc_score
y_predictr_zero_test = grid_cv_zeros.best_estimator_.predict_proba(X_test_zeros)
roc_auc_zero_test = roc_auc_score(y_test, y_predictr_zero_test[:,1])
print roc_auc_zero_test
y_predictr_mean_test = grid_cv_mean.best_estimator_.predict_proba(X_test_mean)
roc_auc_mean_test = roc_auc_score(y_test, y_predictr_mean_test[:,1])
print roc_auc_mean_test
write_answer_1(roc_auc_mean_test, roc_auc_zero_test)