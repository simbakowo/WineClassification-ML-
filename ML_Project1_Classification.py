import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import style
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import randint
from matplotlib.artist import setp
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,recall_score,f1_score,accuracy_score,precision_score



df = pd.read_csv('winequality-white.csv',sep=';', header=0)
#df = df[df['quality'] != 9]
#print(df['quality'].value_counts())

####################3##################identify categorical columns, if any############################
# for attribute in df.columns.tolist():
#     print('Unique values for {0} : {1}'.format(attribute, df[attribute].nunique()))

####################################### Sample a test set, put it aside, and never look at it##########
''' IMPORTANT!
We do not want to separate the X(inputs) from the y(label/target) before splitting into
test and train data if we are planning on having a stratfied split. This is becuase the 
attribute we're planning on stratify splitting on, will be missing from either X or y,
in this case y, becuase we want a balance of all wine qualities in both train and test 
data '''
train_set, test_set = train_test_split(df, test_size=0.2, shuffle=True, stratify=df['quality'])

########################################Scatter Matrix Visualizations################################################
# mpl.style.use('seaborn-dark')  
# #print(plt.style.available) # see what styles are available
# fig, ax = plt.subplots(1, 1)
# ax.set_facecolor('#F0F0F0')
# pd.plotting.scatter_matrix(train_set, alpha = 0.2, figsize = (20,20), ax=ax, diagonal = 'kde', c='#00473C')
# n = len(df.columns)
# axs = pd.plotting.scatter_matrix(train_set, alpha = 0.2, figsize = (8,8), ax=ax, diagonal = 'kde', c='#00473C')
# for x in range(n):
#     for y in range(n):
#         # to get the axis of subplots
#         ax = axs[x, y]
#         # to make x axis name vertical  
#         ax.xaxis.label.set_rotation(90)
#         # to make y axis name horizontal 
#         ax.yaxis.label.set_rotation(0)
#         # to make sure y axis names are outside the plot area
#         ax.yaxis.labelpad = 50
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
# fig = plt.gcf()
# fig.set_size_inches(15, 20, forward=True)
# fig.patch.set_facecolor('#F0F0F0')
# plt.show()
# ######################################Correlation Heatmap Visualization####################################
# sns.set(font_scale=0.8)
# correlation = train_set.corr()
# fig, ax = plt.subplots(1, 1)
# ax.set_facecolor('#F0F0F0')
# fig = plt.gcf()
# fig.set_size_inches(14, 10, forward=True)
# fig.patch.set_facecolor('#F0F0F0')
# # cmap = sns.palplot(sns.diverging_palette(220, 20, n=7))
# heatmap = sns.heatmap(correlation, annot=True, linewidths=1, linecolor='#F0F0F0', vmin=-0.9, vmax=1, cmap='BrBG')
# plt.show()

#######################################FEATURE SELECTION####################################################
'''We will use one of 3 methods, or all (but most probably just one method becaause we dont have that many
features and so we'll only be dropping a few). For classfcation: Method 1, Correlation Method. Method 2. 
Univariate method using SelectKBest. '''
#Method1 was already carried out
#Good idea to split into X and y now as we'll only want study features vs target/feature here
X_train_set = train_set.drop(['quality', 'residual sugar', 'fixed acidity'],1)
y_train_set = train_set['quality']
X_test_set = test_set.drop(['quality', 'residual sugar', 'fixed acidity'],1)
y_test_set = test_set['quality']

#for scorng, we have the option between,chi2, f_classif, mutual_info_classif
#ANOVA will be used(explaned in detail on website, why)
'''Anova Scorng'''
# bestfeatures = SelectKBest(score_func=f_classif, k=11)
# fit = bestfeatures.fit(X_train_set,y_train_set)
# df_scores = pd.DataFrame(fit.scores_)
# df_columns = pd.DataFrame(X_train_set.columns)
# # concatenate dataframes
# feature_scores = pd.concat([df_columns, df_scores],axis=1)
# feature_scores.columns = ['Feature_Name','Score']  # name output columns
# print(feature_scores.nlargest(11,'Score'))  # print 11 best features
'''The last methd we can use of Feature Importance, which is an inbuilt class that comes with Tree Based 
classifiers'''
# rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1) 
# rnd_clf.fit(X_train_set, y_train_set)
# for name, score in zip(X_train_set.columns.tolist(), rnd_clf.feature_importances_):
#     print(name, score)

'''Dropping Features: See README for selection criteria'''
#go back up to line 72 -75 to implement changes

#############################SHORTLISITNG AND COMPARING METHODS#####################################
'''It's worth stating here that are over 20 classifiers/algorithms available for multiclass classification
but in this project, the aim is to compare the 2 ways that can be used implement decision trees:'''
#1. Ensemble Method using a bagclassifier of decision trees
#2.  the RandomForestClassifier class

# bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=11), n_estimators=250, bootstrap=True, n_jobs=-1)
# rnd_clf = RandomForestClassifier(n_estimators=250,max_depth=11, n_jobs=-1) 

# for model in (bag_clf, rnd_clf):
#     print(model.__class__.__name__, cross_val_score(model, X_train_set, y_train_set, scoring="f1_weighted", cv=10).mean())
#when you set cv=integer, integer, to specify the number of folds in a (Stratified)KFold,
#if there arent enoung smaples in a class to be in each stratifed set, a Warning will be raised.

################################FINE TUNING HYPERPARAMETERS#########################################
# #Here we'll use RandomizedSearchCV

# est = RandomForestClassifier(n_jobs=-1, bootstrap=True)
# rf_p_dict = {'max_depth':[7,10,15,None], 'n_estimators':[100,200,300,400,500],'max_features':randint(6,9),
# 'min_samples_leaf':randint(1,4)}



# def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
#     rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
#                                   n_jobs=-1, n_iter=nbr_iter, cv=9)
#     #CV = Cross-Validation ( here using Stratified KFold CV)
#     rdmsearch.fit(X,y)
#     ht_params = rdmsearch.best_params_
#     ht_score = rdmsearch.best_score_
#     return ht_params, ht_score

# rf_parameters, rf_ht_score = hypertuning_rscv(est, rf_p_dict, 40, X_train_set, y_train_set)

# print(rf_parameters)
# print(rf_ht_score)
##################################USING THE BEST HYPERPARAMETERS on the never before seen testset####################################
rnd_clf = RandomForestClassifier(n_estimators=400,max_depth=None, max_features=7, min_samples_leaf=1, n_jobs=-1) 
rnd_clf.fit(X_train_set,y_train_set)
y_pred = rnd_clf.predict(X_test_set)

Accuracy=accuracy_score(y_test_set,y_pred)
Precision=precision_score(y_test_set,y_pred, average='weighted', zero_division=1)
Recall=recall_score(y_test_set,y_pred, average='weighted')
F1=f1_score(y_test_set,y_pred, average='weighted')

print('Accuracy: {0}'.format(Accuracy)) #not a good measure for classification, especally for skewed dataset
print('Precision: {0}'.format(Precision))
print('Recall: {0}'.format(Recall))
print('F1: {0}'.format(F1))

####################################PLOTTING THE CONFUSSION MATRIX#######################################

# # Plot non-normalized confusion matrix
# titles_options = [("Confusion matrix, without normalization", None),
#                   ("Normalized confusion matrix", 'true')]
# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(rnd_clf, X_test_set, y_test_set,
#                                  cmap=plt.cm.Blues,
#                                  normalize=normalize)
#     disp.ax_.set_title(title)

#     print(title)
#     print(disp.confusion_matrix)

# plt.show()
###########################################################################################################
