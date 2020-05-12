This project compares the 2 ways of implementing Random Forrests on the
White Wine Dataset using classification. (10 classes, quality). Could also
be a regression problem.

Denstity and Residual Sugar are collinear. To decide which one to drop, we
can look at which one is less correlated to the output and which one ranks higher 
on the ANOVA F scores and FeatureImportance scores.

Alsp drop fixed acidity based on Feature importance scores.

Because RFs are treebased, they do not require feature scaling

Based on the confusion matrix and the PR curve, we have succesfully prioritised
fraud class (1) >> recall becuase FN are more important