rm(list = ls())
# 8. This problem involves the OJ data set which is part of the ISLR package
library(ISLR)
library(e1071)
#(a) Create a training set containing a random sample of 800
# observations, and a test set containing the remaining observations.
set.seed(5208)

train <- sample(nrow(OJ), 800)
OJ_train <- OJ[train, ]
OJ_test <- OJ[-train, ]

#(b) Fit a support vector classifier to the training data using
# cost=0.01, with Purchase as the response and the other variables
# as predictors. Use the summary() function to produce summary
# statistics, and describe the results obtained.

svm_linear <- svm(Purchase ~ . , kernel = "linear", data = OJ_train, cost = 0.01)
summary(svm_linear)

#(c) What are the training and test error rates?

# calculate error rate
calc_error_rate <- function(svm_model, dataset, true_classes) {
  confusion_matrix <- table(predict(svm_model, dataset), true_classes)
  return(1 - sum(diag(confusion_matrix)) / sum(confusion_matrix))
}

cat("Training Error Rate:", 100 * calc_error_rate(svm_linear, OJ_train, OJ_train$Purchase), "%\n")

#(d) Use the tune() function to select an optimal cost. Consider values in the range 0.01 to 10.
set.seed(5208)

svm_tune <- tune(svm, Purchase ~ . , data = OJ, kernel = "linear", 
                 ranges = list(cost = seq(0.01, 10, length = 100)))
summary(svm_tune)

#(e) Compute the training and test error rates using this new value for cost.
svm_linear <- svm(Purchase ~ . , kernel = "linear", 
                  data = OJ_train, cost = svm_tune$best.parameters$cost)

cat("Training Error Rate:", 100 * calc_error_rate(svm_linear, OJ_train, OJ_train$Purchase), "%\n")

cat("Test Error Rate:", 100 * calc_error_rate(svm_linear, OJ_test, OJ_test$Purchase), "%\n")


#(f) Repeat parts (b) through (e) using a support vector machine with a radial kernel. Use the default value for gamma.

set.seed(5208)
svm_radial <- svm(Purchase ~ . , data = OJ_train, kernel = "radial")
summary(svm_radial)

cat("Training Error Rate:", 100 * calc_error_rate(svm_radial, OJ_train, OJ_train$Purchase), "%\n")
cat("Test Error Rate:", 100 * calc_error_rate(svm_radial, OJ_test, OJ_test$Purchase), "%\n")

#####
set.seed(5208)
svm_tune <- tune(svm, Purchase ~ . , data = OJ_train, kernel = "radial",
                 ranges = list(cost = seq(0.01, 10, length = 100)))
summary(svm_tune)

#####
svm_radial <- svm(Purchase ~ . , data = OJ_train, kernel = "radial",
                  cost = svm_tune$best.parameters$cost)

cat("Training Error Rate:", 100 * calc_error_rate(svm_radial, OJ_train, OJ_train$Purchase), "%\n")
cat("Test Error Rate:", 100 * calc_error_rate(svm_radial, OJ_test, OJ_test$Purchase), "%\n")


#(g) Repeat parts (b) through (e) using a support vector machine with a polynomial kernel. Set degree=2.

set.seed(5208)

svm_poly <- svm(Purchase ~ . , data = OJ_train, kernel = "poly", degree = 2)
summary(svm_poly)

cat("Training Error Rate:", 100 * calc_error_rate(svm_poly, OJ_train, OJ_train$Purchase), "%\n")
cat("Test Error Rate:", 100 * calc_error_rate(svm_poly, OJ_test, OJ_test$Purchase), "%\n")

#####
set.seed(5208)
svm_tune <- tune(svm, Purchase ~ . , data = OJ_train, kernel = "poly", 
                 degree = 2, ranges = list(cost = seq(0.01, 10, length = 100)))
summary(svm_tune)

####
svm_poly <- svm(Purchase ~ . , data = OJ_train, kernel = "poly", 
                degree = 2, cost = svm_tune$best.parameters$cost)

cat("Training Error Rate:", 100 * calc_error_rate(svm_poly, OJ_train, OJ_train$Purchase), "%\n")
cat("Test Error Rate:", 100 * calc_error_rate(svm_poly, OJ_test, OJ_test$Purchase), "%\n")

#(h) Overall, which approach seems to give the best results on this data?
"""
Overall, radial basis kernel seems to be producing minimum misclassification 
error on training set but the linear kernel performs better on test data.

"""
