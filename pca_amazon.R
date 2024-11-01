library(tidyverse)
library(tidymodels)
library(vroom)
library(themis)

amazon_train_data <- vroom("./train.csv")
amazon_test_data <- vroom("./test.csv")

amazon_train_data <- amazon_train_data %>%
  mutate(ACTION = factor(ACTION))

# pca recipe
pca_recipe <- recipe(ACTION~.,data= amazon_train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold = .85)

## SMOTE recipe
SMOTE_recipe <- recipe(ACTION~., data=amazon_train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold = .85) %>%
  step_smote(all_outcomes(), neighbors=4)

logRegModel <- logistic_reg() %>%
  set_engine("glm")

log_wf <- workflow() %>%
  add_recipe(SMOTE_recipe) %>%
  add_model(logRegModel) %>%
  fit(data=amazon_train_data)

log_predictions <- predict(log_wf,
                           new_data=amazon_test_data,
                           type="prob")

logistic_submission <- log_predictions %>%
  bind_cols(.,amazon_test_data) %>%
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1)

vroom_write(x=logistic_submission, file ="./LogisticPreds.csv", delim=",")

## HW 15 Penalized Logistic Regression
log_pen_model <- logistic_reg(mixture=tune(),penalty=tune()) %>%
  set_engine("glmnet")

log_pen_workflow <- workflow() %>%
  add_recipe(SMOTE_recipe) %>%
  add_model(log_pen_model)

log_pen_tuning_grid <- grid_regular(penalty(),
                                    mixture(),
                                    levels = 10)

log_pen_folds <- vfold_cv(amazon_train_data, v = 10, repeats= 1)

log_pen_CV_results <- log_pen_workflow %>%
  tune_grid(resamples=log_pen_folds,
            grid=log_pen_tuning_grid,
            metrics=metric_set(roc_auc))

best_log_pen_tune <- log_pen_CV_results %>%
  select_best(metric = "roc_auc")

final_log_pen_wf <-
  log_pen_workflow %>%
  finalize_workflow(best_log_pen_tune) %>%
  fit(data=amazon_train_data)

log_pen_predictions <- predict(final_log_pen_wf,
                               new_data=amazon_test_data,
                               type="prob")

logistic_pen_submission <- log_pen_predictions %>%
  bind_cols(.,amazon_test_data) %>%
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1)

vroom_write(x=logistic_pen_submission, file ="./Logistic_Pen_Preds.csv", delim=",")


## HW 17 KNN model
library(kknn)

# fit the model
knn_model <- nearest_neighbor(neighbors=tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

# workflow
knn_wf <- workflow() %>%
  add_recipe(SMOTE_recipe) %>%
  add_model(knn_model)

# tuning grid
knn_tuning_grid <- grid_regular(neighbors(),
                                levels = 5)
#folds
knn_folds <- vfold_cv(amazon_train_data, v = 10, repeats= 1)

# cross-validation
knn_CV_results <- knn_wf %>%
  tune_grid(resamples=knn_folds,
            grid=knn_tuning_grid,
            metrics=metric_set(roc_auc))

# pick the best tuning parameter
best_knn_tune <- knn_CV_results %>%
  select_best(metric = "roc_auc")

# finalize the workflow
final_knn_wf <-
  knn_wf %>%
  finalize_workflow(best_knn_tune) %>%
  fit(data=amazon_train_data)

# make the predictions
knn_predictions <- predict(final_knn_wf,
                           new_data=amazon_test_data,
                           type="prob")

# create the file to submit to kaggle
knn_submission <- knn_predictions %>%
  bind_cols(.,amazon_test_data) %>%
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1)

vroom_write(x=knn_submission, file ="./KNN_Preds.csv", delim=",")

## random forest for a binary response
install.packages("ranger")
my_randomforest_model_amazon <- rand_forest(mtry = tune(),
                                            min_n=tune(),
                                            trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# set workflow
randomforest_wf_amazon <- workflow() %>%
  add_recipe(SMOTE_recipe) %>%
  add_model(my_randomforest_model_amazon)

# grid of values to tune over
grid_of_randomforest_tuning_params_amazon <- grid_regular(mtry(range=c(1,10)),
                                                          min_n(),
                                                          levels = 5)

# split data for CV
randomforest_folds_amazon <- vfold_cv(amazon_train_data, v = 10, repeats=1)

# Run the CV
randomforest_CV_results_amazon <- randomforest_wf_amazon %>%
  tune_grid(resamples=randomforest_folds_amazon,
            grid=grid_of_randomforest_tuning_params_amazon,
            metrics=metric_set(roc_auc))

# Find best tuning parameters
best_randomforestTune_amazon <- randomforest_CV_results_amazon %>%
  select_best(metric = "roc_auc")

# Finalize the workflow and fit it
final_randomforest_wf_amazon <- randomforest_wf_amazon %>%
  finalize_workflow(best_randomforestTune_amazon) %>%
  fit(data=amazon_train_data)

randomforest_preds <- predict(final_randomforest_wf_amazon, new_data = amazon_test_data, type = "prob")

# create the file to submit to kaggle
random_forest_submission <- randomforest_preds %>%
  bind_cols(.,amazon_test_data) %>%
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1)

# write out the file
vroom_write(x=random_forest_submission, file ="./random_forest_Preds.csv", delim=",")


## Naive bayes Classifier

# nb model 
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(SMOTE_recipe) %>%
  add_model(nb_model)

# tune smoothness and laplace
# tuning grid
nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)
#folds
nb_folds <- vfold_cv(amazon_train_data, v = 10, repeats= 1)

# cross-validation
nb_CV_results <- nb_wf %>%
  tune_grid(resamples=nb_folds,
            grid=nb_tuning_grid,
            metrics=metric_set(roc_auc))

# pick the best tuning parameter
best_nb_tune <- nb_CV_results %>%
  select_best(metric = "roc_auc")

# # Finalize the workflow and fit it
final_nb_wf <- nb_wf %>%
  finalize_workflow(best_nb_tune) %>%
  fit(data=amazon_train_data)

# make predictions with the model
nb_preds <- predict(final_nb_wf, new_data = amazon_test_data, type = "prob")

# create the file to submit to kaggle
nb_submission <- nb_preds %>%
  bind_cols(.,amazon_test_data) %>%
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1)

# write out the file
vroom_write(x=nb_submission, file ="./nb_Preds.csv", delim=",")




## SVM models
## linear model
svmLinear <- svm_linear(cost=tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_linear_wf <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(svmLinear)

# tune degree and cost
# tuning grid
svm_linear_tuning_grid <- grid_regular(cost(),
                               levels = 5)
#folds
svm_linear_folds <- vfold_cv(amazon_train_data, v = 10, repeats= 1)

# cross-validation
svm_linear_CV_results <- svm_linear_wf %>%
  tune_grid(resamples=svm_linear_folds,
            grid=svm_linear_tuning_grid,
            metrics=metric_set(roc_auc))

# pick the best tuning parameter
best_svm_linear_tune <- svm_linear_CV_results %>%
  select_best(metric = "roc_auc")

# Finalize the workflow and fit it
final_svm_linear_wf <- svm_linear_wf %>%
  finalize_workflow(best_svm_linear_tune) %>%
  fit(data=amazon_train_data)

# make predictions with the model
svm_linear_preds <- predict(final_svm_linear_wf, new_data = amazon_test_data, type = "prob")

# create the file to submit to kaggle
svm_linear_submission <- svm_linear_preds %>%
  bind_cols(.,amazon_test_data) %>%
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1)

# write out the file
vroom_write(x=svm_linear_submission, file ="./svm_linear_Preds.csv", delim=",")




## polynomial model
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_poly_wf <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(svmPoly)

# tune degree and cost
# tuning grid
svm_poly_tuning_grid <- grid_regular(degree(),
                                       cost(),
                                       levels = 5)
#folds
svm_poly_folds <- vfold_cv(amazon_train_data, v = 10, repeats= 1)

# cross-validation
svm_poly_CV_results <- svm_poly_wf %>%
  tune_grid(resamples=svm_poly_folds,
            grid=svm_poly_tuning_grid,
            metrics=metric_set(roc_auc))

# pick the best tuning parameter
best_svm_poly_tune <- svm_poly_CV_results %>%
  select_best(metric = "roc_auc")

# # Finalize the workflow and fit it
final_svm_poly_wf <- svm_poly_wf %>%
  finalize_workflow(best_sv_poly_tune) %>%
  fit(data=amazon_train_data)

# make predictions with the model
svm_poly_preds <- predict(final_svm_poly_wf, new_data = amazon_test_data, type = "prob")

# create the file to submit to kaggle
svm_poly_submission <- svm_poly_preds %>%
  bind_cols(.,amazon_test_data) %>%
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1)

# write out the file
vroom_write(x=svm_poly_submission, file ="./svm_poly_Preds.csv", delim=",")
