library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(embed)

amazon_train_data <- vroom("./train.csv")
amazon_test_data <- vroom("./test.csv")

glimpse(amazon_train_data)
length(unique(amazon_train_data$ROLE_CODE))

## make some bar plots for top 10 categories of department and family
summary_amazon_family <- amazon_train_data %>%
  group_by(ROLE_FAMILY) %>%
  summarise(Count = n(), .groups = 'drop')
summary_amazon_dept <- amazon_train_data %>%
  group_by(ROLE_DEPTNAME) %>%
  summarise(Count = n(), .groups = 'drop')
top_categories_family <- summary_amazon_family %>%
  group_by(ROLE_FAMILY) %>%
  summarise(Total = sum(Count), .groups = 'drop') %>%
  top_n(10, Total)
top_categories_dept <- summary_amazon_dept %>%
  group_by(ROLE_DEPTNAME) %>%
  summarise(Total = sum(Count), .groups = 'drop') %>%
  top_n(10, Total)

ggplot(top_categories_family, aes(x = reorder(ROLE_FAMILY, -Total), y = Total)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +  # Flip the axes for better readability
  labs(title = "Count of Outcomes by Family",
       x = "ROLE_FAMILY",
       y = "Count") +
  theme_minimal()

ggplot(top_categories_dept, aes(x = reorder(ROLE_DEPTNAME, -Total), y = Total)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +  # Flip the axes for better readability
  labs(title = "Count of Outcomes by Department",
       x = "ROLE_DEPTNAME",
       y = "Count") +
  theme_minimal()


# make the other category for those with less than .1% of the observations and make dummy variables
amazon_train_data <- amazon_train_data %>%
  mutate(ACTION = factor(ACTION))

first_recipe <- recipe(ACTION~.,data= amazon_train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors())

amazon_prepped_recipe <- prep(first_recipe)
baked_amazon <- bake(amazon_prepped_recipe, new_data=amazon_train_data)


## let's try our first model  (logistic regression) hw 14
logRegModel <- logistic_reg() %>%
  set_engine("glm")

log_wf <- workflow() %>%
  add_recipe(first_recipe) %>%
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
log_pen_recipe <- recipe(ACTION~.,data= amazon_train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  ## step_other(all_nominal_predictors(), threshold = 0.001)
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

log_pen_model <- logistic_reg(mixture=tune(),penalty=tune()) %>%
  set_engine("glmnet")

log_pen_workflow <- workflow() %>%
  add_recipe(log_pen_recipe) %>%
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

# create the recipe, making sure to normalize for when we calculate distances
knn_recipe <- recipe(ACTION~.,data= amazon_train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  ## step_other(all_nominal_predictors(), threshold = 0.001)
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

# fit the model
knn_model <- nearest_neighbor(neighbors=tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

# workflow
knn_wf <- workflow() %>%
  add_recipe(knn_recipe) %>%
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



