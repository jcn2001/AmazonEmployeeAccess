library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(ggmosaic)
library(embed)

amazon_train_data <- vroom("C:/Users/Josh/Documents/stat348/AmazonEmployee/amazon-employee-access-challenge/train.csv")

glimpse(amazon_train_data)
## 1050  

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
first_recipe <- recipe(ACTION~.,data= amazon_train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors())

prep_amazon <- amazon_prepped_recipe <- prep(first_recipe)
baked_amazon <- bake(amazon_prepped_recipe, new_data=amazon_train_data)