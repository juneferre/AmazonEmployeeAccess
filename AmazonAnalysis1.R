library(vroom)
library(ggmosaic)
library(tidymodels)
library(embed)

train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/train.csv")
test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/test.csv")


my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)



