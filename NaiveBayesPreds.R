# ==============================================================================
# ========================== Naive Bayes Predictions ===========================
# ==============================================================================

library(vroom)
library(tidymodels)
library(discrim)
library(workflows)
library(naivebayes)

# Read in data sets 
train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/train.csv") |>
  mutate(ACTION = as.factor(ACTION))
test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/test.csv")


# Feature engineering
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)


# Define model
nb_mod <- naive_Bayes(Laplace = tune(), smoothness=tune()) |>
  set_mode('classification') |>
  set_engine('naivebayes')

# workflow 
nb_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(nb_mod)


# cross validation to tune neighbors
folds <- vfold_cv(train, v = 5)

## grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels=5)


## run the Cross Validation
CV_results <- nb_wf |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- select_best(CV_results, metric = "roc_auc")
bestTune

# finalize workflow 

final_wf <- nb_wf |>
  finalize_workflow(bestTune) |>
  fit(data = train)

## Predict
preds <- final_wf |>
  predict(new_data = test, 
          type = "prob")


## with type = "prob" amazon_predictions will have 2 columns
## one for Pr(0) and the other for Pr(1)
## with type = "class" it will just have one column (0 or 1)


# Bind the Id column and rename .pred_1 to Action
kaggle_submission <- bind_cols(
  Id = test$id,
  preds) %>%
  rename(Action = .pred_1) %>%  # rename to match Kaggle format exactly
  select(Id, Action)

# Write to CSV
vroom_write(kaggle_submission, file = "./NaiveBayes.csv", delim = ",")
