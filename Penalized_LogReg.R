# ==============================================================================
# ===================== Penalized Logistic Regression ==========================
# ==============================================================================


library(vroom)
#library(ggmosaic)
library(tidymodels)
library(embed)
library(workflows)

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
logRegModel <- logistic_reg(mixture = tune(),
                            penalty = tune()) |>
  set_engine("glmnet")

# workflow 
logReg_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(logRegModel) 


# cross validation to tune mixture and penalty

## grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(train, v = 3, repeats = 1)

## run the Cross Validation
CV_results <- logReg_workflow |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- select_best(CV_results, metric = "roc_auc")
bestTune

# finalize workflow 

final_wf <- logReg_workflow |>
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
  preds
) %>%
  rename(Action = .pred_1) %>%  # rename to match Kaggle format exactly
  select(Id, Action)

# Write to CSV
vroom_write(kaggle_submission, file = "./PenalizedLogRegression1.csv", delim = ",")
