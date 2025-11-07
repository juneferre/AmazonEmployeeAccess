# ==============================================================================
# ====================== Amazon Access Challenge (FAST VERSION) ================
# ==============================================================================

library(vroom)
library(tidymodels)
library(embed)
library(workflows)
library(kernlab)
library(themis)
library(stacks)
library(doParallel)
library(doParallel)
library(dplyr)
library(xgboost)
library(stacks)

set.seed(123)

# -----------------------------------------------------------------------------
# Data Import
# -----------------------------------------------------------------------------
train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/train.csv") |>
  mutate(ACTION = as.factor(ACTION))
test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/test.csv")

# -----------------------------------------------------------------------------
# Parallel Setup (use all but one core)
# -----------------------------------------------------------------------------
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# -----------------------------------------------------------------------------
# Recipe (feature engineering + normalization)
# -----------------------------------------------------------------------------
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_mutate(across(where(is.numeric), ~replace_na(., 0))) %>%
  step_normalize(all_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 5)

# -----------------------------------------------------------------------------
# Cross-Validation (fewer folds for speed)
# -----------------------------------------------------------------------------
folds <- vfold_cv(train, v = 3)

# -----------------------------------------------------------------------------
# Model Definitions (simplified trees for tuning)
# -----------------------------------------------------------------------------

# Random Forest
rf_mod <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 100   # fewer trees during tuning for speed
) %>%
  set_engine("ranger", importance = "impurity", sample.fraction = 0.8) %>%
  set_mode("classification")

# XGBoost
xgb_mod <- boost_tree(
  trees = 300,
  learn_rate = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# -----------------------------------------------------------------------------
# Workflows
# -----------------------------------------------------------------------------
wf_rf <- workflow() %>% add_recipe(my_recipe) %>% add_model(rf_mod)
wf_xgb <- workflow() %>% add_recipe(my_recipe) %>% add_model(xgb_mod)

# -----------------------------------------------------------------------------
# Random Search Grids (small for fast iteration)
# -----------------------------------------------------------------------------
grid_rf <- grid_random(
  mtry(range = c(2, 10)),
  min_n(range = c(2, 8)),
  size = 5
)

grid_xgb <- grid_latin_hypercube(
  learn_rate(range = c(0.01, 0.3)),
  tree_depth(range = c(3, 8)),
  min_n(range = c(2, 10)),
  loss_reduction(range = c(0, 1)),
  sample_size = sample_prop(),
  size = 5
)

library(tidymodels)
library(stacks)

# ----------------------------------------------------------------------------- 
# Control setup for stacking (critical)
# ----------------------------------------------------------------------------- 
ctrl_stack <- control_stack_resamples()

# ----------------------------------------------------------------------------- 
# Tune Random Forest (grid search)
# ----------------------------------------------------------------------------- 
CV_results_rf <- wf_rf %>%
  tune_grid(
    resamples = folds,
    grid = grid_rf,
    metrics = metric_set(roc_auc),
    control = ctrl_stack  # <-- added here
  )

best_rf <- select_best(CV_results_rf, metric = "roc_auc")

final_rf <- finalize_workflow(wf_rf, best_rf) %>%
  fit(data = train)

# ----------------------------------------------------------------------------- 
# Tune XGBoost (Bayesian optimization)
# ----------------------------------------------------------------------------- 
CV_results_xgb <- wf_xgb %>%
  tune_bayes(
    resamples = folds,
    metrics = metric_set(roc_auc),
    initial = 5,
    iter = 10,
    control = control_bayes(no_improve = 5, verbose = TRUE, save_pred = TRUE, save_workflow = TRUE)
  )

best_xgb <- select_best(CV_results_xgb, metric = "roc_auc")

final_xgb <- finalize_workflow(wf_xgb, best_xgb) %>%
  fit(data = train)

# ----------------------------------------------------------------------------- 
# Stacking ensemble (automatically learns best weights)
# ----------------------------------------------------------------------------- 
stack_model <- stacks() %>%
  add_candidates(CV_results_rf) %>%
  add_candidates(CV_results_xgb) %>%
  blend_predictions(metric = metric_set(roc_auc)) %>%
  fit_members()

# ----------------------------------------------------------------------------- 
# Generate predictions on test data
# ----------------------------------------------------------------------------- 

stack_preds <- predict(stack_model, new_data = test, type = "prob")

kaggle_submission <- test %>%
  select(id) %>%
  bind_cols(stack_preds %>% select(.pred_1)) %>%   # use .pred_1 (prob of ACTION = 1)
  rename(ACTION = .pred_1)

vroom_write(kaggle_submission, file = "./Stacked_Ensemble_Submission.csv", delim = ",")
