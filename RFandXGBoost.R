# ==============================================================================
# ========================== Amazon Access Challenge ===========================
# ==============================================================================

library(vroom)
library(tidymodels)
library(embed)
library(workflows)
library(kernlab)
library(themis)
library(dplyr)

# -----------------------------------------------------------------------------
# Read in data sets
# -----------------------------------------------------------------------------
train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/train.csv") |>
  mutate(ACTION = as.factor(ACTION))
test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/test.csv")

# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_mutate(across(where(is.numeric), ~replace_na(., 0))) %>%
  step_normalize(all_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 5)

# -----------------------------------------------------------------------------
# Define models
# -----------------------------------------------------------------------------
rf_mod <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 100
) %>%
  set_engine("ranger", importance = "impurity", sample.fraction = 0.8, respect.unordered.factors = TRUE) %>%
  set_mode("classification")

xgb_mod <- boost_tree(
  trees = 100,
  learn_rate = tune(),
  tree_depth = tune(),
  loss_reduction = tune(),
  min_n = tune(),
  sample_size = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# -----------------------------------------------------------------------------
# Cross-validation setup
# -----------------------------------------------------------------------------
set.seed(123)
folds <- vfold_cv(train, v = 3)

# -----------------------------------------------------------------------------
# Random Forest tuning
# -----------------------------------------------------------------------------
wf_rf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

grid_rf <- grid_random(
  mtry(range = c(2, 20)),
  min_n(range = c(2, 15)),
  size = 25
)

CV_results_rf <- wf_rf %>%
  tune_grid(resamples = folds,
            grid = grid_rf,
            metrics = metric_set(roc_auc))

bestTune_rf <- select_best(CV_results_rf, "roc_auc")

# Fit final random forest model
final_wf_rf <- wf_rf %>%
  finalize_workflow(bestTune_rf) %>%
  fit(data = train)

preds_rf <- final_wf_rf %>%
  predict(new_data = test, type = "prob") %>%
  rename(Action_rf = .pred_1)

# -----------------------------------------------------------------------------
# XGBoost tuning
# -----------------------------------------------------------------------------
wf_xgb <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(xgb_mod)

grid_xgb <- grid_latin_hypercube(
  learn_rate(),
  tree_depth(),
  loss_reduction(),
  min_n(),
  sample_size = sample_prop(),
  size = 25
)

CV_results_xgb <- wf_xgb %>%
  tune_grid(resamples = folds,
            grid = grid_xgb,
            metrics = metric_set(roc_auc))

bestTune_xgb <- select_best(CV_results_xgb, "roc_auc")

# Fit final XGBoost model
final_wf_xgb <- wf_xgb %>%
  finalize_workflow(bestTune_xgb) %>%
  fit(data = train)

preds_xgb <- final_wf_xgb %>%
  predict(new_data = test, type = "prob") %>%
  rename(Action_xgb = .pred_1)

# -----------------------------------------------------------------------------
# 🧩 Ensemble Line — Combine Predictions
# -----------------------------------------------------------------------------
ensemble_preds <- bind_cols(test, preds_rf, preds_xgb) %>%
  mutate(Action = (Action_rf + Action_xgb) / 2)   # simple average ensemble

# optional: weighted average if XGB outperforms slightly
# mutate(Action = 0.6 * Action_xgb + 0.4 * Action_rf)

# -----------------------------------------------------------------------------
# Kaggle Submission
# -----------------------------------------------------------------------------
kaggle_submission <- ensemble_preds %>%
  select(id, Action) %>%
  rename(Id = id)

vroom_write(kaggle_submission, file = "./Ensemble_Submission.csv", delim = ",")
