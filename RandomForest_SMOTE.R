# ==============================================================================
# ========================== Classification Trees ==============================
# ==============================================================================
library(vroom)
library(tidymodels)
library(embed)
library(workflows)
library(kernlab)
library(themis)
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
  step_normalize(all_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 5)   # 💡 New line


prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)


# -----------------------------------------------------------------------------
# Define model
# -----------------------------------------------------------------------------
my_mod <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

# -----------------------------------------------------------------------------
# Parameter tuning setup
# -----------------------------------------------------------------------------
param_set <- parameters(
  finalize(mtry(), baked),
  min_n()
)

grid_of_tuning_params <- grid_regular(param_set, levels = 5)

folds <- vfold_cv(train, v = 3, repeats = 1)

CV_results <- tune_grid(
  wf,
  resamples = folds,
  grid = grid_of_tuning_params,
  metrics = metric_set(roc_auc)
)

bestTune <- select_best(CV_results, metric = "roc_auc")
bestTune

# -----------------------------------------------------------------------------
# finalize workflow 
# -----------------------------------------------------------------------------
final_wf <- wf |>
  finalize_workflow(bestTune) |>
  fit(data = train)

## Predict
preds <- final_wf |>
  predict(new_data = test, 
          type = "prob")


# -----------------------------------------------------------------------------
# Kaggle Submission
# -----------------------------------------------------------------------------
kaggle_submission <- bind_cols(
  Id = test$id,
  preds) %>%
  rename(Action = .pred_1) %>%  # rename to match Kaggle format exactly
  select(Id, Action)

# Write to CSV
vroom_write(kaggle_submission, file = "./ClassificationTrees_SMOTE3.csv", delim = ",")
