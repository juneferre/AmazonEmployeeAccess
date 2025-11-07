# ==============================================================================
# ============================== Neural Networks ===============================
# ==============================================================================

library(vroom)
library(tidymodels)
library(keras)
library(workflows)
library(tensorflow)

# ------------------------------------------------------------------------------
# Read in Data
# ------------------------------------------------------------------------------

train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/train.csv") |>
  mutate(ACTION = as.factor(ACTION))
train <- vroom("./train.csv") |>
  mutate(ACTION = as.factor(ACTION))

test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/test.csv")
test <- vroom("./test.csv")

# ------------------------------------------------------------------------------
# Recipe (Preprocessing)
# ------------------------------------------------------------------------------

my_recipe <- recipe(ACTION ~ ., data = train) |>
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  update_role(MGR_ID, new_role = "MGR_ID") %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)  # scale to [0,1]

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

# ------------------------------------------------------------------------------
# Define the Model
# ------------------------------------------------------------------------------

nn_mod <- mlp(
  hidden_units = tune(),
  epochs = 50) |>
  set_mode("classification") |>
  set_engine("keras")

# ------------------------------------------------------------------------------
# Workflow
# ------------------------------------------------------------------------------

nn_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(nn_mod)

# ------------------------------------------------------------------------------
# Cross-validation Setup
# ------------------------------------------------------------------------------

folds <- vfold_cv(train, v = 5)

# ------------------------------------------------------------------------------
# Grid Search for Hidden Units
# ------------------------------------------------------------------------------

maxHiddenUnits <- 100  # you can change to larger for deeper tuning
tuning_grid <- grid_regular(hidden_units(range = c(1, maxHiddenUnits)), levels = 10)

# ------------------------------------------------------------------------------
# Tune the Model
# ------------------------------------------------------------------------------

CV_results <- nn_wf |>
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(accuracy))

# ------------------------------------------------------------------------------
# Collect and Visualize Results
# ------------------------------------------------------------------------------

tuned_nn <- CV_results |>
  collect_metrics() |>
  filter(.metric == "accuracy")

# Plot accuracy vs. hidden units
tuned_nn |>
  ggplot(aes(x = hidden_units, y = mean)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Neural Network Tuning Results",
    x = "Number of Hidden Units",
    y = "Mean Accuracy")

# ------------------------------------------------------------------------------
# Finalize Workflow and Fit Full Model
# ------------------------------------------------------------------------------

bestTune <- select_best(CV_results, metric = "accuracy")

final_wf <- nn_wf |>
  finalize_workflow(bestTune) |>
  fit(data = train)

# ------------------------------------------------------------------------------
# Predict on Test Set
# ------------------------------------------------------------------------------

preds <- predict(final_wf, new_data = test, type = "prob")

# ------------------------------------------------------------------------------
# Create Kaggle Submission
# ------------------------------------------------------------------------------

kaggle_submission <- bind_cols(
  Id = test$id,
  preds) |>
  rename(Action = .pred_1) |>
  select(Id, Action)

vroom_write(kaggle_submission, file = "./NN_submission.csv", delim = ",")

