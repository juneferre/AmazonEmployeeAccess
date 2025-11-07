# ==============================================================================
# ============================ Logistic Regression =============================
# ==============================================================================


library(vroom)
library(ggmosaic)
library(tidymodels)
library(embed)
library(workflows)
library(themis)

# ------------------------------------------------------------------------------
# Read in Data
# ------------------------------------------------------------------------------
train <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/train.csv") |>
  mutate(ACTION = as.factor(ACTION))
test <- vroom("/Users/juneferre/fall2025/stat348/Kaggle/amazon/test.csv")


# ------------------------------------------------------------------------------
# Recipe (Preprocessing)
# ------------------------------------------------------------------------------
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors()) |>
  # for SMOTE
  step_smote(all_outcomes(), neighbors = 5)
  # for PCR
  # step_normalize(all_predictors()) |>
  # step_pca(all_predictors(), threshold= 0.8)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)



# ------------------------------------------------------------------------------
# Define the Model
# ------------------------------------------------------------------------------
logRegModel <- logistic_reg() |>
  set_engine("glm")

# ------------------------------------------------------------------------------
# Workflow
# ------------------------------------------------------------------------------
logReg_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(logRegModel) |>
  fit(data = train)


# ------------------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------------------
amazon_predictions <- predict(logReg_workflow,
                              new_data = test,
                              type = "prob") # "class" or "prob"


## with type = "prob" amazon_predictions will have 2 columns
## one for Pr(0) and the other for Pr(1)
## with type = "class" it will just have one column (0 or 1)


# ------------------------------------------------------------------------------
# Submission
# ------------------------------------------------------------------------------
kaggle_submission <- bind_cols(
  Id = test$id,
  amazon_predictions) %>%
  rename(Action = .pred_1) %>%  # rename to match Kaggle format exactly
  select(Id, Action)

# Write to CSV
vroom_write(kaggle_submission, file = "./LogRegression_SMOTE.csv", delim = ",")
