# ==============================================================================
# ============================ Logistic Regression =============================
# ==============================================================================


library(vroom)
library(ggmosaic)
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
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)


# Define model
logRegModel <- logistic_reg() |>
  set_engine("glm")

# workflow 
logReg_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(logRegModel) |>
  fit(data = train)


# make predictions

amazon_predictions <- predict(logReg_workflow,
                              new_data = test,
                              type = "prob") # "class" or "prob"


## with type = "prob" amazon_predictions will have 2 columns
## one for Pr(0) and the other for Pr(1)
## with type = "class" it will just have one column (0 or 1)


# Bind the Id column and rename .pred_1 to Action
kaggle_submission <- bind_cols(
  Id = test$id,
  amazon_predictions
) %>%
  rename(Action = .pred_1) %>%  # rename to match Kaggle format exactly
  select(Id, Action)

# Write to CSV
vroom_write(kaggle_submission, file = "./LogRegression1.csv", delim = ",")
