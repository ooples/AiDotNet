# Advanced House Price Prediction

This sample demonstrates advanced regression techniques for predicting house prices, including:

- Feature engineering (polynomial features, interactions)
- Multiple regression algorithm comparison
- Bayesian hyperparameter optimization
- Comprehensive metrics (R2, MAE, RMSE, MAPE)

## What You'll Learn

- How to engineer features with interactions and polynomial terms
- How to compare multiple regression models (Ridge, Lasso, ElasticNet, Gradient Boosting)
- How to use `BayesianOptimizer` for hyperparameter tuning
- How to interpret regression metrics
- How to analyze feature importance

## The Problem

Predict house prices based on:

- **Physical attributes**: Square footage, bedrooms, bathrooms, stories, garage spaces
- **Property characteristics**: Age, lot size
- **Location factors**: School rating, crime rate, distance to city
- **Engineered features**: Sqft per bedroom, bathroom ratio, sqft x school rating

## Regression Models Compared

| Model | Description |
|-------|-------------|
| **Ridge Regression** | L2 regularization - shrinks coefficients but keeps all features |
| **Lasso Regression** | L1 regularization - can zero out unimportant features |
| **Elastic Net** | Combined L1+L2 - best of both worlds |
| **Gradient Boosting** | Ensemble of decision trees - handles non-linearities |

## Bayesian Hyperparameter Optimization

The sample uses Bayesian optimization to find optimal Gradient Boosting hyperparameters:

```csharp
var bayesianOptimizer = new BayesianOptimizer<double, Matrix<double>, Vector<double>>(
    maximize: true,
    acquisitionFunction: AcquisitionFunctionType.ExpectedImprovement,
    nInitialPoints: 5,
    explorationWeight: 2.0);

var searchSpace = new HyperparameterSearchSpace()
    .AddInteger("n_trees", 50, 200)
    .AddInteger("max_depth", 3, 10)
    .AddContinuous("learning_rate", 0.01, 0.3)
    .AddContinuous("subsample", 0.5, 1.0);
```

## Running the Sample

```bash
cd samples/regression/PricePrediction
dotnet run
```

## Expected Output

```
=== AiDotNet Advanced House Price Prediction ===
Comparing multiple regression models with feature engineering and Bayesian optimization

Training set: 800 samples
Test set: 200 samples
Features (13): sqft, bedrooms, bathrooms, age, lot_size, garage, stories, school_rating, crime_rate, distance_city, sqft_per_bed, bath_ratio, sqft_x_school

Price Statistics:
  Min: $95,234
  Max: $1,245,678
  Mean: $478,123
  Median: $456,789

===========================================================================
              MODEL COMPARISON (Standard Hyperparameters)
===========================================================================

Training Ridge Regression...
  R2: 0.8234
  MAE: $45,678
  RMSE: $58,901
  MAPE: 12.34%

Training Lasso Regression...
  R2: 0.8156
  MAE: $47,234
  RMSE: $61,234
  MAPE: 13.12%

Training Elastic Net...
  R2: 0.8201
  MAE: $46,234
  RMSE: $59,876
  MAPE: 12.67%

Training Gradient Boosting...
  R2: 0.9123
  MAE: $32,456
  RMSE: $41,234
  MAPE: 8.45%

===========================================================================
         BAYESIAN HYPERPARAMETER OPTIMIZATION (Gradient Boosting)
===========================================================================

Optimizing hyperparameters using Bayesian optimization...
Search space:
  - Number of trees: [50, 200]
  - Max depth: [3, 10]
  - Learning rate: [0.01, 0.3]
  - Subsample ratio: [0.5, 1.0]
  - Trials: 15

Top 5 Trials:
  Trial 1: R2=0.9234
    n_trees=150, max_depth=6, lr=0.087, subsample=0.85

Best hyperparameters found:
  Number of trees: 150
  Max depth: 6
  Learning rate: 0.0870
  Subsample ratio: 0.850
  Cross-validation R2: 0.9234

===========================================================================
                    FINAL MODEL WITH OPTIMIZED HYPERPARAMETERS
===========================================================================

Optimized Gradient Boosting Performance:
  R2 Score: 0.9312
  MAE: $28,456
  RMSE: $36,789
  MAPE: 7.23%

Improvement over default hyperparameters:
  R2: +1.89%
  MAE reduction: +12.33%

Sample Predictions:
----------------------------------------------------------------------
#    Predicted       Actual          Error %      Status
----------------------------------------------------------------------
1    $425,234        $418,567        1.6%         Good
2    $675,890        $698,234        3.2%         Good
3    $312,456        $345,678        9.6%         Good
...

===========================================================================
                           FEATURE ANALYSIS
===========================================================================

Feature Importance (based on contribution to predictions):
--------------------------------------------------
   1. sqft                 0.8234 ################################
   2. school_rating        0.6789 ###########################
   3. sqft_x_school        0.6234 ########################
   4. bedrooms             0.5123 ####################
   5. bathrooms            0.4567 ##################
...

=== Sample Complete ===
```

## Key Metrics Explained

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **R2 (R-squared)** | Proportion of variance explained | Closer to 1.0 |
| **MAE (Mean Absolute Error)** | Average prediction error in dollars | Lower is better |
| **RMSE (Root Mean Square Error)** | Penalizes large errors more | Lower is better |
| **MAPE (Mean Absolute Percentage Error)** | Average error as percentage | Lower is better |

## Feature Engineering Highlights

```csharp
// Interaction features capture relationships
double sqftPerBedroom = sqft / bedrooms;
double bathroomRatio = bathrooms / bedrooms;
double sqftTimesSchool = sqft * schoolRating / 1000;

// Non-linear price relationships
price += Math.Pow(Math.Max(0, schoolRating - 5), 2) * 15000; // Premium for good schools
price -= Math.Pow(distanceToCity, 1.5) * 300; // Non-linear distance penalty
```

## Key Code Patterns

### Comparing Multiple Models

```csharp
var regressors = new Dictionary<string, Func<IRegressor>>
{
    ["Ridge"] = () => new RidgeRegressionWrapper(new RidgeRegressionOptions<double> { Alpha = 1.0 }),
    ["Lasso"] = () => new LassoRegressionWrapper(new LassoRegressionOptions<double> { Alpha = 0.1 }),
    ["Gradient Boosting"] = () => new GradientBoostingWrapper(new GradientBoostingRegressionOptions
    {
        NumberOfTrees = 100,
        MaxDepth = 5,
        LearningRate = 0.1
    })
};

foreach (var (name, createRegressor) in regressors)
{
    var regressor = createRegressor();
    regressor.Train(trainFeatures, trainPrices);
    var predictions = regressor.Predict(testFeatures);
    var metrics = CalculateMetrics(testPrices, predictions);
}
```

### Bayesian Optimization

```csharp
var result = bayesianOptimizer.Optimize(
    objectiveFunction: (params) => {
        var model = new GradientBoostingWrapper(/*params*/);
        model.Train(trainX, trainY);
        return CalculateCrossValidationR2(model);
    },
    searchSpace: searchSpace,
    nTrials: 15
);

var bestParams = result.BestParameters;
```

## Next Steps

- [DemandForecasting](../DemandForecasting/) - Time-aware regression with prediction intervals
- [BasicRegression](../../getting-started/BasicRegression/) - Simpler regression example
- [IrisClassification](../../classification/MultiClassification/IrisClassification/) - Classification example
