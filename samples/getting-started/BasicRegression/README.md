# Basic Regression - House Price Prediction

This sample demonstrates regression using a house price prediction model.

## What You'll Learn

- How to use `AiModelBuilder` for regression tasks
- How to configure multiple regression algorithms
- How to evaluate regression metrics (R², MAE, RMSE)
- How to compare model performance

## The Problem

Predict house prices based on features like:
- Square footage
- Number of bedrooms
- Number of bathrooms
- Age of the house
- Location score

## Running the Sample

```bash
dotnet run
```

## Expected Output

```
=== AiDotNet Basic Regression ===
Predicting house prices

Generated 500 training samples, 100 test samples

Training Gradient Boosting Regression...
  Training complete!

Model Evaluation:
  R² Score: 0.95
  MAE: $12,345
  RMSE: $18,567

Sample Predictions:
  House 1: Predicted=$425,000, Actual=$418,000 (Error: 1.7%)
  House 2: Predicted=$275,000, Actual=$282,000 (Error: 2.5%)
  ...
```

## Key Code

```csharp
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new GradientBoostingRegression<double>(
        nEstimators: 100,
        maxDepth: 5,
        learningRate: 0.1))
    .ConfigurePreprocessing()  // Auto StandardScaler + Imputer
    .BuildAsync(features, prices);
```

## Next Steps

- [PricePrediction](../../regression/PricePrediction/) - Advanced price prediction with feature engineering
- [DemandForecasting](../../regression/DemandForecasting/) - Time-aware regression
