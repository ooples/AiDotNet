[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9b39ac30ccdf4187b1e05530bfc3c27b)](https://app.codacy.com/gh/ooples/AiDotNet?utm_source=github.com&utm_medium=referral&utm_content=ooples/AiDotNet&utm_campaign=Badge_Grade)
[![CodeQL Analyis](https://github.com/ooples/AiDotNet/actions/workflows/codeql.yml/badge.svg)](https://github.com/ooples/AiDotNet/actions/workflows/codeql.yml)

## Ai.Net

This is a library (currently in preview) for getting the latest and greatest ai algorithms and bringing them directly to the .net community. 
Our approach for this library was to both provide an easy way for beginners to be able to use AI/ML since it usually has a very steep learning curve, 
and an easy way for expert level users to be able to fully customize everything about our algorithms. For now we are showcasing our simplified approach
by providing simple linear regression to get feedback on how we can improve our library. 
We will be adding more algorithms in the future and we are open to any contributions to this library. Please let us know what you think about our approach. 
We will be handling all ai algorithms using the same methods.


### How to use this library

Here is an example to show how easy it is to use this library to get a trained model, get metrics for the trained model, and generate new predictions:

```cs
using AiDotNet.Regression;

var inputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
var outputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

var simpleRegression = new SimpleRegression(inputs, outputs);
var metrics = simpleRegression.Metrics;
var predictions = simpleRegression.Predictions;
```

Here is an example for more advanced users to customize everything used in the algorithm:

```cs
using AiDotNet.Regression;
using AiDotNet.Normalization;

var inputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
var outputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

var advancedSimpleRegression = new SimpleRegression(inputs, outputs, new SimpleRegressionOptions()
{
    TrainingPctSize = 20,
    Normalization = new DecimalNormalization()
});
var metrics2 = advancedSimpleRegression.Metrics;
var predictions2 = advancedSimpleRegression.Predictions;
```
