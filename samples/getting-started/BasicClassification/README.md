# Basic Classification - Iris Dataset

This sample demonstrates multi-class classification using the classic Iris flower dataset.

## What You'll Learn

- How to use `AiModelBuilder` for classification
- How to configure preprocessing pipelines
- How to use cross-validation
- How to evaluate model accuracy

## The Iris Dataset

The Iris dataset contains 150 samples of iris flowers, each with 4 features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The task is to classify each flower into one of 3 species:
- Setosa (0)
- Versicolor (1)
- Virginica (2)

## Running the Sample

```bash
dotnet run
```

## Expected Output

```
Loading Iris dataset...
Loaded 150 samples with 4 features

Building Random Forest classifier...
Training with 5-fold cross-validation...

Cross-Validation Results:
  Fold 1: Accuracy = 96.67%
  Fold 2: Accuracy = 93.33%
  ...
  Mean Accuracy: 95.33% (+/- 2.11%)

Final Model Evaluation:
  Training Accuracy: 100.00%
  Test Accuracy: 96.67%
```

## Code Highlights

```csharp
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing(pipeline => pipeline
        .Add(new StandardScaler<double>()))
    .ConfigureCrossValidation(new KFoldCrossValidator<double>(k: 5))
    .BuildAsync(features, labels);
```

## Next Steps

- [SentimentAnalysis](../../classification/BinaryClassification/SentimentAnalysis/) - Binary classification with text
- [IrisClassification](../../classification/MultiClassification/IrisClassification/) - More detailed Iris analysis
