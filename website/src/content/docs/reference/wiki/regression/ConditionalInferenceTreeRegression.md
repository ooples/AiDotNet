---
title: "ConditionalInferenceTreeRegression<T>"
description: "Represents a conditional inference tree regression model that builds decision trees based on statistical tests."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

Represents a conditional inference tree regression model that builds decision trees based on statistical tests.

## For Beginners

This class creates a special type of decision tree for predicting numerical values.

Think of a decision tree like a flowchart of yes/no questions that helps you make predictions:

- The tree starts with a question (like "Is temperature > 70°F?")
- Based on the answer, it follows different branches
- It continues asking questions until it reaches a final prediction

What makes this tree special is how it chooses the questions:

- It uses statistical tests to find the most meaningful questions to ask
- It avoids favoring certain types of data unfairly
- It provides a measurement of confidence (p-value) for each split

This approach tends to create more reliable and fair prediction models.

## How It Works

A conditional inference tree is a type of decision tree that uses statistical tests to determine optimal
splits in the data. Unlike traditional decision trees that use measures like Gini impurity or information gain,
conditional inference trees use statistical significance testing to create unbiased trees that don't favor
features with many possible split points.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
    new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 }
};
double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new ConditionalInferenceTreeRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained ConditionalInferenceTreeRegression.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConditionalInferenceTreeRegression` | Initializes a new instance with default settings. |
| `ConditionalInferenceTreeRegression(ConditionalInferenceTreeOptions,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the `ConditionalInferenceTreeRegression` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildTreeAsync(Matrix<>,Vector<>,Int32)` | Recursively builds the decision tree by finding optimal splits. |
| `CalculateFeatureImportancesAsync(Int32)` | Asynchronously calculates the importance of each feature in the model. |
| `CalculateFeatureImportancesRecursiveAsync(ConditionalInferenceTreeNode<>,[])` | Recursively calculates feature importances by traversing the tree. |
| `Clone` | Creates a deep copy via serialization to ensure the private _root tree is preserved. |
| `CreateNewInstance` | Creates a new instance of the conditional inference tree regression model with the same configuration. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `DeserializeNode(BinaryReader)` | Deserializes a tree node from a binary reader. |
| `FindBestSplitAsync(Matrix<>,Vector<>)` | Asynchronously finds the best feature and threshold to split the data. |
| `FindBestSplitForFeature(Matrix<>,Vector<>,Int32)` | Finds the best threshold to split data for a specific feature. |
| `GetModelMetadata` | Gets metadata about the regression model. |
| `GetOptions` |  |
| `PredictAsync(Matrix<>)` | Asynchronously predicts target values for new input data. |
| `PredictSingle(Vector<>)` | Predicts a target value for a single input sample. |
| `Serialize` | Serializes the model to a byte array. |
| `SerializeNode(BinaryWriter,ConditionalInferenceTreeNode<>)` | Serializes a tree node to a binary writer. |
| `SplitData(Matrix<>,Vector<>,Int32,)` | Splits the data into left and right subsets based on the feature and threshold. |
| `TrainAsync(Matrix<>,Vector<>)` | Asynchronously trains the regression model on the provided training data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | The options that control the behavior of the conditional inference tree. |
| `_root` | The root node of the decision tree. |

