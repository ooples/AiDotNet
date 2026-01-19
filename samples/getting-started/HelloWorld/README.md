# Hello World - Your First AiDotNet Model

This sample demonstrates the simplest possible AiDotNet model: a neural network that learns the XOR function.

## What You'll Learn

- How to create a `AiModelBuilder`
- How to configure a basic neural network
- How to train and make predictions

## The XOR Problem

XOR (exclusive or) is a classic machine learning problem:

| Input A | Input B | Output |
|---------|---------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

This is not linearly separable, so it requires a neural network with at least one hidden layer.

## Running the Sample

```bash
dotnet run
```

## Expected Output

```
Training neural network on XOR problem...
Epoch 0: Loss = 0.2500
Epoch 200: Loss = 0.1234
Epoch 400: Loss = 0.0567
...
Training complete!

Predictions:
[0, 0] => 0.02 (expected: 0)
[0, 1] => 0.97 (expected: 1)
[1, 0] => 0.98 (expected: 1)
[1, 1] => 0.03 (expected: 0)
```

## Code Walkthrough

The key parts of this sample:

1. **Create training data** - Define inputs and expected outputs
2. **Build the model** - Use `AiModelBuilder` with fluent API
3. **Train** - Call `BuildAsync()` with the training data
4. **Predict** - Use the trained model to make predictions

## Next Steps

- [BasicClassification](../BasicClassification/) - Learn multi-class classification
- [BasicRegression](../BasicRegression/) - Learn regression for continuous values
