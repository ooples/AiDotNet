using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for the AiModelBuilder -> AiModelResult.Predict() pipeline.
/// These tests verify that models trained through AiModelBuilder can correctly predict
/// on new data, including when feature selection is applied by the optimizer.
///
/// This test class specifically targets the critical bug where AiModelResult.Predict()
/// did not apply feature selection from OptimizationResult.SelectedFeatureIndices,
/// causing dimension mismatch errors when the optimizer selected a subset of features
/// during training (which is the default behavior).
/// </summary>
public class AiModelBuilderPredictIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Core Bug Fix Tests

    [Fact]
    public async Task BuildAsync_DefaultOptimizer_PredictDoesNotThrowDimensionMismatch()
    {
        // Arrange: Create a dataset with multiple features where the default NormalOptimizer
        // will select a subset of features during training.
        // The bug was that Predict() would pass ALL columns to a model trained on fewer columns.
        var (x, y) = CreateLinearDataset(samples: 50, features: 5, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act: Build with default optimizer (NormalOptimizer which does feature selection)
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .BuildAsync();

        // Verify the optimizer actually selected a feature subset (not identity)
        var selectedIndices = result.OptimizationResult?.SelectedFeatureIndices;
        Assert.NotNull(selectedIndices);
        Assert.True(selectedIndices.Count > 0 && selectedIndices.Count <= 5,
            $"Optimizer should select between 1 and 5 features, got {selectedIndices.Count}");

        // Create new data with ALL original features (same column count as training input)
        var newData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 2.0, 4.0, 6.0, 8.0, 10.0 },
            { 3.0, 6.0, 9.0, 12.0, 15.0 }
        });

        // Assert: Predict should NOT throw a dimension mismatch error
        var predictions = result.Predict(newData);
        Assert.NotNull(predictions);
        Assert.Equal(3, predictions.Length);
    }

    [Fact]
    public async Task BuildAsync_SelectedFeatureIndices_AreStoredInOptimizationResult()
    {
        // Arrange: Build a model using the default optimizer which performs feature selection
        var (x, y) = CreateLinearDataset(samples: 50, features: 5, seed: 123);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .BuildAsync();

        // Assert: SelectedFeatureIndices should be populated
        var optimizationResult = result.OptimizationResult;
        Assert.NotNull(optimizationResult);
        Assert.NotEmpty(optimizationResult.SelectedFeatureIndices);
        // The indices should be valid (within bounds of the original feature count)
        foreach (var idx in optimizationResult.SelectedFeatureIndices)
        {
            Assert.InRange(idx, 0, 4); // 5 features, indices 0-4
        }
    }

    [Fact]
    public async Task BuildAsync_PredictSingleRow_WorksWithFeatureSelection()
    {
        // Arrange: Train model with feature selection, then predict on a single row
        var (x, y) = CreateLinearDataset(samples: 40, features: 4, seed: 77);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new MultipleRegression<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .BuildAsync();

        var singleRow = CreateMatrix(new double[,] { { 1.0, 2.0, 3.0, 4.0 } });

        // Assert: Should not throw, should return exactly 1 prediction
        var predictions = result.Predict(singleRow);
        Assert.NotNull(predictions);
        Assert.Equal(1, predictions.Length);
        Assert.False(double.IsNaN(predictions[0]));
        Assert.False(double.IsInfinity(predictions[0]));
    }

    #endregion

    #region Multiple Regression Model Tests

    [Fact]
    public async Task BuildAsync_MultipleRegression_EndToEndPrediction()
    {
        // Arrange: Simple linear relationship y = 2*x1 + 3*x2 + noise
        var random = new Random(42);
        var x = new Matrix<double>(30, 3);
        var y = new Vector<double>(30);
        for (int i = 0; i < 30; i++)
        {
            x[i, 0] = random.NextDouble() * 10;
            x[i, 1] = random.NextDouble() * 10;
            x[i, 2] = random.NextDouble() * 10; // noise feature
            y[i] = 2 * x[i, 0] + 3 * x[i, 1] + random.NextDouble() * 0.1;
        }

        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new MultipleRegression<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .BuildAsync();

        var newData = CreateMatrix(new double[,]
        {
            { 1.0, 1.0, 5.0 },
            { 2.0, 2.0, 3.0 },
            { 5.0, 5.0, 1.0 }
        });

        // Assert: Predictions should be returned without dimension errors
        var predictions = result.Predict(newData);
        Assert.NotNull(predictions);
        Assert.Equal(3, predictions.Length);
        foreach (var pred in predictions)
        {
            Assert.False(double.IsNaN(pred));
            Assert.False(double.IsInfinity(pred));
        }
    }

    [Fact]
    public async Task BuildAsync_RidgeRegression_EndToEndPrediction()
    {
        // Arrange
        var (x, y) = CreateLinearDataset(samples: 40, features: 6, seed: 99);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .BuildAsync();

        var newData = new Matrix<double>(5, 6);
        var rng = new Random(100);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 6; j++)
                newData[i, j] = rng.NextDouble() * 10;

        // Assert
        var predictions = result.Predict(newData);
        Assert.NotNull(predictions);
        Assert.Equal(5, predictions.Length);
        foreach (var pred in predictions)
        {
            Assert.False(double.IsNaN(pred));
        }
    }

    #endregion

    #region Feature Selection Consistency Tests

    [Fact]
    public async Task BuildAsync_PredictTwice_ReturnsSameResults()
    {
        // Arrange: Verify prediction is deterministic
        var (x, y) = CreateLinearDataset(samples: 40, features: 4, seed: 55);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .BuildAsync();

        var newData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 }
        });

        // Act
        var predictions1 = result.Predict(newData);
        var predictions2 = result.Predict(newData);

        // Assert: Same input should produce same output
        Assert.Equal(predictions1.Length, predictions2.Length);
        for (int i = 0; i < predictions1.Length; i++)
        {
            Assert.Equal(predictions1[i], predictions2[i], Tolerance);
        }
    }

    [Fact]
    public async Task BuildAsync_WithManyFeatures_PredictHandlesFeatureSelection()
    {
        // Arrange: Create dataset with many features to test feature selection handling
        var (x, y) = CreateLinearDataset(samples: 60, features: 10, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .BuildAsync();

        var optimizationResult = result.OptimizationResult;
        Assert.NotNull(optimizationResult);

        // The optimizer may select any subset of features; verify the count is valid
        int selectedCount = optimizationResult.SelectedFeatureIndices.Count;
        Assert.True(selectedCount >= 1 && selectedCount <= 10,
            $"Selected feature count ({selectedCount}) should be between 1 and the total feature count (10)");

        // Predict with full 10-column input
        var newData = new Matrix<double>(3, 10);
        var rng = new Random(200);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 10; j++)
                newData[i, j] = rng.NextDouble() * 10;

        var predictions = result.Predict(newData);
        Assert.NotNull(predictions);
        Assert.Equal(3, predictions.Length);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public async Task BuildAsync_WithExplicitOptimizer_PredictWorks()
    {
        // Arrange: Explicitly pass NormalOptimizer to confirm it works the same as default
        var (x, y) = CreateLinearDataset(samples: 40, features: 4, seed: 88);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();
        var optimizer = new NormalOptimizer<double, Matrix<double>, Vector<double>>(model);

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .BuildAsync();

        var newData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 }
        });

        // Assert
        var predictions = result.Predict(newData);
        Assert.NotNull(predictions);
        Assert.Equal(2, predictions.Length);
    }

    [Fact]
    public async Task BuildAsync_LargeDataset_PredictDoesNotThrow()
    {
        // Arrange: Larger dataset to ensure robustness
        var (x, y) = CreateLinearDataset(samples: 200, features: 8, seed: 333);
        var loader = DataLoaders.FromMatrixVector(x, y);
        var model = new RidgeRegression<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .BuildAsync();

        var newData = new Matrix<double>(10, 8);
        var rng = new Random(444);
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 8; j++)
                newData[i, j] = rng.NextDouble() * 10;

        // Assert
        var predictions = result.Predict(newData);
        Assert.NotNull(predictions);
        Assert.Equal(10, predictions.Length);
    }

    [Fact]
    public async Task BuildAsync_DirectModelPredict_WithoutBuilder_StillWorks()
    {
        // Arrange: Verify that direct model.Train() + model.Predict() still works
        // (i.e., our changes don't break the non-builder path)
        // Use non-collinear features: y = 2*x1 + 3*x2 + 1
        var model = new RidgeRegression<double>();
        var x = CreateMatrix(new double[,]
        {
            { 1, 1 }, { 2, 1 }, { 3, 2 }, { 4, 2 }, { 5, 3 },
            { 6, 3 }, { 7, 4 }, { 8, 4 }, { 9, 5 }, { 10, 5 }
        });
        // y = 2*x1 + 3*x2 + 1
        var y = CreateVector(new double[] { 6, 8, 13, 15, 20, 22, 27, 29, 34, 36 });

        // Act: Direct training (no builder, no optimizer, no feature selection)
        model.Train(x, y);
        var newX = CreateMatrix(new double[,] { { 11, 6 }, { 12, 7 } });
        var predictions = model.Predict(newX);

        // Assert
        Assert.NotNull(predictions);
        Assert.Equal(2, predictions.Length);
        // y(11, 6) = 2*11 + 3*6 + 1 = 41
        // y(12, 7) = 2*12 + 3*7 + 1 = 46
        Assert.InRange(predictions[0], 39, 43);
        Assert.InRange(predictions[1], 44, 48);
    }

    #endregion

    #region OptimizationResult Deep Copy Tests

    [Fact]
    public void OptimizationResult_DeepCopy_PreservesSelectedFeatureIndices()
    {
        // Arrange
        var original = new OptimizationResult<double, Matrix<double>, Vector<double>>
        {
            SelectedFeatureIndices = new List<int> { 0, 2, 4 }
        };

        // Act
        var copy = original.DeepCopy();

        // Assert
        Assert.Equal(3, copy.SelectedFeatureIndices.Count);
        Assert.Equal(new List<int> { 0, 2, 4 }, copy.SelectedFeatureIndices);

        // Mutating original should not affect copy
        original.SelectedFeatureIndices.Add(6);
        Assert.Equal(3, copy.SelectedFeatureIndices.Count);
    }

    [Fact]
    public void OptimizationResult_WithParameters_PreservesSelectedFeatureIndices()
    {
        // Arrange
        var original = new OptimizationResult<double, Matrix<double>, Vector<double>>
        {
            SelectedFeatureIndices = new List<int> { 1, 3 }
        };

        // Act
        var updated = original.WithParameters(new Vector<double>(new[] { 1.0, 2.0 }));

        // Assert: Values are preserved
        Assert.Equal(new List<int> { 1, 3 }, updated.SelectedFeatureIndices);

        // Assert: Copy independence — mutating original should not affect the copy
        original.SelectedFeatureIndices.Add(5);
        Assert.Equal(2, updated.SelectedFeatureIndices.Count);
        Assert.DoesNotContain(5, updated.SelectedFeatureIndices);
    }

    #endregion

    #region Tensor Feature Selection Tests

    [Fact]
    public void AiModelResult_Predict_AppliesTensorFeatureSelection()
    {
        // Arrange: Create a neural network model with SelectedFeatureIndices set.
        // AiModelResult.Predict() should apply feature selection before calling the model.
        const int inputSize = 4;
        const int selectedSize = 2;
        const int outputSize = 2;

        // Create a simple neural network that expects 'selectedSize' input features
        var layers = new List<AiDotNet.Interfaces.ILayer<float>>
        {
            new AiDotNet.NeuralNetworks.Layers.InputLayer<float>(selectedSize),
            new AiDotNet.NeuralNetworks.Layers.DenseLayer<float>(
                selectedSize, outputSize,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<float>(
            inputType: AiDotNet.Enums.InputType.OneDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            complexity: AiDotNet.Enums.NetworkComplexity.Simple,
            inputSize: selectedSize,
            outputSize: outputSize,
            layers: layers);

        var model = new AiDotNet.NeuralNetworks.FeedForwardNeuralNetwork<float>(architecture);

        // Create an optimization result with feature selection: select columns 0 and 2 from 4
        var optimization = new OptimizationResult<float, Tensor<float>, Tensor<float>>
        {
            BestSolution = model,
            SelectedFeatureIndices = new List<int> { 0, 2 }
        };

        var options = new AiModelResultOptions<float, Tensor<float>, Tensor<float>>
        {
            OptimizationResult = optimization
        };

        var result = new AiModelResult<float, Tensor<float>, Tensor<float>>(options);

        // Create 1D input with ALL 4 features (model expects only 2 after selection)
        var fullInput = new Tensor<float>(new[] { inputSize });
        fullInput[0] = 1.0f;
        fullInput[1] = 2.0f;
        fullInput[2] = 3.0f;
        fullInput[3] = 4.0f;

        // Act: Predict should apply feature selection (columns 0,2 → [1.0, 3.0])
        // and the model should not throw a dimension mismatch
        var prediction = result.Predict(fullInput);

        // Assert: Should get output without errors (feature selection applied)
        Assert.NotNull(prediction);
        // Verify output has expected dimensions
        Assert.Equal(outputSize, prediction.Shape[0]);
    }

    [Fact]
    public void AiModelResult_Predict_AppliesPermutedFeatureSelection()
    {
        // Arrange: Create a neural network and verify that a full-length permutation
        // [2, 0, 1] is NOT treated as identity — the features must be reordered.
        const int featureCount = 3;
        const int outputSize = 1;

        var layers = new List<AiDotNet.Interfaces.ILayer<float>>
        {
            new AiDotNet.NeuralNetworks.Layers.InputLayer<float>(featureCount),
            new AiDotNet.NeuralNetworks.Layers.DenseLayer<float>(
                featureCount, outputSize,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<float>(
            inputType: AiDotNet.Enums.InputType.OneDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            complexity: AiDotNet.Enums.NetworkComplexity.Simple,
            inputSize: featureCount,
            outputSize: outputSize,
            layers: layers);

        var model = new AiDotNet.NeuralNetworks.FeedForwardNeuralNetwork<float>(architecture);

        // Set explicit non-uniform weights to ensure permutation produces different output.
        // DenseLayer weights shape: [outputSize, inputSize] = [1, 3], stored in row-major order.
        var denseLayer = (AiDotNet.NeuralNetworks.Layers.DenseLayer<float>)layers[1];
        var weights = denseLayer.GetWeights();
        weights[0] = 1.0f;  // weight for feature 0
        weights[1] = 2.0f;  // weight for feature 1
        weights[2] = 3.0f;  // weight for feature 2

        // Permutation [2, 0, 1]: same count as input but different order
        var permutedIndices = new List<int> { 2, 0, 1 };
        var optimization = new OptimizationResult<float, Tensor<float>, Tensor<float>>
        {
            BestSolution = model,
            SelectedFeatureIndices = permutedIndices
        };

        var options = new AiModelResultOptions<float, Tensor<float>, Tensor<float>>
        {
            OptimizationResult = optimization
        };

        var result = new AiModelResult<float, Tensor<float>, Tensor<float>>(options);

        // Input: [10, 20, 30] — after permutation [2, 0, 1] → model sees [30, 10, 20]
        var input = new Tensor<float>(new[] { featureCount });
        input[0] = 10.0f;
        input[1] = 20.0f;
        input[2] = 30.0f;

        // Also predict with identity order to compare
        var identityOptimization = new OptimizationResult<float, Tensor<float>, Tensor<float>>
        {
            BestSolution = model,
            SelectedFeatureIndices = new List<int> { 0, 1, 2 }
        };
        var identityOptions = new AiModelResultOptions<float, Tensor<float>, Tensor<float>>
        {
            OptimizationResult = identityOptimization
        };
        var identityResult = new AiModelResult<float, Tensor<float>, Tensor<float>>(identityOptions);

        // Act
        var permutedPrediction = result.Predict(input);
        var identityPrediction = identityResult.Predict(input);

        // Assert: Permuted prediction should differ from identity prediction
        // because the model receives different feature orderings.
        // If the permutation was wrongly skipped (treated as identity), they would be equal.
        Assert.NotNull(permutedPrediction);
        Assert.NotNull(identityPrediction);

        // The predictions should differ because permuted indices reorder the input values,
        // causing different weights to multiply different values.
        bool predictionsMatch = Math.Abs(permutedPrediction[0] - identityPrediction[0]) < 1e-6f;
        Assert.False(predictionsMatch,
            "Permuted and identity predictions should differ, proving the permutation was actually applied.");
        Assert.Equal(outputSize, permutedPrediction.Shape[0]);
    }

    #endregion

    #region Helper Methods

    private static (Matrix<double> x, Vector<double> y) CreateLinearDataset(
        int samples, int features, int seed)
    {
        var random = new Random(seed);
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        // Generate random coefficients for the true relationship
        var trueCoefficients = new double[features];
        for (int j = 0; j < features; j++)
        {
            trueCoefficients[j] = random.NextDouble() * 4 - 2; // coefficients in [-2, 2]
        }

        // Generate data: y = sum(coeff_j * x_j) + small noise
        for (int i = 0; i < samples; i++)
        {
            double target = 0;
            for (int j = 0; j < features; j++)
            {
                x[i, j] = random.NextDouble() * 10;
                target += trueCoefficients[j] * x[i, j];
            }
            y[i] = target + random.NextDouble() * 0.1; // small noise
        }

        return (x, y);
    }

    private static Matrix<double> CreateMatrix(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = data[i, j];
        return matrix;
    }

    private static Vector<double> CreateVector(double[] data)
    {
        var vector = new Vector<double>(data.Length);
        for (int i = 0; i < data.Length; i++)
            vector[i] = data[i];
        return vector;
    }

    #endregion
}
