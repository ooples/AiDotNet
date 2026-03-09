using AiDotNet.Classification.Linear;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Classification.Trees;
using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Tests that verify classification models work end-to-end through AiModelBuilder.
/// Before these tests, ZERO classification tests went through the builder facade.
/// All existing classification tests used direct model.Train()/Predict() calls.
/// This means the entire builder + optimizer + feature selection pipeline was untested
/// for classification — any bug there would go undetected.
/// </summary>
public class AiModelBuilderClassificationTests
{
    [Fact]
    public async Task RidgeClassifier_BuildAndPredict_ProducesValidClassLabels()
    {
        // Arrange: Generate linearly separable 2-class data
        var (x, y) = CreateBinaryClassificationData(
            samplesPerClass: 40, separation: 6.0, features: 2, seed: 42);

        var loader = DataLoaders.FromMatrixVector(x, y);
        var classifier = new RidgeClassifier<double>();

        // Act: Build through the facade
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(classifier)
            .BuildAsync();

        // Predict on held-out test data
        var (testX, testY) = CreateBinaryClassificationData(
            samplesPerClass: 15, separation: 6.0, features: 2, seed: 999);
        var predictions = result.Predict(testX);

        // Assert: Valid predictions and reasonable accuracy
        Assert.Equal(testY.Length, predictions.Length);

        int correct = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction {i} is NaN");
            Assert.False(double.IsInfinity(predictions[i]),
                $"Prediction {i} is Infinity");

            double rounded = Math.Round(predictions[i]);
            if (Math.Abs(rounded - testY[i]) < 0.5)
                correct++;
        }

        double accuracy = (double)correct / predictions.Length;
        Assert.True(accuracy > 0.60,
            $"Accuracy {accuracy:P1} is too low for linearly separable data (expected > 60%)");
    }

    [Fact]
    public async Task DecisionTreeClassifier_BuildAndPredict_ProducesValidClassLabels()
    {
        // Arrange: Generate 2-class data with clear structure
        var (x, y) = CreateBinaryClassificationData(
            samplesPerClass: 50, separation: 5.0, features: 3, seed: 77);

        var loader = DataLoaders.FromMatrixVector(x, y);
        var classifier = new DecisionTreeClassifier<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(classifier)
            .BuildAsync();

        var (testX, testY) = CreateBinaryClassificationData(
            samplesPerClass: 20, separation: 5.0, features: 3, seed: 888);
        var predictions = result.Predict(testX);

        // Assert
        Assert.Equal(testY.Length, predictions.Length);

        int correct = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            double rounded = Math.Round(predictions[i]);
            Assert.True(rounded == 0.0 || rounded == 1.0,
                $"Prediction {predictions[i]} at index {i} is not a valid class label");

            if (Math.Abs(rounded - testY[i]) < 0.5)
                correct++;
        }

        double accuracy = (double)correct / predictions.Length;
        Assert.True(accuracy > 0.70,
            $"Accuracy {accuracy:P1} is too low for separable data (expected > 70%)");
    }

    [Fact]
    public async Task GaussianNaiveBayes_BuildAndPredict_ProducesValidClassLabels()
    {
        // Arrange: Generate Gaussian-distributed 2-class data
        var (x, y) = CreateBinaryClassificationData(
            samplesPerClass: 60, separation: 4.0, features: 2, seed: 55);

        var loader = DataLoaders.FromMatrixVector(x, y);
        var classifier = new GaussianNaiveBayes<double>();

        // Act
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(classifier)
            .BuildAsync();

        var (testX, testY) = CreateBinaryClassificationData(
            samplesPerClass: 20, separation: 4.0, features: 2, seed: 777);
        var predictions = result.Predict(testX);

        // Assert
        Assert.Equal(testY.Length, predictions.Length);

        int correct = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction {i} is NaN");

            double rounded = Math.Round(predictions[i]);
            if (Math.Abs(rounded - testY[i]) < 0.5)
                correct++;
        }

        double accuracy = (double)correct / predictions.Length;
        Assert.True(accuracy > 0.55,
            $"Accuracy {accuracy:P1} is too low for Gaussian-distributed data (expected > 55%)");
    }

    [Fact]
    public async Task Classification_SerializeRoundTrip_PreservesAccuracy()
    {
        // Use DecisionTreeClassifier which works through the builder
        var (x, y) = CreateBinaryClassificationData(
            samplesPerClass: 50, separation: 5.0, features: 3, seed: 33);

        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new DecisionTreeClassifier<double>())
            .BuildAsync();

        var (testX, _) = CreateBinaryClassificationData(
            samplesPerClass: 10, separation: 5.0, features: 3, seed: 444);

        // Get original predictions
        var originalPredictions = result.Predict(testX);

        // Act: Serialize → Deserialize
        var bytes = result.Serialize();
        var restored = new AiModelResult<double, Matrix<double>, Vector<double>>();
        restored.Deserialize(bytes);
        var restoredPredictions = restored.Predict(testX);

        // Assert: Predictions must be identical after round-trip
        Assert.Equal(originalPredictions.Length, restoredPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], restoredPredictions[i], precision: 10);
        }
    }

    #region Helper Methods

    /// <summary>
    /// Creates binary classification data with two Gaussian clusters.
    /// Class 0: centered at origin, Class 1: centered at (separation, separation, ...).
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) CreateBinaryClassificationData(
        int samplesPerClass, double separation, int features, int seed)
    {
        var random = new Random(seed);
        int totalSamples = samplesPerClass * 2;
        var x = new Matrix<double>(totalSamples, features);
        var y = new Vector<double>(totalSamples);

        for (int i = 0; i < totalSamples; i++)
        {
            int classLabel = i < samplesPerClass ? 0 : 1;
            y[i] = classLabel;

            for (int j = 0; j < features; j++)
            {
                // Gaussian noise around class center
                double noise = NextGaussian(random) * 1.0;
                x[i, j] = classLabel * separation + noise;
            }
        }

        return (x, y);
    }

    /// <summary>Box-Muller transform for Gaussian random numbers.</summary>
    private static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    #endregion
}
