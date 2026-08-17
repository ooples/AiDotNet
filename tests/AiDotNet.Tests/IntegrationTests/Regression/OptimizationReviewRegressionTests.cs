using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Regression coverage for model-state and scalability contracts found during optimization review.
/// </summary>
[Trait("Category", "Integration")]
public class OptimizationReviewRegressionTests
{
    [Fact(Timeout = 120000)]
    public async Task QuantileRegression_RejectsDenseProblemAboveConfiguredBudget()
    {
        await Task.Yield();

        var options = new QuantileRegressionOptions<double>
        {
            MaximumDenseLinearProgramEntries = 10,
        };
        var model = new QuantileRegression<double>(options);
        var x = Column(Enumerable.Range(0, 8).Select(i => (double)i).ToArray());
        var y = Vector<double>.FromArray(Enumerable.Range(0, 8).Select(i => 2.0 * i + 1.0).ToArray());

        var exception = Assert.Throws<InvalidOperationException>(() => model.Train(x, y));

        Assert.Contains("exceeds the configured budget", exception.Message, StringComparison.Ordinal);
    }

    [Fact(Timeout = 120000)]
    public async Task QuantileRegression_WithoutIntercept_ConstrainsFitThroughOrigin()
    {
        await Task.Yield();

        var model = new QuantileRegression<double>(new QuantileRegressionOptions<double>
        {
            Quantile = 0.5,
            UseIntercept = false,
        });
        var x = Column(new[] { 1.0, 2.0, 3.0, 4.0 });
        // y = 2x + 5. With no intercept, median regression selects the weighted-median
        // through-origin slope 11/3 instead of the unconstrained slope 2 and intercept 5.
        var y = Vector<double>.FromArray(new[] { 7.0, 9.0, 11.0, 13.0 });

        model.Train(x, y);
        var prediction = model.Predict(Column(new[] { 5.0 }));

        Assert.Equal(55.0 / 3.0, prediction[0], 8);
    }

    [Fact(Timeout = 120000)]
    public async Task SuperLearner_NormalizedNnls_RestoresTargetLocation()
    {
        await Task.Yield();

        var x = Column(Enumerable.Range(0, 30).Select(i => i / 10.0).ToArray());
        var y = Vector<double>.FromArray(Enumerable.Range(0, 30).Select(i => 75.0 + 3.0 * i / 10.0).ToArray());
        var model = new SuperLearner<double>(
            [new RidgeRegression<double>()],
            new SuperLearnerOptions
            {
                NumFolds = 3,
                NormalizeBasePredictions = true,
                MetaLearnerType = SuperLearnerMetaLearner.NonNegativeLeastSquares,
                Seed = 123,
            });

        model.Train(x, y);
        var predictions = model.Predict(x);
        double predictionMean = predictions.ToArray().Average();
        double targetMean = y.ToArray().Average();

        Assert.InRange(predictionMean, targetMean - 1.0, targetMean + 1.0);
    }

    [Fact(Timeout = 120000)]
    public async Task SupportVectorRegression_OwnsTrainingRowsAfterFit()
    {
        await Task.Yield();

        var x = Column(new[] { -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 });
        var y = Vector<double>.FromArray(new[] { 4.0, 1.0, 0.0, 1.0, 4.0, 9.0 });
        var model = new SupportVectorRegression<double>(new SupportVectorRegressionOptions
        {
            KernelType = KernelType.RBF,
            Gamma = 0.5,
            C = 10.0,
            Epsilon = 0.01,
            MaxIterations = 200,
        });
        var probe = Column(new[] { -1.5, 0.5, 2.5 });

        model.Train(x, y);
        var beforeMutation = model.Predict(probe);
        for (int row = 0; row < x.Rows; row++) x[row, 0] = 10_000.0 + row;
        var afterMutation = model.Predict(probe);

        for (int i = 0; i < beforeMutation.Length; i++)
        {
            Assert.Equal(beforeMutation[i], afterMutation[i], 12);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task BayesianRegression_RbfPredictionAndRoundTrip_UseTrainingCentres()
    {
        await Task.Yield();

        var options = new BayesianRegressionOptions<double>
        {
            KernelType = KernelType.RBF,
            Gamma = 0.4,
            UseIntercept = true,
        };
        var x = Column(new[] { -2.0, -1.0, 0.0, 1.0, 2.0 });
        var y = Vector<double>.FromArray(new[] { 4.0, 1.0, 0.0, 1.0, 4.0 });
        var probe = Column(new[] { -1.5, 0.5, 1.5 });
        var model = new BayesianRegression<double>(options);

        model.Train(x, y);
        var expected = model.Predict(probe);
        var restored = new BayesianRegression<double>(new BayesianRegressionOptions<double>
        {
            KernelType = KernelType.RBF,
            Gamma = 0.4,
            UseIntercept = true,
        });
        restored.Deserialize(model.Serialize());
        var actual = restored.Predict(probe);

        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 10);
    }

    [Fact(Timeout = 120000)]
    public async Task BayesianRegression_RejectsMatrixShapeLargerThanRemainingPayload()
    {
        await Task.Yield();

        var model = new BayesianRegression<double>();
        byte[] payload = model.Serialize();
        int baseLength = BitConverter.ToInt32(payload, 0);
        int firstMatrixHeader = sizeof(int) + baseLength;
        BitConverter.GetBytes(int.MaxValue).CopyTo(payload, firstMatrixHeader);
        BitConverter.GetBytes(int.MaxValue).CopyTo(payload, firstMatrixHeader + sizeof(int));

        Assert.Throws<InvalidDataException>(() => new BayesianRegression<double>().Deserialize(payload));
    }

    [Fact(Timeout = 120000)]
    public async Task NeuralNetworkRegression_CloneAndRoundTrip_PreservePredictionScaleAndActivations()
    {
        await Task.Yield();

        var options = new NeuralNetworkRegressionOptions<double, Matrix<double>, Vector<double>>
        {
            Seed = 42,
            LayerSizes = [1, 4, 1],
            Epochs = 12,
            BatchSize = 6,
            LearningRate = 0.02,
            HiddenActivationFunction = new TanhActivation<double>(),
            OutputActivationFunction = new IdentityActivation<double>(),
        };
        var x = Column(new[] { -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 });
        var y = Vector<double>.FromArray(new[] { 93.0, 97.0, 100.0, 104.0, 109.0, 115.0 });
        var model = new NeuralNetworkRegression<double>(options);

        model.Train(x, y);
        var expected = model.Predict(x);
        var clone = (NeuralNetworkRegression<double>)model.Clone();
        var cloned = clone.Predict(x);
        var restored = new NeuralNetworkRegression<double>(new NeuralNetworkRegressionOptions<double, Matrix<double>, Vector<double>>
        {
            LayerSizes = [1, 4, 1],
            HiddenActivationFunction = new TanhActivation<double>(),
            OutputActivationFunction = new IdentityActivation<double>(),
        });
        restored.Deserialize(model.Serialize());
        var roundTripped = restored.Predict(x);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.InRange(expected[i], 50.0, 160.0);
            Assert.Equal(expected[i], cloned[i], 10);
            Assert.Equal(expected[i], roundTripped[i], 10);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task NeuralNetworkRegression_CloneCreatesAnOwnedOptimizer()
    {
        await Task.Yield();

        var owners = new List<IFullModel<double, Matrix<double>, Vector<double>>>();
        var options = new NeuralNetworkRegressionOptions<double, Matrix<double>, Vector<double>>
        {
            LayerSizes = [1, 2, 1],
            OptimizerFactory = owner =>
            {
                owners.Add(owner);
                return new AdamOptimizer<double, Matrix<double>, Vector<double>>(owner);
            },
        };

        var model = new NeuralNetworkRegression<double>(options);
        var clone = (NeuralNetworkRegression<double>)model.Clone();

        Assert.Equal(2, owners.Count);
        Assert.Same(model, owners[0]);
        Assert.Same(clone, owners[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task NeuralNetworkRegression_DeserializeAcceptsPayloadBeforeScalingTrailer()
    {
        await Task.Yield();

        var model = new NeuralNetworkRegression<double>(
            new NeuralNetworkRegressionOptions<double, Matrix<double>, Vector<double>>
            {
                LayerSizes = [1, 2, 1],
            });
        byte[] current = model.Serialize();
        const int trailerBytes = sizeof(int) + sizeof(int) + sizeof(double) + sizeof(double);
        byte[] legacy = current.Take(current.Length - trailerBytes).ToArray();
        var restored = new NeuralNetworkRegression<double>(
            new NeuralNetworkRegressionOptions<double, Matrix<double>, Vector<double>>
            {
                LayerSizes = [1, 2, 1],
            });

        restored.Deserialize(legacy);
        Vector<double> prediction = restored.Predict(Column(new[] { 1.0 }));

        Assert.True(double.IsFinite(prediction[0]));
    }

    private static Matrix<double> Column(double[] values)
    {
        var matrix = new Matrix<double>(values.Length, 1);
        for (int i = 0; i < values.Length; i++) matrix[i, 0] = values[i];
        return matrix;
    }
}
