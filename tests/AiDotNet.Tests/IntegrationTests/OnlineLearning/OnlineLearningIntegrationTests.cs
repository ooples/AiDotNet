using AiDotNet.LinearAlgebra;
using AiDotNet.OnlineLearning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.OnlineLearning;

/// <summary>
/// Integration tests for online learning classes.
/// </summary>
public class OnlineLearningIntegrationTests
{
    #region OnlineSGDClassifier Tests

    [Fact]
    public void OnlineSGDClassifier_Construction_DefaultParams()
    {
        var clf = new OnlineSGDClassifier<double>();
        Assert.NotNull(clf);
        Assert.False(clf.IsTrained);
    }

    [Fact]
    public void OnlineSGDClassifier_PartialFit_DoesNotThrow()
    {
        var clf = new OnlineSGDClassifier<double>(learningRate: 0.1);
        var x = new Vector<double>(new[] { 1.0, 2.0 });
        clf.PartialFit(x, 1.0);
        Assert.True(clf.IsTrained);
    }

    [Fact]
    public void OnlineSGDClassifier_PartialFitBatch_TrainsModel()
    {
        var clf = new OnlineSGDClassifier<double>(learningRate: 0.1, l2Penalty: 0.0);

        // Simple linearly separable data: x[0] > 0 => class 1
        var X = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0 },
            { 2.0, 0.0 },
            { -1.0, 0.0 },
            { -2.0, 0.0 },
        });
        var y = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        // Train multiple epochs
        for (int epoch = 0; epoch < 50; epoch++)
        {
            clf.PartialFit(X, y);
        }

        Assert.True(clf.IsTrained);
    }

    [Fact]
    public void OnlineSGDClassifier_Predict_ReturnsValues()
    {
        var clf = new OnlineSGDClassifier<double>(learningRate: 0.1);

        // Train
        var X = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0 },
            { -1.0, 0.0 },
        });
        var y = new Vector<double>(new[] { 1.0, 0.0 });
        for (int epoch = 0; epoch < 20; epoch++)
            clf.PartialFit(X, y);

        // Predict
        var predictions = clf.Predict(X);
        Assert.Equal(2, predictions.Length);

        // Verify classifier learned the pattern: positive x => class 1, negative x => class 0
        Assert.Equal(1.0, predictions[0]);
        Assert.Equal(0.0, predictions[1]);
    }

    #endregion

    #region OnlineSGDRegressor Tests

    [Fact]
    public void OnlineSGDRegressor_Construction_DefaultParams()
    {
        var reg = new OnlineSGDRegressor<double>();
        Assert.NotNull(reg);
        Assert.False(reg.IsTrained);
    }

    [Fact]
    public void OnlineSGDRegressor_PartialFit_DoesNotThrow()
    {
        var reg = new OnlineSGDRegressor<double>(learningRate: 0.01);
        var x = new Vector<double>(new[] { 1.0, 2.0 });
        reg.PartialFit(x, 3.0);
        Assert.True(reg.IsTrained);
    }

    [Fact]
    public void OnlineSGDRegressor_PartialFitBatch_TrainsModel()
    {
        var reg = new OnlineSGDRegressor<double>(learningRate: 0.01, l2Penalty: 0.0);

        // Simple linear data: y = x[0]
        var X = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 },
        });
        var y = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        for (int epoch = 0; epoch < 100; epoch++)
            reg.PartialFit(X, y);

        Assert.True(reg.IsTrained);

        // Verify regression learned approximately y = x
        var predictions = reg.Predict(X);
        Assert.Equal(3, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
            Assert.True(Math.Abs(predictions[i] - y[i]) < 1.0,
                $"Prediction {predictions[i]} should be close to {y[i]}");
    }

    #endregion

    #region OnlinePassiveAggressiveClassifier Tests

    [Fact]
    public void OnlinePAClassifier_Construction_DefaultParams()
    {
        var clf = new OnlinePassiveAggressiveClassifier<double>();
        Assert.NotNull(clf);
        Assert.False(clf.IsTrained);
    }

    [Fact]
    public void OnlinePAClassifier_PartialFit_DoesNotThrow()
    {
        var clf = new OnlinePassiveAggressiveClassifier<double>();
        var x = new Vector<double>(new[] { 1.0, 2.0 });
        clf.PartialFit(x, 1.0);
        Assert.True(clf.IsTrained);
    }

    [Fact]
    public void OnlinePAClassifier_LinearlySeparableData_ClassifiesCorrectly()
    {
        var clf = new OnlinePassiveAggressiveClassifier<double>();

        // Train on linearly separable data
        for (int epoch = 0; epoch < 20; epoch++)
        {
            clf.PartialFit(new Vector<double>(new[] { 3.0, 0.0 }), 1.0);
            clf.PartialFit(new Vector<double>(new[] { -3.0, 0.0 }), 0.0);
        }

        Assert.True(clf.IsTrained);

        // Verify classifier learned the pattern
        var testX = new Matrix<double>(new double[,]
        {
            { 3.0, 0.0 },
            { -3.0, 0.0 },
        });
        var preds = clf.Predict(testX);
        Assert.Equal(1.0, preds[0]);
        Assert.Equal(0.0, preds[1]);
    }

    #endregion

    #region OnlinePassiveAggressiveRegressor Tests

    [Fact]
    public void OnlinePARegressor_Construction_DefaultParams()
    {
        var reg = new OnlinePassiveAggressiveRegressor<double>();
        Assert.NotNull(reg);
        Assert.False(reg.IsTrained);
    }

    [Fact]
    public void OnlinePARegressor_PartialFit_DoesNotThrow()
    {
        var reg = new OnlinePassiveAggressiveRegressor<double>();
        var x = new Vector<double>(new[] { 1.0, 2.0 });
        reg.PartialFit(x, 3.0);
        Assert.True(reg.IsTrained);
    }

    [Fact]
    public void OnlinePARegressor_LinearData_LearnsApproximately()
    {
        var reg = new OnlinePassiveAggressiveRegressor<double>(c: 1.0, epsilon: 0.1);

        // Simple linear data: y = 2*x
        var X = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 },
            { 4.0 },
        });
        var y = new Vector<double>(new[] { 2.0, 4.0, 6.0, 8.0 });

        for (int epoch = 0; epoch < 100; epoch++)
            reg.PartialFit(X, y);

        Assert.True(reg.IsTrained);

        // Verify regression learned approximately y = 2*x
        var predictions = reg.Predict(X);
        Assert.Equal(4, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
            Assert.True(Math.Abs(predictions[i] - y[i]) < 2.0,
                $"Prediction {predictions[i]} should be close to {y[i]}");
    }

    [Fact]
    public void OnlinePARegressor_EpsilonInsensitiveLoss_DecreasesWithTraining()
    {
        var reg = new OnlinePassiveAggressiveRegressor<double>(c: 1.0, epsilon: 0.1);

        var X = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 },
        });
        var y = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Train a bit
        for (int epoch = 0; epoch < 10; epoch++)
            reg.PartialFit(X, y);

        double earlyLoss = reg.GetEpsilonInsensitiveLoss(X, y);

        // Train more
        for (int epoch = 0; epoch < 100; epoch++)
            reg.PartialFit(X, y);

        double lateLoss = reg.GetEpsilonInsensitiveLoss(X, y);

        // Loss should decrease (or at least not increase significantly) with more training
        Assert.True(lateLoss <= earlyLoss + 0.1,
            $"Loss after more training ({lateLoss}) should not be much greater than earlier loss ({earlyLoss})");
    }

    [Fact]
    public void OnlinePARegressor_Reset_ClearsWeights()
    {
        var reg = new OnlinePassiveAggressiveRegressor<double>();
        var x = new Vector<double>(new[] { 1.0, 2.0 });
        reg.PartialFit(x, 3.0);
        Assert.True(reg.IsTrained);

        reg.Reset();
        Assert.Null(reg.GetWeights());
    }

    [Fact]
    public void OnlinePARegressor_GetWeightsAndBias_ReturnsValues()
    {
        var reg = new OnlinePassiveAggressiveRegressor<double>();
        var x = new Vector<double>(new[] { 1.0, 2.0 });
        reg.PartialFit(x, 5.0);

        var weights = reg.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(2, weights.Length);

        var bias = reg.GetBias();
        Assert.False(double.IsNaN(bias), "Bias should not be NaN");
    }

    #endregion

    #region ADWINDriftDetector Tests

    [Fact]
    public void ADWINDriftDetector_StableStream_NoDrift()
    {
        var detector = new ADWINDriftDetector<double>();
        for (int i = 0; i < 100; i++)
            detector.Update(0.0);

        Assert.False(detector.IsDriftDetected);
    }

    [Fact]
    public void ADWINDriftDetector_Reset_ClearsState()
    {
        var detector = new ADWINDriftDetector<double>();
        for (int i = 0; i < 50; i++)
            detector.Update(1.0);

        detector.Reset();
        Assert.False(detector.IsDriftDetected);
    }

    #endregion
}
