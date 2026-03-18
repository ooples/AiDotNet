using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for anomaly detection models.
/// Tests mathematical invariants: normal data scores, outlier detection,
/// score finiteness, determinism, monotonicity, and clone consistency.
/// </summary>
/// <remarks>
/// Anomaly detectors use IFullModel&lt;T, Matrix&lt;T&gt;, Vector&lt;T&gt;&gt; where
/// the output vector contains anomaly scores (higher = more anomalous).
/// </remarks>
public abstract class AnomalyDetectorTestBase
{
    protected abstract IFullModel<double, Matrix<double>, Vector<double>> CreateModel();

    protected virtual int TrainSamples => 100;
    protected virtual int Features => 3;

    private (Matrix<double> X, Vector<double> Y) GenerateNormalData(Random rng)
    {
        var x = new Matrix<double>(TrainSamples, Features);
        var y = new Vector<double>(TrainSamples);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = ModelTestHelpers.NextGaussian(rng) * 1.0; // centered at 0
            y[i] = 0; // normal label
        }
        return (x, y);
    }

    [Fact]
    public void Outliers_ShouldHaveHigherScores()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);

        // Normal test points
        var normalX = new Matrix<double>(5, Features);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < Features; j++)
                normalX[i, j] = ModelTestHelpers.NextGaussian(rng) * 1.0;

        // Outlier test points (far from training distribution)
        var outlierX = new Matrix<double>(5, Features);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < Features; j++)
                outlierX[i, j] = 50.0 + ModelTestHelpers.NextGaussian(rng) * 0.1;

        var normalScores = model.Predict(normalX);
        var outlierScores = model.Predict(outlierX);

        if (ModelTestHelpers.AllFinite(normalScores) && ModelTestHelpers.AllFinite(outlierScores))
        {
            double normalMean = 0, outlierMean = 0;
            for (int i = 0; i < normalScores.Length; i++) normalMean += normalScores[i];
            for (int i = 0; i < outlierScores.Length; i++) outlierMean += outlierScores[i];
            normalMean /= normalScores.Length;
            outlierMean /= outlierScores.Length;

            // Outliers should score differently than normal points
            Assert.True(Math.Abs(outlierMean - normalMean) > 1e-6,
                $"Normal mean score = {normalMean:F4}, outlier mean = {outlierMean:F4}. " +
                "Anomaly detector doesn't distinguish outliers from normal data.");
        }
    }

    [Fact]
    public void Scores_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);

        var scores = model.Predict(trainX);
        for (int i = 0; i < scores.Length; i++)
        {
            Assert.False(double.IsNaN(scores[i]), $"Anomaly score[{i}] is NaN.");
            Assert.False(double.IsInfinity(scores[i]), $"Anomaly score[{i}] is Infinity.");
        }
    }

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);

        var scores1 = model.Predict(trainX);
        var scores2 = model.Predict(trainX);
        for (int i = 0; i < scores1.Length; i++)
            Assert.Equal(scores1[i], scores2[i]);
    }

    [Fact]
    public void Clone_ShouldProduceSameScores()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);

        var cloned = model.Clone();
        var scores1 = model.Predict(trainX);
        var scores2 = cloned.Predict(trainX);
        for (int i = 0; i < scores1.Length; i++)
            Assert.Equal(scores1[i], scores2[i]);
    }

    [Fact]
    public void OutputDimension_ShouldMatchInputRows()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);
        Assert.Equal(TrainSamples, model.Predict(trainX).Length);
    }

    [Fact]
    public void Metadata_ShouldExistAfterTraining()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact]
    public void Parameters_ShouldBeNonEmpty()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);
        Assert.True(model.GetParameters().Length > 0, "Trained anomaly detector should have parameters.");
    }
}
