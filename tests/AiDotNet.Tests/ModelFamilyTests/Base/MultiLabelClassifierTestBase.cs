using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for multi-label classifiers (BinaryRelevance, ClassifierChain, etc.).
/// These use IFullModel&lt;T, Matrix&lt;T&gt;, Matrix&lt;T&gt;&gt; (Matrix output, not Vector).
/// Tests finite predictions, determinism, and output dimensionality.
/// </summary>
/// <remarks>
/// Multi-label classifiers use Matrix output (one column per label), so they cannot
/// extend ClassificationModelTestBase which expects Vector output.
/// </remarks>
public abstract class MultiLabelClassifierTestBase
{
    protected abstract IFullModel<double, Matrix<double>, Matrix<double>> CreateModel();

    protected virtual int TrainSamples => 80;
    protected virtual int Features => 3;
    protected virtual int NumLabels => 3;

    [Fact]
    public void Predictions_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();

        var x = new Matrix<double>(TrainSamples, Features);
        var y = new Matrix<double>(TrainSamples, NumLabels);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = rng.NextDouble() * 5.0;
            for (int j = 0; j < NumLabels; j++)
                y[i, j] = rng.NextDouble() > 0.5 ? 1.0 : 0.0;
        }

        model.Train(x, y);
        var predictions = model.Predict(x);

        for (int i = 0; i < predictions.Rows; i++)
            for (int j = 0; j < predictions.Columns; j++)
            {
                Assert.False(double.IsNaN(predictions[i, j]),
                    $"Multi-label prediction[{i},{j}] is NaN.");
            }
    }

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();

        var x = new Matrix<double>(TrainSamples, Features);
        var y = new Matrix<double>(TrainSamples, NumLabels);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = rng.NextDouble() * 5.0;
            for (int j = 0; j < NumLabels; j++)
                y[i, j] = rng.NextDouble() > 0.5 ? 1.0 : 0.0;
        }

        model.Train(x, y);
        var pred1 = model.Predict(x);
        var pred2 = model.Predict(x);

        Assert.Equal(pred1.Rows, pred2.Rows);
        Assert.Equal(pred1.Columns, pred2.Columns);
        for (int i = 0; i < pred1.Rows; i++)
            for (int j = 0; j < pred1.Columns; j++)
                Assert.Equal(pred1[i, j], pred2[i, j]);
    }

    [Fact]
    public void OutputDimension_ShouldMatchLabels()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();

        var x = new Matrix<double>(TrainSamples, Features);
        var y = new Matrix<double>(TrainSamples, NumLabels);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = rng.NextDouble() * 5.0;
            for (int j = 0; j < NumLabels; j++)
                y[i, j] = rng.NextDouble() > 0.5 ? 1.0 : 0.0;
        }

        model.Train(x, y);
        var predictions = model.Predict(x);
        Assert.Equal(TrainSamples, predictions.Rows);
    }
}
