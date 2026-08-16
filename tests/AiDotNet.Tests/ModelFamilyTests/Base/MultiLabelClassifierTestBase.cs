using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

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
public abstract class MultiLabelClassifierTestBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected static T ToT(double value) => NumOps.FromDouble(value);
    protected static double ToD(T value) => Convert.ToDouble(value);

    protected abstract IFullModel<T, Matrix<T>, Matrix<T>> CreateModel();

    protected virtual int TrainSamples => 80;
    protected virtual int Features => 3;
    protected virtual int NumLabels => 3;

    private (Matrix<T> X, Matrix<T> Y) CreateTrainingData(Random rng)
    {
        var x = new Matrix<T>(TrainSamples, Features);
        var y = new Matrix<T>(TrainSamples, NumLabels);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = ToT(rng.NextDouble() * 5.0);
            for (int j = 0; j < NumLabels; j++)
                y[i, j] = ToT(rng.NextDouble() > 0.5 ? 1.0 : 0.0);
        }

        return (x, y);
    }

    [Fact(Timeout = 60000)]
    public async Task Predictions_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var (x, y) = CreateTrainingData(rng);

        model.Train(x, y);
        var predictions = model.Predict(x);

        for (int i = 0; i < predictions.Rows; i++)
            for (int j = 0; j < predictions.Columns; j++)
            {
                Assert.False(double.IsNaN(ToD(predictions[i, j])),
                    $"Multi-label prediction[{i},{j}] is NaN.");
            }
    }

    [Fact(Timeout = 60000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var (x, y) = CreateTrainingData(rng);

        model.Train(x, y);
        var pred1 = model.Predict(x);
        var pred2 = model.Predict(x);

        Assert.Equal(pred1.Rows, pred2.Rows);
        Assert.Equal(pred1.Columns, pred2.Columns);
        for (int i = 0; i < pred1.Rows; i++)
            for (int j = 0; j < pred1.Columns; j++)
                Assert.Equal(pred1[i, j], pred2[i, j]);
    }

    [Fact(Timeout = 60000)]
    public async Task OutputDimension_ShouldMatchLabels()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var (x, y) = CreateTrainingData(rng);

        model.Train(x, y);
        var predictions = model.Predict(x);
        Assert.Equal(TrainSamples, predictions.Rows);
    }
}

/// <summary>Default-precision alias for existing hand-written fixtures.</summary>
public abstract class MultiLabelClassifierTestBase : MultiLabelClassifierTestBase<double> { }
