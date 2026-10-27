using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.Training;
using AiDotNet.Regression;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

[Collection("NonParallelIntegration")]
public class HostStateCheckpointTests
{
    [Fact]
    public void RepeatedCapture_ReusesStorageAndKeepsCheckpointIndependent()
    {
        var weights = new List<Matrix<double>> { new(new double[,] { { 1, 2 }, { 3, 4 } }) };
        var biases = new List<Vector<double>> { new(new double[] { 5, 6 }) };
        var savedWeights = HostStateCheckpoint<double>.Capture(weights);
        var savedBiases = HostStateCheckpoint<double>.Capture(biases);
        var matrix = savedWeights[0]; var vector = savedBiases[0];
        weights[0][0, 0] = 7; biases[0][0] = 8;
        Assert.Equal(1, matrix[0, 0]); Assert.Equal(5, vector[0]);
        Assert.Same(savedWeights, HostStateCheckpoint<double>.Capture(weights, savedWeights));
        Assert.Same(savedBiases, HostStateCheckpoint<double>.Capture(biases, savedBiases));
        Assert.Same(matrix, savedWeights[0]); Assert.Same(vector, savedBiases[0]);
        weights[0][0, 0] = 9; biases[0][0] = 10;
        Assert.Equal(7, matrix[0, 0]); Assert.Equal(8, vector[0]);
    }

    [Fact]
    public void LayoutMismatch_DoesNotPartiallyOverwriteCheckpoint()
    {
        var original = new List<Matrix<double>> { new(1, 2), new(2, 1) };
        var saved = HostStateCheckpoint<double>.Capture(original);
        var changed = new List<Matrix<double>> { new(new double[,] { { 4, 5 } }), new(3, 1) };
        Assert.Throws<ArgumentException>(() => HostStateCheckpoint<double>.Capture(changed, saved));
        Assert.Equal(0, saved[0][0, 0]);
    }

    private static (Matrix<double>, Vector<double>) Data()
    {
        var x = new Matrix<double>(16, 2); var y = new Vector<double>(16);
        for (int i = 0; i < 16; i++) { x[i, 0] = i / 16.0; x[i, 1] = Math.Sin(i); y[i] = 1 + i; }
        return (x, y);
    }

    [Fact]
    public async Task DeepHit_RepeatedSnapshotsReuseWeightsAndRestorePredictions()
    {
        var (x, y) = Data();
        using var model = new DeepHit<double>(new DeepHitOptions<double>
        { Epochs = 4, HiddenLayerSize = 4, NumTimeBins = 5, BatchSize = 8, Seed = 42 });
        await model.TrainAsync(x, y);
        var save = typeof(DeepHit<double>).GetMethod("SaveWeights", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var restore = typeof(DeepHit<double>).GetMethod("RestoreWeights", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var first = save.Invoke(model, new object?[] { null })!;
        var second = save.Invoke(model, new[] { first })!;
        var firstWeights = (List<Matrix<double>>)((ITuple)first)[0]!;
        var secondWeights = (List<Matrix<double>>)((ITuple)second)[0]!;
        Assert.Same(firstWeights[0], secondWeights[0]);
        var expected = (await model.PredictAsync(x)).ToArray();
        var live = (List<Matrix<double>>)typeof(DeepHit<double>).GetField("_sharedWeights", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        Assert.NotSame(live[0], secondWeights[0]);
        live[0][0, 0] += 10;
        restore.Invoke(model, new[] { second });
        Assert.Equal(expected, (await model.PredictAsync(x)).ToArray());
    }

    [Fact]
    public async Task DeepSurv_ReusableBestEpochStateProducesFinitePredictions()
    {
        var (x, y) = Data();
        using var model = new DeepSurv<double>(new DeepSurvOptions<double>
        { Epochs = 4, HiddenLayerSize = 4, BatchSize = 8, Seed = 42 });
        await model.TrainAsync(x, y);
        Assert.All((await model.PredictAsync(x)).ToArray(), value => Assert.True(double.IsFinite(value)));
    }

    [Fact]
    public void GaussianMixture_MultipleRestartsPreserveFittedState()
    {
        var (x, y) = Data();
        using var model = new AiDotNet.Clustering.Probabilistic.GaussianMixtureModel<double>(
            new AiDotNet.Clustering.Options.GMMOptions<double>
            { NumComponents = 2, NumInitializations = 3, MaxIterations = 5 });
        model.Train(x, y);
        var labels = model.Predict(x);
        Assert.Equal(x.Rows, labels.Length);
        Assert.All(labels.ToArray(), value => Assert.InRange(value, 0, 1));
    }
}
