using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// The incremental <c>Step</c> APIs of <see cref="GatedDeltaNetLayer{T}"/> and <see cref="MultiLatentAttentionLayer{T}"/>:
/// stepping a sequence one position at a time must give the full causal forward's output at every position.
/// </summary>
public class IncrementalStepTests
{
    private const int Batch = 2, SeqLen = 12, ModelDim = 32, Heads = 4;

    private static Tensor<double> RandomInput(int seed)
    {
        var rng = new Random(seed);
        var data = new double[Batch * SeqLen * ModelDim];
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble() * 2 - 1;
        return new Tensor<double>(new[] { Batch, SeqLen, ModelDim }, new Vector<double>(data));
    }

    private static Tensor<double> Position(Tensor<double> x, int t)
    {
        var slice = new Tensor<double>(new[] { Batch, 1, ModelDim });
        for (int b = 0; b < Batch; b++)
            for (int d = 0; d < ModelDim; d++)
                slice[b, 0, d] = x[b, t, d];
        return slice;
    }

    // Max |forward[:, t] - step_t| over all positions, relative to max |forward|; reset = drop the state before every
    // step (must break parity).
    private static double MaxStepError(LayerBase<double> layer, Func<Tensor<double>, Tensor<double>> step, Action reset, bool resetEveryStep)
    {
        var x = RandomInput(7);
        layer.SetTrainingMode(false);
        var full = layer.Forward(x);
        reset();
        double worst = 0, scale = 0;
        for (int t = 0; t < SeqLen; t++)
        {
            if (resetEveryStep) reset();
            var y = step(Position(x, t));
            Assert.Equal(new[] { Batch, 1, ModelDim }, y.Shape.ToArray());
            for (int b = 0; b < Batch; b++)
                for (int d = 0; d < ModelDim; d++)
                {
                    worst = Math.Max(worst, Math.Abs(full[b, t, d] - y[b, 0, d]));
                    scale = Math.Max(scale, Math.Abs(full[b, t, d]));
                }
        }
        Assert.True(scale > 0, "the forward output is identically zero");
        return worst / scale;
    }

    [Fact]
    public void GatedDeltaNet_Step_MatchesForwardAtEveryPosition()
    {
        var layer = new GatedDeltaNetLayer<double>(SeqLen, ModelDim, Heads, convKernelSize: 4);
        GatedDeltaNetStepState<double> state = layer.CreateStepState(Batch);
        double err = MaxStepError(layer, x => layer.Step(x, state), () => state = layer.CreateStepState(Batch), resetEveryStep: false);
        Assert.True(err < 1e-10, $"relative max |forward - step| = {err}");
        Assert.Equal(SeqLen, state.Position);
    }

    [Fact]
    public void GatedDeltaNet_Step_WithoutCarriedState_DiffersFromForward()
    {
        var layer = new GatedDeltaNetLayer<double>(SeqLen, ModelDim, Heads, convKernelSize: 4);
        var state = layer.CreateStepState(Batch);
        double err = MaxStepError(layer, x => layer.Step(x, state), () => state = layer.CreateStepState(Batch), resetEveryStep: true);
        Assert.True(err > 1e-2, $"dropping the state should change the outputs; relative max diff {err}");
    }

    [Fact]
    public void MultiLatentAttention_Step_MatchesForwardAtEveryPosition()
    {
        var layer = new MultiLatentAttentionLayer<double>(SeqLen, ModelDim, Heads, latentDimension: 8);
        MultiLatentAttentionStepState<double> state = layer.CreateStepState(Batch);
        double err = MaxStepError(layer, x => layer.Step(x, state), () => state = layer.CreateStepState(Batch), resetEveryStep: false);
        Assert.True(err < 1e-10, $"relative max |forward - step| = {err}");
        Assert.Equal(SeqLen, state.Position);
    }

    [Fact]
    public void MultiLatentAttention_Step_WithoutCache_DiffersFromForward()
    {
        var layer = new MultiLatentAttentionLayer<double>(SeqLen, ModelDim, Heads, latentDimension: 8);
        var state = layer.CreateStepState(Batch);
        double err = MaxStepError(layer, x => layer.Step(x, state), () => state = layer.CreateStepState(Batch), resetEveryStep: true);
        Assert.True(err > 1e-2, $"dropping the cache should change the outputs; relative max diff {err}");
    }

    [Fact]
    public void Step_RejectsMoreThanOnePosition()
    {
        var gdn = new GatedDeltaNetLayer<double>(SeqLen, ModelDim, Heads);
        var mla = new MultiLatentAttentionLayer<double>(SeqLen, ModelDim, Heads, latentDimension: 8);
        var two = new Tensor<double>(new[] { Batch, 2, ModelDim });
        Assert.Throws<ArgumentException>(() => gdn.Step(two, gdn.CreateStepState(Batch)));
        Assert.Throws<ArgumentException>(() => mla.Step(two, mla.CreateStepState(Batch)));
    }
}
