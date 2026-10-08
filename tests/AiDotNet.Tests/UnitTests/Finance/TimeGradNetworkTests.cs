using System;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// TimeGrad's sampler reads the context once and then advances the RNN one sampled value at a time. That is only
/// the model's conditioning if stepping the encoder gives the state reading the whole sequence gives.
/// </summary>
public class TimeGradNetworkTests
{
    [Fact]
    public void AdvancingTheHistoryInPieces_MatchesEncodingItAtOnce()
    {
        var network = new TimeGradNetwork<double>(2, 6, 0.0, 1, 2, 4, 2, 4, 8);
        var engine = AiDotNetEngine.Current;
        var rng = RandomHelper.CreateSeededRandom(2292);
        const int batch = 3, length = 9, split = 5;
        var sequence = new Tensor<double>(new[] { batch, length, 1 });
        for (int i = 0; i < sequence.Length; i++) sequence[i] = rng.NextDouble() * 2 - 1;

        var whole = network.EncodeHistory(sequence);
        var expected = new double[batch * 6];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < 6; h++)
                expected[b * 6 + h] = whole[(b * length + length - 1) * 6 + h];

        var hidden = new Tensor<double>?[network.RecurrentLayerCount];
        var cell = new Tensor<double>?[network.RecurrentLayerCount];
        var head = engine.TensorNarrow(sequence, 1, 0, split);
        network.AdvanceHistory(engine, head, hidden, cell);
        Tensor<double> last = head;
        for (int t = split; t < length; t++)
            last = network.AdvanceHistory(engine, engine.TensorNarrow(sequence, 1, t, 1), hidden, cell);

        Assert.Equal(new[] { batch, 6 }, last.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], last[i], 12);
    }

    /// <summary>
    /// The same equivalence in float, in inference mode: there LSTMLayer reads a whole sequence on its fused path, while
    /// stepping runs the per-step loop, so this is the mix the sampler actually uses.
    /// </summary>
    [Fact]
    public void AdvancingTheHistoryInPieces_MatchesEncodingItAtOnce_InFloatInference()
    {
        var network = new TimeGradNetwork<float>(2, 6, 0.0, 1, 2, 4, 2, 4, 8);
        foreach (var layer in network.Layers) layer.SetTrainingMode(false);
        var engine = AiDotNetEngine.Current;
        var rng = RandomHelper.CreateSeededRandom(2293);
        const int batch = 3, length = 9, split = 5;
        var sequence = new Tensor<float>(new[] { batch, length, 1 });
        for (int i = 0; i < sequence.Length; i++) sequence[i] = (float)(rng.NextDouble() * 2 - 1);

        var whole = network.EncodeHistory(sequence);
        var hidden = new Tensor<float>?[network.RecurrentLayerCount];
        var cell = new Tensor<float>?[network.RecurrentLayerCount];
        network.AdvanceHistory(engine, engine.TensorNarrow(sequence, 1, 0, split), hidden, cell);
        Tensor<float> last = sequence;
        for (int t = split; t < length; t++)
            last = network.AdvanceHistory(engine, engine.TensorNarrow(sequence, 1, t, 1), hidden, cell);

        for (int b = 0; b < batch; b++)
            for (int h = 0; h < 6; h++)
                Assert.Equal(whole[(b * length + length - 1) * 6 + h], last[b * 6 + h], 5);
    }
    [Fact]
    public void BindTo_RefusesAListOfTheWrongLength()
    {
        var network = new TimeGradNetwork<double>(2, 6, 0.0, 1, 2, 4, 2, 4, 8);
        var shorter = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<double>>(network.Layers);
        shorter.RemoveAt(shorter.Count - 1);

        Assert.Throws<InvalidOperationException>(() => network.BindTo(shorter));
    }
}
