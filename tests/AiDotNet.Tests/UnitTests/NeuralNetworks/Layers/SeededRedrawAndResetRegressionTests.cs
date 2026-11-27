using System.Linq;
using AiDotNet.Initialization;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public sealed class SeededRedrawAndResetRegressionTests
{
    // Stands in for FromFile: each call reads the next payload from a per-instance cursor.
    private sealed class CursorLoadingStrategy : InitializationStrategyBase<double>
    {
        public int Cursor { get; private set; }

        public override bool IsLazy => false;

        public override bool LoadFromExternal => true;

        public override void InitializeWeights(Tensor<double> weights, int inputSize, int outputSize) => Load(weights);

        public override void InitializeBiases(Tensor<double> biases) => Load(biases);

        private void Load(Tensor<double> tensor)
        {
            int payload = ++Cursor;
            for (int i = 0; i < tensor.Length; i++)
                tensor.SetFlat(i, payload * 100.0 + i);
        }
    }

    [Fact]
    public void SqueezeAndExcitation_SeedAssignment_KeepsExternallyLoadedTensors()
    {
        var strategy = new CursorLoadingStrategy();
        var layer = new SqueezeAndExcitationLayer<double>(
            channels: 8, reductionRatio: 2,
            firstActivation: null, secondActivation: null, initializationStrategy: strategy);
        int cursorAfterConstruction = strategy.Cursor;
        double[] loaded = layer.GetParameters().ToArray();

        layer.RandomSeed = 1234;

        Assert.Equal(cursorAfterConstruction, strategy.Cursor);
        Assert.Equal(loaded, layer.GetParameters().ToArray());
    }

    [Fact]
    public void HtsatFusion_ResetState_RestartsTheSeededLongerRowSequence()
    {
        const int batch = 4, size = 24, patch = 4, width = 8, draws = 8;
        var layer = new HtsatFusionPatchEmbeddingLayer<double>(patch, width) { RandomSeed = 99 };
        var data = RandomHelper.CreateSeededRandom(17);
        var images = new Tensor<double>(new[] { batch, 4, size, size });
        for (int i = 0; i < images.Length; i++)
            images.SetFlat(i, data.NextDouble() - 0.5);

        layer.SetTrainingMode(false);
        layer.Forward(images);
        layer.SetTrainingMode(true);

        double[][] Session() => Enumerable.Range(0, draws).Select(_ => layer.Forward(images).ToArray()).ToArray();

        var first = Session();
        Assert.True(first.Select(o => string.Join(",", o)).Distinct().Count() > 1,
            "the seeded draws should pick different longer rows across forwards.");

        layer.ResetState();
        var second = Session();

        for (int d = 0; d < draws; d++)
            Assert.Equal(first[d], second[d]);
    }
}
