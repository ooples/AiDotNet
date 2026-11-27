using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// A predictor whose layer cannot declare its parameter count yet (a lazy dense layer with no input width)
/// used to recurse without end: <c>ParameterCount</c> fell back to <c>PrepareParameterSurface</c>, which called
/// <c>MaybeEngageWeightStreaming</c>, which read <c>ParameterCount</c> again. Before the fix this overflowed
/// the stack and took the test host down.
/// </summary>
public class NoisePredictorUndeclaredCountTests
{
    private sealed class LazyPredictor : NoisePredictorBase<double>
    {
        // Picked up by the reflected layer walk. Without an input width the layer cannot declare its count.
        private readonly DenseLayer<double> _lazy = new(outputSize: 4);

        public bool LayerDeclaresCount => _lazy.TryGetOwnDeclaredParameterCount(out _, out _);

        public override int InputChannels => 1;
        public override int OutputChannels => 1;
        public override int BaseChannels => 1;
        public override int TimeEmbeddingDim => 1;
        public override bool SupportsCFG => false;
        public override bool SupportsCrossAttention => false;
        public override int ContextDimension => 0;
        public override Tensor<double> PredictNoise(Tensor<double> noisySample, int timestep, Tensor<double>? conditioning = null)
            => noisySample;
    }

    [Fact]
    public void ParameterCount_WithUndeclaredLayer_Terminates()
    {
        var predictor = new LazyPredictor();
        Assert.False(predictor.LayerDeclaresCount);

        long count = predictor.ParameterCount;

        Assert.True(count >= 0);
    }
}
