using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Sdk;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

public class LayerTestBaseHarnessRegressionTests
{
    [Fact]
    public async Task TapeGradientInvariant_HandlesGeneratedPiecewiseLinearFixture()
    {
        var fixture = new AiDotNet.Tests.ModelFamilyTests.Generated.ContinuumMemorySystemLayerTests();

        await fixture.TapeGradient_ShouldReachAtLeastOneTrainableParameter();
    }

    [Fact]
    public async Task SerializationInvariant_AcceptsOneFloatUlpOfReplayDrift()
    {
        var fixture = new ReplayDriftLayerFixture(OneFloatUlpAboveOne - 1.0f);

        await fixture.Serialize_Deserialize_ShouldPreserveBehavior();
    }

    [Fact]
    public async Task SerializationInvariant_RejectsMeaningfulFloatReplayDrift()
    {
        var fixture = new ReplayDriftLayerFixture(1e-3f);

        var error = await Assert.ThrowsAnyAsync<XunitException>(
            fixture.Serialize_Deserialize_ShouldPreserveBehavior);

        Assert.Contains("changed its own output", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task SerializationInvariant_RejectsMatchingNaNOutputs()
    {
        var fixture = new NaNLayerFixture();

        var error = await Assert.ThrowsAnyAsync<XunitException>(
            fixture.Serialize_Deserialize_ShouldPreserveBehavior);

        Assert.Contains("NaN", error.Message, StringComparison.Ordinal);
    }

    private static float OneFloatUlpAboveOne
        => AiDotNet.MixedPrecision.BitConverterHelper.Int32BitsToSingle(
            AiDotNet.MixedPrecision.BitConverterHelper.SingleToInt32Bits(1.0f) + 1);

    private sealed class ReplayDriftLayerFixture : LayerTestBase<float>
    {
        private readonly float _drift;

        public ReplayDriftLayerFixture(float drift) => _drift = drift;

        protected override int[] InputShape => [1];
        protected override bool ExpectsTrainableParameters => false;
        protected override ILayer<float> CreateLayer() => new ResetSensitiveOutputLayer(_drift);
    }

    private sealed class NaNLayerFixture : LayerTestBase<double>
    {
        protected override int[] InputShape => [1];
        protected override bool ExpectsTrainableParameters => false;
        protected override ILayer<double> CreateLayer() => new NaNOutputLayer();
    }

#pragma warning disable AIDN052 // Synthetic test double; production layer metadata would trigger scaffold generation.
    [ElementWiseShape]
    private sealed class NaNOutputLayer : LayerBase<double>
    {
        public NaNOutputLayer() : base([1], [1])
        {
        }

        public override bool SupportsTraining => false;

        protected override Tensor<double> ForwardTraced(Tensor<double> input)
        {
            var output = new Tensor<double>(input.Shape.ToArray());
            output.Fill(double.NaN);
            return output;
        }

        public override void ResetState()
        {
        }
    }
#pragma warning restore AIDN052

#pragma warning disable AIDN052 // Synthetic test double; production layer metadata would trigger scaffold generation.
    [ElementWiseShape]
    private sealed class ResetSensitiveOutputLayer : LayerBase<float>
    {
        private readonly float _drift;
        private bool _wasReset;

        public ResetSensitiveOutputLayer(float drift) : base([1], [1]) => _drift = drift;

        public override bool SupportsTraining => false;

        protected override Tensor<float> ForwardTraced(Tensor<float> input)
        {
            var output = new Tensor<float>(input.Shape.ToArray());
            output.Fill(1.0f + (_wasReset ? _drift : 0.0f));
            return output;
        }

        public override void ResetState() => _wasReset = true;
    }
#pragma warning restore AIDN052
}
