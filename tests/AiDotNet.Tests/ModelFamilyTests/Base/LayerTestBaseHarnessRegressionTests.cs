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
    public async Task SerializationInvariant_RejectsMatchingNaNOutputs()
    {
        var fixture = new NaNLayerFixture();

        var error = await Assert.ThrowsAnyAsync<XunitException>(
            fixture.Serialize_Deserialize_ShouldPreserveBehavior);

        Assert.Contains("NaN", error.Message, StringComparison.Ordinal);
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
}
