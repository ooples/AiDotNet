using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// A model that leaves its layers to the first use (EnsureArchitectureInitialized, as the lazy backbones do) draws their seeds then, long after its constructor armed the
/// construction scope. The scope stays armed with whichever model was constructed last on the thread, so such a
/// model drew its layer seeds from that model's sequence: two models with the same seed got different weights
/// whenever another was built between the first one's constructor and its first use.
/// </summary>
public class LazyLayerSeedSequenceTests
{
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    private sealed class BuildsLayersOnFirstUse : NeuralNetworkBase<double>
    {
        public BuildsLayersOnFirstUse(int seed)
            : base(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 4) { RandomSeed = seed },
                new MeanSquaredErrorLoss<double>())
        {
        }

        // As the lazy backbones (CSPDarknet, VGG16BNBackbone) do on first use.
        public void BuildOnFirstUse() => EnsureArchitectureInitialized();

        protected override void InitializeLayers()
        {
            Layers.Add(new FullyConnectedLayer<double>(4, 4, new AiDotNet.ActivationFunctions.IdentityActivation<double>()));
            Layers.Add(new FullyConnectedLayer<double>(4, 4, new AiDotNet.ActivationFunctions.IdentityActivation<double>()));
        }

        public override IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
            => new BuildsLayersOnFirstUse(Architecture.RandomSeed ?? 0);

        public override ModelMetadata<double> GetModelMetadata() => new()
        {
            Name = nameof(BuildsLayersOnFirstUse),
            Description = "Test double that builds its layers on first use.",
        };
    }

    [Fact]
    public void ModelsWithTheSameSeed_BuildTheSameLayers_WhenAnotherWasConstructedInBetween()
    {
        var first = new BuildsLayersOnFirstUse(3);
        var second = new BuildsLayersOnFirstUse(3);

        // first builds its layers only now, after second re-armed the thread's scope.
        var firstParameters = ParametersAfterFirstUse(first);
        Assert.Equal(40, firstParameters.Length);
        Assert.Equal(firstParameters, ParametersAfterFirstUse(second));
    }

    [Fact]
    public void ModelsWithDifferentSeeds_BuildDifferentLayers()
    {
        var first = new BuildsLayersOnFirstUse(3);
        var second = new BuildsLayersOnFirstUse(4);

        Assert.NotEqual(ParametersAfterFirstUse(first), ParametersAfterFirstUse(second));
    }

    private static double[] ParametersAfterFirstUse(BuildsLayersOnFirstUse model)
    {
        model.BuildOnFirstUse();
        return model.GetParameters().ToArray();
    }
}
