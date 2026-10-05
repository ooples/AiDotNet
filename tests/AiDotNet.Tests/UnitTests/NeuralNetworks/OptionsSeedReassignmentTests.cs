using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Assigning <c>Options</c> applies its seed by restarting the layer initialization scope. A model that assigns
/// the SAME seeded options again after building some layers must not restart it: the layers built afterwards
/// would draw the seeds the earlier ones already used and initialise identically.
/// </summary>
public class OptionsSeedReassignmentTests
{
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    private sealed class ReassignsOptions : NeuralNetworkBase<double>
    {
        private readonly NeuralNetworkOptions _options;

        public ReassignsOptions(NeuralNetworkArchitecture<double> architecture, int seed)
            : base(architecture, new MeanSquaredErrorLoss<double>())
        {
            _options = new NeuralNetworkOptions { Seed = seed };
            Options = _options;
            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            Layers.Add(new FullyConnectedLayer<double>(4, 4, new AiDotNet.ActivationFunctions.IdentityActivation<double>()));

            // The same seeded options, assigned again part-way through construction.
            Options = _options;
            Layers.Add(new FullyConnectedLayer<double>(4, 4, new AiDotNet.ActivationFunctions.IdentityActivation<double>()));
        }

        public override IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
            => new ReassignsOptions(Architecture, _options.Seed ?? 0);

        public override ModelMetadata<double> GetModelMetadata() => new()
        {
            Name = nameof(ReassignsOptions),
            Description = "Test double that reassigns its seeded options between two layers.",
        };
    }

    [Fact]
    public void ReassigningTheSameSeededOptions_DoesNotRepeatTheLayerSeeds()
    {
        var model = new ReassignsOptions(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputSize: 4, outputSize: 4),
            seed: 2290);
        model.MaterializeParameters();

        var first = model.Layers[0].GetParameters().ToArray();
        var second = model.Layers[1].GetParameters().ToArray();

        Assert.Equal(first.Length, second.Length);
        Assert.NotEqual(first, second);
    }
}
