using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Pins the explicit sequential-topology contract used by named activation collection.
/// </summary>
public sealed class SequentialActivationFoldContractTests
{
    [Fact]
    public async Task DirectSequentialModel_UsesLayerFoldWithoutRunningPrediction()
    {
        await Task.Yield();
        var model = new DirectSequentialModel();

        var activations = model.GetNamedLayerActivations(new Tensor<double>(new[] { 1, 2 }));

        Assert.True(model.SequentialFoldEnabled);
        Assert.Single(activations);
        Assert.Equal(0, model.PredictionCalls);
    }

    [Fact]
    public async Task DerivedModel_ConservativelyObservesItsActualForward()
    {
        await Task.Yield();
        var model = new CustomizedDerivedModel();

        var activations = model.GetNamedLayerActivations(new Tensor<double>(new[] { 1, 2 }));

        Assert.False(model.SequentialFoldEnabled);
        Assert.Single(activations);
        Assert.Equal(1, model.PredictionCalls);
    }

    [Fact]
    public async Task AuditedGeneralPurposeModels_ChooseSequentialTopologyBase()
    {
        await Task.Yield();

        Assert.Equal(
            typeof(SequentialVectorModelLayoutBase<double>),
            typeof(NeuralNetwork<double>).BaseType);
        Assert.Equal(
            typeof(SequentialVectorModelLayoutBase<double>),
            typeof(FeedForwardNeuralNetwork<double>).BaseType);
    }

    private class DirectSequentialModel : SequentialVectorModelLayoutBase<double>
    {
        public DirectSequentialModel()
            : base(new MeanSquaredErrorLoss<double>())
        {
            InitializeLayers();
        }

        public bool SequentialFoldEnabled => SupportsSequentialActivationFold;
        public int PredictionCalls { get; protected set; }

        protected override void InitializeLayers()
            => Layers.Add(new ActivationLayer<double>(
                (IActivationFunction<double>)new ReLUActivation<double>()));

        protected override Tensor<double> PredictCore(Tensor<double> input)
        {
            PredictionCalls++;
            return base.PredictCore(input);
        }

        public override ModelMetadata<double> GetModelMetadata()
            => new() { Name = nameof(DirectSequentialModel) };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

        protected override IFullModel<double, Tensor<double>, Tensor<double>> CreateNewInstance()
            => new DirectSequentialModel();
    }

    private sealed class CustomizedDerivedModel : DirectSequentialModel
    {
        protected override Tensor<double> PredictCore(Tensor<double> input)
        {
            PredictionCalls++;
            return Layers[0].Forward(input);
        }
    }
}
