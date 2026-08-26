using System.Collections.Generic;
using System.IO;
using System.Reflection;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class ParameterGradientPublicationTests
{
    [Fact]
    public void FusedCapability_IncludesEagerOnlyExtraTrainableLayer()
    {
        using var network = new PublicationNetwork(useEagerOnlyExtra: true);
        var capabilityCheck = typeof(NeuralNetworkBase<float>).GetMethod(
            "LayersSupportFusedCompiledTraining",
            BindingFlags.Instance | BindingFlags.NonPublic);

        Assert.NotNull(capabilityCheck);
        Assert.False((bool)capabilityCheck.Invoke(network, null)!);
    }

    [Fact]
    public void Publication_UsesCanonicalLayerExtraLayerAndRawTensorOrder()
    {
        using var network = new PublicationNetwork();
        var parameters = network.TrainableTensors;
        var gradients = new Dictionary<Tensor<float>, Tensor<float>>();
        int expectedLength = 0;

        for (int i = 0; i < parameters.Count; i++)
        {
            gradients[parameters[i]] = FilledLike(parameters[i], i + 1);
            expectedLength += parameters[i].Length;
        }

        network.Publish(gradients);
        var flat = network.GetParameterGradients();

        Assert.Equal(expectedLength, flat.Length);
        int offset = 0;
        for (int i = 0; i < parameters.Count; i++)
        {
            for (int j = 0; j < parameters[i].Length; j++)
                Assert.Equal(i + 1, flat[offset++]);
        }
    }

    [Fact]
    public void Publication_ReplacesPreviousSnapshotInsteadOfRetainingStaleLayerSlices()
    {
        using var network = new PublicationNetwork();
        var parameters = network.TrainableTensors;
        network.Publish(parameters.ToDictionary(
            parameter => parameter,
            parameter => FilledLike(parameter, 7)));

        var rawOnly = new Dictionary<Tensor<float>, Tensor<float>>
        {
            [parameters[^1]] = FilledLike(parameters[^1], 11)
        };
        network.Publish(rawOnly);
        var flat = network.GetParameterGradients();

        int rawOffset = flat.Length - parameters[^1].Length;
        for (int i = 0; i < rawOffset; i++) Assert.Equal(0, flat[i]);
        for (int i = rawOffset; i < flat.Length; i++) Assert.Equal(11, flat[i]);
    }

    [Fact]
    public void FlatPublication_RejectsAParameterMisalignedVector()
    {
        using var network = new PublicationNetwork();

        var error = Assert.Throws<ArgumentException>(() =>
            network.PublishFlat(new Vector<float>((int)network.ParameterCount - 1)));

        Assert.Contains("must match ParameterCount", error.Message);
    }

    private static Tensor<float> FilledLike(Tensor<float> parameter, float value)
    {
        var result = new Tensor<float>(parameter.Shape.ToArray());
        for (int i = 0; i < result.Length; i++) result[i] = value;
        return result;
    }

    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
    private sealed class PublicationNetwork : NeuralNetworkBase<float>
    {
        private readonly LayerBase<float> _extraLayer;
        private readonly Tensor<float> _rawParameter;

        public PublicationNetwork(bool useEagerOnlyExtra = false)
            : base(CreateArchitecture(), new MeanSquaredErrorLoss<float>())
        {
            var sample = new Tensor<float>(new[] { 1, 3 });
            var primaryLayer = new DenseLayer<float>(2);
            _ = primaryLayer.Forward(sample);
            Layers.Add(primaryLayer);

            if (useEagerOnlyExtra)
            {
                _extraLayer = new RealGatedLinearRecurrenceLayer<float>(
                    sequenceLength: 1, modelDimension: 3);
            }
            else
            {
                _extraLayer = new DenseLayer<float>(2);
                _ = _extraLayer.Forward(sample);
            }
            _rawParameter = new Tensor<float>(new[] { 2 });
        }

        public IReadOnlyList<Tensor<float>> TrainableTensors => CollectModelTrainableTensors();

        public void Publish(IReadOnlyDictionary<Tensor<float>, Tensor<float>> gradients)
            => PublishParameterGradients(gradients);

        public void PublishFlat(Vector<float> gradients) => PublishFlatParameterGradients(gradients);

        protected override IEnumerable<LayerBase<float>?> GetExtraTrainableLayers()
        {
            yield return _extraLayer;
        }

        protected override IEnumerable<Tensor<float>> GetExtraTrainableTensors()
        {
            yield return _rawParameter;
        }

        protected override void InitializeLayers()
        {
        }

        public override void UpdateParameters(Vector<float> parameters) => SetParameters(parameters);

        public override ModelMetadata<float> GetModelMetadata() => new() { Name = "PublicationNetwork" };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }

        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new PublicationNetwork();

        private static NeuralNetworkArchitecture<float> CreateArchitecture()
            => new(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 3,
                outputSize: 2);
    }
}
