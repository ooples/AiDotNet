using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

public class NeuralNetworkBaseIntegrationTests
{
    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var length = 1;
        foreach (var dim in shape)
        {
            length *= dim;
        }

        var data = new float[length];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)(random.NextDouble() * 2 - 1);
        }

        return new Tensor<float>(data, shape);
    }

    private static TestNeuralNetwork BuildNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var network = new TestNeuralNetwork(architecture);
        network.AddLayer(new DenseLayer<float>(3));
        network.AddLayer(new DenseLayer<float>(2));

        return network;
    }

    [Fact(Timeout = 120000)]
    public async Task NeuralNetworkBase_ForwardWithFeatures_ReturnsExpectedShapes()
    {
        TestNeuralNetwork network = BuildNetwork();
        NeuralNetworkBase<float> baseNetwork = network;

        var input = CreateRandomTensor(new[] { 2, 4 });
        var (output, features) = baseNetwork.ForwardWithFeatures(input, new[] { -1, 0 });

        Assert.Equal(new[] { 2, 2 }, output.Shape.ToArray());
        Assert.Equal(2, features.Count);
        Assert.True(features.ContainsKey(0));
        Assert.True(features.ContainsKey(1));
        Assert.Equal(new[] { 2, 3 }, features[0].Shape.ToArray());
        Assert.Equal(new[] { 2, 2 }, features[1].Shape.ToArray());
    }


    // ComputeInputGradient test removed — method deleted in tape-based autodiff migration

    [Fact(Timeout = 120000)]
    public async Task NeuralNetworkBase_ParameterCount_UpdatesWhenLayersChange()
    {
        await Task.Yield();
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var network = new TestNeuralNetwork(architecture);
        var first = new DenseLayer<float>(3);
        network.AddLayer(first);

        int firstCount = (int)network.ParameterCount;

        var second = new DenseLayer<float>(2);
        network.AddLayer(second);

        Assert.Equal(first.ParameterCount + second.ParameterCount, (int)network.ParameterCount);
        Assert.True(network.RemoveLayer(second));
        Assert.Equal(firstCount, (int)network.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task BuildTrainingObjective_AfterInference_EntersTrainingMode()
    {
        await Task.Yield();
        using var network = BuildNetwork();
        var input = CreateRandomTensor(new[] { 2, 4 });
        var target = CreateRandomTensor(new[] { 2, 2 }, seed: 43);

        network.SetTrainingMode(false);
        Assert.False(network.IsTrainingMode);

        _ = network.BuildTrainingObjective(input, target);

        Assert.True(network.IsTrainingMode);
    }

    [Fact(Timeout = 120000)]
    public async Task BuildTrainingObjective_UsesFamilyInputPreparationHook()
    {
        await Task.Yield();
        using var network = BuildNetwork();
        var input = CreateRandomTensor(new[] { 2, 4 });
        var target = CreateRandomTensor(new[] { 2, 2 }, seed: 44);

        _ = network.BuildTrainingObjective(input, target);

        Assert.Equal(1, network.TrainingInputPreparationCount);
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_WhenPredictCoreIsOverridden_EntersEvalModeAndRestoresCallerMode()
    {
        await Task.Yield();
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 4);
        using var network = new PredictCoreOverrideNetwork(architecture);
        var input = CreateRandomTensor(new[] { 1, 4 });

        Assert.True(network.IsTrainingMode);
        _ = network.Predict(input);

        Assert.False(network.ObservedTrainingModeInsidePredictCore);
        Assert.True(network.IsTrainingMode);
    }

    [Fact(Timeout = 120000)]
    public async Task GeneratedLayerAliases_RebindByIdentityAcrossTopologyChanges()
    {
        await Task.Yield();

        ILayer<float>[] previousLayers =
        [
            new DenseLayer<float>(3),
            new DenseLayer<float>(4),
            new DenseLayer<float>(5)
        ];
        ILayer<float>[] replacementLayers = [new DenseLayer<float>(6)];
        ILayer<float> independent = new DenseLayer<float>(7);

        Assert.Same(
            replacementLayers[0],
            TestNeuralNetwork.RebindOptionalAlias(previousLayers[0], previousLayers, replacementLayers));
        Assert.Null(
            TestNeuralNetwork.RebindOptionalAlias(previousLayers[2], previousLayers, replacementLayers));
        Assert.Same(
            independent,
            TestNeuralNetwork.RebindOptionalAlias(independent, previousLayers, replacementLayers));

        var aliases = new List<ILayer<float>>
        {
            previousLayers[0],
            previousLayers[2],
            independent
        };
        TestNeuralNetwork.RebindCollection(aliases, previousLayers, replacementLayers);

        Assert.Collection(
            aliases,
            alias => Assert.Same(replacementLayers[0], alias),
            alias => Assert.Same(independent, alias));

        var requiredException = Assert.Throws<InvalidOperationException>(() =>
            TestNeuralNetwork.RebindRequiredAlias(
                previousLayers[2], previousLayers, replacementLayers));
        Assert.Contains("Required layer alias", requiredException.Message, StringComparison.Ordinal);

        ILayer<float>[] fixedAliases = [previousLayers[0], previousLayers[2]];
        var arrayException = Assert.Throws<InvalidOperationException>(() =>
            TestNeuralNetwork.RebindCollection(fixedAliases, previousLayers, replacementLayers));
        Assert.Contains("Arrays cannot shrink", arrayException.Message, StringComparison.Ordinal);
    }

    private sealed class TestNeuralNetwork : VectorModelLayoutBase<float>
    {
        public TestNeuralNetwork(NeuralNetworkArchitecture<float> architecture)
            : base(architecture, new MeanSquaredErrorLoss<float>())
        {
        }

        public override bool SupportsTraining => true;
        public int TrainingInputPreparationCount { get; private set; }

        protected override Tensor<float> PrepareInputForTraining(Tensor<float> input)
        {
            TrainingInputPreparationCount++;
            return input;
        }

        public void AddLayer(ILayer<float> layer)
        {
            AddLayerToCollection(layer);
        }

        public bool RemoveLayer(ILayer<float> layer)
        {
            return RemoveLayerFromCollection(layer);
        }

        public static ILayer<float>? RebindOptionalAlias(
            ILayer<float>? alias,
            IReadOnlyList<ILayer<float>> previousLayers,
            IReadOnlyList<ILayer<float>> replacementLayers)
        {
            return RebindLayerAlias(alias, previousLayers, replacementLayers, "testAlias");
        }

        public static ILayer<float> RebindRequiredAlias(
            ILayer<float> alias,
            IReadOnlyList<ILayer<float>> previousLayers,
            IReadOnlyList<ILayer<float>> replacementLayers)
        {
            return RebindRequiredLayerAlias(alias, previousLayers, replacementLayers, "testAlias");
        }

        public static void RebindCollection(
            IEnumerable<ILayer<float>> aliases,
            IReadOnlyList<ILayer<float>> previousLayers,
            IReadOnlyList<ILayer<float>> replacementLayers)
        {
            RebindLayerAliasCollection(aliases, previousLayers, replacementLayers, "testAliases");
        }

        protected override void InitializeLayers()
        {
        }

        public override Tensor<float> Predict(Tensor<float> input)
        {
            bool originalTrainingMode = IsTrainingMode;
            SetTrainingMode(false);

            Tensor<float> current = input;
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }

            SetTrainingMode(originalTrainingMode);
            return current;
        }

        public override void UpdateParameters(Vector<float> parameters)
        {
            SetParameters(parameters);
        }

        public override void Train(Tensor<float> input, Tensor<float> expectedOutput)
        {
            // Use tape-based training via base class
            TrainWithTape(input, expectedOutput);
        }

        public override ModelMetadata<float> GetModelMetadata()
        {
            return new ModelMetadata<float>
            {
                Name = "TestNetwork",
                Version = "1.0",
                FeatureCount = Architecture.InputSize,
                Complexity = (int)ParameterCount,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "LayerCount", Layers.Count }
                }
            };
        }

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }
    }

    [AiDotNet.Attributes.TensorLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Features,
        Direction = AiDotNet.Attributes.TensorLayoutDirection.Input)]
    [AiDotNet.Attributes.TensorLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Features,
        Direction = AiDotNet.Attributes.TensorLayoutDirection.Output)]
    private sealed class PredictCoreOverrideNetwork : NeuralNetworkBase<float>
    {
        public PredictCoreOverrideNetwork(NeuralNetworkArchitecture<float> architecture)
            : base(architecture, new MeanSquaredErrorLoss<float>())
        {
        }

        public bool ObservedTrainingModeInsidePredictCore { get; private set; }
        public override bool SupportsTraining => true;

        protected override void InitializeLayers()
        {
        }

        protected override Tensor<float> PredictCore(Tensor<float> input)
        {
            ObservedTrainingModeInsidePredictCore = IsTrainingMode;
            return input;
        }

        public override ModelMetadata<float> GetModelMetadata()
            => new() { Name = "PredictCoreOverrideNetwork", Version = "1.0" };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }

        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new PredictCoreOverrideNetwork(Architecture);
    }
}
