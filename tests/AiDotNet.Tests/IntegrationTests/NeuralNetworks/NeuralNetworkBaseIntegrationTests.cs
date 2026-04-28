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
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var network = new TestNeuralNetwork(architecture);
        var first = new DenseLayer<float>(3);
        network.AddLayer(first);

        var firstCount = network.ParameterCount;

        var second = new DenseLayer<float>(2);
        network.AddLayer(second);

        Assert.Equal(first.ParameterCount + second.ParameterCount, network.ParameterCount);
        Assert.True(network.RemoveLayer(second));
        Assert.Equal(firstCount, network.ParameterCount);
    }

    private sealed class TestNeuralNetwork : NeuralNetworkBase<float>
    {
        public TestNeuralNetwork(NeuralNetworkArchitecture<float> architecture)
            : base(architecture, new MeanSquaredErrorLoss<float>())
        {
        }

        public override bool SupportsTraining => true;

        public void AddLayer(ILayer<float> layer)
        {
            AddLayerToCollection(layer);
        }

        public bool RemoveLayer(ILayer<float> layer)
        {
            return RemoveLayerFromCollection(layer);
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
                Complexity = ParameterCount,
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

        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
        {
            return new TestNeuralNetwork(Architecture);
        }
    }
}
