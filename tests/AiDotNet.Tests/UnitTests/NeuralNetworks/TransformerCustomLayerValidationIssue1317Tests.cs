using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class TransformerCustomLayerValidationIssue1317Tests
{
    [Fact]
    public void Constructor_CustomShapeCompatibleLayerList_DoesNotRequireBuiltInBoundaryLayerTypes()
    {
        var layers = new List<AiDotNet.Interfaces.ILayer<float>>
        {
            new ShapeCompatibleCustomLayer([1, 16], [1, 256])
        };

        // numEncoderLayers / numDecoderLayers must both be 0 when a custom
        // `layers:` list is provided — the list replaces the auto-built
        // structure (#1382 fail-fast throw). This test verifies #1317:
        // shape-compatible custom layers are accepted.
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 0,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            complexity: NetworkComplexity.Medium,
            inputSize: 16,
            outputSize: 256,
            dropoutRate: 0.0,
            maxSequenceLength: 16,
            vocabularySize: 256,
            usePositionalEncoding: true,
            temperature: 1.0,
            sequencePooling: null,
            layers: layers);

        var transformer = new Transformer<float>(architecture);

        Assert.Same(layers[0], transformer.Layers[0]);
    }

    private sealed class ShapeCompatibleCustomLayer(int[] inputShape, int[] outputShape)
        : LayerBase<float>(inputShape, outputShape)
    {
        public override bool SupportsTraining => false;

        public override Tensor<float> Forward(Tensor<float> input)
        {
            return input;
        }

        public override void UpdateParameters(float learningRate)
        {
        }

        public override Vector<float> GetParameters()
        {
            return Vector<float>.Empty();
        }

        public override void ResetState()
        {
        }
    }
}
