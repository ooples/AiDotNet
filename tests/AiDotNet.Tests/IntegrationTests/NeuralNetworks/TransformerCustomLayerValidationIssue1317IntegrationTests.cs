using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration coverage for issue #1317: custom Transformer layer lists must
/// be validated by shape contract rather than concrete built-in layer types.
/// </summary>
public class TransformerCustomLayerValidationIssue1317IntegrationTests
{
    [Fact]
    public void CustomTransformerLayerStack_ForwardsWithoutBuiltInAttentionOrNormLayers()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [1, 32]),
            new ProjectingCustomLayer([32], [1, 256])
        };

        var model = new Transformer<float>(CreateArchitecture(layers));

        var input = new Tensor<float>([1, 16]);
        for (int i = 0; i < 16; i++)
            input[0, i] = i + 1;

        var output = model.Predict(input);

        Assert.Equal([1, 256], output.Shape);
        Assert.DoesNotContain(model.Layers, layer => layer is MultiHeadAttentionLayer<float>);
        Assert.DoesNotContain(model.Layers, layer => layer is LayerNormalizationLayer<float>);
        Assert.True(output[0, 1] > output[0, 0]);
    }

    [Fact]
    public void CustomTransformerLayerStack_RejectsIncompatibleLayerTransition()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [1, 32]),
            new ProjectingCustomLayer([1, 31], [1, 256])
        };

        var ex = Assert.Throws<ArgumentException>(() => new Transformer<float>(CreateArchitecture(layers)));
        Assert.Contains("Layer 0 is not compatible with Layer 1", ex.Message);
    }

    [Fact]
    public void CustomTransformerLayerStack_RejectsOutputSizeMismatch()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [1, 128])
        };

        var ex = Assert.Throws<ArgumentException>(() => new Transformer<float>(CreateArchitecture(layers)));
        Assert.Contains("must match the architecture output size (256)", ex.Message);
    }

    [Fact]
    public void DefaultTransformerArchitecture_StillBuildsStandardTransformerLayers()
    {
        var model = new Transformer<float>(CreateArchitecture(layers: null));

        Assert.Contains(model.Layers, layer => layer is MultiHeadAttentionLayer<float>);
        Assert.Contains(model.Layers, layer => layer is LayerNormalizationLayer<float>);
    }

    private static TransformerArchitecture<float> CreateArchitecture(List<ILayer<float>>? layers)
        => new(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1,
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

    private sealed class ProjectingCustomLayer(int[] inputShape, int[] outputShape)
        : LayerBase<float>(inputShape, outputShape)
    {
        public override bool SupportsTraining => false;

        public override Tensor<float> Forward(Tensor<float> input)
        {
            int batch = input.Shape.Length == 0 ? 1 : input.Shape[0];
            int outputWidth = GetOutputShape()[^1];
            var output = new Tensor<float>([batch, outputWidth]);

            float sum = 0f;
            for (int i = 0; i < input.Length; i++)
                sum += input[i];

            for (int b = 0; b < batch; b++)
            for (int j = 0; j < outputWidth; j++)
                output[b, j] = sum + j;

            return output;
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
