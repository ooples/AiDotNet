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

        var model = new Transformer<float>(CreateCustomLayerArchitecture(layers));

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

        var ex = Assert.Throws<ArgumentException>(() => new Transformer<float>(CreateCustomLayerArchitecture(layers)));
        Assert.Contains("Layer 0 is not compatible with Layer 1", ex.Message);
    }

    [Fact]
    public void CustomTransformerLayerStack_RejectsOutputSizeMismatch()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [1, 128])
        };

        var ex = Assert.Throws<ArgumentException>(() => new Transformer<float>(CreateCustomLayerArchitecture(layers)));
        Assert.Contains("must match the architecture output size (256)", ex.Message);
    }

    [Fact]
    public void CustomTransformerLayerStack_RejectsFlattenedSizeOnlyTransitionWithoutExplicitReshape()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [2, 8]),
            new ProjectingCustomLayer([16], [1, 256])
        };

        var ex = Assert.Throws<ArgumentException>(() => new Transformer<float>(CreateCustomLayerArchitecture(layers)));
        Assert.Contains("Layer 0 is not compatible with Layer 1", ex.Message);
    }

    [Fact]
    public void CustomTransformerLayerStack_AllowsFlattenedTransitionThroughExplicitReshapeLayer()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [2, 8]),
            new ReshapeLayer<float>([16]),
            new ProjectingCustomLayer([16], [1, 256])
        };

        var model = new Transformer<float>(CreateCustomLayerArchitecture(layers));

        Assert.Same(layers[1], model.Layers[1]);
    }

    [Fact]
    public void CustomTransformerLayerStack_RejectsTransitionInputShapeMetadataFailures()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [1, 32]),
            new ThrowingInputShapeLayer([1, 32], [1, 256])
        };

        var ex = Assert.Throws<ArgumentException>(() => new Transformer<float>(CreateCustomLayerArchitecture(layers)));

        Assert.Contains("Failed to resolve shape metadata from layer", ex.Message);
        Assert.IsType<InvalidOperationException>(ex.InnerException);
    }

    [Fact]
    public void CustomTransformerLayerStack_RejectsOutputShapeMetadataFailures()
    {
        var layers = new List<ILayer<float>>
        {
            new ThrowingOutputShapeLayer([1, 16], [1, 256])
        };

        var ex = Assert.Throws<ArgumentException>(() => new Transformer<float>(CreateCustomLayerArchitecture(layers)));

        Assert.Contains("Failed to resolve shape metadata from layer", ex.Message);
        Assert.IsType<InvalidOperationException>(ex.InnerException);
    }

    [Fact]
    public void DefaultTransformerArchitecture_StillBuildsStandardTransformerLayers()
    {
        var model = new Transformer<float>(CreateDefaultTransformerArchitecture());

        Assert.Contains(model.Layers, layer => layer is MultiHeadAttentionLayer<float>);
        Assert.Contains(model.Layers, layer => layer is LayerNormalizationLayer<float>);
    }

    private static TransformerArchitecture<float> CreateCustomLayerArchitecture(List<ILayer<float>> layers)
        => CreateArchitecture(InputType.OneDimensional, layers);

    private static TransformerArchitecture<float> CreateDefaultTransformerArchitecture()
        => CreateArchitecture(InputType.TwoDimensional, layers: null);

    private static TransformerArchitecture<float> CreateArchitecture(InputType inputType, List<ILayer<float>>? layers)
        => new(
            inputType: inputType,
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

    private class ProjectingCustomLayer(int[] inputShape, int[] outputShape)
        : LayerBase<float>(inputShape, outputShape)
    {
        public override bool SupportsTraining => false;

        public override Tensor<float> Forward(Tensor<float> input)
        {
            var output = new Tensor<float>(GetOutputShape());

            float sum = 0f;
            for (int i = 0; i < input.Length; i++)
                sum += input[i];

            for (int i = 0; i < output.Length; i++)
                output[i] = sum + i;

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

    private sealed class ThrowingInputShapeLayer(int[] inputShape, int[] outputShape)
        : ProjectingCustomLayer(inputShape, outputShape)
    {
        public override int[] GetInputShape()
        {
            throw new InvalidOperationException("Input shape metadata is unavailable.");
        }
    }

    private sealed class ThrowingOutputShapeLayer(int[] inputShape, int[] outputShape)
        : ProjectingCustomLayer(inputShape, outputShape), ILayer<float>
    {
        int[] ILayer<float>.GetOutputShape()
        {
            throw new InvalidOperationException("Output shape metadata is unavailable.");
        }
    }
}
