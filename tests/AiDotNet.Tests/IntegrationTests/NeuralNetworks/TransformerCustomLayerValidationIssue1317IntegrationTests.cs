using AiDotNet.ActivationFunctions;
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

        Assert.Equal(new[] { 1, 256 }, output.Shape.ToArray());
        Assert.DoesNotContain(model.Layers, layer => layer is MultiHeadAttentionLayer<float>);
        Assert.DoesNotContain(model.Layers, layer => layer is LayerNormalizationLayer<float>);
        Assert.True(output[0, 1] > output[0, 0]);
    }

    [Fact]
    public void CustomTransformerLayerStack_RejectsFirstLayerInputShapeMismatch()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([2, 8], [1, 32]),
            new ProjectingCustomLayer([32], [1, 256])
        };

        var ex = Assert.Throws<ArgumentException>(() => new Transformer<float>(CreateCustomLayerArchitecture(layers)));
        Assert.Contains("first layer's input shape", ex.Message);
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
        // The enriched validator diagnostic now injects type + shape info between the layer
        // indices, e.g. "Layer 0 (ProjectingCustomLayer, output [..]) is not compatible with
        // Layer 1 (..)." — assert the two structural tokens rather than the old monolithic
        // substring (matches the Issue1323 regression guard's approach to the same upgrade).
        Assert.Contains("Layer 0", ex.Message);
        Assert.Contains("is not compatible with Layer 1", ex.Message);
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
        // The enriched validator diagnostic now injects type + shape info between the layer
        // indices, e.g. "Layer 0 (ProjectingCustomLayer, output [..]) is not compatible with
        // Layer 1 (..)." — assert the two structural tokens rather than the old monolithic
        // substring (matches the Issue1323 regression guard's approach to the same upgrade).
        Assert.Contains("Layer 0", ex.Message);
        Assert.Contains("is not compatible with Layer 1", ex.Message);
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
    public void CustomTransformerLayerStack_AllowsExplicitReshapeBeforeDynamicLeadingBatchAxis()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [2, 8]),
            new ReshapeLayer<float>([16]),
            new ProjectingCustomLayer([-1, 16], [1, 256])
        };

        var model = new Transformer<float>(CreateCustomLayerArchitecture(layers));

        Assert.Same(layers[1], model.Layers[1]);
        Assert.Same(layers[2], model.Layers[2]);
    }

    [Fact]
    public void CustomTransformerLayerStack_AllowsDynamicLeadingBatchAxis()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([-1, 16], [-1, 32]),
            new ProjectingCustomLayer([32], [1, 256])
        };

        var model = new Transformer<float>(CreateCustomLayerArchitecture(layers));

        Assert.Same(layers[0], model.Layers[0]);
        Assert.Same(layers[1], model.Layers[1]);
    }

    [Fact]
    public void CustomTransformerLayerStack_RejectsPartiallyKnownShapeMismatch()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [64, -1]),
            new ProjectingCustomLayer([128, -1], [1, 256])
        };

        var ex = Assert.Throws<ArgumentException>(() => new Transformer<float>(CreateCustomLayerArchitecture(layers)));
        // The enriched validator diagnostic now injects type + shape info between the layer
        // indices, e.g. "Layer 0 (ProjectingCustomLayer, output [..]) is not compatible with
        // Layer 1 (..)." — assert the two structural tokens rather than the old monolithic
        // substring (matches the Issue1323 regression guard's approach to the same upgrade).
        Assert.Contains("Layer 0", ex.Message);
        Assert.Contains("is not compatible with Layer 1", ex.Message);
    }

    [Fact]
    public void CustomTransformerLayerStack_AllowsPartiallyKnownShapeWhenKnownDimensionsMatch()
    {
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [64, -1]),
            new ProjectingCustomLayer([64, -1], [1, 256])
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

        // The default transformer encoder is now assembled as TransformerEncoderBlock
        // composites (the canonical Pre-LN block: self-attention + FFN, each wrapped in
        // a residual connection with LayerNorm — #1380). Self-attention and layer
        // normalization live INSIDE that block rather than as separate top-level layers,
        // so assert the standard encoder block is present (it encapsulates both).
        Assert.Contains(model.Layers, layer => layer is TransformerEncoderBlock<float>);
    }

    [Fact]
    public void CustomTransformerLayerStack_AcceptsFlashAttentionLayerAsDropInReplacement()
    {
        // Regression coverage: FlashAttentionLayer is documented as a drop-in
        // replacement for MultiHeadAttentionLayer ("FlashAttentionLayer provides
        // the same functionality as MultiHeadAttentionLayer but uses the Flash
        // Attention algorithm which is 2-4x faster and uses significantly less
        // memory. It can be used as a drop-in replacement in transformer
        // architectures.") The relaxed ValidateCustomLayers contract (#1317 /
        // #1320) accepts shape-compatible layer stacks; this test pins down
        // that the type-based "must include MultiHeadAttentionLayer" check is
        // gone for good and FlashAttention specifically rides the same path.
        const int seqLen = 16;
        const int dModel = 16;
        const int heads = 2;
        const int vocab = 256;

        var layers = new List<ILayer<float>>
        {
            new EmbeddingLayer<float>(vocab, dModel),
            new FlashAttentionLayer<float>(seqLen, dModel, heads),
            new LayerNormalizationLayer<float>(),
            new DenseLayer<float>(dModel, (IActivationFunction<float>)new ReLUActivation<float>()),
            new LayerNormalizationLayer<float>(),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(vocab, (IActivationFunction<float>)new IdentityActivation<float>())
        };

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            // Custom layers: REPLACE the auto-built encoder, so numEncoderLayers must be 0 (#1382).
            numEncoderLayers: 0,
            numDecoderLayers: 0,
            numHeads: heads,
            modelDimension: dModel,
            feedForwardDimension: dModel,
            complexity: NetworkComplexity.Medium,
            inputSize: seqLen,
            outputSize: vocab,
            dropoutRate: 0.0,
            maxSequenceLength: seqLen,
            vocabularySize: vocab,
            usePositionalEncoding: true,
            temperature: 1.0,
            sequencePooling: null,
            layers: layers);

        var model = new Transformer<float>(arch);

        Assert.Contains(model.Layers, layer => layer is FlashAttentionLayer<float>);
        Assert.DoesNotContain(model.Layers, layer => layer is MultiHeadAttentionLayer<float>);
    }

    [Fact]
    public void CustomTransformerArchitecture_WithoutAnyAttentionLayer_StillPassesValidation()
    {
        // The validator contract is shape-based, not type-based — a research
        // architecture that replaces attention with, e.g., an SSM/MLP-only
        // stack must be accepted as long as the shapes line up. This test
        // pins down that no concrete-attention-type requirement leaks back
        // in.
        var layers = new List<ILayer<float>>
        {
            new ProjectingCustomLayer([1, 16], [1, 32]),
            new ProjectingCustomLayer([32], [1, 256])
        };

        var model = new Transformer<float>(CreateCustomLayerArchitecture(layers));

        Assert.DoesNotContain(model.Layers, layer => layer is MultiHeadAttentionLayer<float>);
        Assert.DoesNotContain(model.Layers, layer => layer is FlashAttentionLayer<float>);
    }

    private static TransformerArchitecture<float> CreateCustomLayerArchitecture(List<ILayer<float>> layers)
        => CreateArchitecture(InputType.OneDimensional, layers);

    private static TransformerArchitecture<float> CreateDefaultTransformerArchitecture()
        => CreateArchitecture(InputType.TwoDimensional, layers: null);

    private static TransformerArchitecture<float> CreateArchitecture(InputType inputType, List<ILayer<float>>? layers)
        => new(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            // A custom `layers:` list REPLACES the auto-built encoder, so the constructor (correctly,
            // per #1382) rejects supplying both a custom list AND numEncoderLayers > 0. These tests
            // build custom-layer-only stacks, so use 0 encoder layers when layers: is supplied; the
            // default-architecture case (layers == null) keeps the standard single encoder layer.
            numEncoderLayers: layers is null ? 1 : 0,
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
