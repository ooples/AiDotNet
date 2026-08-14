using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.LazyShape;

/// <summary>
/// Validates the lazy-shape contract introduced by issue #1209 — layers constructed
/// with only their kernel/output dims (no input H/W) resolve their shapes from the
/// actual input on the first forward pass, and report <c>IsShapeResolved</c>
/// accordingly.
/// </summary>
public class LayerShapeResolutionTests
{
    [Fact]
    public void Conv_BeforeForward_IsShapeResolvedIsFalse()
    {
        var conv = new ConvolutionalLayer<double>(outputDepth: 8, kernelSize: 3);
        Assert.False(conv.IsShapeResolved);
        // Output shape contains -1 placeholder for H/W
        var outShape = conv.GetOutputShape();
        Assert.Contains(-1, outShape);
    }

    [Fact]
    public void Conv_AfterFirstForward_ResolvesShapeFromInput()
    {
        var conv = new ConvolutionalLayer<double>(outputDepth: 8, kernelSize: 3, stride: 1, padding: 1);
        var input = new Tensor<double>(new[] { 1, 4, 16, 16 });

        var output = conv.Forward(input);

        Assert.True(conv.IsShapeResolved);
        // Output shape resolved: [outputDepth=8, H=16 (with stride=1+pad=1), W=16]
        var outShape = conv.GetOutputShape();
        Assert.Equal(3, outShape.Length);
        Assert.Equal(8, outShape[0]);
        Assert.Equal(16, outShape[1]);
        Assert.Equal(16, outShape[2]);
    }

    [Fact]
    public void Conv_DifferentInputSizes_SameInstance_BothResolveCorrectly()
    {
        // Per the lazy contract, the FIRST forward resolves the shape; subsequent
        // forwards with the same channel count + same kernel/stride/padding still
        // produce correct outputs (just the spatial dims change). A model that
        // legitimately needs to handle multiple input sizes constructs the layer
        // once and lets the first forward pin the shape — variable spatial dims
        // are handled by the convolution arithmetic itself, not by re-resolving
        // weight shapes.
        var conv = new ConvolutionalLayer<double>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1);
        var input1 = new Tensor<double>(new[] { 1, 3, 32, 32 });
        var input2 = new Tensor<double>(new[] { 1, 3, 64, 64 });

        var out1 = conv.Forward(input1);
        var out2 = conv.Forward(input2);

        Assert.Equal(4, out1.Shape[1]);
        Assert.Equal(32, out1.Shape[2]);
        Assert.Equal(4, out2.Shape[1]);
        Assert.Equal(64, out2.Shape[2]);
        Assert.True(conv.IsShapeResolved);
    }

    [Fact]
    public void Conv_GetParameters_BeforeForward_ReturnsEmpty()
    {
        // Lazy contract: a layer that hasn't seen its first input has no
        // weights yet, but GetParameters / ParameterCount must remain
        // safely callable so chain-walkers (optimizers, exporters,
        // composite SetParameters slicers) can compose with lazy children
        // without first having to drive a forward. Returning an empty
        // vector — and a 0 ParameterCount — is the documented contract.
        var conv = new ConvolutionalLayer<double>(outputDepth: 8, kernelSize: 3);
        var pre = conv.GetParameters();
        Assert.Equal(0, pre.Length);
        Assert.Equal(0, conv.ParameterCount);

        // After first Forward, weights are allocated and GetParameters
        // returns the real flat parameter vector.
        var input = new Tensor<double>(new[] { 1, 4, 8, 8 });
        conv.Forward(input);
        var post = conv.GetParameters();
        Assert.True(post.Length > 0);
        Assert.Equal(post.Length, conv.ParameterCount);
    }

    [Fact]
    public void Conv_RejectsInvalidCtorArgs()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConvolutionalLayer<double>(outputDepth: 0, kernelSize: 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConvolutionalLayer<double>(outputDepth: 8, kernelSize: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConvolutionalLayer<double>(outputDepth: 8, kernelSize: 3, stride: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConvolutionalLayer<double>(outputDepth: 8, kernelSize: 3, padding: -1));
    }

    [Fact]
    public void Conv_RejectsBadInputRank()
    {
        var conv = new ConvolutionalLayer<double>(outputDepth: 8, kernelSize: 3);
        var rank2 = new Tensor<double>(new[] { 16, 16 });
        Assert.Throws<ArgumentException>(() => conv.Forward(rank2));
    }

    [Fact]
    public void LayerNorm_ShapeOnlyGuess_ReconcilesWithFirstRealFeatureWidth()
    {
        using var layer = new LayerNormalizationLayer<double>();

        // A network topology walker sees only an approximate sequential width. Custom and
        // branched forwards may route a different tensor to this layer at execution time.
        layer.ResolveShapesOnly([32]);
        Assert.Equal(64, layer.ParameterCount);

        var actualInput = new Tensor<double>([2, 7, 192]);
        var output = layer.Forward(actualInput);

        Assert.Equal(actualInput.Shape, output.Shape);
        Assert.Equal(192, layer.GetGammaTensor().Length);
        Assert.Equal(192, layer.GetBetaTensor().Length);
        Assert.Equal(384, layer.ParameterCount);
        Assert.Equal(384, layer.GetParameters().Length);
    }

    [Fact]
    public void LayerNorm_EagerFeatureWidth_RemainsBinding()
    {
        using var layer = new LayerNormalizationLayer<double>(featureSize: 32);
        var incompatibleInput = new Tensor<double>([2, 192]);

        Assert.ThrowsAny<ArgumentException>(() => layer.Forward(incompatibleInput));
        Assert.Equal(32, layer.GetGammaTensor().Length);
        Assert.Equal(32, layer.GetBetaTensor().Length);
    }

    [Fact]
    public void Dense_ShapeOnlyGuess_AllocatesOnceAtFirstRealFeatureWidth()
    {
        using var layer = new DenseLayer<double>(outputSize: 8);

        layer.ResolveShapesOnly([32]);
        Assert.False(layer.IsInitialized);
        Assert.Equal((32 * 8) + 8, layer.ParameterCount);

        var output = layer.Forward(new Tensor<double>([2, 7, 192]));

        Assert.Equal(new[] { 2, 7, 8 }, output.Shape);
        Assert.Equal(192, layer.GetInputShape()[0]);
        Assert.Equal((192 * 8) + 8, layer.ParameterCount);
        Assert.Equal(layer.ParameterCount, layer.GetParameters().Length);
    }

    [Fact]
    public void Architecture_DynamicSpatialDims_CreateAndValidate()
    {
        var arch = NeuralNetworkArchitecture<double>.CreateDynamicSpatial(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            channels: 3,
            outputSize: 10);

        Assert.True(arch.HasDynamicSpatialDims);
        Assert.Equal(-1, arch.InputHeight);
        Assert.Equal(-1, arch.InputWidth);
        Assert.Equal(3, arch.InputDepth);
    }

    [Fact]
    public void Architecture_HalfDynamic_Rejected()
    {
        // H = 224, W = -1 — half-dynamic — must throw.
        Assert.Throws<ArgumentException>(() => new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 224,
            inputWidth: -1,
            inputDepth: 3,
            outputSize: 10));
    }
}
