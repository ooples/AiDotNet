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
    public void Conv_GetParameters_BeforeForward_Throws()
    {
        // Per PyTorch UninitializedParameter semantics, a lazy layer that has
        // not yet seen any input must reject GetParameters / SetParameters /
        // ParameterCount calls — its weight shape is not yet known.
        var conv = new ConvolutionalLayer<double>(outputDepth: 8, kernelSize: 3);
        Assert.Throws<InvalidOperationException>(() => conv.GetParameters());
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
