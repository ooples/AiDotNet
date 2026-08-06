using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Holds every <see cref="IShapeContract"/> to the shape its type's REAL forward pass produces.
/// </summary>
/// <remarks>
/// <para>
/// The point of the whole shape system. A declared relation that is never executed against the
/// implementation is a comment with syntax highlighting — and worse than no annotation, because the next
/// reader believes it. These tests run the forward and compare, so a relation that drifts from the code
/// fails here rather than misleading an inference pass months later.
/// </para>
/// <para>
/// The convolution cases deliberately use inputs where the naive "divide by stride" shortcut and the
/// real formula DISAGREE. A suite built only on sizes divisible by the stride would pass against an
/// off-by-one implementation, which is precisely the error a shape system exists to catch.
/// </para>
/// </remarks>
public class ShapeContractConformanceTests
{
    private static Tensor<double> Input(params int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = ((i * 31) % 13) / 13.0;
        return t;
    }

    /// <summary>Runs the layer for real and asserts the declared contract predicted that exact shape.</summary>
    private static void AssertContractMatchesForward(object layer, params int[] inputShape)
    {
        var predicted = ShapeInference.InferOutputShape(layer, inputShape);
        Assert.True(
            predicted is not null,
            $"{layer.GetType().Name} declares a shape contract but it could not resolve for input "
            + $"[{string.Join(",", inputShape)}]. An unresolvable contract infers nothing, so every "
            + "caller silently falls back to running the model to find out.");

        var actual = ((ILayer<double>)layer).Forward(Input(inputShape)).Shape.ToArray();

        Assert.True(
            predicted!.SequenceEqual(actual),
            $"{layer.GetType().Name} for input [{string.Join(",", inputShape)}]: the shape contract "
            + $"predicts [{string.Join(",", predicted)}] but the forward pass produced "
            + $"[{string.Join(",", actual)}]. The declaration and the implementation disagree; one of "
            + "them is wrong and the declaration is the one nothing else was checking.");
    }

    // ---------------------------------------------------------------------------------------------
    // Convolution: the layer whose output extent the coarse relation kind could name but never compute
    // ---------------------------------------------------------------------------------------------

    [Theory]
    // stride 1, "same" padding — extent preserved
    [InlineData(8, 3, 3, 1, 1, 16, 16)]
    // stride 2 with padding: 32 -> 16, the case a scale factor also gets right
    [InlineData(4, 3, 3, 2, 1, 32, 32)]
    // stride 2, ODD input: 33 -> 17. "Divide by stride" says 16. This is the disagreement.
    [InlineData(4, 3, 3, 2, 1, 33, 33)]
    // stride 2, NO padding: 32 -> 15, not 16. The other disagreement.
    [InlineData(4, 3, 3, 2, 0, 32, 32)]
    // non-square input, so Height and Width cannot be conflated
    [InlineData(6, 3, 5, 2, 2, 24, 40)]
    // 1x1 projection — channels change, extent does not
    [InlineData(7, 3, 1, 1, 0, 12, 20)]
    public void Convolution_ContractPredictsTheForwardShape(
        int outChannels, int inChannels, int kernel, int stride, int padding, int height, int width)
    {
        var layer = new ConvolutionalLayer<double>(
            outputDepth: outChannels, kernelSize: kernel, stride: stride, padding: padding);

        AssertContractMatchesForward(layer, inChannels, height, width);
    }

    [Fact]
    public void Convolution_BatchedAndUnbatched_BothPredictCorrectly()
    {
        // Batch-optional is declared on the layout, so the contract must answer at both ranks — and must
        // NOT invent a batch axis for the unbatched form.
        var unbatched = new ConvolutionalLayer<double>(
            outputDepth: 5, kernelSize: 3, stride: 2, padding: 1);
        AssertContractMatchesForward(unbatched, 3, 16, 16);

        var batched = new ConvolutionalLayer<double>(
            outputDepth: 5, kernelSize: 3, stride: 2, padding: 1);
        AssertContractMatchesForward(batched, 2, 3, 16, 16);
    }

    [Fact]
    public void Convolution_ContractReadsTheInstance_NotTheType()
    {
        // Two layers of the same TYPE with different strides must report different relations. If the
        // contract were static per type - the failing of the coarse relation kind - these would agree,
        // and the whole mechanism would be decorative.
        var strideOne = new ConvolutionalLayer<double>(
            outputDepth: 4, kernelSize: 3, stride: 1, padding: 1);
        var strideTwo = new ConvolutionalLayer<double>(
            outputDepth: 4, kernelSize: 3, stride: 2, padding: 1);

        var a = ShapeInference.InferOutputShape(strideOne, new[] { 3, 32, 32 });
        var b = ShapeInference.InferOutputShape(strideTwo, new[] { 3, 32, 32 });

        Assert.Equal(new[] { 4, 32, 32 }, a);
        Assert.Equal(new[] { 4, 16, 16 }, b);
    }

    // ---------------------------------------------------------------------------------------------
    // Dense: feature-last, and rank-polymorphic
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void Dense_RankTwo_PredictsTheForwardShape()
    {
        AssertContractMatchesForward(new DenseLayer<double>(outputSize: 5), 4, 12);
    }

    [Fact]
    public void Dense_RankThree_PreservesTimeAndPredictsTheForwardShape()
    {
        // The sequence length is not the layer's to fix. A contract that pinned it would make a correct
        // layer appear to reject valid inputs.
        AssertContractMatchesForward(new DenseLayer<double>(outputSize: 5), 2, 7, 12);
    }

    [Fact]
    public void Dense_AcceptsAnySequenceLength_WithoutRedeclaring()
    {
        var layer = new DenseLayer<double>(outputSize: 5);

        Assert.Equal(new[] { 2, 3, 5 }, ShapeInference.InferOutputShape(layer, new[] { 2, 3, 12 }));
        Assert.Equal(new[] { 2, 50, 5 }, ShapeInference.InferOutputShape(layer, new[] { 2, 50, 12 }));
    }

    // ---------------------------------------------------------------------------------------------
    // The two declarations a type makes about itself must agree with each other
    // ---------------------------------------------------------------------------------------------

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void Convolution_ContractAgreesWithItsDeclaredLayout(int rank)
    {
        var layer = new ConvolutionalLayer<double>(
            outputDepth: 8, kernelSize: 3, stride: 1, padding: 1);

        Assert.True(
            ShapeInference.ContractMatchesLayout(layer, rank, out var mismatch),
            mismatch ?? string.Empty);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void Dense_ContractAgreesWithItsDeclaredLayout(int rank)
    {
        var layer = new DenseLayer<double>(outputSize: 5);

        Assert.True(
            ShapeInference.ContractMatchesLayout(layer, rank, out var mismatch),
            mismatch ?? string.Empty);
    }

    [Fact]
    public void ContractAndCoarseRelationKind_CannotDriftApart()
    {
        // ShapeRelationKind predates the symbolic contract and carries none of its terms. Deriving it
        // rather than maintaining it separately is what stops the two from becoming independent claims
        // that disagree.
        var conv = new ConvolutionalLayer<double>(
            outputDepth: 8, kernelSize: 3, stride: 2, padding: 1);

        Assert.Equal(
            ShapeRelationKind.Convolutional,
            ShapeInference.DeriveRelationKind(((IShapeContract)conv).OutputAxesFor(4)!));

        var dense = new DenseLayer<double>(outputSize: 5);
        Assert.Equal(
            ShapeRelationKind.FeatureOnly,
            ShapeInference.DeriveRelationKind(((IShapeContract)dense).OutputAxesFor(3)!));
    }

    // ---------------------------------------------------------------------------------------------
    // Declining beats guessing
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void UnsupportedRank_InfersNothing_RatherThanSomethingPlausible()
    {
        var conv = new ConvolutionalLayer<double>(
            outputDepth: 8, kernelSize: 3, stride: 1, padding: 1);

        // A rank-2 tensor is not an image. Returning a shape here would be a guess a caller cannot
        // distinguish from knowledge.
        Assert.Null(ShapeInference.InferOutputShape(conv, new[] { 16, 16 }));
    }

    [Fact]
    public void UnannotatedType_InfersNothing()
    {
        Assert.Null(ShapeInference.InferOutputShape(new object(), new[] { 1, 2, 3 }));
    }

    [Fact]
    public void WindowThatDoesNotFit_IsRefused_NotSilentlyClamped()
    {
        // A 7x7 kernel over a 4x4 input with no padding has no valid position. Reporting a size of 1
        // (or 0, or a negative clamped to 0) would hide a genuine misconfiguration.
        var relation = AxisRelation.Window(TensorAxis.Height, kernel: 7, stride: 1, padding: 0);
        var axes = new Dictionary<TensorAxis, int> { [TensorAxis.Height] = 4 };

        Assert.False(relation.TryResolve(axes, out _));
    }

    [Fact]
    public void ScaleThatDoesNotDivideEvenly_IsRefused()
    {
        // 5 halved is not an integer. A shape system that rounded here would disagree with whatever the
        // kernel actually does, in a direction nobody declared.
        var relation = AxisRelation.Scaled(TensorAxis.Width, numerator: 1, denominator: 2);
        var axes = new Dictionary<TensorAxis, int> { [TensorAxis.Width] = 5 };

        Assert.False(relation.TryResolve(axes, out _));
    }

    [Fact]
    public void UnknownRelation_CarriesItsReason_AndResolvesToNothing()
    {
        var relation = AxisRelation.Unknown("CTC collapses repeat runs, so the token count is data-dependent");

        Assert.False(relation.TryResolve(new Dictionary<TensorAxis, int>(), out _));
        Assert.Contains("data-dependent", relation.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void RelationsReadableAsFormulas()
    {
        // These strings end up in failure messages, so they have to say something a reader can act on.
        Assert.Equal(
            "floor((in.Height + 2*1 - 1*(3-1) - 1) / 2) + 1",
            AxisRelation.Window(TensorAxis.Height, kernel: 3, stride: 2, padding: 1).ToString());
        Assert.Equal("2 * in.Width", AxisRelation.Scaled(TensorAxis.Width, 2).ToString());
        Assert.Equal("in.Batch", AxisRelation.Same(TensorAxis.Batch).ToString());
    }
}
