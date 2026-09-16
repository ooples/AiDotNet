using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video;
using AiDotNet.Video.Motion;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video.Motion;

/// <summary>
/// Optical-flow models must predict on a tensor of their own declared input shape.
/// </summary>
/// <remarks>
/// <para>
/// These models declare their input through the architecture as <c>InputType.ThreeDimensional</c> with
/// <c>InputDepth = 6</c> - a frame PAIR stacked on the channel axis - so <c>GetInputShape()</c> is the
/// unbatched <c>[6, H, W]</c>. Their shared <see cref="OpticalFlowBase{T}"/> <c>PredictCore</c> rejected
/// every rank-3 input with "Input must be rank 4" instead of adding the batch axis, so none of them could
/// run on the shape it advertises.
/// </para>
/// <para>
/// A small spatial size and a narrow, shallow stack keep this cheap; the defect is independent of both.
/// </para>
/// </remarks>
public class OpticalFlowDeclaredInputTests
{
    private const int Size = 16;

    private static NeuralNetworkArchitecture<double> PairArchitecture(int depth = 6) => new(
        inputType: InputType.ThreeDimensional,
        taskType: NeuralNetworkTaskType.Regression,
        inputHeight: Size, inputWidth: Size, inputDepth: depth,
        outputSize: 2);

    private static readonly Dictionary<string, Func<OpticalFlowBase<double>>> PairDeclaringModels = new()
    {
        ["DKM"] = () => new DKM<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["DPFlow"] = () => new DPFlow<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["FlowDiffuser"] = () => new FlowDiffuser<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["FlowFormerPlusPlus"] = () => new FlowFormerPlusPlus<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["MemFlow"] = () => new MemFlow<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["NeuFlowV2"] = () => new NeuFlowV2<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["RoMa"] = () => new RoMa<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["RPKNet"] = () => new RPKNet<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["SEARAFT"] = () => new SEARAFT<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["SKFlow"] = () => new SKFlow<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["UFM"] = () => new UFM<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["UniMatch"] = () => new UniMatch<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
        ["VideoFlow"] = () => new VideoFlow<double>(PairArchitecture(), numFeatures: 8, numLayers: 2),
    };

    public static IEnumerable<object[]> PairDeclaringModelNames()
        => PairDeclaringModels.Keys.Select(name => new object[] { name });

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<double>(shape);
        var span = t.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = rng.NextDouble();
        }
        return t;
    }

    private static Tensor<double> WithBatchAxis(Tensor<double> unbatched)
    {
        var shape = new int[unbatched.Rank + 1];
        shape[0] = 1;
        for (int i = 0; i < unbatched.Rank; i++) shape[i + 1] = unbatched.Shape[i];
        var batched = new Tensor<double>(shape);
        unbatched.Data.Span.CopyTo(batched.Data.Span);
        return batched;
    }

    private static void AssertSameValues(Tensor<double> expected, Tensor<double> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        var e = expected.Data.Span;
        var a = actual.Data.Span;
        for (int i = 0; i < e.Length; i++)
        {
            Assert.Equal(e[i], a[i], 10);
        }
    }

    [Theory]
    [MemberData(nameof(PairDeclaringModelNames))]
    public void Predict_OnTheDeclaredInputShape_ReturnsAnUnbatchedFlowField(string modelName)
    {
        var model = PairDeclaringModels[modelName]();

        int[] declared = model.GetArchitecture().GetInputShape();
        Assert.Equal(new[] { 6, Size, Size }, declared);

        var pair = Random(declared, seed: 5);
        var flow = model.Predict(pair);

        // Unbatched in, unbatched out: one [dx, dy] field at the input resolution.
        Assert.Equal(new[] { 2, Size, Size }, flow.Shape.ToArray());

        // And it is the SAME field the batched form produces for that one sample - the promotion adds
        // an axis, it does not change the computation.
        var batchedFlow = model.Predict(WithBatchAxis(pair));
        Assert.Equal(new[] { 1, 2, Size, Size }, batchedFlow.Shape.ToArray());
        AssertSameValues(batchedFlow, flow);
    }

    [Fact]
    public void RAPIDFlow_PredictsOnAnUnbatchedFramePair()
    {
        // RAPIDFlow is the family's exception: its architecture's InputDepth is the PER-FRAME channel count
        // (3), so its declared shape [3, H, W] describes one frame rather than the pair Predict takes (see
        // the TestScaffoldGenerator note on RAPIDFlow). What the shared base owes it is the same as every
        // other member: an unbatched pair [2*3, H, W] must be accepted, not rejected for its rank.
        var model = new RAPIDFlow<double>(PairArchitecture(depth: 3), numRefinementIterations: 1);

        var pair = Random([6, Size, Size], seed: 7);
        var flow = model.Predict(pair);

        Assert.Equal(new[] { 2, Size, Size }, flow.Shape.ToArray());
        AssertSameValues(model.Predict(WithBatchAxis(pair)), flow);
    }

    [Fact]
    public void RAFT_PredictsOnAnUnbatchedFramePair()
    {
        // RAFT overrides PredictCore but inherits the family's batch-optional input layout, so it owes the
        // same unbatched pair [2*C, H, W]. It indexed Shape[3] unconditionally and threw IndexOutOfRange.
        const int size = 32;
        var model = new RAFT<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputHeight: size, inputWidth: size, inputDepth: 3,
                outputSize: 2),
            numFeatures: 16, correlationLevels: 2, correlationRadius: 1, numIterations: 1);

        var pair = Random([6, size, size], seed: 9);
        var flow = model.Predict(pair);

        Assert.Equal(new[] { 2, size, size }, flow.Shape.ToArray());
        AssertSameValues(model.Predict(WithBatchAxis(pair)), flow);
    }
}
