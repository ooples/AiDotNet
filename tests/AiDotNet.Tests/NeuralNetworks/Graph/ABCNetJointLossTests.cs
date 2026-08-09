using System;
using System.Linq;
using AiDotNet.ComputerVision.OCR.EndToEnd;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Holds ABCNet to its paper's JOINT objective — specifically, that the recognition branch is trained.
/// </summary>
/// <remarks>
/// <para>
/// Liu et al. (CVPR 2020) train detection and recognition together. Before this, ABCNet's training
/// forward returned only the detection tensor, so the two recognition convolutions and the per-column
/// classifier received no gradient at all: the model trained its detector and left its recognizer at
/// initialisation. Nothing caught it, because the loss still fell and every generic invariant still
/// passed — an untrained branch is invisible to a test that only watches the loss.
/// </para>
/// <para>
/// So the assertion here is on PARAMETERS, not on loss. A falling loss proves something is learning; only
/// a changed weight proves that THIS branch is.
/// </para>
/// </remarks>
public class ABCNetJointLossTests
{
    private readonly ITestOutputHelper _out;
    public ABCNetJointLossTests(ITestOutputHelper output) => _out = output;

    private static ABCNet<double> Model() =>
        new(new ABCNetOptions<double> { InputHeight = 32, InputWidth = 32, FeatureChannels = 16 });

    private static Tensor<double> Image()
    {
        var t = new Tensor<double>(new[] { 3, 32, 32 });
        for (int i = 0; i < t.Length; i++) t[i] = ((i * 13) % 23) / 23.0;
        return t;
    }

    [Fact]
    public void DeclaresBothHeadsOfThePapersObjective()
    {
        var declared = ((ICompositeLoss<double>)Model()).DeclaredOutputs;

        Assert.Equal(2, declared.Count);
        Assert.Contains(declared, o => o.Name == "detection");
        Assert.Contains(declared, o => o.Name == "recognition");

        // Equal weights: the paper states an unweighted sum. Declared rather than buried in a loss so it
        // can be read off and checked against the paper.
        Assert.All(declared, o => Assert.Equal(1.0, o.Weight));
    }

    [Fact]
    public void ComputeOutputsProducesBothHeads()
    {
        var outputs = ((ICompositeLoss<double>)Model()).ComputeOutputs(Image());

        Assert.Equal(2, outputs.Count);
        Assert.All(outputs, o => Assert.True(o is not null && o.Length > 0));

        _out.WriteLine($"detection   [{string.Join(",", outputs[0].Shape.ToArray())}]");
        _out.WriteLine($"recognition [{string.Join(",", outputs[1].Shape.ToArray())}]");
    }

    [Fact]
    public void TrainingMovesTheRECOGNITIONWeights_NotJustTheDetector()
    {
        // THE REGRESSION GUARD. These are layers 5-7 - the recognition convolutions and the per-column
        // classifier. Before the joint objective they were untouched by training, and the only way to see
        // that is to look at their weights before and after.
        var model = Model();
        var input = Image();

        // ONE TARGET PER DECLARED OUTPUT, read off the declaration - which is the point of declaring.
        // A single detection-shaped tensor cannot supervise a [32, 97] recognition head, so that head
        // would fall back to a self-target and contribute no gradient at all.
        var targets = ((ICompositeLoss<double>)model).ComputeOutputs(input)
            .Select(o =>
            {
                var t = new Tensor<double>(o.Shape.ToArray());
                for (int i = 0; i < t.Length; i++) t[i] = 0.5;
                return t;
            })
            .ToList();

        var recognitionLayers = model.Layers
            .Skip(ABCNet<double>.ExpectedLayerCount - 3)
            .ToList();

        var before = recognitionLayers.Select(l => l.GetParameters().ToArray()).ToList();

        // The probe must be measuring SOMETHING. These layers are lazy, so if their parameters are not
        // materialised the comparison loop below iterates zero times and reports 0.0 no matter what
        // training did - a test that cannot fail for the right reason and cannot pass for one either.
        for (int i = 0; i < before.Count; i++)
        {
            _out.WriteLine($"  {recognitionLayers[i].GetType().Name}: {before[i].Length} params");
        }

        Assert.True(
            before.Sum(p => p.Length) > 0,
            "the recognition layers report zero parameters, so this test measures nothing. Materialise "
            + "them before comparing, or the assertion below is vacuous.");
        model.Train(input, targets);
        var after = recognitionLayers.Select(l => l.GetParameters().ToArray()).ToList();

        double moved = 0.0;
        for (int i = 0; i < before.Count; i++)
        {
            for (int k = 0; k < before[i].Length && k < after[i].Length; k++)
            {
                moved = Math.Max(moved, Math.Abs(after[i][k] - before[i][k]));
            }
        }

        _out.WriteLine($"largest recognition-weight change after one step: {moved:E3}");

        Assert.True(
            moved > 0.0,
            "not one recognition-branch weight changed after a training step, so the branch received no "
            + "gradient. That is the paper gap this objective exists to close: ABCNet trains detection "
            + "and recognition JOINTLY, and a detection-only forward leaves the recognizer at its "
            + "initialisation while the loss still falls and every other invariant still passes.");
    }

    [Fact]
    public void RecognitionGradientReachesTheCoordinateHead()
    {
        // THE COUPLING, which is what makes ABCNet one model rather than a detector bolted to a
        // recognizer. BezierAlign samples the trunk THROUGH the regressed control points, so a
        // recognition term must move the Bezier head too. Sampling through detached coordinates would
        // reproduce the arithmetic and lose the paper's actual argument.
        var model = Model();
        var input = Image();

        var targets = ((ICompositeLoss<double>)model).ComputeOutputs(input)
            .Select(o =>
            {
                var t = new Tensor<double>(o.Shape.ToArray());
                for (int i = 0; i < t.Length; i++) t[i] = 0.5;
                return t;
            })
            .ToList();

        // Layer index 4 is the Bezier coordinate head (backbone 0-2, score head 3, Bezier head 4).
        var bezierHead = model.Layers[4];
        var before = bezierHead.GetParameters().ToArray();
        model.Train(input, targets);
        var after = bezierHead.GetParameters().ToArray();

        double moved = 0.0;
        for (int k = 0; k < before.Length && k < after.Length; k++)
        {
            moved = Math.Max(moved, Math.Abs(after[k] - before[k]));
        }

        _out.WriteLine($"largest Bezier-head change: {moved:E3}");
        Assert.True(moved > 0.0, "the Bezier coordinate head received no gradient at all.");
    }
}
