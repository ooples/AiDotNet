using System.Threading.Tasks;
using AiDotNet.DistributedTraining;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LossFunctions;

/// <summary>
/// The distributed token count comes from the loss that scored the objective, with the layout it scored when the
/// objective ran (a compiled plan's trace), not from whatever the same loss instance scored since.
/// </summary>
public class MeanReductionCaptureTests
{
    // Four rows of three classes; rows 1 and 3 carry the ignore index (-1).
    private static Tensor<float> ClassIndexTargets() =>
        new(new[] { 4 }, new Vector<float>(new[] { 0f, -1f, 2f, -1f }));

    private static Tensor<float> Logits(int rows, int classes)
    {
        var values = new float[rows * classes];
        for (int i = 0; i < values.Length; i++) values[i] = (i % 5) * 0.1f;
        return new Tensor<float>(new[] { rows, classes }, new Vector<float>(values));
    }

    [Fact(Timeout = 60000)]
    public async Task ScoredObjective_CountsTheSupervisedRowsOfItsOwnLayout()
    {
        await Task.CompletedTask;
        var loss = new CrossEntropyWithLogitsLoss<float>();
        var target = ClassIndexTargets();
        MeanReductionRecord<float>? scored;
        using (var scope = MeanReductionCapture<float>.Begin())
        {
            loss.ComputeTapeLoss(Logits(4, 3), target);
            scored = scope.Captured;
        }

        Assert.NotNull(scored);
        Assert.Same(loss, scored!.Loss);
        Assert.Equal(2L, DistributedTapeCoordinator<float>.CountValidElements(target, scored));

        // The same instance now scores a different layout (dense [4, 4] rows). The recorded reduction still counts the
        // objective's two supervised rows; counting with the loss's latest layout no longer recognizes the target.
        loss.ComputeTapeLoss(Logits(4, 4), new Tensor<float>(new[] { 4, 4 }));
        Assert.Equal(2L, DistributedTapeCoordinator<float>.CountValidElements(target, scored));
        Assert.Equal(4L, DistributedTapeCoordinator<float>.CountValidElements(target, (AiDotNet.Interfaces.ILossFunction<float>)loss));
    }

    [Fact(Timeout = 60000)]
    public async Task ObjectiveWithoutAReportingLoss_CountsEveryTargetElement()
    {
        await Task.CompletedTask;
        var target = ClassIndexTargets();
        MeanReductionRecord<float>? scored;
        using (var scope = MeanReductionCapture<float>.Begin())
        {
            new MeanSquaredErrorLoss<float>().ComputeTapeLoss(
                new Tensor<float>(new[] { 4 }), new Tensor<float>(new[] { 4 }));
            scored = scope.Captured;
        }

        Assert.Null(scored);
        Assert.Equal(4L, DistributedTapeCoordinator<float>.CountValidElements(target, scored));
    }

    [Fact(Timeout = 60000)]
    public async Task ScoringOutsideAScope_IsNotRecorded_AndNestedScopesKeepTheirOwn()
    {
        await Task.CompletedTask;
        var outerLoss = new CrossEntropyWithLogitsLoss<float>();
        var innerLoss = new CrossEntropyWithLogitsLoss<float>();
        outerLoss.ComputeTapeLoss(Logits(4, 3), ClassIndexTargets());   // no scope open: nothing to record into

        using var outer = MeanReductionCapture<float>.Begin();
        using (var inner = MeanReductionCapture<float>.Begin())
        {
            innerLoss.ComputeTapeLoss(Logits(4, 3), ClassIndexTargets());
            Assert.Same(innerLoss, inner.Captured!.Loss);
        }
        Assert.Null(outer.Captured);
        outerLoss.ComputeTapeLoss(Logits(4, 3), ClassIndexTargets());
        Assert.Same(outerLoss, outer.Captured!.Loss);
    }
}
