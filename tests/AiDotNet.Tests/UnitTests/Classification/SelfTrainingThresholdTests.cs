using AiDotNet.Classification.SemiSupervised;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Classification;

/// <summary>
/// The pseudo-label eligibility test is the double comparison <c>confidence &gt;= threshold</c>, whatever the
/// element type; comparing against the threshold rounded to T moves the boundary by one ulp.
/// </summary>
public class SelfTrainingThresholdTests
{
    [Fact(Timeout = 60000)]
    public async Task AtLeast_Float_ThresholdRoundedDown_RejectsTheRoundedValue()
    {
        await Task.CompletedTask;

        // (float)0.7 = 0.699999988 < 0.7, so it is not confident enough; the next float up is.
        var values = new Tensor<float>(new[] { 0.7f, 0.70000005f, 0.6999999f }, new[] { 3 });

        var eligible = SelfTrainingClassifier<float>.AtLeast(values, 0.7);

        Assert.Equal(new[] { 0f, 1f, 0f }, eligible.ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task AtLeast_Float_ThresholdRoundedUp_KeepsTheRoundedValue()
    {
        await Task.CompletedTask;

        // (float)0.1 = 0.100000001 >= 0.1 is accepted; the next float down (0.099999994) is not.
        var values = new Tensor<float>(new[] { 0.1f, 0.10000001f, 0.099999994f }, new[] { 3 });

        var eligible = SelfTrainingClassifier<float>.AtLeast(values, 0.1);

        Assert.Equal(new[] { 1f, 1f, 0f }, eligible.ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task ConfidenceFromProbabilities_RowCountMismatch_Throws()
    {
        await Task.CompletedTask;

        // Two probability rows for three samples: reducing them gave two confidences, which the pseudo-label
        // selection then paired with the wrong samples.
        var probabilities = new Matrix<double>(new double[,] { { 0.2, 0.8 }, { 0.6, 0.4 } });

        Assert.Throws<InvalidOperationException>(() => SelfTrainingClassifier<double>.ConfidenceFromProbabilities(probabilities, 3));
        Assert.Throws<InvalidOperationException>(() => SelfTrainingClassifier<double>.ConfidenceFromProbabilities(null, 3));
    }

    [Fact(Timeout = 60000)]
    public async Task ConfidenceFromProbabilities_IsTheRowMaximumFlooredAtZero()
    {
        await Task.CompletedTask;
        var probabilities = new Matrix<double>(new double[,] { { 0.2, 0.8 }, { 0.6, 0.4 }, { -0.5, -0.1 } });

        var confidence = SelfTrainingClassifier<double>.ConfidenceFromProbabilities(probabilities, 3);

        Assert.Equal(new[] { 0.8, 0.6, 0.0 }, confidence.ToArray());
        Assert.Equal(new double[2], SelfTrainingClassifier<double>.ConfidenceFromProbabilities(new Matrix<double>(2, 0), 2).ToArray());
    }
}
