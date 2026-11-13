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
}
