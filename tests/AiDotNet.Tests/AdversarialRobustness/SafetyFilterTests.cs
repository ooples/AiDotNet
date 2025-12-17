using AiDotNet.AdversarialRobustness.Safety;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.AdversarialRobustness;

public class SafetyFilterTests
{
    [Fact]
    public void ValidateInput_WithNaN_MarksInvalid()
    {
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.0, double.NaN, 1.0 });

        var result = filter.ValidateInput(input);

        Assert.False(result.IsValid);
        Assert.Contains(result.Issues, i => i.Type == "InvalidValue");
    }

    [Fact]
    public void ValidateInput_WithExcessLength_MarksInvalid()
    {
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 2
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.0, 1.0, 2.0 });

        var result = filter.ValidateInput(input);

        Assert.False(result.IsValid);
        Assert.Contains(result.Issues, i => i.Type == "LengthExceeded");
    }

    [Fact]
    public void ComputeSafetyScore_ReturnsBoundedScore()
    {
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            EnableOutputFiltering = true,
            MaxInputLength = 10
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(new[] { 0.0, 1.0, 2.0 });

        var score = filter.ComputeSafetyScore(input);

        Assert.True(score >= 0.0);
        Assert.True(score <= 1.0);
    }
}

