using System;
using AiDotNet.Helpers;
using AiDotNet.ReinforcementLearning;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

public sealed class MaskNumericContractTests
{
    [Fact]
    public void Decimal_masked_logits_are_explicitly_unsupported()
    {
        var ops = MathHelper.GetNumericOperations<decimal>();
        var logits = new Vector<decimal>(new[] { 1m, 2m });
        Assert.Same(logits, ActionMasking.MaskLogits(logits, null, ops));
        var error = Assert.Throws<NotSupportedException>(() =>
            ActionMasking.MaskLogits(logits, new[] { true, false }, ops));
        Assert.Contains("Decimal", error.Message);
    }

    [Fact]
    public void Floating_point_masks_keep_infinity_and_legal_probability_support()
    {
        Assert.True(double.IsNegativeInfinity(ActionMasking.NegativeInfinity(MathHelper.GetNumericOperations<double>())));
        Assert.True(float.IsNegativeInfinity(ActionMasking.NegativeInfinity(MathHelper.GetNumericOperations<float>())));
        var probabilities = ActionMasking.MaskProbabilities(new[] { 0.25, 0.25, 0.25, 0.25 },
            new[] { false, true, false, true });
        Assert.Equal(new[] { 0d, 0.5, 0d, 0.5 }, probabilities);
    }
}
