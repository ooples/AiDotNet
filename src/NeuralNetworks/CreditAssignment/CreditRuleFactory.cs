using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Maps the public <see cref="CreditRule"/> selector to a concrete <see cref="ICreditRule{T}"/> instance.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public static class CreditRuleFactory<T>
{
    /// <summary>
    /// Creates the built-in credit rule for <paramref name="rule"/>. Returns <c>null</c> for
    /// <see cref="CreditRule.Backprop"/> — the default reverse-mode path is used unchanged in that case.
    /// </summary>
    public static ICreditRule<T>? Create(CreditRule rule) => rule switch
    {
        CreditRule.Backprop => null,
        CreditRule.FeedbackAlignment => new FeedbackAlignmentCreditRule<T>(),
        CreditRule.DirectFeedbackAlignment => new DirectFeedbackAlignmentCreditRule<T>(),
        CreditRule.SignSymmetric => new SignSymmetricCreditRule<T>(),
        _ => throw new ArgumentOutOfRangeException(nameof(rule), rule, "Unknown credit rule."),
    };
}
