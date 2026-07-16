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
    /// <param name="rule">The credit rule selector.</param>
    /// <param name="seed">Optional RNG seed for reproducible fixed feedback matrices.</param>
    public static ICreditRule<T>? Create(CreditRule rule, int? seed = null) => rule switch
    {
        CreditRule.Backprop => null,
        CreditRule.FeedbackAlignment => new FeedbackAlignmentCreditRule<T>(seed),
        CreditRule.DirectFeedbackAlignment => new DirectFeedbackAlignmentCreditRule<T>(seed),
        CreditRule.SignSymmetric => new SignSymmetricCreditRule<T>(seed),
        CreditRule.KolenPollack => new KolenPollackCreditRule<T>(seed),
        CreditRule.DirectKolenPollack => new DirectKolenPollackCreditRule<T>(seed),
        CreditRule.DRTP => new DrtpCreditRule<T>(seed),
        CreditRule.DFANormalized => new NormalizedDfaCreditRule<T>(seed),
        CreditRule.LocalErrorSignal => new LocalErrorSignalCreditRule<T>(seed),
        _ => throw new ArgumentOutOfRangeException(nameof(rule), rule, "Unknown credit rule."),
    };
}
