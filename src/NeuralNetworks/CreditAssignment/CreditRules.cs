using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Factory methods for the built-in credit-assignment (learning) rules, returning configurable
/// <see cref="ICreditRule{T}"/> instances for use with <c>AiModelBuilder.ConfigureCreditRule(ICreditRule&lt;T&gt;)</c>.
/// </summary>
/// <remarks>
/// <para>
/// This is the instance-based counterpart to the <c>CreditRule</c> enum overload of <c>ConfigureCreditRule</c>.
/// Use it when you want to configure a rule (e.g. fix its random seed) or pass a rule around as a value:
/// <code>
/// builder.ConfigureCreditRule(CreditRules.DirectFeedbackAlignment&lt;float&gt;(seed: 42));
/// </code>
/// The type argument is required because C# cannot infer it from the return type alone.
/// </para>
/// </remarks>
public static class CreditRules
{
    /// <summary>Standard reverse-mode back-propagation (the default learning rule).</summary>
    public static ICreditRule<T> Backprop<T>() => new BackpropCreditRule<T>();

    /// <summary>Feedback Alignment (Lillicrap et al., 2016): sequential fixed random feedback.</summary>
    /// <param name="seed">Optional RNG seed for the fixed feedback matrices (reproducibility).</param>
    public static ICreditRule<T> FeedbackAlignment<T>(int? seed = null) => new FeedbackAlignmentCreditRule<T>(seed);

    /// <summary>
    /// Direct Feedback Alignment (Nøkland, 2016): the global output error is projected directly onto each hidden
    /// layer through a fixed random matrix. Scales to Transformers/attention (Launay et al., 2020).
    /// </summary>
    /// <param name="seed">Optional RNG seed for the fixed feedback matrices (reproducibility).</param>
    public static ICreditRule<T> DirectFeedbackAlignment<T>(int? seed = null) => new DirectFeedbackAlignmentCreditRule<T>(seed);

    /// <summary>
    /// Sign-Symmetric feedback (Liao et al., 2016): the error is routed back through the sign of the transpose
    /// weights. Dense layers only (not defined for attention/normalization blocks).
    /// </summary>
    /// <param name="seed">Unused (kept for signature symmetry); Sign-Symmetric holds no random state.</param>
    public static ICreditRule<T> SignSymmetric<T>(int? seed = null) => new SignSymmetricCreditRule<T>(seed);
}
