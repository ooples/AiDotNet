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

    /// <summary>
    /// Kolen-Pollack (Kolen &amp; Pollack, 1994; Akrout et al., 2019): sequential feedback whose matrices are
    /// <i>learned</i> — each converges to its forward weight — so credit assignment approaches back-propagation on
    /// deep dense stacks.
    /// </summary>
    /// <param name="seed">Optional RNG seed for the initial feedback matrices (reproducibility).</param>
    /// <param name="feedbackLearningRate">Step size for the feedback-matrix (alignment) update.</param>
    /// <param name="weightDecay">Weight decay applied to the feedback matrices (drives convergence to the forward weights).</param>
    public static ICreditRule<T> KolenPollack<T>(int? seed = null, double feedbackLearningRate = 0.05, double weightDecay = 0.001)
        => new KolenPollackCreditRule<T>(seed, feedbackLearningRate, weightDecay);

    /// <summary>
    /// Direct Kolen-Pollack: Kolen-Pollack learning in the Direct Feedback Alignment topology (direct
    /// output→layer feedback matrices that are <i>learned</i> rather than fixed). Scales to attention like DFA.
    /// </summary>
    /// <param name="seed">Optional RNG seed for the initial feedback matrices (reproducibility).</param>
    /// <param name="feedbackLearningRate">Step size for the feedback-matrix (alignment) update.</param>
    /// <param name="weightDecay">Weight decay applied to the feedback matrices.</param>
    public static ICreditRule<T> DirectKolenPollack<T>(int? seed = null, double feedbackLearningRate = 0.05, double weightDecay = 0.001)
        => new DirectKolenPollackCreditRule<T>(seed, feedbackLearningRate, weightDecay);

    /// <summary>
    /// Direct Random Target Projection (Frenkel &amp; Bol, 2021): each hidden layer's teaching signal is a fixed
    /// random projection of the one-hot target — no backward error path is needed.
    /// </summary>
    /// <param name="seed">Optional RNG seed for the fixed feedback matrices (reproducibility).</param>
    public static ICreditRule<T> DRTP<T>(int? seed = null) => new DrtpCreditRule<T>(seed);

    /// <summary>
    /// Normalized Direct Feedback Alignment (Launay et al., 2020 style): DFA with unit-norm feedback columns and
    /// per-layer teaching-signal rescaling so the signal magnitude stays stable across depth.
    /// </summary>
    /// <param name="seed">Optional RNG seed for the fixed feedback matrices (reproducibility).</param>
    public static ICreditRule<T> DFANormalized<T>(int? seed = null) => new NormalizedDfaCreditRule<T>(seed);

    /// <summary>
    /// <b>Local Error Signals</b> (Nøkland &amp; Eidnes, 2019): a backprop-free supervised rule where every hidden
    /// layer carries its own learned linear classifier to the labels and is trained by the gradient of its own
    /// cross-entropy. Deep routing layers (e.g. attention) far from the readout still receive a strong supervised
    /// signal, and the rule applies to non-contiguous trainable layers.
    /// </summary>
    public static ICreditRule<T> LocalErrorSignal<T>(int? seed = null, double classifierLearningRate = 0.05, double weightDecay = 0.0)
        => new LocalErrorSignalCreditRule<T>(seed, classifierLearningRate, weightDecay);
}
