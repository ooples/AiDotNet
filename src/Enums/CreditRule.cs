namespace AiDotNet.Enums;

/// <summary>
/// Selects the built-in <b>credit-assignment</b> ("learning") rule used to produce per-parameter updates
/// during neural-network training via <c>AiModelBuilder.ConfigureCreditRule(...)</c>. The rule decides how
/// the error signal is routed back to each layer; the forward pass, optimizer, batching and scheduler are
/// unchanged. Use the <c>ICreditRule&lt;T&gt;</c> overload of <c>ConfigureCreditRule</c> to supply a custom rule.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> when a network makes a mistake, it must figure out how much each weight was to
/// blame so it knows how to change it. Standard <see cref="Backprop"/> sends the blame backwards through
/// the exact transpose of every weight. The alternative rules here send it through a <i>fixed random</i>
/// (or sign-only) shortcut instead — a family of "biologically plausible" methods that surprisingly still
/// learn because the forward weights rotate to align with the random feedback over training.
/// </para>
/// </remarks>
public enum CreditRule
{
    /// <summary>
    /// Standard reverse-mode back-propagation (the default). Error is routed back through the exact
    /// transpose of each weight matrix. Selecting this is identical to not configuring a credit rule at all.
    /// </summary>
    Backprop = 0,

    /// <summary>
    /// <b>Feedback Alignment (FA)</b> — Lillicrap et al., 2016. Replaces each layer's transpose-weight
    /// backward path with a <i>fixed random</i> feedback matrix of the same shape. The error is still
    /// propagated layer-by-layer (sequentially), but through random matrices; the forward weights learn to
    /// align with them.
    /// </summary>
    FeedbackAlignment = 1,

    /// <summary>
    /// <b>Direct Feedback Alignment (DFA)</b> — Nøkland, 2016. Projects the <i>global</i> output error
    /// directly to every hidden layer through a per-layer fixed random matrix (no sequential backward
    /// chain). Each layer's teaching signal depends only on the output error, enabling parallel/local
    /// updates.
    /// </summary>
    DirectFeedbackAlignment = 2,

    /// <summary>
    /// <b>Sign-Symmetric feedback</b> — Liao et al., 2016 / Xiao et al., 2018. Routes the error back through
    /// the <i>sign</i> of the transpose weights (magnitude discarded, sign of the true weight kept). Unlike
    /// FA/DFA the feedback tracks the live weight signs each step rather than being fixed at random.
    /// </summary>
    SignSymmetric = 3,
}
