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

    /// <summary>
    /// <b>Kolen-Pollack (KP)</b> — Kolen &amp; Pollack, 1994; Akrout et al., 2019 ("Deep Learning without Weight
    /// Transport"). Sequential feedback like FA, but the feedback matrices are <i>learned</i> rather than fixed:
    /// each step every feedback matrix receives the same outer-product increment (plus weight decay) that its
    /// forward weight receives, so the feedback converges to the transpose of the forward weight and credit
    /// assignment approaches back-propagation quality. Unlike FA/DFA this closes the alignment gap on deep nets.
    /// </summary>
    KolenPollack = 4,

    /// <summary>
    /// <b>Direct Kolen-Pollack (DKP)</b> — Kolen-Pollack learning applied in the Direct Feedback Alignment
    /// topology: the global output error is projected directly onto every hidden layer through per-layer
    /// feedback matrices, but those matrices are <i>learned</i> (each aligns to the effective forward Jacobian
    /// from its layer to the output) rather than fixed at random. Combines DFA's parallel/attention-friendly
    /// routing with KP's learned alignment.
    /// </summary>
    DirectKolenPollack = 5,

    /// <summary>
    /// <b>Direct Random Target Projection (DRTP)</b> — Frenkel, Lefebvre &amp; Bol, 2021 ("Learning without
    /// Feedback"). Each hidden layer's teaching signal is a <i>fixed random projection of the one-hot target</i>
    /// (not of the output error). Requires no backward error signal at all, making it the most hardware-friendly
    /// of the family; the target's sign supplies a valid descent direction on average.
    /// </summary>
    DRTP = 6,

    /// <summary>
    /// <b>Normalized Direct Feedback Alignment</b> — Launay et al., 2020 style. Vanilla DFA with per-layer
    /// normalization of the feedback projection so the teaching-signal magnitude stays stable across depth
    /// (the feedback columns are unit-normalized and each layer's teaching signal is rescaled to the output
    /// error's per-sample magnitude). This is the DFA variant intended to train deep / Transformer networks.
    /// </summary>
    DFANormalized = 7,
}
