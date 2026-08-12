namespace AiDotNet.Models;

/// <summary>
/// Controls what a <c>Clone</c> carries across from the original.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Cloning a model gives you a second, separate copy. This class lets you say
/// how much of the original the copy should bring with it. You almost never need to set any of
/// this — <c>model.Clone()</c> already does the sensible thing, giving you a complete independent
/// copy that behaves exactly like the original.
/// </para>
/// <para>
/// The default is deliberately stronger than the equivalent in other libraries. In PyTorch,
/// <c>copy.deepcopy(model)</c> cannot carry optimizer state, because the optimizer is a separate
/// object that merely holds references to the model's parameters — so a copy taken mid-training
/// restarts its optimizer from scratch. That is a structural limitation rather than a considered
/// choice, so this library does not reproduce it: <see cref="Full"/> carries optimizer state, and
/// a clone taken mid-training resumes as the original would.
/// </para>
/// <para>
/// The one thing the default does <b>not</b> share is the random number stream. Two models drawing
/// from the same stream produce identical dropout masks and identical shuffles forever, so their
/// training silently correlates and nothing in the results reveals it. The clone instead gets a
/// fresh stream derived from the original's seed, which stays reproducible without the coupling.
/// Set <see cref="ShareRandomState"/> when you genuinely want a bit-identical twin.
/// </para>
/// <para>
/// The contract in one line: <i>a clone trains as if it were the original, but its randomness does
/// not track it.</i>
/// </para>
/// </remarks>
public sealed record CloneOptions
{
    /// <summary>
    /// Gets the default: a complete, independent copy with a freshly derived random stream.
    /// </summary>
    /// <value>
    /// Configuration, learned parameters, optimizer state, buffers and trainability flags are all
    /// carried; the random stream is derived rather than shared.
    /// </value>
    /// <remarks>
    /// This is what a bare <c>Clone()</c> uses. It is the least surprising reading of the word
    /// "clone" — the same thing, separately — and is strictly stronger than a deep copy in PyTorch,
    /// which cannot reach optimizer state at all.
    /// </remarks>
    public static CloneOptions Full { get; } = new();

    /// <summary>
    /// Gets a configuration-only copy: same architecture and settings, nothing learned.
    /// </summary>
    /// <value>
    /// Configuration and trainability flags are carried; parameters, optimizer state, buffers and
    /// the random stream are not.
    /// </value>
    /// <remarks>
    /// <para>
    /// This matches scikit-learn's <c>clone()</c>, which deliberately returns an <i>unfitted</i>
    /// estimator carrying the same hyperparameters. Use it to run the same architecture on
    /// different data, or to restart training from a fresh initialization.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as copying the recipe but not the cake.</para>
    /// </remarks>
    public static CloneOptions Architecture { get; } = new()
    {
        IncludeParameters = false,
        IncludeOptimizerState = false,
        IncludeBuffers = false,
    };
    /// <summary>
    /// Gets a copy that shares each weight tensor's storage until either side writes to it.
    /// </summary>
    /// <value>Everything <see cref="Full"/> carries, taken by copy-on-write rather than eagerly.</value>
    /// <remarks>
    /// Observationally identical to <see cref="Full"/> and O(1) until the first write, which then
    /// splits the two. This is not a new behaviour: it is what the library already did by default,
    /// decided by the <c>AIDOTNET_COW_DEEPCOPY</c> environment variable. A per-process switch is the
    /// wrong place for a per-call decision, so it is a value here.
    /// </remarks>
    public static CloneOptions CopyOnWrite { get; } = new() { Mode = CloneMode.CopyOnWrite };

    /// <summary>
    /// Gets an ALIAS: the copy points at the same parameters as the original.
    /// </summary>
    /// <value>Everything <see cref="Full"/> carries, shared rather than copied.</value>
    /// <remarks>
    /// <para>
    /// <b>Training this copy trains the original.</b> This is not a copy in any safe sense; it is a
    /// second handle on one model, for read-only fan-out such as evaluating the same weights on
    /// several inputs at once.
    /// </para>
    /// <para>
    /// Deliberately not reachable from a friendly-sounding preset. If you are unsure which of these
    /// you want, you want <see cref="Full"/> or <see cref="CopyOnWrite"/>.
    /// </para>
    /// </remarks>
    public static CloneOptions Shared { get; } = new() { Mode = CloneMode.Shared };

    /// <summary>
    /// Gets how much of the original's storage the copy is allowed to share.
    /// </summary>
    /// <value>Defaults to <see cref="CloneMode.Deep"/>.</value>
    /// <remarks>
    /// Orthogonal to <see cref="IncludeParameters"/> and the rest: those decide WHAT is carried,
    /// this decides whether what is carried is copied or shared. <see cref="CloneMode.Shared"/> with
    /// <see cref="IncludeParameters"/> off is meaningless, since there is nothing left to share.
    /// </remarks>
    public CloneMode Mode { get; init; } = CloneMode.Deep;


    /// <summary>
    /// Gets a value indicating whether configuration is carried. Always <see langword="true"/>.
    /// </summary>
    /// <value>Always <see langword="true"/>.</value>
    /// <remarks>
    /// Configuration is what makes the clone the same <i>kind</i> of thing as the original, so
    /// there is no meaningful clone without it. It is exposed as a property only so that reading
    /// a <see cref="CloneOptions"/> shows the complete picture rather than an implied part.
    /// </remarks>
    public bool IncludeConfiguration => true;

    /// <summary>
    /// Gets a value indicating whether learned parameters are carried. Defaults to <see langword="true"/>.
    /// </summary>
    /// <value><see langword="true"/> to copy trained weights; otherwise <see langword="false"/>.</value>
    /// <remarks>
    /// Parameters are read through <c>GetParameters()</c> and written through
    /// <c>UpdateParameters(Vector&lt;T&gt;)</c> — the same contract training uses on every step, so
    /// a clone cannot disagree with training about what the parameters are.
    /// </remarks>
    public bool IncludeParameters { get; init; } = true;

    /// <summary>
    /// Gets a value indicating whether optimizer state is carried. Defaults to <see langword="true"/>.
    /// </summary>
    /// <value><see langword="true"/> to copy momentum, moment estimates and step counts.</value>
    /// <remarks>
    /// <para>
    /// Carrying this is what lets a clone taken mid-training continue rather than restart. Adam's
    /// first and second moment estimates take many steps to warm up, so a copy without them
    /// behaves markedly differently from the original for a while — an effect easily mistaken for
    /// a difference in the model itself.
    /// </para>
    /// <para>
    /// This is the setting PyTorch cannot offer, since its optimizer lives outside the module.
    /// </para>
    /// </remarks>
    public bool IncludeOptimizerState { get; init; } = true;

    /// <summary>
    /// Gets a value indicating whether non-gradient learned state is carried. Defaults to <see langword="true"/>.
    /// </summary>
    /// <value><see langword="true"/> to copy running statistics such as batch-normalization means.</value>
    /// <remarks>
    /// Batch normalization's running mean and variance are learned from data but never receive a
    /// gradient. Dropping them leaves a clone that trains identically yet <i>evaluates</i>
    /// differently, which is a particularly hard difference to trace back to the clone.
    /// </remarks>
    public bool IncludeBuffers { get; init; } = true;

    /// <summary>
    /// Gets a value indicating whether the clone shares the original's random stream rather than
    /// deriving its own. Defaults to <see langword="false"/>.
    /// </summary>
    /// <value><see langword="true"/> for a bit-identical twin; <see langword="false"/> to derive a fresh stream.</value>
    /// <remarks>
    /// <para>
    /// Left <see langword="false"/>, the clone seeds itself deterministically from the original, so
    /// runs stay reproducible while the two models draw different dropout masks and shuffles.
    /// </para>
    /// <para>
    /// Set it <see langword="true"/> only when you want the two to behave identically down to their
    /// randomness — comparing an optimization change, for instance, where any difference in the
    /// random stream would confound the comparison.
    /// </para>
    /// </remarks>
    public bool ShareRandomState { get; init; }
}
