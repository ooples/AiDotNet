namespace AiDotNet.Enums;

/// <summary>
/// Says where a cloned architecture's WEIGHTS come from: an independent draw, or the same draw the source
/// made.
/// </summary>
/// <remarks>
/// <para>
/// Cloning an architecture rebuilds its layers through their constructors, so the new model is initialized
/// rather than having weights copied into it. That leaves a real question — should the new model start from
/// the same numbers as the original, or from its own? — which the caller is in a position to answer and the
/// clone is not.
/// </para>
/// <para>
/// <b>For Beginners:</b> Two networks that start from identical random weights are not two samples of
/// anything; they are one network twice over. Some designs need exactly that (a target network, which is
/// about to be overwritten from its online copy anyway) and some are destroyed by it (a twin critic, whose
/// whole purpose is that the two estimates disagree).
/// </para>
/// </remarks>
public enum CloneInitialization
{
    /// <summary>
    /// The clone draws its own weights, deterministically. <b>The default.</b>
    /// </summary>
    /// <remarks>
    /// <para>
    /// When the source architecture carries a seed, the clone's seed is derived from it by splitting — so
    /// the clone's weights differ from the source's and from every other clone of that source, while still
    /// being reproducible run to run and independent of the order in which the clones were created. This is
    /// the model <c>jax.random.split</c> uses.
    /// </para>
    /// <para>
    /// When the source carries no seed there is nothing to split; each rebuild already draws from the shared
    /// non-deterministic generator, which is independent by construction.
    /// </para>
    /// </remarks>
    Independent = 0,

    /// <summary>
    /// The clone reproduces the source's initialization, giving a model that starts from the same weights.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this only when two models are deliberately meant to begin identically. It is honest about its one
    /// limitation: the clone re-runs the source's initialization rather than copying tensors, so this
    /// reproduces the source's weights <b>when the source's initialization was seeded</b>. With no seed there
    /// is nothing to reproduce and the clone gets a fresh draw.
    /// </para>
    /// <para>
    /// That caveat is not unique to this library: scikit-learn's <c>clone</c> draws the same distinction,
    /// returning an "exact clone" when <c>random_state</c> is an integer and a "statistical clone" — a
    /// genuinely different model — when it is a generator instance.
    /// </para>
    /// </remarks>
    Identical = 1,
}
