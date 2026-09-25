namespace AiDotNet.Enums;

/// <summary>
/// Selects how a Soft Actor-Critic policy produces the SPREAD of its action distribution — the standard
/// deviation of the Gaussian it samples from.
/// </summary>
/// <remarks>
/// <para>
/// SAC's policy is a diagonal Gaussian, <c>pi(.|s) = N(mu(s), diag(sigma^2))</c>. Both members below share
/// the same mean head <c>mu(s)</c>; they differ only in where <c>sigma</c> comes from, which is what decides
/// whether the agent can be MORE uncertain in some market states than others.
/// </para>
/// <para>
/// <b>For Beginners:</b> The mean is the position the agent wants to take; the spread is how much it is
/// willing to jitter around it while exploring. The question this setting answers is whether that jitter is
/// one fixed amount for the whole run, or something the network decides per state.
/// </para>
/// </remarks>
public enum SacActorHead
{
    /// <summary>
    /// One learned log standard deviation per action dimension, shared across every state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The actor emits exactly <c>ActionSize</c> outputs (the mean), and the spread lives in a separate
    /// learned parameter vector. This is the state-independent log-std used by the reference PPO
    /// implementations in OpenAI Baselines, and it is the default here because it keeps the actor's output
    /// width at <c>ActionSize</c> — the shape every existing caller already builds its architecture against.
    /// </para>
    /// <para>
    /// The spread still learns (it trades entropy against value through the temperature alpha), but it
    /// cannot differ between a calm market and a volatile one.
    /// </para>
    /// </remarks>
    StateIndependentLogStd = 0,

    /// <summary>
    /// The actor predicts the log standard deviation from the state, alongside the mean. <b>This is the head
    /// described in the SAC paper</b> (Haarnoja et al. 2018).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The actor's output width becomes <c>2 * ActionSize</c>: the first <c>ActionSize</c> values are the
    /// mean and the remainder are the per-dimension log standard deviation, so the policy can be uncertain
    /// in states where the value of an action is unclear and confident where it is not. This is what the
    /// paper's stochastic actor does and what makes the entropy term genuinely state-dependent.
    /// </para>
    /// <para>
    /// <b>This changes the actor architecture's required output size</b>, so an architecture built for
    /// <c>ActionSize</c> outputs will be rejected. Opt in deliberately.
    /// </para>
    /// </remarks>
    StateConditionedGaussian = 1,
}
