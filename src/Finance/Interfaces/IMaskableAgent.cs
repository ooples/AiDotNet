using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// An agent whose discrete action selection can be restricted to a legal subset.
/// </summary>
/// <remarks>
/// <para><b>Pre-shielding, not post-filtering.</b> The safety-RL literature distinguishes two shapes: a
/// <i>post-shield</i> replaces an unsafe action after the policy chose it, while a <i>pre-shield</i> supplies
/// the legal set so the policy chooses within it. Only the second is sound for learning — a policy whose
/// choice is overridden afterwards is trained toward actions it is then denied, so it learns from a payoff it
/// can never realise. This interface is the pre-shield.</para>
///
/// <para><b>Opt-in, because most agents cannot use it.</b> Masking is meaningful only over a discrete index
/// set. Continuous policies (SAC, CQL, IQL, DDPG) emit a real-valued vector — often a Gaussian mean and
/// log-sigma — where "legal action" is a feasible region, not a set of indices, and constraining that is
/// projection rather than masking. Those agents deliberately do not implement this interface.</para>
///
/// <para><b>An unsupported mask is REFUSED, not dropped.</b> A caller holding a non-null mask and an agent
/// that does not implement this interface must throw, not fall back to unmasked selection. That fallback
/// looks like graceful degradation and is not: the mask exists because some actions are unsafe or
/// unrepresentable, so selecting without it returns an ILLEGAL action to a caller who asked for a legal one,
/// with nothing in the result distinguishing the two. Only a <see langword="null"/> mask — meaning no
/// restriction was asked for — permits the unmasked path. <c>FinRLAgent&lt;T&gt;</c> is the worked
/// example: it forwards to a maskable inner agent and throws when its inner agent is continuous.</para>
///
/// <para><b>Why a separate interface rather than widening
/// <c>SelectAction</c>.</b> That method is overridden 61 times across this library — MuZero, Dreamer,
/// Decision Transformer, the tabular agents, the bandits, the policy classes — and adding a parameter to the
/// abstract declaration forces every one of them to change for a capability almost none can honour. The
/// surveyed ecosystem reaches the same conclusion by a different route: sb3-contrib ships a separate
/// <c>MaskablePPO</c> rather than altering <c>PPO.predict</c>.</para>
/// </remarks>
/// <typeparam name="T">Element type.</typeparam>
public interface IMaskableAgent<T>
{
    /// <summary>
    /// Selects an action restricted to the legal set.
    /// </summary>
    /// <param name="state">The current observation.</param>
    /// <param name="training">Whether exploration applies.</param>
    /// <param name="legalActions">
    /// One flag per action, <see langword="true"/> where selectable. Length must equal the agent's action
    /// size. <see langword="null"/> means unrestricted and must behave exactly as the unmasked call does.
    /// </param>
    /// <returns>
    /// For successful discrete selections, a one-hot action vector whose set index is always legal.
    /// A null mask preserves the native unmasked result, including continuous output for configurable agents.
    /// </returns>
    /// <remarks>
    /// <para><b>The contract is that the returned action is legal — at every selection site.</b> An
    /// implementation that masks its greedy branch but leaves its exploration draw unmasked satisfies the
    /// signature and violates the contract: an epsilon-greedy agent explores into illegal actions at rate
    /// epsilon, which early in training is nearly every step.</para>
    /// <para>Configurable agents may implement this interface in continuous mode, but must reject every
    /// non-null mask in that mode. Interface detection alone does not establish the active configuration.</para>
    /// </remarks>
    Vector<T> SelectAction(Vector<T> state, bool training, bool[]? legalActions);
}
