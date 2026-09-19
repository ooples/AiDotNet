using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// An environment that can say which discrete actions are legal in the current state.
/// </summary>
/// <remarks>
/// <para><b>Why the ENVIRONMENT owns legality.</b> Whether an action can be taken is a property of the world,
/// not of the policy: a chess position has legal moves, an account has structures it is cleared to trade. The
/// agent is the wrong place to encode that, and a post-hoc filter applied after selection is worse still —
/// the policy is then trained toward actions it is subsequently denied, so it learns from a payoff it can
/// never realise.</para>
///
/// <para><b>This follows the established convention rather than inventing one.</b> PettingZoo carries the
/// mask in the observation (<c>observation["action_mask"]</c>); Shimmy's OpenSpiel environments and RLlib
/// carry it alongside the observation in the info/dict channel; OpenSpiel exposes <c>legal_actions()</c> on
/// the state. All of them put legality with the world and let the policy consult it. The one shape deliberately
/// NOT copied is folding the mask into the observation vector itself — that would change
/// <see cref="IEnvironment{T}.ObservationSpaceDimension"/> and therefore every agent's input width, which is
/// exactly why RLlib uses a Dict space instead.</para>
///
/// <para><b>Opt-in, so nothing existing changes.</b> This is a separate interface rather than a new member on
/// <see cref="IEnvironment{T}"/>: an environment with no notion of illegal actions implements nothing, and the
/// 60-odd agents that cannot use a mask are untouched. A consumer tests for the interface and falls back to
/// the unmasked path.</para>
///
/// <para><b>Convention for the info dictionary.</b> Implementations SHOULD also publish the same array under
/// <see cref="AiDotNet.ReinforcementLearning.ActionMasking.ActionMaskKey"/> (spelled <c>action_mask</c>) in the
/// dictionary returned by <see cref="IEnvironment{T}.Step"/>, so a consumer
/// that only has the step result can read it without holding the environment reference. The property is the
/// authority; the info entry is a mirror. The property also covers the case the info channel cannot:
/// <see cref="IEnvironment{T}.Reset"/> returns only an observation, so an agent choosing its FIRST action has
/// no step result to read.</para>
/// </remarks>
/// <typeparam name="T">Element type.</typeparam>
public interface IMaskedActionEnvironment<T> : IEnvironment<T>
{
    /// <summary>
    /// One flag per discrete action: <see langword="true"/> where the action may be selected in the current
    /// state, <see langword="false"/> where it may not. Length equals
    /// <see cref="IEnvironment{T}.ActionSpaceSize"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>At least one action must remain legal.</b> A fully-masked state leaves the policy with nothing
    /// to choose and no honest way to proceed; an environment that can reach such a state should model it as
    /// terminal, or keep an explicit no-op legal. Consumers are entitled to treat an all-false mask as a bug
    /// rather than silently picking an illegal action.</para>
    ///
    /// <para>Meaningful only for discrete action spaces. When
    /// <see cref="IEnvironment{T}.IsContinuousActionSpace"/> is <see langword="true"/> there is no index set to
    /// mask, and an implementation should return <see langword="null"/> rather than pretend otherwise —
    /// constraining a continuous action is projection onto a feasible region, which is a different operation
    /// and not this one.</para>
    /// </remarks>
    bool[]? LegalActionMask { get; }
}
