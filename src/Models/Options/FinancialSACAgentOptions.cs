namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Financial SAC (Soft Actor-Critic) trading agent.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAC is an off-policy actor-critic algorithm that maximizes
/// both reward and entropy (exploration). It is well-suited for continuous action
/// spaces like portfolio weight allocation. These options extend the base trading
/// agent options with SAC-specific parameters.
/// </para>
/// </remarks>
public class FinancialSACAgentOptions<T> : TradingAgentOptions<T>
{
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public FinancialSACAgentOptions() { }

    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public FinancialSACAgentOptions(FinancialSACAgentOptions<T> other) : base(other)
    {
        // Base properties are copied by the base copy constructor; only SAC's own are listed here.
        // ActorHead in particular was missing, so a copied StateConditionedGaussian silently reverted to
        // StateIndependentLogStd — and that changes the actor output width the agent demands.
        InitialLogAlpha = other.InitialLogAlpha;
        TargetEntropyRatio = other.TargetEntropyRatio;
        ActorHead = other.ActorHead;
    }

    /// <summary>
    /// Initial log alpha value for automatic temperature tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how much the agent explores vs exploits.
    /// When auto-tuning is enabled, this is just the starting point.
    /// </para>
    /// </remarks>
    public double InitialLogAlpha { get; set; } = 0.0;

    /// <summary>
    /// Target entropy ratio relative to action dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sets the target randomness level as a fraction of
    /// the maximum possible entropy. Values closer to 1.0 encourage more exploration.
    /// </para>
    /// </remarks>
    public double TargetEntropyRatio { get; set; } = -1.0;

    /// <summary>
    /// Where the policy's standard deviation comes from. Defaults to
    /// <see cref="AiDotNet.Enums.SacActorHead.StateIndependentLogStd"/>, which keeps the actor's output width
    /// at <c>ActionSize</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Set this to <see cref="AiDotNet.Enums.SacActorHead.StateConditionedGaussian"/> for the head described
    /// in the SAC paper (Haarnoja et al. 2018), where the network predicts the spread from the state.
    /// <b>That widens the actor's required output size to <c>2 * ActionSize</c></b>, so an architecture built
    /// for <c>ActionSize</c> outputs will be rejected with a message saying so — which is why the
    /// non-breaking head is the default rather than the paper's.
    /// </para>
    /// <para><b>For Beginners:</b> "Spread" is how much the agent experiments around its chosen position.
    /// The default keeps one spread for every market condition. The paper's head lets the network decide
    /// the spread from what it is currently looking at — so it can commit in a clear market and hedge in a
    /// murky one — at the cost of needing an actor twice as wide.</para>
    /// </remarks>
    /// <value>
    /// A <see cref="AiDotNet.Enums.SacActorHead"/>. Defaults to
    /// <see cref="AiDotNet.Enums.SacActorHead.StateIndependentLogStd"/>, which requires an actor with
    /// <c>ActionSize</c> outputs; <see cref="AiDotNet.Enums.SacActorHead.StateConditionedGaussian"/>
    /// requires <c>2 * ActionSize</c> outputs and is rejected at construction otherwise.
    /// </value>
    public AiDotNet.Enums.SacActorHead ActorHead { get; set; } = AiDotNet.Enums.SacActorHead.StateIndependentLogStd;
}
