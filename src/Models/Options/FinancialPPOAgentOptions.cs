namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Financial PPO (Proximal Policy Optimization) trading agent.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// The defaults are Schulman et al. 2017's MuJoCo setting (Table 3), the one for separate fully-connected policy and
/// value networks: a horizon of 2048 steps, Adam at 3e-4, 10 epochs of minibatches of 64, γ = 0.99, λ = 0.95, ε = 0.2
/// and no entropy bonus. Baselines' <c>ppo1</c> (the paper's code) supplies the rest: Adam's ε = 1e-5, observations
/// clipped to ±5 after standardizing, and the value loss at weight 1. CleanRL's additions are available as options:
/// <see cref="MaxGradientNorm"/> = 0.5 and <see cref="TradingAgentOptions{T}.EntropyCoefficient"/> = 0.01 with
/// <see cref="TradingAgentOptions{T}.ValueCoefficient"/> = 0.5.
/// </para>
/// <para>
/// <b>For Beginners:</b> PPO is a robust policy gradient algorithm that prevents
/// large policy updates, leading to stable training. These options extend the base
/// trading agent options with PPO-specific parameters.
/// </para>
/// </remarks>
public class FinancialPPOAgentOptions<T> : TradingAgentOptions<T>
{
    /// <summary>Creates the paper's defaults.</summary>
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public FinancialPPOAgentOptions()
    {
        // Schulman et al. 2017, section 6.1: "a fully-connected MLP with two hidden layers of 64 units, and
        // tanh nonlinearities". The agent applies tanh; this is the paper's width.
        HiddenLayers = new[] { 64, 64 };
        // Table 3 (MuJoCo) has no entropy term: the paper adds c₂·S only where policy and value share parameters.
        // Baselines' run_mujoco passes entcoeff = 0.0.
        EntropyCoefficient = 0.0;
        // Baselines adds the value loss at weight 1 (total_loss = pol_surr + pol_entpen + vf_loss); the paper's c₁
        // only matters for shared parameters, which this agent does not have.
        ValueCoefficient = 1.0;
    }

    /// <summary>Copies every option, the base's included.</summary>
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public FinancialPPOAgentOptions(FinancialPPOAgentOptions<T> other) : base(other)
    {
        // This used to chain to this() and copy only the PPO-specific properties, so a copy silently lost
        // every base property — including PPOClipRange and GAELambda, the ones PPO itself trains with.
        NumEpochs = other.NumEpochs;
        RolloutLength = other.RolloutLength;
        MiniBatchSize = other.MiniBatchSize;
        AdamEpsilon = other.AdamEpsilon;
        ObservationClipRange = other.ObservationClipRange;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    /// <summary>
    /// Optimization epochs over each rollout (K, 10 in the paper's MuJoCo setting).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many times to iterate over each batch of experiences.
    /// More epochs extract more learning but risk overfitting to the batch.
    /// </para>
    /// </remarks>
    public int NumEpochs { get; set; } = 10;

    /// <summary>
    /// Steps collected between updates (the horizon T, 2048 in the paper's MuJoCo setting). The rollout runs across
    /// episode boundaries; <see cref="Interfaces.IRLAgent{T}.Train"/> does nothing until it is full.
    /// </summary>
    public int RolloutLength { get; set; } = 2048;

    /// <summary>
    /// Steps per Adam update (64 in the paper's MuJoCo setting). A rollout that does not divide evenly drops its last
    /// partial minibatch each epoch, as Baselines' iterator does.
    /// </summary>
    public int MiniBatchSize { get; set; } = 64;

    /// <summary>Adam's ε (1e-5, Baselines' adam_epsilon).</summary>
    public double AdamEpsilon { get; set; } = 1e-5;

    /// <summary>
    /// The bound observations are clipped to after standardizing by their running statistics (5, Baselines'
    /// <c>clip_by_value(obz, -5, 5)</c>).
    /// </summary>
    public double ObservationClipRange { get; set; } = 5.0;

    /// <summary>
    /// The global gradient-norm bound over both networks, or null for none (the paper's and Baselines' setting).
    /// CleanRL clips at 0.5.
    /// </summary>
    public double? MaxGradientNorm { get; set; }
}
