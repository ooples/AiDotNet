using AiDotNet.Enums;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// Base class for deep learning-based causal discovery algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Deep learning methods learn causal structure by training neural networks that
/// parameterize the structural equation model. The DAG constraint is typically
/// enforced through continuous relaxation (e.g., NOTEARS-style) during training.
/// </para>
/// <para>
/// <b>For Beginners:</b> These methods use neural networks to discover causal relationships.
/// They can capture complex nonlinear effects but require more data and computation than
/// traditional methods. Think of them as "letting the neural network figure out which
/// variables cause which" by training it on the data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class DeepCausalBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.DeepLearning;

    /// <summary>
    /// Number of hidden units in neural network layers.
    /// </summary>
    protected int HiddenUnits { get; set; } = 64;

    /// <summary>
    /// Learning rate for gradient-based optimization.
    /// </summary>
    protected double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Maximum training epochs.
    /// </summary>
    protected int MaxEpochs { get; set; } = 100;

    /// <summary>
    /// Edge weight threshold for post-training pruning.
    /// </summary>
    protected double EdgeThreshold { get; set; } = 0.1;

    /// <summary>
    /// Initial log-variance for variational parameters.
    /// </summary>
    protected double InitialLogVariance { get; set; } = -4.0;

    /// <summary>
    /// Default KL divergence weight for variational regularization.
    /// </summary>
    protected double DefaultKlWeight { get; set; } = 0.01;

    /// <summary>
    /// Maximum KL divergence weight after warm-up schedule.
    /// </summary>
    protected double MaxKlWeight { get; set; } = 0.25;

    /// <summary>
    /// Whether to use KL weight warm-up schedule to prevent posterior collapse.
    /// </summary>
    protected bool UseKlWarmUp { get; set; } = true;

    /// <summary>
    /// Maximum penalty parameter (rho_max) for augmented Lagrangian methods.
    /// </summary>
    protected double MaxPenaltyValue { get; set; } = 1e+16;

    /// <summary>
    /// Applies deep learning options.
    /// </summary>
    protected void ApplyDeepOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.MaxIterations.HasValue) MaxEpochs = options.MaxIterations.Value;
        if (options.MaxEpochs.HasValue) MaxEpochs = options.MaxEpochs.Value;
        if (options.LearningRate.HasValue) LearningRate = options.LearningRate.Value;
        if (options.HiddenUnits.HasValue)
        {
            if (options.HiddenUnits.Value <= 0)
                throw new ArgumentException("HiddenUnits must be greater than 0.");
            HiddenUnits = options.HiddenUnits.Value;
        }
        if (options.EdgeThreshold.HasValue) EdgeThreshold = options.EdgeThreshold.Value;
        if (options.InitialLogVariance.HasValue) InitialLogVariance = options.InitialLogVariance.Value;
        if (options.DefaultKlWeight.HasValue) DefaultKlWeight = options.DefaultKlWeight.Value;
        if (options.MaxKlWeight.HasValue) MaxKlWeight = options.MaxKlWeight.Value;
        if (options.UseKlWarmUp.HasValue) UseKlWarmUp = options.UseKlWarmUp.Value;
        if (options.MaxPenalty.HasValue) MaxPenaltyValue = options.MaxPenalty.Value;
    }

}
