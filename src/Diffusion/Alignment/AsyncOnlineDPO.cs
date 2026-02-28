using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Alignment;

/// <summary>
/// Asynchronous Online DPO for diffusion model alignment with on-policy generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Async Online DPO extends standard DPO by generating new preference pairs during training
/// rather than relying on a static dataset. It asynchronously generates image pairs from the
/// current policy, obtains preference labels (from a reward model or human feedback), and
/// uses these fresh pairs for DPO updates. This on-policy approach leads to better alignment
/// than offline DPO trained on stale data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular DPO learns from a fixed dataset of preferred/dispreferred image
/// pairs collected beforehand. Async Online DPO is smarter — it continuously generates new images
/// with the current model, gets feedback on them, and learns from that fresh feedback. It's like
/// a student who keeps practicing and getting real-time feedback rather than studying old examples.
/// </para>
/// <para>
/// Reference: Calandriello et al., "Human Alignment of Large Language Models through Online
/// Preference Optimization", 2024; adapted for diffusion models
/// </para>
/// </remarks>
public class AsyncOnlineDPO<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDiffusionModel<T> _model;
    private readonly IDiffusionModel<T> _referenceModel;
    private readonly double _beta;
    private readonly double _onlineMixingRatio;
    private readonly int _bufferSize;
    private int _totalUpdates;

    /// <summary>
    /// Gets the temperature parameter.
    /// </summary>
    public double Beta => _beta;

    /// <summary>
    /// Gets the ratio of online vs offline samples used in each batch.
    /// </summary>
    public double OnlineMixingRatio => _onlineMixingRatio;

    /// <summary>
    /// Gets the total number of DPO updates performed.
    /// </summary>
    public int TotalUpdates => _totalUpdates;

    /// <summary>
    /// Initializes a new Async Online DPO trainer.
    /// </summary>
    /// <param name="model">The diffusion model to align.</param>
    /// <param name="referenceModel">Frozen reference model for KL regularization.</param>
    /// <param name="beta">Temperature parameter (default: 5000.0).</param>
    /// <param name="onlineMixingRatio">Fraction of batch from online generation (default: 0.5).</param>
    /// <param name="bufferSize">Size of the online preference buffer (default: 1000).</param>
    public AsyncOnlineDPO(
        IDiffusionModel<T> model,
        IDiffusionModel<T> referenceModel,
        double beta = 5000.0,
        double onlineMixingRatio = 0.5,
        int bufferSize = 1000)
    {
        _model = model;
        _referenceModel = referenceModel;
        _beta = beta;
        _onlineMixingRatio = onlineMixingRatio;
        _bufferSize = bufferSize;
        _totalUpdates = 0;
    }

    /// <summary>
    /// Computes the online DPO loss combining online and offline preference pairs.
    /// </summary>
    /// <param name="preferredModelLogProb">Log-probability of preferred sample under current model.</param>
    /// <param name="disPreferredModelLogProb">Log-probability of dispreferred sample under current model.</param>
    /// <param name="preferredRefLogProb">Log-probability of preferred sample under reference model.</param>
    /// <param name="disPreferredRefLogProb">Log-probability of dispreferred sample under reference model.</param>
    /// <param name="isOnlineSample">Whether this pair was generated online (for weighting).</param>
    /// <returns>Weighted DPO loss.</returns>
    public T ComputeOnlineDPOLoss(
        T preferredModelLogProb,
        T disPreferredModelLogProb,
        T preferredRefLogProb,
        T disPreferredRefLogProb,
        bool isOnlineSample = true)
    {
        var prefDiff = NumOps.Subtract(preferredModelLogProb, preferredRefLogProb);
        var disPrefDiff = NumOps.Subtract(disPreferredModelLogProb, disPreferredRefLogProb);
        var logitDiff = NumOps.Subtract(prefDiff, disPrefDiff);
        var scaled = NumOps.Multiply(NumOps.FromDouble(_beta), logitDiff);

        var negScaled = NumOps.Negate(scaled);
        var expNeg = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(negScaled)));
        var sigmoid = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));

        var logSigmoid = NumOps.FromDouble(Math.Log(Math.Max(1e-10, NumOps.ToDouble(sigmoid))));
        var loss = NumOps.Negate(logSigmoid);

        // Apply importance weighting for online samples
        double weight = isOnlineSample ? _onlineMixingRatio : (1.0 - _onlineMixingRatio);
        return NumOps.Multiply(NumOps.FromDouble(weight), loss);
    }

    /// <summary>
    /// Increments the update counter after a training step.
    /// </summary>
    public void RecordUpdate()
    {
        _totalUpdates++;
    }

    /// <summary>
    /// Determines if the reference model should be refreshed based on update count.
    /// </summary>
    /// <param name="refreshInterval">Number of updates between reference refreshes (default: 100).</param>
    /// <returns>True if the reference model should be updated.</returns>
    public bool ShouldRefreshReference(int refreshInterval = 100)
    {
        return _totalUpdates > 0 && _totalUpdates % refreshInterval == 0;
    }
}
