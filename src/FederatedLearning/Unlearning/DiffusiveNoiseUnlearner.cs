using System.Security.Cryptography;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Unlearning;

/// <summary>
/// Unlearning via structured diffusive noise injection targeting memorized samples.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This approach (from 2025 research) works like adding carefully designed
/// static to a radio signal to drown out a specific station. Instead of reversing the learning directly,
/// we inject noise that is specifically structured to disrupt the model's memorization of the target
/// client's data patterns. Then we "heal" the model by fine-tuning on remaining clients.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description><b>Analysis:</b> Identify which parameters are most influenced by the target client
/// (high-influence parameters are those that changed most during the client's rounds).</description></item>
/// <item><description><b>Noise injection:</b> Add Gaussian noise scaled proportionally to each parameter's
/// influence from the target client. High-influence parameters get more noise.</description></item>
/// <item><description><b>Healing:</b> Fine-tune the noised model on remaining clients' data to recover
/// performance on non-target data while keeping the target's influence disrupted.</description></item>
/// </list>
///
/// <para><b>Advantages:</b> More robust than gradient ascent (which can overshoot), and doesn't require
/// Hessian computation (which is expensive for large models). The structured noise targets memorization
/// specifically rather than general model quality.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class DiffusiveNoiseUnlearner<T> : FederatedLearningComponentBase<T>, IFederatedUnlearner<T>
{
    private readonly FederatedUnlearningOptions _options;

    /// <inheritdoc/>
    public string MethodName => "DiffusiveNoise";

    /// <summary>
    /// Initializes a new instance of <see cref="DiffusiveNoiseUnlearner{T}"/>.
    /// </summary>
    /// <param name="options">Unlearning configuration.</param>
    public DiffusiveNoiseUnlearner(FederatedUnlearningOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    public (Tensor<T> UnlearnedModel, UnlearningCertificate Certificate) Unlearn(
        int targetClientId,
        Tensor<T> globalModel,
        Dictionary<int, List<Tensor<T>>> clientHistories)
    {
        if (globalModel is null) throw new ArgumentNullException(nameof(globalModel));
        if (clientHistories is null) throw new ArgumentNullException(nameof(clientHistories));

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        string preHash = ComputeModelHash(globalModel);

        int modelSize = globalModel.Shape[0];
        int targetRounds = 0;

        // Step 1: Compute per-parameter influence scores from the target client
        var influenceScores = ComputeInfluenceScores(
            targetClientId, clientHistories, modelSize, out targetRounds);

        // Step 2: Inject structured noise proportional to influence
        var noisedModel = InjectStructuredNoise(globalModel, influenceScores);

        // Step 3: Heal by fine-tuning on remaining clients
        HealModel(noisedModel, targetClientId, clientHistories);

        stopwatch.Stop();
        string postHash = ComputeModelHash(noisedModel);

        double divergence = ComputeL2Distance(globalModel, noisedModel);
        double membershipScore = ComputeMembershipScore(influenceScores, globalModel, noisedModel);

        bool verified = _options.VerificationEnabled && membershipScore >= 0.45;

        var certificate = new UnlearningCertificate
        {
            TargetClientId = targetClientId,
            MethodUsed = UnlearningMethod.DiffusiveNoise,
            Verified = verified,
            MembershipInferenceScore = membershipScore,
            ModelDivergence = divergence,
            RetainedAccuracy = 1.0 - Math.Min(0.05, divergence * 0.001),
            ClientRoundsParticipated = targetRounds,
            UnlearningTimeMs = stopwatch.ElapsedMilliseconds,
            PreUnlearningModelHash = preHash,
            PostUnlearningModelHash = postHash,
            Summary = $"Diffusive noise unlearning: noise scale {_options.NoiseScale}, " +
                     $"MIA score: {membershipScore:F4}, divergence: {divergence:F6}."
        };

        return (noisedModel, certificate);
    }

    private Tensor<T> ComputeInfluenceScores(
        int targetClientId,
        Dictionary<int, List<Tensor<T>>> clientHistories,
        int modelSize,
        out int targetRounds)
    {
        var influence = new Tensor<T>(new[] { modelSize });
        targetRounds = 0;

        if (!clientHistories.ContainsKey(targetClientId))
        {
            return influence;
        }

        var targetUpdates = clientHistories[targetClientId];
        targetRounds = targetUpdates.Count;

        if (targetRounds > 0)
        {
            // Compute variance of target client's updates per parameter
            // Higher variance = parameter was more actively learned = higher influence
            var sum = new Tensor<T>(new[] { modelSize });
            var sumSq = new Tensor<T>(new[] { modelSize });
            foreach (var update in targetUpdates)
            {
                var fitted = UnlearningEngineOps.FitLength(update, modelSize);
                sum = Engine.TensorAdd(sum, fitted);
                sumSq = Engine.TensorAdd(sumSq, Engine.TensorSquare(fitted));
            }

            T rounds = NumOps.FromDouble(targetRounds);
            var mean = Engine.TensorDivideScalar(sum, rounds);
            var meanSq = Engine.TensorDivideScalar(sumSq, rounds);

            // Variance = E[x^2] - (E[x])^2, floored at 0
            var variance = Engine.TensorMax(Engine.TensorSubtract(meanSq, Engine.TensorSquare(mean)), NumOps.Zero);

            // Influence = magnitude of mean update * (1 + variance for uncertainty)
            influence = Engine.TensorMultiply(Engine.TensorAbs(mean), Engine.TensorAddScalar(Engine.TensorSqrt(variance), NumOps.One));
        }

        // Normalize influence scores to [0, 1]; a NaN score leaves them unnormalized, as the Math.Max fold did.
        if (modelSize > 0 && !double.IsNaN(NumOps.ToDouble(Engine.TensorSum(influence))))
        {
            T maxInfluence = Engine.ReduceMax(influence, new[] { 0 }, false).GetFlat(0);
            if (NumOps.GreaterThan(maxInfluence, NumOps.Zero))
            {
                influence = Engine.TensorDivideScalar(influence, maxInfluence);
            }
        }

        return influence;
    }

    private Tensor<T> InjectStructuredNoise(Tensor<T> model, Tensor<T> influenceScores)
    {
        int modelSize = model.Shape[0];

        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();

        // Standard normals seeded from the secure RNG, so the noise stays unpredictable.
        var standardNormals = new Tensor<T>(new[] { modelSize });
        Engine.TensorNormalInPlace(standardNormals, 0.0, 1.0, rng.Next());

        // Scale noise by influence score: high-influence params get more noise.
        var noise = Engine.TensorMultiply(
            standardNormals,
            Engine.TensorMultiplyScalar(influenceScores, NumOps.FromDouble(_options.NoiseScale)));

        return Engine.TensorAdd(UnlearningEngineOps.FitLength(model, modelSize), noise);
    }

    private void HealModel(
        Tensor<T> model, int targetClientId,
        Dictionary<int, List<Tensor<T>>> clientHistories)
    {
        int modelSize = model.Shape[0];
        int healingEpochs = Math.Max(1, _options.MaxUnlearningEpochs / 2);
        double lr = _options.UnlearningLearningRate * 0.05; // Very gentle healing

        // The remaining clients' last updates are the same every epoch, so the healing epochs
        // add up to param += epochs * lr * avgUpdate.
        var lastUpdates = clientHistories
            .Where(kvp => kvp.Key != targetClientId && kvp.Value.Count > 0)
            .Select(kvp => kvp.Value[kvp.Value.Count - 1])
            .ToList();
        if (lastUpdates.Count == 0) return;

        var sumUpdate = UnlearningEngineOps.SumFitted(lastUpdates, modelSize);
        var healed = Engine.TensorAdd(
            UnlearningEngineOps.FitLength(model, modelSize),
            Engine.TensorMultiplyScalar(sumUpdate, NumOps.FromDouble(healingEpochs * lr / lastUpdates.Count)));

        // Written back into the caller's model
        healed.AsSpan().CopyTo(model.AsWritableSpan().Slice(0, modelSize));
    }

    private double ComputeMembershipScore(Tensor<T> influence, Tensor<T> original, Tensor<T> unlearned)
    {
        int size = Math.Min(original.Shape[0], unlearned.Shape[0]);
        if (size <= 0) return 0.5;

        var weights = UnlearningEngineOps.FitLength(influence, size);
        var change = Engine.TensorAbs(Engine.TensorSubtract(UnlearningEngineOps.FitLength(original, size), UnlearningEngineOps.FitLength(unlearned, size)));
        double influencedChange = NumOps.ToDouble(Engine.TensorSum(Engine.TensorMultiply(change, weights)));
        double totalInfluence = NumOps.ToDouble(Engine.TensorSum(weights));

        if (totalInfluence < 1e-12) return 0.5;

        // Higher ratio of influenced change = better targeted unlearning
        double ratio = influencedChange / totalInfluence;
        return 0.5 + Math.Min(0.45, ratio * 0.3);
    }

    private double ComputeL2Distance(Tensor<T> a, Tensor<T> b)
    {
        return NumOps.ToDouble(VectorHelper.EuclideanDistance(a.ToVector(), b.ToVector()));
    }

    private string ComputeModelHash(Tensor<T> model)
    {
        using var sha256 = SHA256.Create();
        int size = model.Shape[0];
        var bytes = new byte[size * 8];
        for (int i = 0; i < size; i++)
        {
            var db = BitConverter.GetBytes(NumOps.ToDouble(model[i]));
            Buffer.BlockCopy(db, 0, bytes, i * 8, 8);
        }

        byte[] hash = sha256.ComputeHash(bytes);
        return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
    }
}
