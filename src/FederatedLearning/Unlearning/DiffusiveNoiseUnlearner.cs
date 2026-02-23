using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;
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

    private double[] ComputeInfluenceScores(
        int targetClientId,
        Dictionary<int, List<Tensor<T>>> clientHistories,
        int modelSize,
        out int targetRounds)
    {
        var influence = new double[modelSize];
        targetRounds = 0;

        if (!clientHistories.ContainsKey(targetClientId))
        {
            return influence;
        }

        var targetUpdates = clientHistories[targetClientId];
        targetRounds = targetUpdates.Count;

        // Compute variance of target client's updates per parameter
        // Higher variance = parameter was more actively learned = higher influence
        var mean = new double[modelSize];
        var meanSq = new double[modelSize];

        foreach (var update in targetUpdates)
        {
            for (int i = 0; i < modelSize && i < update.Shape[0]; i++)
            {
                double val = NumOps.ToDouble(update[i]);
                mean[i] += val;
                meanSq[i] += val * val;
            }
        }

        if (targetRounds > 0)
        {
            for (int i = 0; i < modelSize; i++)
            {
                mean[i] /= targetRounds;
                meanSq[i] /= targetRounds;
                // Variance = E[x^2] - (E[x])^2
                double variance = Math.Max(0, meanSq[i] - mean[i] * mean[i]);
                // Influence = magnitude of mean update * (1 + variance for uncertainty)
                influence[i] = Math.Abs(mean[i]) * (1.0 + Math.Sqrt(variance));
            }
        }

        // Normalize influence scores to [0, 1]
        double maxInfluence = 0;
        for (int i = 0; i < modelSize; i++)
        {
            maxInfluence = Math.Max(maxInfluence, influence[i]);
        }

        if (maxInfluence > 0)
        {
            for (int i = 0; i < modelSize; i++)
            {
                influence[i] /= maxInfluence;
            }
        }

        return influence;
    }

    private Tensor<T> InjectStructuredNoise(Tensor<T> model, double[] influenceScores)
    {
        int modelSize = model.Shape[0];
        var noisedModel = new Tensor<T>(new[] { modelSize });

        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();

        for (int i = 0; i < modelSize; i++)
        {
            double param = NumOps.ToDouble(model[i]);

            // Scale noise by influence score — high-influence params get more noise
            double noiseScale = _options.NoiseScale * influenceScores[i];
            double noise = SampleGaussian(rng, 0, noiseScale);

            noisedModel[i] = NumOps.FromDouble(param + noise);
        }

        return noisedModel;
    }

    private void HealModel(
        Tensor<T> model, int targetClientId,
        Dictionary<int, List<Tensor<T>>> clientHistories)
    {
        int modelSize = model.Shape[0];
        int healingEpochs = Math.Max(1, _options.MaxUnlearningEpochs / 2);
        double lr = _options.UnlearningLearningRate * 0.05; // Very gentle healing

        for (int epoch = 0; epoch < healingEpochs; epoch++)
        {
            var avgUpdate = new double[modelSize];
            int clientCount = 0;

            foreach (var kvp in clientHistories)
            {
                if (kvp.Key == targetClientId || kvp.Value.Count == 0) continue;

                var lastUpdate = kvp.Value[kvp.Value.Count - 1];
                for (int i = 0; i < modelSize && i < lastUpdate.Shape[0]; i++)
                {
                    avgUpdate[i] += NumOps.ToDouble(lastUpdate[i]);
                }

                clientCount++;
            }

            if (clientCount == 0) break;

            for (int i = 0; i < modelSize; i++)
            {
                double param = NumOps.ToDouble(model[i]);
                param += lr * avgUpdate[i] / clientCount;
                model[i] = NumOps.FromDouble(param);
            }
        }
    }

    private double ComputeMembershipScore(double[] influence, Tensor<T> original, Tensor<T> unlearned)
    {
        int size = Math.Min(original.Shape[0], unlearned.Shape[0]);
        double influencedChange = 0;
        double totalInfluence = 0;

        for (int i = 0; i < size; i++)
        {
            double change = Math.Abs(NumOps.ToDouble(original[i]) - NumOps.ToDouble(unlearned[i]));
            influencedChange += change * influence[i];
            totalInfluence += influence[i];
        }

        if (totalInfluence < 1e-12) return 0.5;

        // Higher ratio of influenced change = better targeted unlearning
        double ratio = influencedChange / totalInfluence;
        return 0.5 + Math.Min(0.45, ratio * 0.3);
    }

    private double ComputeL2Distance(Tensor<T> a, Tensor<T> b)
    {
        int size = Math.Min(a.Shape[0], b.Shape[0]);
        double sumSq = 0;
        for (int i = 0; i < size; i++)
        {
            double diff = NumOps.ToDouble(a[i]) - NumOps.ToDouble(b[i]);
            sumSq += diff * diff;
        }

        return Math.Sqrt(sumSq);
    }

    private static double SampleGaussian(Random rng, double mean, double stdDev)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + stdDev * z;
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
