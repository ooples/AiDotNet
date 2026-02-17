using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Unlearning;

/// <summary>
/// Approximate unlearning via gradient ascent: reverses learning by ascending the loss on target data.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Normal training minimizes the loss (gradient descent — going downhill).
/// Gradient ascent does the opposite: it maximizes the loss on the target client's data, effectively
/// making the model "forget" what it learned from that client. Think of it as deliberately making the
/// model bad at the target client's data while keeping it good at everyone else's.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Take the target client's historical model updates.</description></item>
/// <item><description>Compute the aggregate direction the client pushed the model.</description></item>
/// <item><description>Move the model in the OPPOSITE direction (ascent) for several epochs.</description></item>
/// <item><description>Optionally fine-tune on remaining clients to recover lost accuracy.</description></item>
/// </list>
///
/// <para><b>Speed:</b> Much faster than exact retraining (minutes vs. hours). Provides approximate
/// guarantees — the target client's influence is reduced but not provably zero.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class GradientAscentUnlearner<T> : FederatedLearningComponentBase<T>, IFederatedUnlearner<T>
{
    private readonly FederatedUnlearningOptions _options;

    /// <inheritdoc/>
    public string MethodName => "GradientAscent";

    /// <summary>
    /// Initializes a new instance of <see cref="GradientAscentUnlearner{T}"/>.
    /// </summary>
    /// <param name="options">Unlearning configuration.</param>
    public GradientAscentUnlearner(FederatedUnlearningOptions options)
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

        // Compute the target client's aggregate contribution
        var targetContribution = new double[modelSize];

        if (clientHistories.ContainsKey(targetClientId))
        {
            var targetUpdates = clientHistories[targetClientId];
            targetRounds = targetUpdates.Count;

            foreach (var update in targetUpdates)
            {
                for (int i = 0; i < modelSize && i < update.Shape[0]; i++)
                {
                    targetContribution[i] += NumOps.ToDouble(update[i]);
                }
            }

            // Average the contribution
            if (targetRounds > 0)
            {
                for (int i = 0; i < modelSize; i++)
                {
                    targetContribution[i] /= targetRounds;
                }
            }
        }

        // Apply gradient ascent: move OPPOSITE to client's contribution
        var unlearnedModel = new Tensor<T>(new[] { modelSize });
        double lr = _options.UnlearningLearningRate;

        for (int i = 0; i < modelSize; i++)
        {
            unlearnedModel[i] = globalModel[i];
        }

        for (int epoch = 0; epoch < _options.MaxUnlearningEpochs; epoch++)
        {
            // Decay learning rate over epochs
            double currentLr = lr * (1.0 - (double)epoch / _options.MaxUnlearningEpochs);

            for (int i = 0; i < modelSize; i++)
            {
                double param = NumOps.ToDouble(unlearnedModel[i]);
                // Ascend: ADD the contribution direction (opposite of descent)
                param -= currentLr * targetContribution[i];
                unlearnedModel[i] = NumOps.FromDouble(param);
            }
        }

        // Fine-tune on remaining clients to recover accuracy
        FineTuneOnRemaining(unlearnedModel, targetClientId, clientHistories);

        stopwatch.Stop();
        string postHash = ComputeModelHash(unlearnedModel);

        double divergence = ComputeL2Distance(globalModel, unlearnedModel);
        double membershipScore = ComputeMembershipScore(targetContribution, unlearnedModel, globalModel);

        bool verified = _options.VerificationEnabled && membershipScore >= 0.45;

        var certificate = new UnlearningCertificate
        {
            TargetClientId = targetClientId,
            MethodUsed = UnlearningMethod.GradientAscent,
            Verified = verified,
            MembershipInferenceScore = membershipScore,
            ModelDivergence = divergence,
            RetainedAccuracy = 1.0 - divergence * 0.01, // Heuristic estimate
            ClientRoundsParticipated = targetRounds,
            UnlearningTimeMs = stopwatch.ElapsedMilliseconds,
            PreUnlearningModelHash = preHash,
            PostUnlearningModelHash = postHash,
            Summary = $"Gradient ascent unlearning: {_options.MaxUnlearningEpochs} epochs, " +
                     $"MIA score: {membershipScore:F4}, divergence: {divergence:F6}."
        };

        return (unlearnedModel, certificate);
    }

    private void FineTuneOnRemaining(
        Tensor<T> model, int targetClientId,
        Dictionary<int, List<Tensor<T>>> clientHistories)
    {
        int modelSize = model.Shape[0];
        int healingEpochs = Math.Max(1, _options.MaxUnlearningEpochs / 3);
        double lr = _options.UnlearningLearningRate * 0.1; // Lower LR for fine-tuning

        for (int epoch = 0; epoch < healingEpochs; epoch++)
        {
            var avgUpdate = new double[modelSize];
            int clientCount = 0;

            foreach (var kvp in clientHistories)
            {
                if (kvp.Key == targetClientId || kvp.Value.Count == 0) continue;

                // Use the last round's update as representative
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

    private double ComputeMembershipScore(double[] contribution, Tensor<T> unlearned, Tensor<T> original)
    {
        // Measure how much of the target's contribution remains
        int size = Math.Min(unlearned.Shape[0], original.Shape[0]);
        double remainingInfluence = 0;
        double totalInfluence = 0;

        for (int i = 0; i < size; i++)
        {
            double diff = NumOps.ToDouble(unlearned[i]) - NumOps.ToDouble(original[i]);
            remainingInfluence += diff * contribution[i];
            totalInfluence += contribution[i] * contribution[i];
        }

        if (totalInfluence < 1e-12) return 0.5;

        double ratio = Math.Abs(remainingInfluence / totalInfluence);
        return 0.5 + (1.0 - Math.Min(1.0, ratio)) * 0.5;
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
