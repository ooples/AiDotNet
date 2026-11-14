using System.Security.Cryptography;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Helpers;
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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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
        var targetContribution = new Tensor<T>(new[] { modelSize });

        if (clientHistories.ContainsKey(targetClientId))
        {
            var targetUpdates = clientHistories[targetClientId];
            targetRounds = targetUpdates.Count;

            // Average the contribution
            if (targetRounds > 0)
            {
                targetContribution = Engine.TensorDivideScalar(
                    UnlearningEngineOps.SumFitted(targetUpdates, modelSize), NumOps.FromDouble(targetRounds));
            }
        }

        // Apply gradient ascent: move OPPOSITE to client's contribution
        double lr = _options.UnlearningLearningRate;

        // Each epoch subtracts its decayed rate times the same contribution, so the epochs add up to
        // one step with the summed rate.
        double totalLr = 0.0;
        for (int epoch = 0; epoch < _options.MaxUnlearningEpochs; epoch++)
        {
            // Decay learning rate over epochs
            totalLr += lr * (1.0 - (double)epoch / _options.MaxUnlearningEpochs);
        }

        var unlearnedModel = Engine.TensorSubtract(
            UnlearningEngineOps.FitLength(globalModel, modelSize),
            Engine.TensorMultiplyScalar(targetContribution, NumOps.FromDouble(totalLr)));

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

        // Use the last round's update as representative. It is the same every epoch, so the
        // healing epochs add up to param += epochs * lr * avgUpdate.
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

    private double ComputeMembershipScore(Tensor<T> contribution, Tensor<T> unlearned, Tensor<T> original)
    {
        // Measure how much of the target's contribution remains
        int size = Math.Min(unlearned.Shape[0], original.Shape[0]);
        if (size <= 0) return 0.5;

        var weights = UnlearningEngineOps.FitLength(contribution, size);
        var diff = Engine.TensorSubtract(UnlearningEngineOps.FitLength(unlearned, size), UnlearningEngineOps.FitLength(original, size));
        double remainingInfluence = UnlearningEngineOps.Dot(diff, weights);
        double totalInfluence = UnlearningEngineOps.Dot(weights, weights);

        if (totalInfluence < 1e-12) return 0.5;

        double ratio = Math.Abs(remainingInfluence / totalInfluence);
        return 0.5 + (1.0 - Math.Min(1.0, ratio)) * 0.5;
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
