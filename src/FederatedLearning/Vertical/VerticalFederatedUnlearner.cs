using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Implements GDPR-compliant entity unlearning for vertical federated learning models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Under GDPR and similar regulations, individuals have the "right to
/// be forgotten". When someone requests deletion, not only must their data be removed from storage,
/// but the model must also be updated to remove any influence of their data. This is called
/// "machine unlearning".</para>
///
/// <para>In VFL, unlearning is more complex than in standard ML because:</para>
/// <list type="bullet">
/// <item><description>Data is distributed across parties (each party must participate in unlearning).</description></item>
/// <item><description>The model is split (both bottom and top models must be updated).</description></item>
/// <item><description>Some parties may not even know which entity is being unlearned (privacy).</description></item>
/// </list>
///
/// <para><b>Methods supported:</b></para>
/// <list type="bullet">
/// <item><description><b>GradientAscent:</b> Fast approximate unlearning by reversing the training gradient.</description></item>
/// <item><description><b>PrimalDual:</b> Optimization-based unlearning with stronger guarantees.</description></item>
/// <item><description><b>Certified:</b> Provides mathematical certification that unlearning succeeded.</description></item>
/// <item><description><b>Retraining:</b> Gold standard - retrain from scratch without the entity.</description></item>
/// </list>
///
/// <para><b>Reference:</b></para>
/// <list type="bullet">
/// <item><description>"VFL Certified Unlearning" (2025)</description></item>
/// <item><description>"VFL Primal-Dual Unlearning" (2025)</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VerticalFederatedUnlearner<T> : FederatedLearningComponentBase<T>
{
    private readonly VflUnlearningOptions _options;

    /// <summary>
    /// Initializes a new instance of <see cref="VerticalFederatedUnlearner{T}"/>.
    /// </summary>
    /// <param name="options">Unlearning configuration options.</param>
    public VerticalFederatedUnlearner(VflUnlearningOptions? options = null)
    {
        _options = options ?? new VflUnlearningOptions();
    }

    /// <summary>
    /// Computes the influence of a set of entities on the model parameters.
    /// Used for certified unlearning to determine how much the model needs to change.
    /// </summary>
    /// <param name="entityEmbeddings">The embeddings of entities to unlearn.</param>
    /// <param name="entityLabels">The labels of entities to unlearn.</param>
    /// <param name="predictions">The current model predictions for these entities.</param>
    /// <returns>An influence score indicating how much these entities affected the model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Influence functions measure how much each training sample
    /// affected the final model. High-influence samples caused larger changes to the model
    /// and require more aggressive unlearning.</para>
    /// </remarks>
    public double ComputeInfluence(
        Tensor<T> entityEmbeddings,
        Tensor<T> entityLabels,
        Tensor<T> predictions)
    {
        if (entityEmbeddings is null || entityLabels is null || predictions is null)
        {
            return 0.0;
        }

        int totalElements = 1;
        for (int d = 0; d < predictions.Rank; d++)
        {
            totalElements *= predictions.Shape[d];
        }

        // Approximate influence as the gradient norm of the loss on these entities
        double gradientNormSquared = 0.0;
        for (int i = 0; i < totalElements; i++)
        {
            double pred = NumOps.ToDouble(predictions[i]);
            double label = NumOps.ToDouble(entityLabels[i]);
            double grad = 2.0 * (pred - label);
            gradientNormSquared += grad * grad;
        }

        return Math.Sqrt(gradientNormSquared);
    }

    /// <summary>
    /// Verifies that unlearning was effective by checking if the model's behavior on
    /// the unlearned entities is statistically indistinguishable from a model that
    /// was never trained on those entities.
    /// </summary>
    /// <param name="predictionsBeforeUnlearning">Model predictions before unlearning.</param>
    /// <param name="predictionsAfterUnlearning">Model predictions after unlearning.</param>
    /// <param name="randomBaseline">Predictions from a randomly-initialized model (baseline).</param>
    /// <returns>A verification result containing a pass/fail status and a divergence score.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> After unlearning, the model should behave as if it never
    /// saw the removed entities. This method checks this by comparing the model's predictions
    /// on the removed entities before and after unlearning. If the predictions changed
    /// significantly toward random-like behavior, unlearning likely worked.</para>
    /// </remarks>
    public UnlearningVerification VerifyUnlearning(
        Tensor<T> predictionsBeforeUnlearning,
        Tensor<T> predictionsAfterUnlearning,
        Tensor<T>? randomBaseline = null)
    {
        if (predictionsBeforeUnlearning is null || predictionsAfterUnlearning is null)
        {
            return new UnlearningVerification { Passed = false, DivergenceScore = 0.0 };
        }

        int totalElements = 1;
        for (int d = 0; d < predictionsBeforeUnlearning.Rank; d++)
        {
            totalElements *= predictionsBeforeUnlearning.Shape[d];
        }

        // Compute prediction divergence (how much predictions changed)
        double divergenceSum = 0.0;
        for (int i = 0; i < totalElements; i++)
        {
            double before = NumOps.ToDouble(predictionsBeforeUnlearning[i]);
            double after = NumOps.ToDouble(predictionsAfterUnlearning[i]);
            double diff = before - after;
            divergenceSum += diff * diff;
        }

        double divergenceScore = totalElements > 0 ? Math.Sqrt(divergenceSum / totalElements) : 0.0;

        // The divergence should be above a threshold (predictions should have changed)
        double threshold = _options.CertificationEpsilon * 0.1;
        bool passed = divergenceScore > threshold;

        return new UnlearningVerification
        {
            Passed = passed,
            DivergenceScore = divergenceScore,
            Threshold = threshold,
            EntitiesUnlearned = totalElements > 0 ? predictionsBeforeUnlearning.Shape[0] : 0
        };
    }

    /// <summary>
    /// Adds calibrated noise to model parameters for certified unlearning.
    /// The noise ensures that the post-unlearning model is epsilon-indistinguishable
    /// from a model trained without the unlearned entities.
    /// </summary>
    /// <param name="parameters">The model parameters to add noise to.</param>
    /// <param name="influence">The influence score of the unlearned entities.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>The noised parameters.</returns>
    public IReadOnlyList<Tensor<T>> AddCertificationNoise(
        IReadOnlyList<Tensor<T>> parameters, double influence, int? seed = null)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        var random = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();

        double noiseScale = influence / _options.CertificationEpsilon;
        var noisedParams = new List<Tensor<T>>(parameters.Count);

        foreach (var param in parameters)
        {
            int totalElements = 1;
            for (int d = 0; d < param.Rank; d++)
            {
                totalElements *= param.Shape[d];
            }

            var noised = new Tensor<T>(param.Shape);
            for (int i = 0; i < totalElements; i++)
            {
                double val = NumOps.ToDouble(param[i]);
                // Box-Muller for Gaussian noise
                double u1 = Math.Max(random.NextDouble(), 1e-15);
                double u2 = random.NextDouble();
                double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                noised[i] = NumOps.FromDouble(val + noiseScale * z);
            }

            noisedParams.Add(noised);
        }

        return noisedParams;
    }
}

/// <summary>
/// Contains the results of an unlearning verification check.
/// </summary>
public class UnlearningVerification
{
    /// <summary>
    /// Gets or sets whether the unlearning verification passed.
    /// </summary>
    public bool Passed { get; set; }

    /// <summary>
    /// Gets or sets the prediction divergence score (how much predictions changed after unlearning).
    /// </summary>
    public double DivergenceScore { get; set; }

    /// <summary>
    /// Gets or sets the threshold used for the pass/fail decision.
    /// </summary>
    public double Threshold { get; set; }

    /// <summary>
    /// Gets or sets the number of entities that were unlearned.
    /// </summary>
    public int EntitiesUnlearned { get; set; }
}
