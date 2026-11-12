using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.ContinualLearning;

/// <summary>
/// Implements FedAGC — Adaptive Gradient Correction for federated continual learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a federated model learns new tasks, it tends to forget
/// old ones (catastrophic forgetting). FedAGC adaptively corrects the gradient during training:
/// it identifies which gradient directions would harm old task performance and reduces their
/// magnitude, while allowing gradients that help the new task without hurting old tasks to
/// pass through freely. The correction strength adapts based on how much conflict exists
/// between old and new task gradients.</para>
///
/// <para>Correction:</para>
/// <code>
/// conflict = dot(grad_new, grad_old_importance)
/// if conflict &lt; 0:
///     grad_corrected = grad_new - (conflict / ||grad_old||²) * grad_old_importance
/// else:
///     grad_corrected = grad_new  (no correction needed)
/// </code>
///
/// <para>Reference: FedAGC: Adaptive Gradient Correction for Federated Continual Learning (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FedAGCContinualLearning<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedContinualLearningStrategy<T>
{
    private readonly double _correctionStrength;
    [AiDotNet.Attributes.Scratch]
    private Vector<T>? _accumulatedImportance;

    /// <summary>
    /// Creates a new FedAGC strategy.
    /// </summary>
    /// <param name="correctionStrength">How aggressively to correct conflicting gradients.
    /// 1.0 = full projection, 0.0 = no correction. Default: 0.8.</param>
    public FedAGCContinualLearning(double correctionStrength = 0.8)
    {
        if (correctionStrength < 0 || correctionStrength > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(correctionStrength), "Must be in [0, 1].");
        }

        _correctionStrength = correctionStrength;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeImportance(Vector<T> modelParameters, Matrix<T> taskData)
    {
        Guard.NotNull(modelParameters);
        Guard.NotNull(taskData);
        if (modelParameters.Length == 0)
        {
            throw new ArgumentException("Model parameters cannot be empty.", nameof(modelParameters));
        }

        if (taskData.Rows == 0 || taskData.Columns == 0)
        {
            throw new ArgumentException("Task data cannot be empty.", nameof(taskData));
        }

        // Approximate Fisher information: E[∇logp(y|x,θ)²] ≈ (1/N) Σ (∂L/∂θ)² for N samples.
        // The surrogate loss of a sample x is L(θ) = (θ·x)² over the first min(d, features) entries, and the
        // gradient of a randomly sampled parameter is its forward difference (L(θ + ε e_idx) - L(θ)) / ε,
        // which is exactly 2 (θ·x) x_idx + ε x_idx² (zero for parameters past the feature width).
        int d = modelParameters.Length;
        int numSamples = taskData.Rows;
        int columns = taskData.Columns;
        int featureDim = Math.Min(d, columns);

        double epsilon = 1e-5;
        int paramSubsample = Math.Min(d, 200); // Subsample parameters for efficiency.
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);

        // Draw the sampled parameter of every (sample, draw) pair in the original order.
        var sampledParams = new int[numSamples * paramSubsample];
        var flatFeatureIndex = new int[numSamples * paramSubsample];
        var inFeatures = new int[numSamples * paramSubsample];
        for (int s = 0; s < numSamples; s++)
        {
            for (int p = 0; p < paramSubsample; p++)
            {
                int idx = rng.Next(d);
                int k = s * paramSubsample + p;
                sampledParams[k] = idx;
                inFeatures[k] = idx < featureDim ? 1 : 0;
                flatFeatureIndex[k] = s * columns + (idx < featureDim ? idx : 0);
            }
        }

        // b_s = θ[:f] · x_s[:f] for every sample.
        var data = Tensor<T>.FromMatrix(taskData);
        var features = Engine.TensorSlice(data, new[] { 0, 0 }, new[] { numSamples, featureDim });
        var theta = Engine.TensorSlice(modelParameters.AsTensor(), new[] { 0 }, new[] { featureDim }).Reshape(new[] { featureDim, 1 });
        var baseDots = Engine.TensorMatMul(features, theta); // [numSamples, 1]

        // x_idx of every draw (0 outside the feature width), then the forward-difference gradient.
        int draws = numSamples * paramSubsample;
        var drawShape = new[] { numSamples, paramSubsample };
        var sampledFeatures = Engine.TensorMultiply(
            Engine.TensorGather(data.Reshape(new[] { numSamples * columns }), new Tensor<int>(flatFeatureIndex, new[] { draws }), 0),
            new Tensor<int>(inFeatures, new[] { draws }).Cast<T>()).Reshape(drawShape);
        var twoBase = Engine.TensorMultiplyScalar(baseDots, NumOps.FromDouble(2.0));
        var gradients = Engine.TensorAdd(
            Engine.TensorMultiply(sampledFeatures, twoBase),
            Engine.TensorMultiplyScalar(Engine.TensorSquare(sampledFeatures), NumOps.FromDouble(epsilon)));

        // Fisher ≈ gradient squared (diagonal approximation), accumulated per parameter and normalized by
        // the number of samples.
        var squared = Engine.TensorSquare(gradients).Reshape(new[] { draws });
        var importance = Engine.TensorIndexAdd(new Tensor<T>(new[] { d }), 0, new Tensor<int>(sampledParams, new[] { draws }), squared);
        var resultVec = Engine.TensorMultiplyScalar(importance, NumOps.FromDouble(1.0 / numSamples)).ToVector();

        // Accumulate importance across tasks (EWC-style online accumulation).
        if (_accumulatedImportance == null)
        {
            _accumulatedImportance = resultVec;
        }
        else
        {
            _accumulatedImportance = Engine.Add(_accumulatedImportance, resultVec);
        }

        return resultVec;
    }

    /// <inheritdoc/>
    public T ComputeRegularizationPenalty(
        Vector<T> currentParameters, Vector<T> referenceParameters,
        Vector<T> importanceWeights, double regularizationStrength)
    {
        Guard.NotNull(currentParameters);
        Guard.NotNull(referenceParameters);
        Guard.NotNull(importanceWeights);

        if (currentParameters.Length != referenceParameters.Length ||
            currentParameters.Length != importanceWeights.Length)
        {
            throw new ArgumentException(
                $"Parameter vectors must have equal length. Got current={currentParameters.Length}, " +
                $"reference={referenceParameters.Length}, importance={importanceWeights.Length}.");
        }

        if (currentParameters.Length == 0)
        {
            return NumOps.Zero;
        }

        var diff = Engine.Subtract(currentParameters, referenceParameters);
        T penalty = Engine.Sum(Engine.Multiply(importanceWeights, Engine.Multiply(diff, diff)));
        return NumOps.Multiply(penalty, NumOps.FromDouble(regularizationStrength * 0.5));
    }

    /// <inheritdoc/>
    public Vector<T> ProjectGradient(Vector<T> gradient, Vector<T> importanceWeights)
    {
        Guard.NotNull(gradient);
        Guard.NotNull(importanceWeights);

        if (gradient.Length == 0)
        {
            return gradient.Clone();
        }

        // Compute conflict: dot product of gradient with importance direction. The importance vector is first
        // divided by its largest magnitude; the projection (g . u / |u|^2) u is invariant to that scale and the
        // squared norm cannot overflow narrow T.
        T maxImportance = Engine.ReduceMax(Engine.Abs(importanceWeights).AsTensor(), new[] { 0 }, true)[0];
        if (NumOps.GreaterThan(maxImportance, NumOps.Zero))
        {
            var unit = Engine.Divide(importanceWeights, maxImportance);
            T dot = Engine.DotProduct(gradient, unit);
            T importNorm2 = Engine.DotProduct(unit, unit);
            if (NumOps.LessThan(dot, NumOps.Zero))
            {
                // Conflict detected: project out the conflicting component.
                T scale = NumOps.Multiply(NumOps.FromDouble(_correctionStrength), NumOps.Divide(dot, importNorm2));
                return Engine.Subtract(gradient, Engine.Multiply(unit, scale));
            }
        }

        // No conflict: pass gradient through unchanged.
        return gradient.Clone();
    }

    /// <inheritdoc/>
    public Vector<T> AggregateImportance(
        Dictionary<int, Vector<T>> clientImportances,
        Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientImportances);
        if (clientImportances.Count == 0)
        {
            throw new ArgumentException("Client importances cannot be empty.", nameof(clientImportances));
        }

        int d = clientImportances.Values.First().Length;
        var aggregated = new Vector<T>(d);
        double totalWeight = clientWeights?.Values.Sum() ?? clientImportances.Count;
        if (totalWeight <= 0)
        {
            totalWeight = clientImportances.Count;
        }

        foreach (var (clientId, importance) in clientImportances)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            var clientImportance = importance.Length == d ? importance : importance.GetSubVector(0, d);
            aggregated = Engine.Add(aggregated, Engine.Multiply(clientImportance, NumOps.FromDouble(w / totalWeight)));
        }

        return aggregated;
    }

    /// <summary>
    /// Computes adaptive per-parameter correction strength based on how important each parameter
    /// is for old tasks vs how much the new task gradient wants to change it.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Not all parameters need the same amount of correction. Parameters
    /// that are very important for old tasks (high Fisher information) and face large conflicting
    /// gradients should be corrected more aggressively. Parameters that aren't important for old
    /// tasks can change freely. This adaptive approach balances plasticity (learning new tasks)
    /// with stability (remembering old tasks) at the per-parameter level.</para>
    /// </remarks>
    /// <param name="gradient">New task gradient.</param>
    /// <param name="importanceWeights">Accumulated importance from old tasks.</param>
    /// <returns>Per-parameter correction strength in [0, correctionStrength].</returns>
    public Vector<T> ComputeAdaptiveCorrectionStrength(Vector<T> gradient, Vector<T> importanceWeights)
    {
        int d = gradient.Length;
        var importance = importanceWeights.Length == d ? importanceWeights : importanceWeights.GetSubVector(0, d);
        var baseStrength = NumOps.FromDouble(_correctionStrength);
        if (d == 0)
        {
            return new Vector<T>(0);
        }

        // Find max importance for normalization.
        var absImportance = Engine.Abs(importance);
        T maxImportance = Engine.ReduceMax(absImportance.AsTensor(), new[] { 0 }, true)[0];

        if (!NumOps.GreaterThan(maxImportance, NumOps.Zero))
        {
            // No importance data — use uniform base strength.
            return Engine.TensorFullLike(gradient.AsTensor(), baseStrength).ToVector();
        }

        // Conflict indicator: the gradient opposes the importance direction (sign test, so the product
        // cannot overflow narrow T).
        var conflict = Engine.TensorLessThan(
            Engine.TensorMultiply(Engine.TensorSign(gradient.AsTensor()), Engine.TensorSign(importance.AsTensor())), NumOps.Zero);

        // Adaptive strength = base * normalizedImportance * conflictIndicator
        var normalized = Engine.Multiply(absImportance, NumOps.Divide(baseStrength, maxImportance));
        return Engine.TensorMultiply(normalized.AsTensor(), conflict).ToVector();
    }

    /// <summary>
    /// Projects gradient using adaptive per-parameter correction strengths.
    /// </summary>
    /// <remarks>
    /// The per-parameter projection of g onto the importance direction is (g * imp / imp²) * imp = g, so a
    /// conflicting parameter (g * imp &lt; 0) with strength alpha &gt; 0 becomes g * (1 - alpha), whatever
    /// the magnitude of its importance.
    /// </remarks>
    /// <param name="gradient">New task gradient.</param>
    /// <param name="importanceWeights">Accumulated importance from old tasks.</param>
    /// <param name="adaptiveStrengths">Per-parameter correction strengths from ComputeAdaptiveCorrectionStrength.</param>
    /// <returns>Corrected gradient.</returns>
    public Vector<T> ProjectGradientAdaptive(
        Vector<T> gradient, Vector<T> importanceWeights, Vector<T> adaptiveStrengths)
    {
        int d = gradient.Length;
        var importance = importanceWeights.Length == d ? importanceWeights : importanceWeights.GetSubVector(0, d);
        var strengths = adaptiveStrengths.Length == d ? adaptiveStrengths : adaptiveStrengths.GetSubVector(0, d);
        if (d == 0)
        {
            return new Vector<T>(0);
        }

        var g = gradient.AsTensor();
        var conflict = Engine.TensorLessThan(
            Engine.TensorMultiply(Engine.TensorSign(g), Engine.TensorSign(importance.AsTensor())), NumOps.Zero);
        var active = Engine.TensorMultiply(conflict, Engine.TensorGreaterThan(strengths.AsTensor(), NumOps.Zero));

        // Reduce conflicting component proportionally to adaptive strength.
        var reduced = Engine.TensorSubtract(g, Engine.TensorMultiply(strengths.AsTensor(), g));
        return Engine.TensorWhere(active, reduced, g).ToVector();
    }

    /// <summary>Gets the accumulated importance from all previous tasks.</summary>
    public Vector<T>? AccumulatedImportance => _accumulatedImportance;

    /// <summary>Gets the correction strength.</summary>
    public double CorrectionStrength => _correctionStrength;
}
