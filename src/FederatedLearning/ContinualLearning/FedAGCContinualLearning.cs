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
public class FedAGCContinualLearning<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedContinualLearningStrategy<T>
{
    private readonly double _correctionStrength;
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
        // We use finite differences with the task data to estimate squared gradients per parameter.
        int d = modelParameters.Length;
        int numSamples = taskData.Rows;
        var importance = new double[d];

        // For each data sample, approximate the gradient via finite differences
        // on a subset of parameters (full sweep too expensive for large models).
        double epsilon = 1e-5;
        int paramSubsample = Math.Min(d, 200); // Subsample parameters for efficiency.
        var rng = new Random(42);

        for (int s = 0; s < numSamples; s++)
        {
            // Extract sample features as a proxy "loss" input.
            // Use the dot product of parameters with the sample as a surrogate loss.
            double baseLoss = 0;
            int featureDim = Math.Min(d, taskData.Columns);
            for (int i = 0; i < featureDim; i++)
            {
                baseLoss += NumOps.ToDouble(modelParameters[i]) * NumOps.ToDouble(taskData[s, i]);
            }

            baseLoss = baseLoss * baseLoss; // Squared loss surrogate.

            for (int p = 0; p < paramSubsample; p++)
            {
                int idx = rng.Next(d);
                double origVal = NumOps.ToDouble(modelParameters[idx]);

                // Forward difference: (L(θ+ε) - L(θ)) / ε
                double perturbedLoss = 0;
                for (int i = 0; i < featureDim; i++)
                {
                    double paramVal = i == idx ? origVal + epsilon : NumOps.ToDouble(modelParameters[i]);
                    perturbedLoss += paramVal * NumOps.ToDouble(taskData[s, i]);
                }

                perturbedLoss = perturbedLoss * perturbedLoss;
                double grad = (perturbedLoss - baseLoss) / epsilon;

                // Fisher ≈ gradient squared (diagonal approximation).
                importance[idx] += grad * grad;
            }
        }

        // Normalize by number of samples.
        double invSamples = numSamples > 0 ? 1.0 / numSamples : 0;
        var result = new T[d];
        for (int i = 0; i < d; i++)
        {
            result[i] = NumOps.FromDouble(importance[i] * invSamples);
        }

        var resultVec = new Vector<T>(result);

        // Accumulate importance across tasks (EWC-style online accumulation).
        if (_accumulatedImportance == null)
        {
            _accumulatedImportance = resultVec;
        }
        else
        {
            var acc = new T[d];
            for (int i = 0; i < d; i++)
            {
                acc[i] = NumOps.Add(_accumulatedImportance[i], resultVec[i]);
            }

            _accumulatedImportance = new Vector<T>(acc);
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

        T penalty = NumOps.Zero;
        for (int i = 0; i < currentParameters.Length; i++)
        {
            var diff = NumOps.Subtract(currentParameters[i], referenceParameters[i]);
            penalty = NumOps.Add(penalty,
                NumOps.Multiply(importanceWeights[i], NumOps.Multiply(diff, diff)));
        }

        return NumOps.Multiply(penalty, NumOps.FromDouble(regularizationStrength * 0.5));
    }

    /// <inheritdoc/>
    public Vector<T> ProjectGradient(Vector<T> gradient, Vector<T> importanceWeights)
    {
        int d = gradient.Length;
        var projected = new T[d];

        // Compute conflict: dot product of gradient with importance direction.
        double dot = 0, importNorm2 = 0;
        for (int i = 0; i < d; i++)
        {
            double g = NumOps.ToDouble(gradient[i]);
            double imp = NumOps.ToDouble(importanceWeights[i]);
            dot += g * imp;
            importNorm2 += imp * imp;
        }

        if (dot < 0 && importNorm2 > 0)
        {
            // Conflict detected: project out the conflicting component.
            double scale = _correctionStrength * dot / importNorm2;
            for (int i = 0; i < d; i++)
            {
                double g = NumOps.ToDouble(gradient[i]);
                double imp = NumOps.ToDouble(importanceWeights[i]);
                projected[i] = NumOps.FromDouble(g - scale * imp);
            }
        }
        else
        {
            // No conflict: pass gradient through unchanged.
            for (int i = 0; i < d; i++)
            {
                projected[i] = gradient[i];
            }
        }

        return new Vector<T>(projected);
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
        var aggregated = new T[d];
        double totalWeight = clientWeights?.Values.Sum() ?? clientImportances.Count;
        if (totalWeight <= 0)
        {
            totalWeight = clientImportances.Count;
        }

        foreach (var (clientId, importance) in clientImportances)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            var wT = NumOps.FromDouble(w / totalWeight);
            for (int i = 0; i < d; i++)
            {
                aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(importance[i], wT));
            }
        }

        return new Vector<T>(aggregated);
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
        var strength = new T[d];

        // Find max importance for normalization.
        double maxImportance = 0;
        for (int i = 0; i < d; i++)
        {
            double imp = Math.Abs(NumOps.ToDouble(importanceWeights[i]));
            if (imp > maxImportance) maxImportance = imp;
        }

        if (maxImportance <= 0)
        {
            // No importance data — use uniform base strength.
            for (int i = 0; i < d; i++)
            {
                strength[i] = NumOps.FromDouble(_correctionStrength);
            }

            return new Vector<T>(strength);
        }

        for (int i = 0; i < d; i++)
        {
            double g = NumOps.ToDouble(gradient[i]);
            double imp = NumOps.ToDouble(importanceWeights[i]);
            double normalizedImp = Math.Abs(imp) / maxImportance;

            // Conflict indicator: negative means gradient opposes importance direction.
            double conflict = g * imp < 0 ? 1.0 : 0.0;

            // Adaptive strength = base * normalizedImportance * conflictIndicator
            strength[i] = NumOps.FromDouble(_correctionStrength * normalizedImp * conflict);
        }

        return new Vector<T>(strength);
    }

    /// <summary>
    /// Projects gradient using adaptive per-parameter correction strengths.
    /// </summary>
    /// <param name="gradient">New task gradient.</param>
    /// <param name="importanceWeights">Accumulated importance from old tasks.</param>
    /// <param name="adaptiveStrengths">Per-parameter correction strengths from ComputeAdaptiveCorrectionStrength.</param>
    /// <returns>Corrected gradient.</returns>
    public Vector<T> ProjectGradientAdaptive(
        Vector<T> gradient, Vector<T> importanceWeights, Vector<T> adaptiveStrengths)
    {
        int d = gradient.Length;
        var corrected = new T[d];

        for (int i = 0; i < d; i++)
        {
            double g = NumOps.ToDouble(gradient[i]);
            double imp = NumOps.ToDouble(importanceWeights[i]);
            double alpha = NumOps.ToDouble(adaptiveStrengths[i]);

            if (g * imp < 0 && alpha > 0)
            {
                // Reduce conflicting component proportionally to adaptive strength.
                double impNorm2 = imp * imp;
                if (impNorm2 > 1e-10)
                {
                    double projection = g * imp / impNorm2;
                    corrected[i] = NumOps.FromDouble(g - alpha * projection * imp);
                }
                else
                {
                    corrected[i] = NumOps.FromDouble(g);
                }
            }
            else
            {
                corrected[i] = NumOps.FromDouble(g);
            }
        }

        return new Vector<T>(corrected);
    }

    /// <summary>Gets the accumulated importance from all previous tasks.</summary>
    public Vector<T>? AccumulatedImportance => _accumulatedImportance;

    /// <summary>Gets the correction strength.</summary>
    public double CorrectionStrength => _correctionStrength;
}
