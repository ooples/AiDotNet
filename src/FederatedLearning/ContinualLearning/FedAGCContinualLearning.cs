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
        // Approximate Fisher information: squared gradient magnitude per parameter.
        int d = modelParameters.Length;
        var importance = new T[d];

        // Use parameter magnitude as proxy for importance (simplified).
        for (int i = 0; i < d; i++)
        {
            double v = NumOps.ToDouble(modelParameters[i]);
            importance[i] = NumOps.FromDouble(v * v);
        }

        var result = new Vector<T>(importance);

        // Accumulate importance across tasks.
        if (_accumulatedImportance == null)
        {
            _accumulatedImportance = result;
        }
        else
        {
            var acc = new T[d];
            for (int i = 0; i < d; i++)
            {
                acc[i] = NumOps.Add(_accumulatedImportance[i], result[i]);
            }

            _accumulatedImportance = new Vector<T>(acc);
        }

        return result;
    }

    /// <inheritdoc/>
    public T ComputeRegularizationPenalty(
        Vector<T> currentParameters, Vector<T> referenceParameters,
        Vector<T> importanceWeights, double regularizationStrength)
    {
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
        int d = clientImportances.Values.First().Length;
        var aggregated = new T[d];
        double totalWeight = clientWeights?.Values.Sum() ?? clientImportances.Count;

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

    /// <summary>Gets the correction strength.</summary>
    public double CorrectionStrength => _correctionStrength;
}
