namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Specifies the FedSAM variant to use.
/// </summary>
public enum FedSamVariant
{
    /// <summary>Base FedSAM — standard sharpness-aware minimization per client.</summary>
    Base,
    /// <summary>FedSMOO — simultaneous global and local flatness optimization.</summary>
    FedSMOO,
    /// <summary>FedSpeed — efficient SAM with gradient perturbation approximation.</summary>
    FedSpeed,
    /// <summary>FedLESAM — locally estimated SAM to reduce overhead.</summary>
    FedLESAM,
    /// <summary>FedSCAM — stochastic controlled averaging for SAM in FL.</summary>
    FedSCAM
}

/// <summary>
/// Implements FedSAM (Sharpness-Aware Minimization for Federated Learning) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Regular optimization finds a low point (minimum) in the loss
/// landscape, but this point might be "sharp" — a small change in parameters causes a large
/// change in loss. FedSAM instead seeks "flat" minima that are more robust, which is especially
/// important in FL where each client's data creates a different loss landscape.</para>
///
/// <para>Local training uses a two-step process per batch:</para>
/// <code>
/// 1. Compute gradient g at current weights w
/// 2. Perturb weights: w_perturbed = w + rho * g / ||g||
/// 3. Compute gradient g' at w_perturbed
/// 4. Update: w = w - lr * g' (use perturbed gradient for actual update)
/// </code>
///
/// <para>Variants:</para>
/// <list type="bullet">
/// <item><b>FedSMOO</b>: Optimizes for both global and local flatness using a dual perturbation
/// with separate global perturbation radius.</item>
/// <item><b>FedSpeed</b>: Approximates the perturbation gradient using a running average,
/// avoiding the costly second forward-backward pass.</item>
/// <item><b>FedLESAM</b>: Uses locally estimated perturbation direction from gradient
/// history for efficient SAM without extra computation.</item>
/// <item><b>FedSCAM</b>: Applies stochastic controlled averaging with variance reduction
/// on the SAM perturbation direction.</item>
/// </list>
///
/// <para>Reference: Caldarola, D., et al. (2022). "Improving Generalization in Federated Learning
/// by Seeking Flat Minima."</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedSamAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _perturbationRadius;
    private readonly FedSamVariant _variant;
    private readonly double _globalPerturbationRadius;
    private readonly double _approximationCoeff;
    private readonly double _controlCoeff;

    // State for FedSpeed/FedLESAM gradient history.
    private Dictionary<string, T[]>? _previousGradient;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedSamAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="perturbationRadius">Radius for the SAM perturbation step (rho). Default: 0.05 per paper.</param>
    /// <param name="variant">The FedSAM variant to use. Default: Base.</param>
    /// <param name="globalPerturbationRadius">Global perturbation radius for FedSMOO. Default: 0.01.</param>
    /// <param name="approximationCoeff">Approximation coefficient for FedSpeed/FedLESAM. Default: 0.5.</param>
    /// <param name="controlCoeff">Stochastic control coefficient for FedSCAM. Default: 0.1.</param>
    public FedSamAggregationStrategy(
        double perturbationRadius = 0.05,
        FedSamVariant variant = FedSamVariant.Base,
        double globalPerturbationRadius = 0.01,
        double approximationCoeff = 0.5,
        double controlCoeff = 0.1)
    {
        if (perturbationRadius <= 0)
        {
            throw new ArgumentException("Perturbation radius must be positive.", nameof(perturbationRadius));
        }

        if (globalPerturbationRadius < 0)
        {
            throw new ArgumentException("Global perturbation radius must be non-negative.", nameof(globalPerturbationRadius));
        }

        _perturbationRadius = perturbationRadius;
        _variant = variant;
        _globalPerturbationRadius = globalPerturbationRadius;
        _approximationCoeff = approximationCoeff;
        _controlCoeff = controlCoeff;
    }

    /// <summary>
    /// Aggregates client models using standard weighted averaging.
    /// </summary>
    /// <remarks>
    /// <para><b>By design</b>, FedSAM uses standard FedAvg for server-side aggregation. The
    /// sharpness-aware behavior is applied during <em>local training</em> via <see cref="ComputePerturbation"/>,
    /// <see cref="ApplyPerturbation"/>, and <see cref="ComputeSAMGradient"/>. The server aggregation
    /// step is unchanged from FedAvg, as described in the FedSAM paper.</para>
    /// </remarks>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Computes the SAM perturbation direction from the current gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes the gradient computed at the current weights
    /// and produces the perturbation to add to the weights. The model is then temporarily moved
    /// to w + perturbation before computing the gradient used for the actual update. This
    /// "look-ahead" finds a direction that works well even when parameters are slightly different,
    /// leading to better generalization.</para>
    /// </remarks>
    /// <param name="gradient">Current gradient as a parameter dictionary.</param>
    /// <returns>Perturbation dictionary: epsilon = rho * g / ||g|| for each layer.</returns>
    public Dictionary<string, T[]> ComputePerturbation(Dictionary<string, T[]> gradient)
    {
        Guard.NotNull(gradient);
        return _variant switch
        {
            FedSamVariant.Base => ComputeBasePerturbation(gradient),
            FedSamVariant.FedSMOO => ComputeSMOOPerturbation(gradient),
            FedSamVariant.FedSpeed => ComputeSpeedPerturbation(gradient),
            FedSamVariant.FedLESAM => ComputeLESAMPerturbation(gradient),
            FedSamVariant.FedSCAM => ComputeSCAMPerturbation(gradient),
            _ => throw new ArgumentOutOfRangeException(nameof(_variant), $"Unknown FedSAM variant: {_variant}.")
        };
    }

    /// <summary>
    /// Applies a perturbation to model parameters: w_perturbed = w + epsilon.
    /// </summary>
    /// <param name="parameters">Current model parameters.</param>
    /// <param name="perturbation">Perturbation to apply.</param>
    /// <returns>Perturbed parameters.</returns>
    public Dictionary<string, T[]> ApplyPerturbation(
        Dictionary<string, T[]> parameters,
        Dictionary<string, T[]> perturbation)
    {
        Guard.NotNull(parameters);
        Guard.NotNull(perturbation);
        var perturbed = new Dictionary<string, T[]>(parameters.Count);
        foreach (var (layerName, weights) in parameters)
        {
            if (!perturbation.TryGetValue(layerName, out var eps))
            {
                perturbed[layerName] = (T[])weights.Clone();
                continue;
            }

            if (eps.Length != weights.Length)
            {
                throw new ArgumentException(
                    $"Perturbation layer '{layerName}' length {eps.Length} differs from parameter length {weights.Length}.");
            }

            var result = new T[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                result[i] = NumOps.Add(weights[i], eps[i]);
            }

            perturbed[layerName] = result;
        }

        return perturbed;
    }

    /// <summary>
    /// Updates the gradient history (used by FedSpeed/FedLESAM for approximation).
    /// Call this after each training step.
    /// </summary>
    /// <param name="gradient">The gradient from the current step.</param>
    public void UpdateGradientHistory(Dictionary<string, T[]> gradient)
    {
        _previousGradient = new Dictionary<string, T[]>(gradient.Count);
        foreach (var (layerName, grad) in gradient)
        {
            _previousGradient[layerName] = (T[])grad.Clone();
        }
    }

    /// <summary>
    /// Computes the complete SAM-modified gradient for a training step.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main method to use during local training.
    /// It takes the gradient at the current weights, computes the perturbation, and returns
    /// what the gradient would be at the perturbed point. For efficiency variants (FedSpeed,
    /// FedLESAM), it approximates this without requiring a second forward-backward pass.</para>
    /// </remarks>
    /// <param name="currentGradient">Gradient at the current weights.</param>
    /// <param name="perturbedGradient">Gradient recomputed at w + perturbation (null for
    /// FedSpeed/FedLESAM which approximate this).</param>
    /// <returns>The gradient to use for the actual parameter update.</returns>
    public Dictionary<string, T[]> ComputeSAMGradient(
        Dictionary<string, T[]> currentGradient,
        Dictionary<string, T[]>? perturbedGradient)
    {
        if (_variant == FedSamVariant.FedSpeed || _variant == FedSamVariant.FedLESAM)
        {
            // Approximate: g_sam ≈ g + alpha * (g - g_prev)
            return ApproximatePerturbedGradient(currentGradient);
        }

        // For Base/FedSMOO/FedSCAM: use the actual gradient at the perturbed point.
        if (perturbedGradient == null)
        {
            throw new ArgumentNullException(nameof(perturbedGradient),
                $"Perturbed gradient is required for variant {_variant}. " +
                "Compute gradient at w + perturbation and pass it here.");
        }

        if (_variant == FedSamVariant.FedSCAM)
        {
            return ComputeSCAMGradient(currentGradient, perturbedGradient);
        }

        // Base and FedSMOO: use perturbed gradient directly.
        UpdateGradientHistory(currentGradient);
        return perturbedGradient;
    }

    private Dictionary<string, T[]> ComputeBasePerturbation(Dictionary<string, T[]> gradient)
    {
        double globalNorm = ComputeGlobalNorm(gradient);
        double scale = globalNorm > 1e-12 ? _perturbationRadius / globalNorm : 0.0;

        var perturbation = new Dictionary<string, T[]>(gradient.Count);
        var scaleT = NumOps.FromDouble(scale);

        foreach (var (layerName, grad) in gradient)
        {
            var eps = new T[grad.Length];
            for (int i = 0; i < grad.Length; i++)
            {
                eps[i] = NumOps.Multiply(grad[i], scaleT);
            }

            perturbation[layerName] = eps;
        }

        return perturbation;
    }

    /// <remarks>
    /// Simplified FedSMOO: uses a combined perturbation radius (local + global) on the local
    /// gradient as a surrogate. Full FedSMOO requires the global gradient, which is unavailable
    /// at the client during local training. This approximation is valid when local and global
    /// gradients are well-aligned (early training or IID data).
    /// </remarks>
    private Dictionary<string, T[]> ComputeSMOOPerturbation(Dictionary<string, T[]> gradient)
    {
        // Simplified FedSMOO: combined radius on local gradient as surrogate for dual perturbation.
        double globalNorm = ComputeGlobalNorm(gradient);
        double combinedRadius = _perturbationRadius + _globalPerturbationRadius;
        double scale = globalNorm > 1e-12 ? combinedRadius / globalNorm : 0.0;

        var perturbation = new Dictionary<string, T[]>(gradient.Count);
        var scaleT = NumOps.FromDouble(scale);

        foreach (var (layerName, grad) in gradient)
        {
            var eps = new T[grad.Length];
            for (int i = 0; i < grad.Length; i++)
            {
                eps[i] = NumOps.Multiply(grad[i], scaleT);
            }

            perturbation[layerName] = eps;
        }

        return perturbation;
    }

    private Dictionary<string, T[]> ComputeSpeedPerturbation(Dictionary<string, T[]> gradient)
    {
        // FedSpeed: use gradient + approximation from history to avoid second pass.
        // Perturbation = rho * (g + alpha * (g - g_prev)) / ||g + alpha * (g - g_prev)||
        var adjustedGrad = AdjustGradientWithHistory(gradient);
        double norm = ComputeGlobalNorm(adjustedGrad);
        double scale = norm > 1e-12 ? _perturbationRadius / norm : 0.0;

        var perturbation = new Dictionary<string, T[]>(adjustedGrad.Count);
        var scaleT = NumOps.FromDouble(scale);

        foreach (var (layerName, grad) in adjustedGrad)
        {
            var eps = new T[grad.Length];
            for (int i = 0; i < grad.Length; i++)
            {
                eps[i] = NumOps.Multiply(grad[i], scaleT);
            }

            perturbation[layerName] = eps;
        }

        return perturbation;
    }

    /// <remarks>
    /// FedLESAM uses the same gradient-history approximation as FedSpeed. The distinction is
    /// in how the approximation coefficient is interpreted: FedLESAM treats it as a local
    /// estimation weight, while FedSpeed treats it as a momentum term. With a shared
    /// <see cref="_approximationCoeff"/>, both produce identical perturbations. To differentiate,
    /// configure a different <c>approximationCoeff</c> value for each variant.
    /// </remarks>
    private Dictionary<string, T[]> ComputeLESAMPerturbation(Dictionary<string, T[]> gradient)
    {
        return ComputeSpeedPerturbation(gradient);
    }

    private Dictionary<string, T[]> ComputeSCAMPerturbation(Dictionary<string, T[]> gradient)
    {
        // FedSCAM: standard perturbation (uses control variate during gradient computation).
        return ComputeBasePerturbation(gradient);
    }

    private Dictionary<string, T[]> ComputeSCAMGradient(
        Dictionary<string, T[]> currentGradient,
        Dictionary<string, T[]> perturbedGradient)
    {
        // FedSCAM: g_sam = g_perturbed + control * (g_current - g_prev)
        var result = new Dictionary<string, T[]>(perturbedGradient.Count);
        var coeff = NumOps.FromDouble(_controlCoeff);

        foreach (var (layerName, pGrad) in perturbedGradient)
        {
            var cGrad = currentGradient[layerName];
            var output = new T[pGrad.Length];

            if (_previousGradient != null && _previousGradient.TryGetValue(layerName, out var prevGrad))
            {
                for (int i = 0; i < pGrad.Length; i++)
                {
                    var correction = NumOps.Multiply(NumOps.Subtract(cGrad[i], prevGrad[i]), coeff);
                    output[i] = NumOps.Add(pGrad[i], correction);
                }
            }
            else
            {
                Array.Copy(pGrad, output, pGrad.Length);
            }

            result[layerName] = output;
        }

        UpdateGradientHistory(currentGradient);
        return result;
    }

    private Dictionary<string, T[]> AdjustGradientWithHistory(Dictionary<string, T[]> gradient)
    {
        if (_previousGradient == null)
        {
            return gradient;
        }

        var adjusted = new Dictionary<string, T[]>(gradient.Count);
        var alpha = NumOps.FromDouble(_approximationCoeff);

        foreach (var (layerName, grad) in gradient)
        {
            var result = new T[grad.Length];

            if (_previousGradient.TryGetValue(layerName, out var prevGrad) && prevGrad.Length == grad.Length)
            {
                for (int i = 0; i < grad.Length; i++)
                {
                    // adjusted = g + alpha * (g - g_prev)
                    var diff = NumOps.Subtract(grad[i], prevGrad[i]);
                    result[i] = NumOps.Add(grad[i], NumOps.Multiply(diff, alpha));
                }
            }
            else
            {
                Array.Copy(grad, result, grad.Length);
            }

            adjusted[layerName] = result;
        }

        return adjusted;
    }

    private Dictionary<string, T[]> ApproximatePerturbedGradient(Dictionary<string, T[]> currentGradient)
    {
        // g_sam ≈ g + alpha * (g - g_prev)
        var result = AdjustGradientWithHistory(currentGradient);
        UpdateGradientHistory(currentGradient);
        return result;
    }

    private static double ComputeGlobalNorm(Dictionary<string, T[]> gradient)
    {
        double normSq = 0;
        foreach (var (_, grad) in gradient)
        {
            for (int i = 0; i < grad.Length; i++)
            {
                double v = NumOps.ToDouble(grad[i]);
                normSq += v * v;
            }
        }

        return Math.Sqrt(normSq);
    }

    /// <summary>Gets the SAM perturbation radius (rho).</summary>
    public double PerturbationRadius => _perturbationRadius;

    /// <summary>Gets the FedSAM variant being used.</summary>
    public FedSamVariant Variant => _variant;

    /// <summary>Gets the global perturbation radius (FedSMOO).</summary>
    public double GlobalPerturbationRadius => _globalPerturbationRadius;

    /// <summary>Gets the approximation coefficient (FedSpeed/FedLESAM).</summary>
    public double ApproximationCoefficient => _approximationCoeff;

    /// <summary>Gets the stochastic control coefficient (FedSCAM).</summary>
    public double ControlCoefficient => _controlCoeff;

    /// <inheritdoc/>
    public override string GetStrategyName() => _variant == FedSamVariant.Base
        ? $"FedSAM(\u03c1={_perturbationRadius})"
        : $"FedSAM-{_variant}(\u03c1={_perturbationRadius})";
}
