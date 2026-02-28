namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Implements FedMeZO — Memory-efficient Zeroth-Order optimization for federated LLM fine-tuning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Training large language models normally requires storing gradients
/// for all parameters, which takes enormous memory (often 3-4x the model size). Zeroth-order (ZO)
/// optimization estimates gradients by evaluating the loss at two slightly perturbed points —
/// no backpropagation needed. This reduces memory to just the model size plus a random seed.
/// FedMeZO brings this to federated learning: clients only need to share the scalar loss
/// difference and the random seed, making it extremely communication-efficient.</para>
///
/// <para>ZO gradient estimate (SPSA — Simultaneous Perturbation Stochastic Approximation):</para>
/// <code>
/// z ~ N(0, I)   // shared via seed
/// grad_i ≈ (L(w + epsilon*z) - L(w - epsilon*z)) / (2 * epsilon) * z_i
/// </code>
///
/// <para>Communication per client: {loss_diff: double, seed: int} — just 12 bytes instead of
/// the full parameter vector (millions to billions of doubles).</para>
///
/// <para>Multi-query ZO: For better gradient estimates, multiple perturbation directions can be
/// sampled per step, averaging the estimates.</para>
///
/// <para>Reference: Malladi, S., et al. (2024). "Fine-Tuning Language Models with Just Forward
/// Passes." NeurIPS 2023. FedMeZO extension for federated settings (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FedMeZO<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly double _perturbationScale;
    private readonly double _learningRate;
    private readonly int _modelDim;
    private readonly int _numPerturbations;

    /// <inheritdoc/>
    public int AdapterParameterCount => _modelDim;

    /// <inheritdoc/>
    public double CompressionRatio => 1.0; // ZO operates on full model but communicates only scalars.

    /// <summary>
    /// Creates a new FedMeZO strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="perturbationScale">Scale of the random perturbation (epsilon). Default: 0.001 per paper.</param>
    /// <param name="learningRate">Server learning rate for applying aggregated ZO gradients. Default: 0.001.</param>
    /// <param name="numPerturbations">Number of random perturbation directions per ZO step (q).
    /// More directions give better gradient estimates at the cost of q extra forward passes. Default: 1.</param>
    public FedMeZO(int modelDim, double perturbationScale = 0.001, double learningRate = 0.001, int numPerturbations = 1)
    {
        if (modelDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(modelDim), "Model dimension must be positive.");
        }

        if (perturbationScale <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(perturbationScale), "Perturbation scale must be positive.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        if (numPerturbations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numPerturbations), "Number of perturbations must be at least 1.");
        }

        _modelDim = modelDim;
        _perturbationScale = perturbationScale;
        _learningRate = learningRate;
        _numPerturbations = numPerturbations;
    }

    /// <summary>
    /// Generates a random perturbation vector z ~ N(0, I) from a seed.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Both the client and server can regenerate the exact same random
    /// direction from the same seed. This means the client only needs to send the seed (4 bytes)
    /// instead of the full perturbation vector (millions of doubles). This is the key insight
    /// that makes ZO optimization communication-efficient.</para>
    /// </remarks>
    /// <param name="seed">Random seed (shared between client and server).</param>
    /// <returns>Random perturbation vector of dimension <see cref="_modelDim"/>.</returns>
    public Vector<T> GeneratePerturbation(int seed)
    {
        var rng = new Random(seed);
        var z = new T[_modelDim];

        for (int i = 0; i < _modelDim; i++)
        {
            // Box-Muller for standard normal.
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            z[i] = NumOps.FromDouble(normal);
        }

        return new Vector<T>(z);
    }

    /// <summary>
    /// Computes the perturbed parameter vectors w+ = w + epsilon*z and w- = w - epsilon*z.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> To estimate the gradient without backpropagation, we "probe"
    /// the loss function in a random direction. We add a tiny perturbation (+epsilon*z) to the
    /// weights and evaluate the loss, then subtract it (-epsilon*z) and evaluate again. The
    /// difference tells us how the loss changes along direction z, which is our gradient estimate.</para>
    /// </remarks>
    /// <param name="weights">Current model weights.</param>
    /// <param name="seed">Random seed to generate perturbation z.</param>
    /// <returns>Tuple of (w_plus, w_minus) perturbed weight vectors.</returns>
    public (Vector<T> wPlus, Vector<T> wMinus) ComputePerturbedWeights(Vector<T> weights, int seed)
    {
        var z = GeneratePerturbation(seed);
        var eps = NumOps.FromDouble(_perturbationScale);

        var wPlus = new T[_modelDim];
        var wMinus = new T[_modelDim];

        for (int i = 0; i < _modelDim; i++)
        {
            var perturbation = NumOps.Multiply(z[i], eps);
            wPlus[i] = NumOps.Add(weights[i], perturbation);
            wMinus[i] = NumOps.Subtract(weights[i], perturbation);
        }

        return (new Vector<T>(wPlus), new Vector<T>(wMinus));
    }

    /// <summary>
    /// Estimates the gradient from a single perturbation direction using the loss difference.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Given the loss at w+epsilon*z and at w-epsilon*z, the gradient
    /// estimate is: grad ≈ (L+ - L-) / (2*epsilon) * z. This is the SPSA (Simultaneous Perturbation
    /// Stochastic Approximation) estimator. The scalar (L+ - L-) / (2*epsilon) is the directional
    /// derivative along z, and multiplying by z gives the full gradient estimate.</para>
    /// </remarks>
    /// <param name="lossPlus">Loss evaluated at w + epsilon*z.</param>
    /// <param name="lossMinus">Loss evaluated at w - epsilon*z.</param>
    /// <param name="seed">The seed used to generate the perturbation z.</param>
    /// <returns>Estimated gradient vector.</returns>
    public Vector<T> EstimateGradient(double lossPlus, double lossMinus, int seed)
    {
        double lossDiff = lossPlus - lossMinus;
        double scale = lossDiff / (2.0 * _perturbationScale);

        var z = GeneratePerturbation(seed);
        var grad = new T[_modelDim];
        var scaleT = NumOps.FromDouble(scale);

        for (int i = 0; i < _modelDim; i++)
        {
            grad[i] = NumOps.Multiply(z[i], scaleT);
        }

        return new Vector<T>(grad);
    }

    /// <summary>
    /// Estimates the gradient using multiple perturbation directions for better accuracy.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A single random direction gives a noisy gradient estimate.
    /// By averaging estimates from multiple random directions, we get a much more accurate
    /// approximation — similar to how polling more people gives a better average opinion.
    /// Each additional direction costs one extra forward pass pair but no backpropagation.</para>
    /// </remarks>
    /// <param name="lossResults">Array of (lossPlus, lossMinus, seed) tuples from multiple perturbation evaluations.</param>
    /// <returns>Averaged gradient estimate over all perturbation directions.</returns>
    public Vector<T> EstimateGradientMultiQuery((double lossPlus, double lossMinus, int seed)[] lossResults)
    {
        if (lossResults.Length == 0)
        {
            throw new ArgumentException("At least one perturbation result is required.", nameof(lossResults));
        }

        var avgGrad = new double[_modelDim];
        int q = lossResults.Length;

        foreach (var (lossPlus, lossMinus, seed) in lossResults)
        {
            double lossDiff = lossPlus - lossMinus;
            double scale = lossDiff / (2.0 * _perturbationScale);

            var z = GeneratePerturbation(seed);
            for (int i = 0; i < _modelDim; i++)
            {
                avgGrad[i] += NumOps.ToDouble(z[i]) * scale / q;
            }
        }

        var result = new T[_modelDim];
        for (int i = 0; i < _modelDim; i++)
        {
            result[i] = NumOps.FromDouble(avgGrad[i]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Creates the minimal message a client sends to the server in the FedMeZO protocol.
    /// Instead of sending millions of parameters, the client sends only scalar loss differences
    /// and seeds.
    /// </summary>
    /// <param name="lossPlus">Loss at w + epsilon*z.</param>
    /// <param name="lossMinus">Loss at w - epsilon*z.</param>
    /// <param name="seed">The perturbation seed.</param>
    /// <returns>A compact message object.</returns>
    public static ZOClientMessage CreateClientMessage(double lossPlus, double lossMinus, int seed)
    {
        return new ZOClientMessage(lossPlus - lossMinus, seed);
    }

    /// <summary>
    /// Reconstructs the gradient estimate on the server from a client's compact message.
    /// </summary>
    /// <param name="message">The compact ZO client message.</param>
    /// <returns>Full gradient estimate vector (reconstructed from seed).</returns>
    public Vector<T> ReconstructGradientFromMessage(ZOClientMessage message)
    {
        double scale = message.LossDifference / (2.0 * _perturbationScale);
        var z = GeneratePerturbation(message.Seed);
        var grad = new T[_modelDim];
        var scaleT = NumOps.FromDouble(scale);

        for (int i = 0; i < _modelDim; i++)
        {
            grad[i] = NumOps.Multiply(z[i], scaleT);
        }

        return new Vector<T>(grad);
    }

    /// <summary>
    /// Aggregates ZO gradient estimates from multiple client messages on the server.
    /// </summary>
    /// <param name="clientMessages">Dictionary of client ID to their ZO messages.</param>
    /// <param name="clientWeights">Optional per-client weights.</param>
    /// <returns>Aggregated gradient estimate.</returns>
    public Vector<T> AggregateFromMessages(
        Dictionary<int, ZOClientMessage[]> clientMessages,
        Dictionary<int, double>? clientWeights = null)
    {
        Guard.NotNull(clientMessages);
        if (clientMessages.Count == 0)
        {
            throw new ArgumentException("No client messages provided.", nameof(clientMessages));
        }

        var aggregated = new double[_modelDim];
        double totalWeight = 0;

        foreach (var (clientId, messages) in clientMessages)
        {
            if (messages.Length == 0)
            {
                continue; // Skip clients with no perturbation results.
            }

            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            // Each client may have multiple perturbation queries.
            int q = messages.Length;
            foreach (var msg in messages)
            {
                double scale = w * msg.LossDifference / (2.0 * _perturbationScale * q);
                var z = GeneratePerturbation(msg.Seed);

                for (int i = 0; i < _modelDim; i++)
                {
                    aggregated[i] += NumOps.ToDouble(z[i]) * scale;
                }
            }
        }

        if (totalWeight > 0)
        {
            for (int i = 0; i < _modelDim; i++)
            {
                aggregated[i] /= totalWeight;
            }
        }

        var result = new T[_modelDim];
        for (int i = 0; i < _modelDim; i++)
        {
            result[i] = NumOps.FromDouble(aggregated[i]);
        }

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public Vector<T> ExtractAdapterParameters(Vector<T> fullModelParameters)
    {
        // ZO operates on the full parameter vector (no separate adapters).
        var result = new T[fullModelParameters.Length];
        for (int i = 0; i < fullModelParameters.Length; i++)
        {
            result[i] = fullModelParameters[i];
        }

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public Vector<T> MergeAdapterParameters(Vector<T> fullModelParameters, Vector<T> aggregatedAdapters)
    {
        // Apply the aggregated ZO gradient update: w_new = w - lr * grad_estimate.
        var merged = new T[fullModelParameters.Length];
        var lr = NumOps.FromDouble(_learningRate);

        for (int i = 0; i < fullModelParameters.Length; i++)
        {
            merged[i] = NumOps.Subtract(fullModelParameters[i], NumOps.Multiply(aggregatedAdapters[i], lr));
        }

        return new Vector<T>(merged);
    }

    /// <inheritdoc/>
    public Vector<T> AggregateAdapters(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientAdapters);
        if (clientAdapters.Count == 0)
        {
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        }

        // Weighted average of ZO gradient estimates from clients.
        int paramLen = clientAdapters.Values.First().Length;
        foreach (var (clientId, adapters) in clientAdapters)
        {
            if (adapters.Length != paramLen)
            {
                throw new ArgumentException(
                    $"Client {clientId} adapter length {adapters.Length} differs from expected {paramLen}.");
            }
        }
        var aggregated = new T[paramLen];
        double totalWeight = 0;

        foreach (var (clientId, gradEstimate) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            var wT = NumOps.FromDouble(w);
            for (int i = 0; i < paramLen; i++)
            {
                aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(gradEstimate[i], wT));
            }
        }

        var invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < paramLen; i++)
        {
            aggregated[i] = NumOps.Multiply(aggregated[i], invTotal);
        }

        return new Vector<T>(aggregated);
    }

    /// <summary>Gets the ZO perturbation scale (epsilon).</summary>
    public double PerturbationScale => _perturbationScale;

    /// <summary>Gets the server learning rate.</summary>
    public double LearningRate => _learningRate;

    /// <summary>Gets the number of perturbation directions per step.</summary>
    public int NumPerturbations => _numPerturbations;
}

/// <summary>
/// Compact message from a client in the FedMeZO protocol.
/// Contains only the loss difference and seed — the server reconstructs the gradient.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of sending millions of parameter values, each client
/// sends just two numbers: how much the loss changed when we wiggled the model in a random
/// direction (lossDifference) and which random direction we used (seed). The server can
/// recreate the random direction from the seed and compute the full gradient estimate.
/// This makes communication ~1,000,000x cheaper.</para>
/// </remarks>
public class ZOClientMessage
{
    /// <summary>Creates a new ZO client message.</summary>
    /// <param name="lossDifference">L(w + epsilon*z) - L(w - epsilon*z).</param>
    /// <param name="seed">The random seed used to generate perturbation z.</param>
    public ZOClientMessage(double lossDifference, int seed)
    {
        LossDifference = lossDifference;
        Seed = seed;
    }

    /// <summary>The loss difference: L(w + epsilon*z) - L(w - epsilon*z).</summary>
    public double LossDifference { get; }

    /// <summary>The random seed that generates the perturbation vector z.</summary>
    public int Seed { get; }
}
