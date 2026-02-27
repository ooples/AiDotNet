namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Implements FLoRA — Federated Low-Rank Adaptation with stacked lossless aggregation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard federated LoRA averages the A and B matrices from
/// different clients, which introduces approximation error. FLoRA instead <em>stacks</em>
/// the local LoRA updates — each client's (B_k, A_k) pair is concatenated vertically/horizontally,
/// preserving all information. The server then uses an SVD-based compression to bring the stacked
/// result back to the target rank. This gives lossless aggregation without the information loss
/// of simple averaging.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// 1. Stack: B_stacked = [B_1; B_2; ...; B_K], A_stacked = [A_1; A_2; ...; A_K]
/// 2. Compute ΔW = B_stacked * A_stacked (full rank update)
/// 3. SVD: ΔW = U Σ V^T, truncate to rank r
/// 4. Return B_new = U[:, :r] * sqrt(Σ[:r]), A_new = sqrt(Σ[:r]) * V[:, :r]^T
/// </code>
///
/// <para>Reference: Wang, Y., et al. (2024). "FLoRA: Federated Fine-Tuning Large Language
/// Models with Heterogeneous Low-Rank Adaptations." arXiv:2405.14739.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FLoRA<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly int _rank;
    private readonly double _alpha;
    private readonly int _modelDim;
    private readonly int _numAdaptedLayers;
    private readonly int _layerInputDim;
    private readonly int _layerOutputDim;

    /// <inheritdoc/>
    public int AdapterParameterCount { get; }

    /// <inheritdoc/>
    public double CompressionRatio { get; }

    /// <summary>
    /// Creates a new FLoRA strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="rank">Target LoRA rank after aggregation. Default: 8.</param>
    /// <param name="alpha">LoRA scaling factor. Default: 16.</param>
    /// <param name="numAdaptedLayers">Number of adapted layers. Default: 4.</param>
    /// <param name="layerInputDim">Input dimension of adapted layers. Default: 768.</param>
    /// <param name="layerOutputDim">Output dimension of adapted layers. Default: 768.</param>
    public FLoRA(
        int modelDim,
        int rank = 8,
        double alpha = 16.0,
        int numAdaptedLayers = 4,
        int layerInputDim = 768,
        int layerOutputDim = 768)
    {
        if (modelDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(modelDim), "Model dimension must be positive.");
        }

        if (rank <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), "Rank must be positive.");
        }

        _rank = rank;
        _alpha = alpha;
        _modelDim = modelDim;
        _numAdaptedLayers = numAdaptedLayers;
        _layerInputDim = layerInputDim;
        _layerOutputDim = layerOutputDim;

        int paramsPerLayer = _layerOutputDim * _rank + _rank * _layerInputDim;
        AdapterParameterCount = _numAdaptedLayers * paramsPerLayer;
        CompressionRatio = _modelDim > 0 ? (double)AdapterParameterCount / _modelDim : 0;
    }

    /// <inheritdoc/>
    public Vector<T> ExtractAdapterParameters(Vector<T> fullModelParameters)
    {
        int totalParams = fullModelParameters.Length;
        int adapterCount = Math.Min(AdapterParameterCount, totalParams);
        int start = totalParams - adapterCount;

        var adapterParams = new T[adapterCount];
        for (int i = 0; i < adapterCount; i++)
        {
            adapterParams[i] = fullModelParameters[start + i];
        }

        return new Vector<T>(adapterParams);
    }

    /// <inheritdoc/>
    public Vector<T> MergeAdapterParameters(Vector<T> fullModelParameters, Vector<T> aggregatedAdapters)
    {
        int totalParams = fullModelParameters.Length;
        int adapterCount = aggregatedAdapters.Length;
        int start = totalParams - adapterCount;

        var merged = new T[totalParams];
        for (int i = 0; i < start; i++)
        {
            merged[i] = fullModelParameters[i];
        }

        double scale = _alpha / _rank;
        var scaleT = NumOps.FromDouble(scale);
        for (int i = 0; i < adapterCount; i++)
        {
            merged[start + i] = NumOps.Multiply(aggregatedAdapters[i], scaleT);
        }

        return new Vector<T>(merged);
    }

    /// <inheritdoc/>
    public Vector<T> AggregateAdapters(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights)
    {
        if (clientAdapters.Count == 0)
        {
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        }

        int adapterLen = clientAdapters.Values.First().Length;
        int paramsPerLayer = (_layerOutputDim * _rank) + (_rank * _layerInputDim);
        int bSize = _layerOutputDim * _rank; // B matrix: [out_dim x rank]
        int aSize = _rank * _layerInputDim;  // A matrix: [rank x in_dim]

        double totalWeight = 0;
        foreach (var (clientId, _) in clientAdapters)
        {
            totalWeight += clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
        }

        var aggregated = new T[adapterLen];

        // Process each adapted layer independently.
        for (int layer = 0; layer < _numAdaptedLayers; layer++)
        {
            int layerOffset = layer * paramsPerLayer;

            if (layerOffset + paramsPerLayer > adapterLen)
            {
                break;
            }

            // Step 1: Reconstruct weighted sum of ΔW = sum(w_k * B_k * A_k).
            // ΔW is [out_dim x in_dim].
            var deltaW = new double[_layerOutputDim * _layerInputDim];

            foreach (var (clientId, adapters) in clientAdapters)
            {
                double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
                double normalizedW = w / totalWeight;

                // Extract B_k [out_dim x rank] and A_k [rank x in_dim].
                // Compute B_k * A_k and add weighted result to deltaW.
                for (int i = 0; i < _layerOutputDim; i++)
                {
                    for (int j = 0; j < _layerInputDim; j++)
                    {
                        double sum = 0;
                        for (int r = 0; r < _rank; r++)
                        {
                            double bVal = NumOps.ToDouble(adapters[layerOffset + i * _rank + r]);
                            double aVal = NumOps.ToDouble(adapters[layerOffset + bSize + r * _layerInputDim + j]);
                            sum += bVal * aVal;
                        }

                        deltaW[i * _layerInputDim + j] += normalizedW * sum;
                    }
                }
            }

            // Step 2: SVD of ΔW to re-decompose into rank-r factors.
            // Use power iteration method for truncated SVD (efficient for low rank).
            var (newB, newA) = TruncatedSVD(deltaW, _layerOutputDim, _layerInputDim, _rank);

            // Step 3: Write B_new and A_new back to the aggregated adapter vector.
            for (int i = 0; i < _layerOutputDim; i++)
            {
                for (int r = 0; r < _rank; r++)
                {
                    aggregated[layerOffset + i * _rank + r] = NumOps.FromDouble(newB[i * _rank + r]);
                }
            }

            for (int r = 0; r < _rank; r++)
            {
                for (int j = 0; j < _layerInputDim; j++)
                {
                    aggregated[layerOffset + bSize + r * _layerInputDim + j] = NumOps.FromDouble(newA[r * _layerInputDim + j]);
                }
            }
        }

        return new Vector<T>(aggregated);
    }

    /// <summary>
    /// Truncated SVD via power iteration. Decomposes M [rows x cols] into
    /// B [rows x rank] and A [rank x cols] such that M ≈ B * A.
    /// Singular values are split evenly: B absorbs sqrt(sigma), A absorbs sqrt(sigma).
    /// </summary>
    private static (double[] B, double[] A) TruncatedSVD(double[] matrix, int rows, int cols, int rank)
    {
        const int maxIter = 50;
        const double tolerance = 1e-8;
        var rng = new Random(42);

        var B = new double[rows * rank];
        var A = new double[rank * cols];

        // Compute each singular vector via power iteration.
        // Work on residual matrix to get successive components.
        var residual = (double[])matrix.Clone();

        for (int r = 0; r < rank; r++)
        {
            // Initialize random vector v [cols].
            var v = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                v[j] = rng.NextDouble() - 0.5;
            }

            Normalize(v);

            double sigma = 0;

            for (int iter = 0; iter < maxIter; iter++)
            {
                // u = M * v (u is [rows])
                var u = new double[rows];
                for (int i = 0; i < rows; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < cols; j++)
                    {
                        sum += residual[i * cols + j] * v[j];
                    }

                    u[i] = sum;
                }

                double newSigma = Normalize(u);

                // v = M^T * u (v is [cols])
                var vNew = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < rows; i++)
                    {
                        sum += residual[i * cols + j] * u[i];
                    }

                    vNew[j] = sum;
                }

                Normalize(vNew);

                if (Math.Abs(newSigma - sigma) < tolerance * Math.Max(sigma, 1e-10))
                {
                    sigma = newSigma;
                    v = vNew;
                    break;
                }

                sigma = newSigma;
                v = vNew;
            }

            // Split sigma evenly: B gets sqrt(sigma) * u, A gets sqrt(sigma) * v^T.
            double sqrtSigma = Math.Sqrt(Math.Max(sigma, 0));

            // Recompute u for final sigma.
            var uFinal = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                double sum = 0;
                for (int j = 0; j < cols; j++)
                {
                    sum += residual[i * cols + j] * v[j];
                }

                uFinal[i] = sum;
            }

            double norm = Normalize(uFinal);
            if (norm < 1e-15)
            {
                // Remaining singular values are essentially zero.
                break;
            }

            for (int i = 0; i < rows; i++)
            {
                B[i * rank + r] = sqrtSigma * uFinal[i];
            }

            for (int j = 0; j < cols; j++)
            {
                A[r * cols + j] = sqrtSigma * v[j];
            }

            // Deflate: residual -= sigma * u * v^T.
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    residual[i * cols + j] -= sigma * uFinal[i] * v[j];
                }
            }
        }

        return (B, A);
    }

    private static double Normalize(double[] vec)
    {
        double norm = 0;
        for (int i = 0; i < vec.Length; i++)
        {
            norm += vec[i] * vec[i];
        }

        norm = Math.Sqrt(norm);
        if (norm > 1e-15)
        {
            for (int i = 0; i < vec.Length; i++)
            {
                vec[i] /= norm;
            }
        }

        return norm;
    }

    /// <summary>Gets the target LoRA rank.</summary>
    public int Rank => _rank;

    /// <summary>Gets the LoRA alpha scaling factor.</summary>
    public double Alpha => _alpha;
}
