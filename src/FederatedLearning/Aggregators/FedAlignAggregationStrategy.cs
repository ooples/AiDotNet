namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedAlign (Feature Alignment) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different clients with different data can learn different "languages"
/// for representing the same concepts. FedAlign adds a regularizer during local training that
/// forces each client's feature space to align with shared anchor representations, so all clients
/// speak the same "language" when their models are combined.</para>
///
/// <para>Local training objective:</para>
/// <code>L = L_task + alpha * D(f_local(anchors), f_global(anchors))</code>
/// <para>where anchors are shared reference inputs and D measures representation distance
/// using either L2 distance or CKA (Centered Kernel Alignment).</para>
///
/// <para>Protocol:</para>
/// <list type="number">
/// <item>Server generates or selects anchor inputs (small shared dataset)</item>
/// <item>Server sends global model + anchors to clients</item>
/// <item>Each client computes f_global(anchors) and f_local(anchors) during training</item>
/// <item>Alignment loss penalizes distance between the two representations</item>
/// </list>
///
/// <para>Reference: Mendieta, M., et al. (2022). "Local Learning Matters: Rethinking Data Heterogeneity
/// in Federated Learning." CVPR 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedAlignAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _alignmentWeight;
    private readonly AlignmentDistanceMetric _distanceMetric;
    private Matrix<T>? _anchorInputs;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedAlignAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="alignmentWeight">Weight of the alignment loss (alpha). Default: 1.0 per paper.</param>
    /// <param name="distanceMetric">Distance metric for feature alignment. Default: L2.</param>
    public FedAlignAggregationStrategy(
        double alignmentWeight = 1.0,
        AlignmentDistanceMetric distanceMetric = AlignmentDistanceMetric.L2)
    {
        if (alignmentWeight < 0)
        {
            throw new ArgumentException("Alignment weight must be non-negative.", nameof(alignmentWeight));
        }

        _alignmentWeight = alignmentWeight;
        _distanceMetric = distanceMetric;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Registers anchor inputs that all clients use for alignment.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Anchors are a small set of reference inputs that every client
    /// and the global model process. By comparing how each client represents these anchors
    /// versus how the global model represents them, we can measure and correct feature misalignment.</para>
    /// </remarks>
    /// <param name="anchors">Anchor input matrix [numAnchors x inputDim].</param>
    public void SetAnchors(Matrix<T> anchors)
    {
        if (anchors.Rows == 0 || anchors.Columns == 0)
        {
            throw new ArgumentException("Anchor matrix must not be empty.", nameof(anchors));
        }

        _anchorInputs = anchors.Clone();
    }

    /// <summary>
    /// Gets the registered anchor inputs, or null if none have been set.
    /// </summary>
    public Matrix<T>? Anchors => _anchorInputs;

    /// <summary>
    /// Computes the feature alignment loss between local and global model representations on anchors.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Both the local model and the global model process the same
    /// anchor inputs to produce feature representations. This method measures how different
    /// those representations are. A large distance means the local model has "drifted" from
    /// the global model's representation space, and the loss encourages it to stay aligned.</para>
    /// </remarks>
    /// <param name="localFeatures">Local model features on anchors [numAnchors x featureDim].</param>
    /// <param name="globalFeatures">Global model features on anchors [numAnchors x featureDim].</param>
    /// <returns>The alignment loss: alpha * D(f_local, f_global).</returns>
    public T ComputeAlignmentLoss(Matrix<T> localFeatures, Matrix<T> globalFeatures)
    {
        if (localFeatures.Rows != globalFeatures.Rows || localFeatures.Columns != globalFeatures.Columns)
        {
            throw new ArgumentException(
                $"Feature matrices must have the same shape. Local: [{localFeatures.Rows}x{localFeatures.Columns}], " +
                $"Global: [{globalFeatures.Rows}x{globalFeatures.Columns}].");
        }

        double distance = _distanceMetric switch
        {
            AlignmentDistanceMetric.L2 => ComputeL2Distance(localFeatures, globalFeatures),
            AlignmentDistanceMetric.CKA => ComputeCKADistance(localFeatures, globalFeatures),
            AlignmentDistanceMetric.MMD => ComputeMMDDistance(localFeatures, globalFeatures),
            _ => ComputeL2Distance(localFeatures, globalFeatures)
        };

        return NumOps.FromDouble(_alignmentWeight * distance);
    }

    /// <summary>
    /// Computes the complete local training loss including task loss and alignment loss.
    /// </summary>
    /// <param name="taskLoss">The base task loss.</param>
    /// <param name="localFeatures">Local model features on anchors.</param>
    /// <param name="globalFeatures">Global model features on anchors.</param>
    /// <returns>L_total = L_task + alpha * D(f_local, f_global).</returns>
    public T ComputeTotalLoss(T taskLoss, Matrix<T> localFeatures, Matrix<T> globalFeatures)
    {
        var alignmentLoss = ComputeAlignmentLoss(localFeatures, globalFeatures);
        return NumOps.Add(taskLoss, alignmentLoss);
    }

    /// <summary>
    /// Mean squared L2 distance: (1/N) * sum_i ||f_local_i - f_global_i||^2.
    /// </summary>
    private static double ComputeL2Distance(Matrix<T> local, Matrix<T> global)
    {
        int n = local.Rows;
        int d = local.Columns;
        double totalSq = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double diff = NumOps.ToDouble(local[i, j]) - NumOps.ToDouble(global[i, j]);
                totalSq += diff * diff;
            }
        }

        return totalSq / n;
    }

    /// <summary>
    /// Centered Kernel Alignment distance: 1 - CKA(local, global).
    /// CKA measures representational similarity invariant to invertible linear transformations.
    /// </summary>
    private static double ComputeCKADistance(Matrix<T> local, Matrix<T> global)
    {
        int n = local.Rows;
        int d = local.Columns;

        // Center features (subtract column means).
        var localCentered = CenterFeatures(local, n, d);
        var globalCentered = CenterFeatures(global, n, d);

        // Linear CKA: HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
        // where HSIC with linear kernel = ||X^T Y||_F^2 / (n-1)^2
        double crossNorm = FrobeniusNormOfProduct(localCentered, globalCentered, n, d);
        double localNorm = FrobeniusNormOfProduct(localCentered, localCentered, n, d);
        double globalNorm = FrobeniusNormOfProduct(globalCentered, globalCentered, n, d);

        double denom = Math.Sqrt(localNorm * globalNorm);
        double cka = denom > 1e-10 ? crossNorm / denom : 0.0;

        return 1.0 - cka;
    }

    /// <summary>
    /// Maximum Mean Discrepancy with RBF kernel: MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')].
    /// </summary>
    private static double ComputeMMDDistance(Matrix<T> local, Matrix<T> global)
    {
        int n = local.Rows;
        int d = local.Columns;

        // Compute median of pairwise distances for bandwidth selection.
        double medianDist = EstimateMedianPairwiseDistance(local, n, d);
        double bandwidth = Math.Max(medianDist, 1e-8);
        double gamma = 1.0 / (2.0 * bandwidth * bandwidth);

        double kxx = 0, kyy = 0, kxy = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    continue;
                }

                double distXX = PairwiseL2Sq(local, local, i, j, d);
                double distYY = PairwiseL2Sq(global, global, i, j, d);
                double distXY = PairwiseL2Sq(local, global, i, j, d);

                kxx += Math.Exp(-gamma * distXX);
                kyy += Math.Exp(-gamma * distYY);
                kxy += Math.Exp(-gamma * distXY);
            }
        }

        double normFactor = n * (n - 1);
        return Math.Max(kxx / normFactor - 2.0 * kxy / normFactor + kyy / normFactor, 0.0);
    }

    private static double[][] CenterFeatures(Matrix<T> features, int n, int d)
    {
        var means = new double[d];
        for (int j = 0; j < d; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += NumOps.ToDouble(features[i, j]);
            }

            means[j] = sum / n;
        }

        var centered = new double[n][];
        for (int i = 0; i < n; i++)
        {
            centered[i] = new double[d];
            for (int j = 0; j < d; j++)
            {
                centered[i][j] = NumOps.ToDouble(features[i, j]) - means[j];
            }
        }

        return centered;
    }

    private static double FrobeniusNormOfProduct(double[][] a, double[][] b, int n, int d)
    {
        // ||A^T B||_F^2 = sum_{ij} (sum_k a_ki * b_kj)^2
        // Efficient: compute G = A^T B [d x d], then sum squares.
        double normSq = 0;
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double g = 0;
                for (int k = 0; k < n; k++)
                {
                    g += a[k][i] * b[k][j];
                }

                normSq += g * g;
            }
        }

        return normSq;
    }

    private static double PairwiseL2Sq(Matrix<T> a, Matrix<T> b, int i, int j, int d)
    {
        double sq = 0;
        for (int k = 0; k < d; k++)
        {
            double diff = NumOps.ToDouble(a[i, k]) - NumOps.ToDouble(b[j, k]);
            sq += diff * diff;
        }

        return sq;
    }

    private static double EstimateMedianPairwiseDistance(Matrix<T> features, int n, int d)
    {
        // Use a sample of pairwise distances for efficiency.
        int maxPairs = Math.Min(n * (n - 1) / 2, 500);
        var distances = new List<double>(maxPairs);
        int step = Math.Max(1, n * (n - 1) / 2 / maxPairs);
        int count = 0;

        for (int i = 0; i < n && distances.Count < maxPairs; i++)
        {
            for (int j = i + 1; j < n && distances.Count < maxPairs; j++)
            {
                count++;
                if (count % step == 0)
                {
                    double sq = 0;
                    for (int k = 0; k < d; k++)
                    {
                        double diff = NumOps.ToDouble(features[i, k]) - NumOps.ToDouble(features[j, k]);
                        sq += diff * diff;
                    }

                    distances.Add(Math.Sqrt(sq));
                }
            }
        }

        if (distances.Count == 0)
        {
            return 1.0;
        }

        distances.Sort();
        return distances[distances.Count / 2];
    }

    /// <summary>Gets the feature alignment weight (alpha).</summary>
    public double AlignmentWeight => _alignmentWeight;

    /// <summary>Gets the distance metric used for alignment.</summary>
    public AlignmentDistanceMetric DistanceMetric => _distanceMetric;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedAlign(\u03b1={_alignmentWeight},{_distanceMetric})";
}

/// <summary>
/// Distance metric for feature alignment in FedAlign.
/// </summary>
public enum AlignmentDistanceMetric
{
    /// <summary>Mean squared L2 distance between representations.</summary>
    L2,

    /// <summary>Centered Kernel Alignment (CKA) — invariant to invertible linear transforms.</summary>
    CKA,

    /// <summary>Maximum Mean Discrepancy (MMD) with RBF kernel.</summary>
    MMD
}
