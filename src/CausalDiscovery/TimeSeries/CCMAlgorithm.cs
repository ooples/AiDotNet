using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// CCM — Convergent Cross-Mapping for detecting causation in nonlinear dynamical systems.
/// </summary>
/// <remarks>
/// <para>
/// CCM is based on Takens' theorem from dynamical systems theory. If X causes Y, then
/// the shadow manifold reconstructed from Y should contain information about X, and
/// cross-mapping accuracy should improve (converge) with longer time series.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>For each pair (i,j), construct delay embedding M_j from variable j with lag tau and dimension E</item>
/// <item>For each point in M_j, find E+1 nearest neighbors in M_j</item>
/// <item>Compute simplex weights from distances: w_k = exp(-d_k / d_1)</item>
/// <item>Cross-map: predict x_i(t) as weighted combination of x_i at neighbor times</item>
/// <item>Compute correlation rho between predicted and actual x_i</item>
/// <item>If rho converges (improves) with increasing library size L, j cross-maps i → i causes j</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CCM tests causation by checking whether one variable's history
/// can "predict" another variable using nearest-neighbor reconstruction in delay-coordinate
/// space. Crucially, if X causes Y, then Y's history cross-maps to X (not the other way),
/// which is the opposite of Granger causality's logic.
/// </para>
/// <para>
/// Reference: Sugihara et al. (2012), "Detecting Causality in Complex Ecosystems", Science.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Detecting Causality in Complex Ecosystems", "https://doi.org/10.1126/science.1227079", Year = 2012, Authors = "George Sugihara, Robert May, Hao Ye, Chih-hao Hsieh, Ethan Deyle, Michael Fogarty, Stephan Munch")]
public class CCMAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CCM";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    private readonly double _convergenceThreshold;
    private readonly double _correlationThreshold;

    public CCMAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        _convergenceThreshold = options?.EdgeThreshold ?? 0.05;
        _correlationThreshold = options?.CorrelationThreshold ?? 0.1;
        if (_convergenceThreshold < 0 || _convergenceThreshold > 1)
            throw new ArgumentException("EdgeThreshold (convergence threshold) must be between 0 and 1.");
        if (_correlationThreshold < 0 || _correlationThreshold > 1)
            throw new ArgumentException("CorrelationThreshold must be between 0 and 1.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        // Use MaxLag + 1 as embedding dimension (uncapped)
        int embDim = MaxLag + 1;
        int tau = 1;
        int minLib = embDim + 2;
        // The half-library convergence test requires validN/2 >= minLib,
        // so validN >= 2*minLib, meaning n >= 2*minLib + (embDim-1)*tau
        int minSamples = 2 * minLib + (embDim - 1) * tau;

        if (d < 2)
            throw new ArgumentException($"CCM requires at least 2 variables, got {d}.");
        if (n < minSamples)
            throw new ArgumentException($"CCM requires at least {minSamples} samples for embedding dimension {embDim}, got {n}.");

        var result = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;

                int validN = n - (embDim - 1) * tau;
                if (validN < minLib) continue;

                // Build delay embedding from variable j
                var embeddings = new Matrix<T>(validN, embDim);
                for (int t = 0; t < validN; t++)
                    for (int e = 0; e < embDim; e++)
                        embeddings[t, e] = data[t + (embDim - 1) * tau - e * tau, j];

                // Cross-map at full and half library to test convergence
                double rhoFull = ComputeCrossMapCorrelation(data, embeddings, i, validN, embDim);
                double rhoHalf = ComputeCrossMapCorrelation(data, embeddings, i, validN / 2, embDim);

                double convergence = rhoFull - rhoHalf;
                // Accept edge if: convergence is positive (standard CCM) OR
                // rhoFull is very high (near-perfect prediction, common for deterministic data)
                if ((convergence > _convergenceThreshold || rhoFull > 0.95) && rhoFull > _correlationThreshold)
                    result[i, j] = NumOps.FromDouble(rhoFull);
            }

        return result;
    }

    private double ComputeCrossMapCorrelation(Matrix<T> data, Matrix<T> embeddings, int targetCol, int libSize, int embDim)
    {
        if (libSize < embDim + 2) return 0;
        int offset = embDim - 1;
        int numNeighbors = embDim + 1;

        var predictions = new Vector<T>(libSize);
        var actuals = new Vector<T>(libSize);

        for (int t = 0; t < libSize; t++)
        {
            var queryVec = new Vector<T>(embDim);
            for (int e = 0; e < embDim; e++)
                queryVec[e] = embeddings[t, e];

            // Find nearest neighbors using Engine-accelerated distance
            var distances = new double[libSize];
            for (int s = 0; s < libSize; s++)
            {
                if (s == t) { distances[s] = double.MaxValue; continue; }
                var candidateVec = new Vector<T>(embDim);
                for (int e = 0; e < embDim; e++)
                    candidateVec[e] = embeddings[s, e];
                distances[s] = NumOps.ToDouble(VectorHelper.EuclideanDistance(queryVec, candidateVec));
            }

            // Select numNeighbors nearest
            var neighborIdx = new int[numNeighbors];
            var neighborDist = new double[numNeighbors];
            for (int k = 0; k < numNeighbors; k++)
            {
                double minDist = double.MaxValue;
                int minIdx = 0;
                for (int s = 0; s < libSize; s++)
                {
                    if (distances[s] < minDist)
                    {
                        bool used = false;
                        for (int prev = 0; prev < k; prev++)
                            if (neighborIdx[prev] == s) { used = true; break; }
                        if (used) continue;
                        minDist = distances[s];
                        minIdx = s;
                    }
                }
                neighborIdx[k] = minIdx;
                neighborDist[k] = minDist;
            }

            // Simplex projection weights
            double dMin = Math.Max(neighborDist[0], 1e-15);
            double sumW = 0;
            var weights = new double[numNeighbors];
            for (int k = 0; k < numNeighbors; k++)
            {
                weights[k] = Math.Exp(-neighborDist[k] / dMin);
                sumW += weights[k];
            }

            T pred = NumOps.Zero;
            for (int k = 0; k < numNeighbors; k++)
            {
                T w = NumOps.FromDouble(weights[k] / (sumW + 1e-15));
                pred = NumOps.Add(pred, NumOps.Multiply(w, data[neighborIdx[k] + offset, targetCol]));
            }

            predictions[t] = pred;
            actuals[t] = data[t + offset, targetCol];
        }

        // Compute Pearson correlation using Engine-accelerated dot products
        T nT = NumOps.FromDouble(libSize);
        T sumP = NumOps.Zero, sumA = NumOps.Zero;
        for (int t = 0; t < libSize; t++)
        {
            sumP = NumOps.Add(sumP, predictions[t]);
            sumA = NumOps.Add(sumA, actuals[t]);
        }
        T meanP = NumOps.Divide(sumP, nT);
        T meanA = NumOps.Divide(sumA, nT);

        // Center vectors and use Engine.DotProduct
        var centeredP = new Vector<T>(libSize);
        var centeredA = new Vector<T>(libSize);
        for (int t = 0; t < libSize; t++)
        {
            centeredP[t] = NumOps.Subtract(predictions[t], meanP);
            centeredA[t] = NumOps.Subtract(actuals[t], meanA);
        }

        T covPA = Engine.DotProduct(centeredP, centeredA);
        T varP = Engine.DotProduct(centeredP, centeredP);
        T varA = Engine.DotProduct(centeredA, centeredA);

        double dVarP = NumOps.ToDouble(varP);
        double dVarA = NumOps.ToDouble(varA);
        double denom = Math.Sqrt(Math.Max(dVarP, 1e-15) * Math.Max(dVarA, 1e-15));
        return NumOps.ToDouble(covPA) / denom;
    }
}
