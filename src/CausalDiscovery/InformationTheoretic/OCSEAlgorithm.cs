using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.InformationTheoretic;

/// <summary>
/// oCSE — optimal Causation Entropy for causal network inference.
/// </summary>
/// <remarks>
/// <para>
/// oCSE uses causation entropy — a measure of the information a variable provides about
/// another variable's transition — to identify causal links. It greedily selects the
/// optimal conditioning set that maximizes the causation entropy criterion.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>For each target Y, compute transition: delta_Y[t] = Y[t+1] - Y[t]</item>
/// <item>For each candidate cause X, compute causation entropy:
///   CE(X→Y|S) = MI(delta_Y ; X | S) where S is the current conditioning set</item>
/// <item>Greedily add variables to S that maximize CE until no variable exceeds threshold</item>
/// <item>Variables remaining outside S with significant CE are causal parents of Y</item>
/// <item>Edge weight = causation entropy value (Gaussian MI approximation)</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> oCSE measures how much a variable helps predict another variable's
/// CHANGES over time (not just its values). This is closer to true causation — a cause
/// should affect how the effect changes.
/// </para>
/// <para>
/// Reference: Sun et al. (2015), "Causal Network Inference by Optimal Causation Entropy",
/// SIAM Journal on Applied Dynamical Systems.
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
[ModelPaper("Causal Network Inference by Optimal Causation Entropy", "https://doi.org/10.1137/140956166", Year = 2015, Authors = "Jie Sun, Dane Taylor, Erik M. Bollt")]
public class OCSEAlgorithm<T> : InfoTheoreticBase<T>
{
    /// <inheritdoc/>
    public override string Name => "oCSE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;

    private readonly double _threshold;

    public OCSEAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyInfoOptions(options);
        _threshold = options?.EdgeThreshold ?? 0.05;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int effectiveN = n - 1; // We need transitions

        if (d < 2)
            throw new ArgumentException($"oCSE requires at least 2 variables, got {d}.");
        if (effectiveN < d + 3)
            throw new ArgumentException($"oCSE requires at least {d + 3 + 1} time points for {d} variables, got {n}.");

        // Compute transitions: delta_Y[t] = Y[t+1] - Y[t]
        var transitions = new Matrix<T>(effectiveN, d);
        for (int t = 0; t < effectiveN; t++)
            for (int j = 0; j < d; j++)
                transitions[t, j] = NumOps.Subtract(data[t + 1, j], data[t, j]);

        var result = new Matrix<T>(d, d);

        // For each target variable
        for (int target = 0; target < d; target++)
        {
            // Build transition vector for target
            var deltaY = new Vector<T>(effectiveN);
            for (int t = 0; t < effectiveN; t++)
                deltaY[t] = transitions[t, target];

            // Greedy forward selection of causal parents
            var selectedParents = new List<int>();
            var candidateSet = new List<int>();
            for (int i = 0; i < d; i++)
                if (i != target) candidateSet.Add(i);

            while (candidateSet.Count > 0)
            {
                // Find candidate with highest causation entropy conditioned on selected parents
                int bestCandidate = -1;
                double bestCE = _threshold;

                foreach (int candidate in candidateSet)
                {
                    double ce = ComputeCausationEntropy(data, deltaY, candidate, selectedParents, effectiveN, d);
                    if (ce > bestCE)
                    {
                        bestCE = ce;
                        bestCandidate = candidate;
                    }
                }

                if (bestCandidate < 0)
                    break; // No candidate exceeds threshold

                selectedParents.Add(bestCandidate);
                candidateSet.Remove(bestCandidate);
            }

            // Set edge weights for selected parents
            foreach (int parent in selectedParents)
            {
                // Compute final causation entropy as edge weight
                var otherParents = selectedParents.Where(p => p != parent).ToList();
                double ce = ComputeCausationEntropy(data, deltaY, parent, otherParents, effectiveN, d);
                if (ce > _threshold)
                    result[parent, target] = NumOps.FromDouble(ce);
            }
        }

        return result;
    }

    private double ComputeCausationEntropy(Matrix<T> data, Vector<T> deltaY,
        int candidate, List<int> condSet, int effectiveN, int d)
    {
        // CE(X→Y|S) = MI(deltaY; X | S)
        // Gaussian approximation: MI(A;B|C) = 0.5 * log(det(Sigma_AC) * det(Sigma_BC) / (det(Sigma_C) * det(Sigma_ABC)))
        // Simplified via partial correlation: MI ≈ -0.5 * log(1 - partialCorr^2)

        if (condSet.Count == 0)
        {
            // Simple MI between deltaY and X (no conditioning)
            return ComputeGaussianMIVectors(deltaY, data, candidate, effectiveN);
        }

        // Partial correlation approach: regress both deltaY and X on S, then correlate residuals
        var residDelta = RegressOut(deltaY, data, condSet, effectiveN);
        var xVec = new Vector<T>(effectiveN);
        for (int t = 0; t < effectiveN; t++)
            xVec[t] = data[t, candidate];
        var residX = RegressOut(xVec, data, condSet, effectiveN);

        // Correlation between residuals
        T covRR = Engine.DotProduct(residDelta, residX);
        T varD = Engine.DotProduct(residDelta, residDelta);
        T varX = Engine.DotProduct(residX, residX);

        double dVarD = Math.Max(NumOps.ToDouble(varD), 1e-15);
        double dVarX = Math.Max(NumOps.ToDouble(varX), 1e-15);
        double corr = NumOps.ToDouble(covRR) / Math.Sqrt(dVarD * dVarX);
        corr = Math.Max(-0.9999, Math.Min(0.9999, corr));

        return -0.5 * Math.Log(1 - corr * corr);
    }

    private double ComputeGaussianMIVectors(Vector<T> deltaY, Matrix<T> data, int col, int n)
    {
        // Center both vectors
        T nT = NumOps.FromDouble(n);
        T sumD = NumOps.Zero, sumX = NumOps.Zero;
        for (int t = 0; t < n; t++)
        {
            sumD = NumOps.Add(sumD, deltaY[t]);
            sumX = NumOps.Add(sumX, data[t, col]);
        }
        T meanD = NumOps.Divide(sumD, nT);
        T meanX = NumOps.Divide(sumX, nT);

        var centD = new Vector<T>(n);
        var centX = new Vector<T>(n);
        for (int t = 0; t < n; t++)
        {
            centD[t] = NumOps.Subtract(deltaY[t], meanD);
            centX[t] = NumOps.Subtract(data[t, col], meanX);
        }

        T covDX = Engine.DotProduct(centD, centX);
        T varD = Engine.DotProduct(centD, centD);
        T varX = Engine.DotProduct(centX, centX);

        double dVarD = Math.Max(NumOps.ToDouble(varD), 1e-15);
        double dVarX = Math.Max(NumOps.ToDouble(varX), 1e-15);
        double corr = NumOps.ToDouble(covDX) / Math.Sqrt(dVarD * dVarX);
        corr = Math.Max(-0.9999, Math.Min(0.9999, corr));

        return -0.5 * Math.Log(1 - corr * corr);
    }

    private Vector<T> RegressOut(Vector<T> y, Matrix<T> data, List<int> condSet, int n)
    {
        int p = condSet.Count;
        if (p == 0) return y;

        // Build normal equations using Engine.DotProduct for each column pair
        var XtX = new Matrix<T>(p, p);
        var Xty = new Vector<T>(p);

        // Extract conditioning columns
        var cols = new Vector<T>[p];
        for (int c = 0; c < p; c++)
        {
            cols[c] = new Vector<T>(n);
            for (int t = 0; t < n; t++)
                cols[c][t] = data[t, condSet[c]];
        }

        for (int a = 0; a < p; a++)
        {
            Xty[a] = Engine.DotProduct(cols[a], y);
            for (int b = a; b < p; b++)
            {
                T dot = Engine.DotProduct(cols[a], cols[b]);
                XtX[a, b] = dot;
                XtX[b, a] = dot;
            }
        }

        // Ridge for stability
        T ridge = NumOps.FromDouble(1e-10);
        for (int a = 0; a < p; a++)
            XtX[a, a] = NumOps.Add(XtX[a, a], ridge);

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX, Xty, MatrixDecompositionType.Lu);

        // Compute residuals
        var residuals = new Vector<T>(n);
        for (int t = 0; t < n; t++)
        {
            T pred = NumOps.Zero;
            for (int c = 0; c < p; c++)
                pred = NumOps.Add(pred, NumOps.Multiply(cols[c][t], beta[c]));
            residuals[t] = NumOps.Subtract(y[t], pred);
        }

        return residuals;
    }
}
