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
/// <item>Variables in S are the discovered causal parents of Y</item>
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
[ResearchPaper("Causal Network Inference by Optimal Causation Entropy", "https://doi.org/10.1137/140956166", Year = 2015, Authors = "Jie Sun, Dane Taylor, Erik M. Bollt")]
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
            throw new ArgumentException($"oCSE requires at least {d + 4} time points for {d} variables, got {n}.");

        // Compute transitions: delta_Y[t] = Y[t+1] - Y[t]; the regressors are the rows t < effectiveN
        var x = Tensor<T>.FromMatrix(data);
        var head = Engine.TensorSlice(x, new[] { 0, 0 }, new[] { effectiveN, d });
        var transitions = Engine.TensorSubtract(Engine.TensorSlice(x, new[] { 1, 0 }, new[] { effectiveN, d }), head);
        T constantTolerance = NumOps.FromDouble(1e-10);

        var weights = new Dictionary<int, double>();

        // For each target variable
        for (int target = 0; target < d; target++)
        {
            // Build transition vector for target
            var deltaY = Engine.TensorSliceAxis(transitions, 1, target);

            // Check if transitions are near-constant (deterministic trend).
            // If so, skip this target since constant transitions carry no MI.
            var deltaVar = Engine.ReduceMean(Engine.TensorSquare(
                Engine.TensorSubtractScalar(deltaY, Engine.TensorMean(deltaY))), new[] { 0 }, true);
            if (CausalTensorHelpers.AnyBelow(deltaVar, constantTolerance))
            {
                // Transitions are near-constant — no temporal variation to explain.
                // Skip this target: constant transitions carry zero mutual information,
                // so no causal parents can be identified. Leave W[*,target] as zero.
                continue;
            }

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
                    double ce = ComputeCausationEntropy(head, deltaY, candidate, selectedParents);
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
                double ce = ComputeCausationEntropy(head, deltaY, parent, otherParents);
                if (ce > _threshold)
                    weights[parent * d + target] = ce;
            }
        }

        return CausalTensorHelpers.FromStatistics<T>(d, d, weights);
    }

    private double ComputeCausationEntropy(Tensor<T> head, Tensor<T> deltaY, int candidate, List<int> condSet)
    {
        // CE(X→Y|S) = MI(deltaY; X | S)
        // Gaussian approximation: MI(A;B|C) = 0.5 * log(det(Sigma_AC) * det(Sigma_BC) / (det(Sigma_C) * det(Sigma_ABC)))
        // Simplified via partial correlation: MI ≈ -0.5 * log(1 - partialCorr^2)

        var xVec = Engine.TensorSliceAxis(head, 1, candidate);
        if (condSet.Count == 0)
        {
            // Simple MI between deltaY and X (no conditioning)
            return ComputeGaussianMIVectors(deltaY, xVec);
        }

        // Partial correlation approach: regress both deltaY and X on S, then correlate residuals
        var residDelta = RegressOut(deltaY, head, condSet);
        var residX = RegressOut(xVec, head, condSet);
        return ClampedCorrelationMI(residDelta, residX);
    }

    private double ComputeGaussianMIVectors(Tensor<T> deltaY, Tensor<T> x)
    {
        // Center both vectors
        var centD = Engine.TensorSubtractScalar(deltaY, Engine.TensorMean(deltaY));
        var centX = Engine.TensorSubtractScalar(x, Engine.TensorMean(x));
        return ClampedCorrelationMI(centD, centX);
    }

    /// <summary>
    /// -0.5 log(1 - r^2) with r = sum(a b) / sqrt(max(sum(a^2), 1e-15) max(sum(b^2), 1e-15)) clamped to
    /// [-0.9999, 0.9999].
    /// </summary>
    private double ClampedCorrelationMI(Tensor<T> a, Tensor<T> b)
    {
        T cov = Engine.TensorSum(Engine.TensorMultiply(a, b));
        T floor = NumOps.FromDouble(1e-15);
        T varA = Engine.TensorSum(Engine.TensorSquare(a));
        T varB = Engine.TensorSum(Engine.TensorSquare(b));
        if (NumOps.LessThan(varA, floor)) varA = floor;
        if (NumOps.LessThan(varB, floor)) varB = floor;
        T corr = NumOps.Divide(cov, NumOps.Sqrt(NumOps.Multiply(varA, varB)));
        T bound = NumOps.FromDouble(0.9999);
        if (NumOps.GreaterThan(corr, bound)) corr = bound;
        if (NumOps.LessThan(corr, NumOps.Negate(bound))) corr = NumOps.Negate(bound);

        return NumOps.ToDouble(NumOps.Multiply(NumOps.FromDouble(-0.5),
            NumOps.Log(NumOps.Subtract(NumOps.One, NumOps.Multiply(corr, corr)))));
    }

    private Tensor<T> RegressOut(Tensor<T> y, Tensor<T> head, List<int> condSet)
    {
        int p = condSet.Count;
        if (p == 0) return y;
        int n = y.Length;

        // Center y and the conditioning columns
        var centY = Engine.TensorSubtractScalar(y, Engine.TensorMean(y)).Reshape(new[] { n, 1 });
        var cols = CenterColumns(SelectColumns(head, condSet.ToArray()));

        // Build normal equations on centered data, ridge for stability
        var colsT = Engine.TensorTranspose(cols);
        var XtX = CausalTensorHelpers.Symmetrize(Engine.TensorMatMul(colsT, cols));
        XtX = Engine.TensorAdd(XtX, Engine.TensorMultiplyScalar(Engine.TensorEye<T>(p), NumOps.FromDouble(1e-10)));
        var Xty = Engine.TensorMatMul(colsT, centY).Reshape(new[] { p }).ToVector();

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX.ToMatrix(), Xty, MatrixDecompositionType.Lu);

        // Compute residuals from centered regression
        var prediction = Engine.TensorMatMul(cols, beta.AsTensor().Reshape(new[] { p, 1 }));
        return Engine.TensorSubtract(centY, prediction).Reshape(new[] { n });
    }
}
