using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums;
using AiDotNet.Extensions;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// Base class for continuous optimization causal discovery methods (NOTEARS, DAGMA, GOLEM).
/// </summary>
/// <remarks>
/// <para>
/// These methods formulate the DAG learning problem as a continuous optimization:
/// minimize a loss function (e.g., least squares) subject to a smooth acyclicity constraint.
/// The key innovation is replacing the combinatorial DAG constraint with a differentiable function.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional methods check all possible graph structures to find the best one,
/// which is extremely slow for many variables. Continuous optimization methods instead use calculus-based
/// optimization (like gradient descent) to smoothly search for the best graph, which is much faster.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class ContinuousOptimizationBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.ContinuousOptimization;

    /// <summary>
    /// L1 sparsity penalty (lambda1). Controls sparsity of the learned graph.
    /// </summary>
    protected double Lambda1 { get; set; } = 0.1;

    /// <summary>
    /// Edge weight threshold for post-optimization pruning. Default matches the NOTEARS
    /// reference implementation (0.3).
    /// </summary>
    protected double WThreshold { get; set; } = 0.3;

    /// <summary>
    /// Maximum number of outer loop iterations.
    /// </summary>
    protected int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Convergence tolerance for the acyclicity constraint h(W).
    /// </summary>
    protected double HTolerance { get; set; } = 1e-8;

    /// <summary>
    /// Modulus used to cycle column-specific perturbations during data standardization.
    /// The value is the largest prime under 100, chosen so column indices that are
    /// multiples of small numbers (2, 3, 5) do not collide. Bounds the maximum
    /// perturbation at <c>1 + 1e-8 * 97 ≈ 1 + 9.7e-7</c>, which stays numerically
    /// negligible regardless of the number of columns.
    /// </summary>
    private const int PerturbationCycleModulus = 97;

    /// <summary>
    /// Loss type: "l2" (least squares), "logistic", or "poisson".
    /// </summary>
    protected string LossType { get; set; } = "l2";

    /// <summary>
    /// Applies common options from <see cref="CausalDiscoveryOptions"/> to the algorithm
    /// parameters. All values are validated before assignment so that invalid external
    /// input fails fast with a clear message instead of silently corrupting optimization
    /// behavior.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when any provided option is outside its accepted range.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <c>LossType</c> is not one of <c>"l2"</c>, <c>"logistic"</c>, <c>"poisson"</c>.
    /// </exception>
    protected void ApplyOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;

        if (options.SparsityPenalty.HasValue)
        {
            if (options.SparsityPenalty.Value < 0)
                throw new ArgumentOutOfRangeException(
                    nameof(options.SparsityPenalty),
                    $"SparsityPenalty must be >= 0; got {options.SparsityPenalty.Value}.");
            Lambda1 = options.SparsityPenalty.Value;
        }
        if (options.EdgeThreshold.HasValue)
        {
            if (options.EdgeThreshold.Value < 0)
                throw new ArgumentOutOfRangeException(
                    nameof(options.EdgeThreshold),
                    $"EdgeThreshold must be >= 0; got {options.EdgeThreshold.Value}.");
            WThreshold = options.EdgeThreshold.Value;
        }
        if (options.MaxIterations.HasValue)
        {
            if (options.MaxIterations.Value <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(options.MaxIterations),
                    $"MaxIterations must be > 0; got {options.MaxIterations.Value}.");
            MaxIterations = options.MaxIterations.Value;
        }
        if (options.AcyclicityTolerance.HasValue)
        {
            if (options.AcyclicityTolerance.Value <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(options.AcyclicityTolerance),
                    $"AcyclicityTolerance must be > 0; got {options.AcyclicityTolerance.Value}.");
            HTolerance = options.AcyclicityTolerance.Value;
        }
        if (options.LossType != null)
        {
            // Normalize to lowercase + trim so callers passing "L2" or " l2 " hit the
            // same code path as "l2". The downstream loss dispatch uses lowercase tokens.
            var normalized = options.LossType.Trim().ToLowerInvariant();
            if (normalized is not ("l2" or "logistic" or "poisson"))
                throw new ArgumentException(
                    $"LossType must be one of 'l2', 'logistic', 'poisson' " +
                    $"(case-insensitive, whitespace trimmed); got '{options.LossType}'.",
                    nameof(options.LossType));
            LossType = normalized;
        }
    }

    /// <summary>
    /// Computes the L2 loss: (1/2n) * ||X - XW||²_F and its gradient.
    /// </summary>
    /// <param name="X">Data matrix [n x d].</param>
    /// <param name="W">Weighted adjacency matrix [d x d].</param>
    /// <returns>Tuple of (loss value, gradient matrix [d x d]).</returns>
    protected (double Loss, Matrix<T> Gradient) ComputeL2Loss(Matrix<T> X, Matrix<T> W)
    {
        int n = X.Rows;

        // Vectorized: R = X - X @ W using Engine.TensorMatMul
        var xTensor = Tensor<T>.FromMatrix(X);
        var wTensor = Tensor<T>.FromMatrix(W);
        var xwTensor = Engine.TensorMatMul(xTensor, wTensor);
        var rTensor = Engine.TensorSubtract(xTensor, xwTensor);

        // Loss = (1/2n) * ||R||²_F
        double loss = NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(rTensor)));
        loss *= 0.5 / n;

        // Vectorized: Gradient = -(1/n) * X^T @ R
        var xT = xTensor.Transpose([1, 0]);
        var gradTensor = Engine.TensorMatMul(xT, rTensor);
        T negInvN = NumOps.Negate(NumOps.FromDouble(1.0 / n));
        gradTensor = Engine.TensorMultiplyScalar<T>(gradTensor, negInvN);

        return (loss, gradTensor.ToMatrix());
    }

    /// <summary>
    /// Computes the NOTEARS acyclicity constraint h(W) = tr(e^(W∘W)) - d
    /// and its gradient dh/dW = 2 * e^(W∘W) ∘ W.
    /// </summary>
    /// <param name="W">Weighted adjacency matrix [d x d].</param>
    /// <param name="taylorOrder">Number of Taylor series terms for matrix exponential.</param>
    /// <returns>Tuple of (h value, gradient matrix [d x d]).</returns>
    protected (double H, Matrix<T> Gradient) ComputeNOTEARSConstraint(Matrix<T> W, int taylorOrder = 8)
    {
        int d = W.Rows;

        // W ∘ W (element-wise square)
        var wTensor = Tensor<T>.FromMatrix(W);
        var wSquared = Engine.TensorMultiply(wTensor, wTensor).ToMatrix();

        // e^(W∘W) via shared MatrixExponential extension
        var expMatrix = wSquared.MatrixExponential(taylorOrder);

        // h = tr(expMatrix) - d
        double h = NumOps.ToDouble(expMatrix.Trace()) - d;

        // Gradient: dh/dW = 2 * (e^(W∘W))^T ∘ W
        // The transpose is required because h(W) = tr(exp(W∘W)) - d differentiates to
        // (exp(W∘W))^T element-wise multiplied with d(W∘W)/dW = 2W. Reading expMatrix[i, j]
        // without the transpose flips the derivative direction and steers NOTEARS away
        // from the acyclic optimum.
        T two = NumOps.FromDouble(2.0);
        var gradient = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorTranspose(Tensor<T>.FromMatrix(expMatrix)), wTensor), two);

        return (h, gradient.ToMatrix());
    }

    /// <summary>
    /// Computes the L1 norm of a matrix: sum of absolute values.
    /// </summary>
    protected double ComputeL1Norm(Matrix<T> W)
    {
        return NumOps.ToDouble(Engine.TensorSum(Engine.TensorAbs(Tensor<T>.FromMatrix(W))));
    }

    /// <summary>
    /// Applies threshold to W: sets entries with |W[i,j]| &lt; threshold to 0.
    /// Also zeros out the diagonal.
    /// </summary>
    protected Matrix<T> ThresholdAndClean(Matrix<T> W, double threshold)
    {
        var values = Tensor<T>.FromMatrix(W);
        var keep = Engine.TensorMultiply(
            Engine.TensorGreaterEqual(Engine.TensorAbs(values), Engine.TensorFullLike(values, NumOps.FromDouble(threshold))),
            OffDiagonalMask(W.Rows));
        return Engine.TensorWhere(keep, values, Engine.TensorZerosLike(values)).ToMatrix();
    }

    /// <summary>
    /// One Adam step on a parameter tensor:
    /// m = b1*m + (1-b1)*g, v = b2*v + (1-b2)*g², p - lr * (m/bc1) / (sqrt(v/bc2) + eps).
    /// </summary>
    /// <returns>The updated parameter and the new first and second moments.</returns>
    private protected (Tensor<T> Parameter, Tensor<T> M, Tensor<T> V) AdamUpdate(
        Tensor<T> parameter, Tensor<T> gradient, Tensor<T> m, Tensor<T> v,
        double beta1, double beta2, double eps, double lr, double biasCorr1, double biasCorr2)
    {
        var newM = Engine.TensorAdd(Engine.TensorMultiplyScalar(m, NumOps.FromDouble(beta1)),
            Engine.TensorMultiplyScalar(gradient, NumOps.FromDouble(1.0 - beta1)));
        var newV = Engine.TensorAdd(Engine.TensorMultiplyScalar(v, NumOps.FromDouble(beta2)),
            Engine.TensorMultiplyScalar(Engine.TensorSquare(gradient), NumOps.FromDouble(1.0 - beta2)));
        var mHat = Engine.TensorDivideScalar(newM, NumOps.FromDouble(biasCorr1));
        var vHat = Engine.TensorDivideScalar(newV, NumOps.FromDouble(biasCorr2));
        var step = Engine.TensorDivide(Engine.TensorMultiplyScalar(mHat, NumOps.FromDouble(lr)),
            Engine.TensorAddScalar(Engine.TensorSqrt(vHat), NumOps.FromDouble(eps)));
        return (Engine.TensorSubtract(parameter, step), newM, newV);
    }

    /// <summary>
    /// Projects a weighted directed graph onto a DAG by retaining the strongest
    /// edges that do not introduce a directed cycle.
    /// </summary>
    /// <remarks>
    /// Continuous acyclicity penalties approach zero numerically but do not make
    /// every finite-iteration solution exactly acyclic. Thresholding such a
    /// solution can therefore leave a small residual cycle. The standard
    /// strongest-edge projection below preserves as much of the learned weighted
    /// structure as possible while enforcing the public DAG contract.
    ///
    /// COST, and the variable count it is sized for: one reachability traversal runs per candidate
    /// edge, so a dense d-node graph costs O(d^2) edges times O(d^2) traversal. That is quartic. At
    /// d = 50 it is roughly 6 million operations and invisible next to the solver itself, which runs
    /// hundreds of augmented-Lagrangian iterations over the same data. At d = 200 it is on the order
    /// of 1.6 billion and would dominate.
    ///
    /// This is sized for the regime continuous-optimization structure learning is used in --
    /// NOTEARS, DAGMA and MCSL are all evaluated at tens of variables, and their own inner solves are
    /// already O(d^3) per iteration -- so the projection is not the bottleneck there. A caller who
    /// needs hundreds of variables should expect this step to become significant, and the fix is a
    /// union-find over a topological order or an incremental reachability matrix, either of which
    /// removes the quartic term.
    /// </remarks>
    protected Matrix<T> EnforceAcyclic(Matrix<T> weights)
    {
        int d = weights.Rows;
        var flat = Tensor<T>.FromMatrix(weights).Reshape(new[] { d * d });
        var magnitude = Engine.TensorAbs(flat);

        // Candidate edges: off-diagonal entries with a non-zero magnitude, in row-major (from, to) order.
        var candidate = Engine.TensorMultiply(
            Engine.TensorGreaterThan(magnitude, NumOps.Zero), OffDiagonalMask(d).Reshape(new[] { d * d }));
        int[] positions = Engine.TensorNonzero(candidate).ToArray();
        var dag = new Tensor<T>(new[] { d * d });
        if (positions.Length == 0)
            return dag.Reshape(new[] { d, d }).ToMatrix();

        // Strongest first; the stable sort keeps (from, to) order among equal magnitudes.
        var positionIndex = new Tensor<int>(positions, new[] { positions.Length });
        int[] order = Engine.TensorArgsort(Engine.TensorIndexSelect(magnitude, positionIndex, 0), -1, true).ToArray();

        var reachable = new bool[d, d];
        var kept = new List<int>();
        foreach (int rank in order)
        {
            int position = positions[rank];
            int from = position / d, to = position % d;

            // Adding from -> to creates a cycle exactly when to already reaches from.
            if (!HasDirectedPath(reachable, to, from))
            {
                reachable[from, to] = true;
                kept.Add(position);
            }
        }

        if (kept.Count == 0)
            return dag.Reshape(new[] { d, d }).ToMatrix();

        var keptIndex = new Tensor<int>(kept.ToArray(), new[] { kept.Count });
        dag = Engine.TensorScatter(dag, keptIndex, Engine.TensorIndexSelect(flat, keptIndex, 0), 0);
        return dag.Reshape(new[] { d, d }).ToMatrix();
    }

    private static bool HasDirectedPath(bool[,] graph, int start, int target)
    {
        if (start == target) return true;

        int d = graph.GetLength(0);
        var visited = new bool[d];
        var pending = new Stack<int>();
        pending.Push(start);

        while (pending.Count > 0)
        {
            int node = pending.Pop();
            if (node == target) return true;
            if (visited[node]) continue;
            visited[node] = true;

            for (int next = 0; next < d; next++)
            {
                if (!visited[next] && graph[node, next])
                    pending.Push(next);
            }
        }

        return false;
    }

    /// <summary>
    /// Thresholds the weight matrix, with covariance-based fallback when no edges survive.
    /// Use this instead of ThresholdAndClean when the original data is available.
    /// </summary>
    protected Matrix<T> ThresholdWithFallback(Matrix<T> W, double threshold, Matrix<T> data)
    {
        var result = ThresholdAndClean(W, threshold);

        // Check if any edges survived
        int d = W.Rows;
        var offDiagonal = OffDiagonalMask(d);
        bool hasEdges = Engine.TensorCountNonzero(
            Engine.TensorMultiply(Tensor<T>.FromMatrix(result), offDiagonal)) > 0;

        if (!hasEdges)
        {
            // No edges survived — use PARTIAL correlations to identify direct edges.
            // Partial correlation removes indirect effects (e.g., X0↔X2 through X1)
            // by conditioning on all other variables. Only direct relationships survive.
            // This is computed from the precision matrix (inverse covariance):
            // pcor(i,j) = -P[i,j] / sqrt(P[i,i] * P[j,j])
            var cov = ComputeCovarianceMatrix(data);

            // Compute precision matrix via regularized inverse
            var precision = Tensor<T>.FromMatrix(InvertWithRidge(cov, d, NumOps.FromDouble(1e-4)));
            var covariance = Tensor<T>.FromMatrix(cov);
            var shape = new[] { d, d };
            var ones = Engine.TensorOnesLike(precision);

            // Pairs need a positive precision diagonal on both ends (an entry with P[i,i] <= 0 is skipped).
            var precisionDiagonal = Engine.TensorDiagonal(precision);
            var positive = Engine.TensorSubtract(Engine.TensorOnesLike(precisionDiagonal),
                Engine.TensorLessEqual(precisionDiagonal, Engine.TensorZerosLike(precisionDiagonal)));
            var include = Engine.TensorMultiply(Engine.TensorOuter(positive, positive), offDiagonal);

            // Partial correlation from precision matrix:
            // pcor(i,j) = -P[i,j] / sqrt(P[i,i] * P[j,j])
            var precisionScale = Engine.TensorSqrt(Engine.TensorOuter(precisionDiagonal, precisionDiagonal));
            var partialCorr = Engine.TensorDivide(Engine.TensorNegate(precision),
                Engine.TensorWhere(include, precisionScale, ones));
            include = Engine.TensorMultiply(include, Engine.TensorSubtract(ones,
                Engine.TensorLessThan(Engine.TensorAbs(partialCorr), Engine.TensorFullLike(partialCorr, NumOps.FromDouble(threshold)))));

            // Direction: use variance ratio from raw covariance (varI = cov[i,i] per row, varJ = cov[j,j] per column)
            var variance = Engine.TensorDiagonal(covariance);
            var varI = Engine.TensorBroadcastTo(variance.Reshape(new[] { d, 1 }), shape);
            var varJ = Engine.TensorBroadcastTo(variance.Reshape(new[] { 1, d }), shape);
            var tiny = Engine.TensorFullLike(precision, NumOps.FromDouble(1e-10));
            include = Engine.TensorMultiply(include, Engine.TensorSubtract(ones, Engine.TensorLessThan(varI, tiny)));

            var weight = Engine.TensorDivide(covariance, Engine.TensorWhere(include, varI, ones));
            var reverseValid = Engine.TensorGreaterThan(varJ, tiny);
            var reverseWeight = Engine.TensorWhere(reverseValid,
                Engine.TensorAbs(Engine.TensorDivide(covariance, Engine.TensorWhere(reverseValid, varJ, ones))),
                Engine.TensorZerosLike(precision));

            var absWeight = Engine.TensorAbs(weight);
            var stronger = Engine.TensorGreaterThan(absWeight, reverseWeight);
            var tiedUpper = Engine.TensorMultiply(
                Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorSubtract(absWeight, reverseWeight)), tiny),
                Engine.TensorTriu(ones, 1));
            var assign = Engine.TensorMultiply(include, Engine.TensorMax(stronger, tiedUpper));

            result = Engine.TensorWhere(assign, weight, Tensor<T>.FromMatrix(result)).ToMatrix();
        }

        return result;
    }

    /// <summary>
    /// Standardizes data to zero mean and unit variance per column, with a small
    /// column-specific perturbation to break exact collinearity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// All continuous optimization methods (NOTEARS, DAGMA, etc.) assume standardized data
    /// for numerical stability. Without standardization, the loss gradients can be proportional
    /// to the data magnitude, causing the acyclicity constraint tr(exp(W∘W)) to explode
    /// for even moderate step sizes during optimization.
    /// </para>
    /// <para>
    /// The column-specific perturbation is essential for identifiability. When multiple
    /// variables are exact linear transformations of each other (e.g., Y = 2X + 1),
    /// standardization makes their columns identical, creating symmetric saddle points
    /// in the acyclicity constraint landscape. The perturbation breaks this symmetry,
    /// allowing gradient-based optimization to distinguish between causal directions.
    /// This is standard practice in NOTEARS implementations.
    /// </para>
    /// </remarks>
    protected Matrix<T> StandardizeData(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        var shape = new[] { n, d };

        var centered = CenterColumns(Tensor<T>.FromMatrix(data));
        var variance = Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, true);
        var std = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-15)));

        // Standardize and add tiny column-specific perturbation to break exact collinearity.
        // The perturbation must stay numerically negligible regardless of dimensionality,
        // so we use a small base step and modulo-cycle the column index by a prime
        // (PerturbationCycleModulus = 97). This keeps the maximum perturbation bounded
        // at ~ 1e-6 even at d ≈ 100 columns, instead of growing linearly to 1e-2 with
        // the previous (1 + 1e-4 * (j+1)) scheme.
        const double perturbBase = 1e-8;
        var cycle = new int[d];
        for (int j = 0; j < d; j++)
            cycle[j] = (j % PerturbationCycleModulus) + 1;
        var perturbScale = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(new Tensor<int>(cycle, new[] { 1, d }).Cast<T>(), NumOps.FromDouble(perturbBase)),
            NumOps.One);

        var standardized = Engine.TensorDivide(centered, Engine.TensorBroadcastTo(std, shape));
        return Engine.TensorMultiply(standardized, Engine.TensorBroadcastTo(perturbScale, shape)).ToMatrix();
    }

    /// <summary>
    /// Fallback: uses pairwise correlation to detect edges when continuous optimization
    /// fails to find structure (e.g., due to near-degenerate data).
    /// </summary>
    protected Matrix<T> FallbackCorrelationGraph(Matrix<T> data)
    {
        // |corr(a, b)| for a < b, kept when it reaches WThreshold.
        var correlation = Tensor<T>.FromMatrix(ComputeAbsoluteCorrelationMatrix(data));
        var keep = Engine.TensorMultiply(
            Engine.TensorGreaterEqual(correlation, Engine.TensorFullLike(correlation, NumOps.FromDouble(WThreshold))),
            Engine.TensorTriu(Engine.TensorOnesLike(correlation), 1));
        return Engine.TensorWhere(keep, correlation, Engine.TensorZerosLike(correlation)).ToMatrix();
    }

    /// <summary>
    /// Computes the precision matrix (inverse covariance) with ridge regularization
    /// using the library's Matrix type and numeric operations.
    /// </summary>
    /// <param name="S">Covariance matrix to invert.</param>
    /// <param name="d">Dimension.</param>
    /// <param name="ridge">Ridge regularization to ensure invertibility.</param>
    /// <returns>The precision matrix (S + ridge*I)^{-1}.</returns>
    private Matrix<T> InvertWithRidge(Matrix<T> S, int d, T ridge)
    {
        // (S + ridge * I)^{-1} through the library's LU-based inverse.
        var regularized = Engine.TensorAdd(Tensor<T>.FromMatrix(S),
            Engine.TensorMultiplyScalar(Engine.TensorEye<T>(d), ridge));
        return LinalgDispatch<T>.Inv(regularized).ToMatrix();
    }
}
