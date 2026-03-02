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
    /// Edge weight threshold for post-optimization pruning.
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
    /// Loss type: "l2" (least squares), "logistic", or "poisson".
    /// </summary>
    protected string LossType { get; set; } = "l2";

    /// <summary>
    /// Applies common options from CausalDiscoveryOptions to the algorithm parameters.
    /// </summary>
    protected void ApplyOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;

        if (options.SparsityPenalty.HasValue)
            Lambda1 = options.SparsityPenalty.Value;
        if (options.EdgeThreshold.HasValue)
            WThreshold = options.EdgeThreshold.Value;
        if (options.MaxIterations.HasValue)
            MaxIterations = options.MaxIterations.Value;
        if (options.AcyclicityTolerance.HasValue)
            HTolerance = options.AcyclicityTolerance.Value;
        if (options.LossType != null)
            LossType = options.LossType;
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
        int d = X.Columns;

        // Residual R = X - X @ W
        var R = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                T xw = NumOps.Zero;
                for (int k = 0; k < d; k++)
                    xw = NumOps.Add(xw, NumOps.Multiply(X[i, k], W[k, j]));
                R[i, j] = NumOps.Subtract(X[i, j], xw);
            }
        }

        // Loss = (1/2n) * ||R||²_F
        double loss = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
            {
                double r = NumOps.ToDouble(R[i, j]);
                loss += r * r;
            }
        loss *= 0.5 / n;

        // Gradient = -(1/n) * X^T @ R = (1/n) * X^T @ (XW - X)
        T nT = NumOps.FromDouble(n);
        var grad = new Matrix<T>(d, d);
        for (int k = 0; k < d; k++)
        {
            for (int j = 0; j < d; j++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < n; i++)
                    sum = NumOps.Add(sum, NumOps.Multiply(X[i, k], R[i, j]));
                grad[k, j] = NumOps.Negate(NumOps.Divide(sum, nT));
            }
        }

        return (loss, grad);
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
        var wSquared = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                wSquared[i, j] = NumOps.Multiply(W[i, j], W[i, j]);

        // e^(W∘W) via shared MatrixExponential extension
        var expMatrix = wSquared.MatrixExponential(taylorOrder);

        // h = tr(expMatrix) - d
        double h = NumOps.ToDouble(expMatrix.Trace()) - d;

        // Gradient: dh/dW = 2 * e^(W∘W) ∘ W
        T two = NumOps.FromDouble(2.0);
        var gradient = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                gradient[i, j] = NumOps.Multiply(two, NumOps.Multiply(expMatrix[i, j], W[i, j]));

        return (h, gradient);
    }

    /// <summary>
    /// Computes the L1 norm of a matrix: sum of absolute values.
    /// </summary>
    protected double ComputeL1Norm(Matrix<T> W)
    {
        int d = W.Rows;
        double sum = 0;
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                sum += Math.Abs(NumOps.ToDouble(W[i, j]));
        return sum;
    }

    /// <summary>
    /// Applies threshold to W: sets entries with |W[i,j]| &lt; threshold to 0.
    /// Also zeros out the diagonal.
    /// </summary>
    protected Matrix<T> ThresholdAndClean(Matrix<T> W, double threshold)
    {
        int d = W.Rows;
        var result = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (i != j && Math.Abs(NumOps.ToDouble(W[i, j])) >= threshold)
                    result[i, j] = W[i, j];

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
        var result = new Matrix<T>(n, d);
        T nT = NumOps.FromDouble(n);

        for (int j = 0; j < d; j++)
        {
            T mean = NumOps.Zero;
            for (int i = 0; i < n; i++)
                mean = NumOps.Add(mean, data[i, j]);
            mean = NumOps.Divide(mean, nT);

            T variance = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(data[i, j], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            variance = NumOps.Divide(variance, nT);
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-15)));

            // Standardize and add tiny column-specific perturbation to break exact collinearity.
            // Scale factor: (1 + 1e-4 * (j+1)) makes each column slightly different,
            // breaking symmetric saddle points without changing statistical properties.
            double perturbScale = 1.0 + 1e-4 * (j + 1);
            T perturbT = NumOps.FromDouble(perturbScale);
            for (int i = 0; i < n; i++)
                result[i, j] = NumOps.Multiply(NumOps.Divide(NumOps.Subtract(data[i, j], mean), std), perturbT);
        }

        return result;
    }

    /// <summary>
    /// Fallback: uses pairwise correlation to detect edges when continuous optimization
    /// fails to find structure (e.g., due to near-degenerate data).
    /// </summary>
    protected Matrix<T> FallbackCorrelationGraph(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        T nT = NumOps.FromDouble(n);
        var result = new Matrix<T>(d, d);

        // Compute column means
        var means = new T[d];
        for (int j = 0; j < d; j++)
        {
            means[j] = NumOps.Zero;
            for (int i = 0; i < n; i++)
                means[j] = NumOps.Add(means[j], data[i, j]);
            means[j] = NumOps.Divide(means[j], nT);
        }

        // Compute pairwise correlations
        for (int a = 0; a < d; a++)
        {
            for (int b = a + 1; b < d; b++)
            {
                T sxy = NumOps.Zero, sxx = NumOps.Zero, syy = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    T dx = NumOps.Subtract(data[i, a], means[a]);
                    T dy = NumOps.Subtract(data[i, b], means[b]);
                    sxy = NumOps.Add(sxy, NumOps.Multiply(dx, dy));
                    sxx = NumOps.Add(sxx, NumOps.Multiply(dx, dx));
                    syy = NumOps.Add(syy, NumOps.Multiply(dy, dy));
                }

                double sxxD = NumOps.ToDouble(sxx), syyD = NumOps.ToDouble(syy);
                if (sxxD > 1e-10 && syyD > 1e-10)
                {
                    double corr = Math.Abs(NumOps.ToDouble(sxy) / Math.Sqrt(sxxD * syyD));
                    if (corr >= WThreshold)
                        result[a, b] = NumOps.FromDouble(corr);
                }
            }
        }

        return result;
    }
}
