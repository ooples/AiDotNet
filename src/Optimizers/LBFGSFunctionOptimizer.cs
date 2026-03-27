using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// L-BFGS (Limited-memory BFGS) optimizer for minimizing a scalar function of a vector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Implements the L-BFGS two-loop recursion (Nocedal and Wright, "Numerical Optimization",
/// Algorithm 7.4) with backtracking line search using the Armijo condition.</para>
///
/// <para><b>For Beginners:</b> L-BFGS is a fast optimizer that uses information from recent
/// iterations to approximate the curvature of the objective function. This lets it take
/// much better steps than simple gradient descent, converging in far fewer iterations.</para>
///
/// <para>This implementation operates on generic <see cref="Vector{T}"/> using
/// <see cref="INumericOperations{T}"/> for all arithmetic, making it work with any
/// numeric type (float, double, decimal, etc.).</para>
/// </remarks>
public class LBFGSFunctionOptimizer<T> : IFunctionOptimizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _memorySize;
    private readonly int _maxLineSearchSteps;

    /// <summary>
    /// Creates a new L-BFGS function optimizer.
    /// </summary>
    /// <param name="memorySize">Number of recent iterations to store for Hessian approximation (default: 10).</param>
    /// <param name="maxLineSearchSteps">Maximum backtracking steps in line search (default: 20).</param>
    public LBFGSFunctionOptimizer(int memorySize = 10, int maxLineSearchSteps = 20)
    {
        _memorySize = memorySize;
        _maxLineSearchSteps = maxLineSearchSteps;
    }

    /// <inheritdoc/>
    public Vector<T> Minimize(
        Vector<T> initialParameters,
        Func<Vector<T>, (T objective, Vector<T> gradient)> objectiveAndGradient,
        int maxIterations,
        T tolerance)
    {
        int n = initialParameters.Length;
        var param = new Vector<T>(n);
        for (int i = 0; i < n; i++) param[i] = initialParameters[i];

        var sVectors = new List<Vector<T>>();
        var yVectors = new List<Vector<T>>();
        Vector<T>? prevGrad = null;
        Vector<T>? prevParam = null;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var (objective, gradient) = objectiveAndGradient(param);

            // Check convergence: ||gradient||_inf < tolerance
            T maxGrad = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T absG = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(gradient[i])));
                if (NumOps.GreaterThan(absG, maxGrad))
                    maxGrad = absG;
            }

            if (!NumOps.GreaterThan(maxGrad, tolerance))
                break;

            // Compute L-BFGS search direction
            var direction = TwoLoopRecursion(gradient, sVectors, yVectors, n);

            // Backtracking line search with Armijo condition
            T step = NumOps.One;
            T c1 = NumOps.FromDouble(1e-4);

            T directionalDeriv = NumOps.Zero;
            for (int i = 0; i < n; i++)
                directionalDeriv = NumOps.Add(directionalDeriv, NumOps.Multiply(gradient[i], direction[i]));

            bool stepAccepted = false;
            for (int ls = 0; ls < _maxLineSearchSteps; ls++)
            {
                var trial = new Vector<T>(n);
                for (int i = 0; i < n; i++)
                    trial[i] = NumOps.Add(param[i], NumOps.Multiply(step, direction[i]));

                var (trialObj, _) = objectiveAndGradient(trial);

                T armijo = NumOps.Add(objective, NumOps.Multiply(c1, NumOps.Multiply(step, directionalDeriv)));

                if (!NumOps.GreaterThan(trialObj, armijo))
                {
                    // Update L-BFGS memory
                    if (prevParam is not null && prevGrad is not null)
                    {
                        var s = new Vector<T>(n);
                        var (_, newGrad) = objectiveAndGradient(trial);
                        var y = new Vector<T>(n);
                        T sy = NumOps.Zero;

                        for (int i = 0; i < n; i++)
                        {
                            s[i] = NumOps.Subtract(trial[i], prevParam[i]);
                            y[i] = NumOps.Subtract(newGrad[i], prevGrad[i]);
                            sy = NumOps.Add(sy, NumOps.Multiply(s[i], y[i]));
                        }

                        if (NumOps.GreaterThan(sy, NumOps.FromDouble(1e-10)))
                        {
                            sVectors.Add(s);
                            yVectors.Add(y);
                            if (sVectors.Count > _memorySize)
                            {
                                sVectors.RemoveAt(0);
                                yVectors.RemoveAt(0);
                            }
                        }
                    }

                    prevParam = new Vector<T>(n);
                    for (int i = 0; i < n; i++) prevParam[i] = param[i];
                    prevGrad = gradient;

                    param = trial;
                    stepAccepted = true;
                    break;
                }

                step = NumOps.Multiply(step, NumOps.FromDouble(0.5));
            }

            if (!stepAccepted)
            {
                // Line search failed — take tiny step in descent direction
                T fallbackStep = NumOps.FromDouble(1e-4);
                for (int i = 0; i < n; i++)
                    param[i] = NumOps.Add(param[i], NumOps.Multiply(fallbackStep, direction[i]));
            }
        }

        return param;
    }

    /// <summary>
    /// L-BFGS two-loop recursion to compute search direction.
    /// Per Nocedal and Wright, "Numerical Optimization" Algorithm 7.4.
    /// </summary>
    private Vector<T> TwoLoopRecursion(Vector<T> gradient, List<Vector<T>> sVectors, List<Vector<T>> yVectors, int n)
    {
        int m = sVectors.Count;
        var q = new Vector<T>(n);
        for (int i = 0; i < n; i++) q[i] = gradient[i];

        if (m == 0)
        {
            // Steepest descent (negated)
            for (int i = 0; i < n; i++) q[i] = NumOps.Negate(q[i]);
            return q;
        }

        var alphas = new T[m];
        var rhos = new T[m];

        for (int k = 0; k < m; k++)
        {
            T dot = NumOps.Zero;
            for (int j = 0; j < n; j++)
                dot = NumOps.Add(dot, NumOps.Multiply(sVectors[k][j], yVectors[k][j]));
            rhos[k] = NumOps.GreaterThan(dot, NumOps.FromDouble(1e-10))
                ? NumOps.Divide(NumOps.One, dot)
                : NumOps.Zero;
        }

        // Backward pass
        for (int k = m - 1; k >= 0; k--)
        {
            T dot = NumOps.Zero;
            for (int j = 0; j < n; j++)
                dot = NumOps.Add(dot, NumOps.Multiply(sVectors[k][j], q[j]));
            alphas[k] = NumOps.Multiply(rhos[k], dot);
            for (int j = 0; j < n; j++)
                q[j] = NumOps.Subtract(q[j], NumOps.Multiply(alphas[k], yVectors[k][j]));
        }

        // Initial Hessian approximation: H0 = (s^T y) / (y^T y) * I
        T sTy = NumOps.Zero;
        T yTy = NumOps.Zero;
        for (int j = 0; j < n; j++)
        {
            sTy = NumOps.Add(sTy, NumOps.Multiply(sVectors[m - 1][j], yVectors[m - 1][j]));
            yTy = NumOps.Add(yTy, NumOps.Multiply(yVectors[m - 1][j], yVectors[m - 1][j]));
        }
        T gamma = NumOps.GreaterThan(yTy, NumOps.FromDouble(1e-10))
            ? NumOps.Divide(sTy, yTy) : NumOps.One;
        for (int j = 0; j < n; j++)
            q[j] = NumOps.Multiply(q[j], gamma);

        // Forward pass
        for (int k = 0; k < m; k++)
        {
            T dot = NumOps.Zero;
            for (int j = 0; j < n; j++)
                dot = NumOps.Add(dot, NumOps.Multiply(yVectors[k][j], q[j]));
            T beta = NumOps.Multiply(rhos[k], dot);
            for (int j = 0; j < n; j++)
                q[j] = NumOps.Add(q[j], NumOps.Multiply(NumOps.Subtract(alphas[k], beta), sVectors[k][j]));
        }

        // Negate for descent direction
        for (int j = 0; j < n; j++)
            q[j] = NumOps.Negate(q[j]);

        return q;
    }
}
