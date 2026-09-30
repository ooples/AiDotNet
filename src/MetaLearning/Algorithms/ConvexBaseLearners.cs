using System;
using System.Collections.Generic;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// The convex base learners of MetaOptNet (Lee et al. 2019), solved numerically in double precision, each returning
/// the constant matrix that turns its solution into a differentiable one.
/// </summary>
/// <remarks>
/// <para>
/// A base learner solves <c>theta* = argmin L(theta; Z, Y)</c> for the support embeddings <c>Z</c>. The meta-gradient
/// needs <c>d theta* / d Z</c>, which the paper obtains by the implicit function theorem on the KKT conditions
/// (section 3.2) rather than by differentiating the solver's iterations. Each solver here therefore returns its
/// solution together with the inverse of the KKT Jacobian at that solution; the caller then writes
/// <c>theta = theta* - Jinv * F(theta*, Z)</c> with the residual <c>F</c> built from engine ops. That expression
/// equals <c>theta*</c> (the residual is zero at the optimum) and differentiates to the implicit gradient, so the
/// tape carries one multiplication by a constant instead of a solver.
/// </para>
/// <para>All arithmetic here is plain double precision: it is a numerical solve, not part of any tape.</para>
/// </remarks>
internal static class ConvexBaseLearners
{
    /// <summary>Solves <c>A X = B</c> for a general square <c>A</c> by LU with partial pivoting.</summary>
    /// <exception cref="ArgumentException">The matrix is singular to working precision.</exception>
    internal static double[,] Solve(double[,] a, double[,] b)
    {
        int n = a.GetLength(0), m = b.GetLength(1);
        var lu = (double[,])a.Clone();
        var x = (double[,])b.Clone();
        var pivot = new int[n];
        for (int i = 0; i < n; i++) pivot[i] = i;

        for (int k = 0; k < n; k++)
        {
            int best = k;
            double biggest = Math.Abs(lu[k, k]);
            for (int i = k + 1; i < n; i++)
            {
                double candidate = Math.Abs(lu[i, k]);
                if (candidate > biggest) { biggest = candidate; best = i; }
            }

            if (biggest < 1e-300)
            {
                throw new ArgumentException("The base learner's system is singular to working precision.", nameof(a));
            }

            if (best != k)
            {
                for (int j = 0; j < n; j++) (lu[k, j], lu[best, j]) = (lu[best, j], lu[k, j]);
                for (int j = 0; j < m; j++) (x[k, j], x[best, j]) = (x[best, j], x[k, j]);
            }

            for (int i = k + 1; i < n; i++)
            {
                double factor = lu[i, k] / lu[k, k];
                if (factor == 0) continue;
                lu[i, k] = factor;
                for (int j = k + 1; j < n; j++) lu[i, j] -= factor * lu[k, j];
                for (int j = 0; j < m; j++) x[i, j] -= factor * x[k, j];
            }
        }

        for (int k = n - 1; k >= 0; k--)
        {
            for (int j = 0; j < m; j++)
            {
                double sum = x[k, j];
                for (int i = k + 1; i < n; i++) sum -= lu[k, i] * x[i, j];
                x[k, j] = sum / lu[k, k];
            }
        }

        return x;
    }

    /// <summary>The inverse of a square matrix, by solving against the identity.</summary>
    internal static double[,] Invert(double[,] a)
    {
        int n = a.GetLength(0);
        var identity = new double[n, n];
        for (int i = 0; i < n; i++) identity[i, i] = 1.0;
        return Solve(a, identity);
    }

    /// <summary>
    /// Ridge regression in the dual (the reference implementation's head): <c>M = (K + lambda I)^-1 Y</c>, from which
    /// the classifier is <c>W = Z' M</c> and the query logits are <c>K_query M</c>.
    /// </summary>
    /// <param name="kernel">The support Gram matrix <c>K = Z Z'</c>, <c>[n, n]</c>.</param>
    /// <param name="oneHot">The one-hot support labels, <c>[n, classes]</c>.</param>
    /// <param name="lambda">The ridge regularization.</param>
    /// <returns>
    /// The dual coefficients and the inverse of <c>K + lambda I</c>, the KKT Jacobian of the residual
    /// <c>(K + lambda I) M - Y</c>.
    /// </returns>
    internal static (double[,] Solution, double[,] JacobianInverse) SolveRidgeDual(double[,] kernel, double[,] oneHot, double lambda)
    {
        int n = kernel.GetLength(0);
        var system = (double[,])kernel.Clone();
        for (int i = 0; i < n; i++) system[i, i] += lambda;
        var inverse = Invert(system);
        int classes = oneHot.GetLength(1);
        var solution = new double[n, classes];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < classes; c++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++) sum += inverse[i, j] * oneHot[j, c];
                solution[i, c] = sum;
            }

        return (solution, inverse);
    }

    /// <summary>
    /// The Crammer and Singer multi-class SVM dual (eq. 10), solved by coordinate descent over examples: each step
    /// re-optimises one example's block <c>alpha_n</c> exactly, which is the classical solver for this dual.
    /// </summary>
    /// <param name="kernel">The support Gram matrix, <c>[n, n]</c>.</param>
    /// <param name="labels">Each support example's class column.</param>
    /// <param name="classes">Number of classes in the episode.</param>
    /// <param name="cost">The regularization parameter C.</param>
    /// <param name="maxIterations">Sweeps over the support set.</param>
    /// <param name="tolerance">Stops when no block moves by more than this.</param>
    /// <remarks>
    /// The dual constrains each example's block by <c>alpha_nk &lt;= C [k = y_n]</c> and <c>sum_k alpha_nk = 0</c>.
    /// With a fixed example's block the objective is a separable quadratic over that simplex-like set, whose exact
    /// minimiser follows from a one-dimensional multiplier found by bisection on the equality constraint.
    /// </remarks>
    internal static double[,] SolveCrammerSingerDual(
        double[,] kernel, int[] labels, int classes, double cost, int maxIterations, double tolerance)
    {
        int n = labels.Length;
        var alpha = new double[n, classes];
        var gradient = new double[n, classes]; // d objective / d alpha_nk = (K alpha)_nk - [k = y_n]

        for (int i = 0; i < n; i++)
            for (int k = 0; k < classes; k++) gradient[i, k] = k == labels[i] ? -1.0 : 0.0;

        for (int sweep = 0; sweep < maxIterations; sweep++)
        {
            double largestMove = 0;
            for (int i = 0; i < n; i++)
            {
                double diagonal = kernel[i, i];
                if (diagonal < 1e-12) continue;

                // Minimise 0.5 * d_ii * ||a||^2 + (g_i - d_ii * alpha_i) . a over the block's constraint set.
                var linear = new double[classes];
                for (int k = 0; k < classes; k++) linear[k] = gradient[i, k] - diagonal * alpha[i, k];

                var updated = ProjectBlock(linear, diagonal, labels[i], classes, cost);
                for (int k = 0; k < classes; k++)
                {
                    double delta = updated[k] - alpha[i, k];
                    if (delta == 0) continue;
                    largestMove = Math.Max(largestMove, Math.Abs(delta));
                    alpha[i, k] = updated[k];
                    for (int j = 0; j < n; j++) gradient[j, k] += kernel[j, i] * delta;
                }
            }

            if (largestMove < tolerance) break;
        }

        return alpha;
    }

    /// <summary>
    /// The exact minimiser of <c>0.5 * d ||a||^2 + linear . a</c> over <c>a_k &lt;= C [k = label]</c>,
    /// <c>sum_k a_k = 0</c>: <c>a_k = min(upper_k, -(linear_k + nu) / d)</c> with the multiplier <c>nu</c> found by
    /// bisection on the sum.
    /// </summary>
    private static double[] ProjectBlock(double[] linear, double diagonal, int label, int classes, double cost)
    {
        double Sum(double nu)
        {
            double total = 0;
            for (int k = 0; k < classes; k++)
            {
                double upper = k == label ? cost : 0.0;
                total += Math.Min(upper, -(linear[k] + nu) / diagonal);
            }

            return total;
        }

        // The sum is continuous and non-increasing in nu; bracket the root, then bisect.
        double low = -1.0, high = 1.0;
        for (int i = 0; i < 200 && Sum(low) < 0; i++) low *= 2.0;
        for (int i = 0; i < 200 && Sum(high) > 0; i++) high *= 2.0;
        for (int i = 0; i < 200; i++)
        {
            double middle = 0.5 * (low + high);
            if (Sum(middle) > 0) low = middle; else high = middle;
        }

        double nuStar = 0.5 * (low + high);
        var block = new double[classes];
        for (int k = 0; k < classes; k++)
        {
            double upper = k == label ? cost : 0.0;
            block[k] = Math.Min(upper, -(linear[k] + nuStar) / diagonal);
        }

        return block;
    }

    /// <summary>
    /// The inverse KKT Jacobian of the Crammer and Singer dual at <paramref name="alpha"/>, over the free variables:
    /// the constraints active at the solution (a block entry at its upper bound) are held fixed, and each example's
    /// equality constraint is kept, which is the active-set form of the implicit function theorem.
    /// </summary>
    /// <returns>
    /// <c>[n * classes, n * classes]</c>: the inverse KKT Jacobian over the free variables, so that the caller's
    /// <c>alpha* - Jinv F</c> carries the implicit derivative - the same convention as the ridge and logistic
    /// solvers, which return the inverse of their own Jacobian. Rows and columns of variables pinned by an active
    /// bound are zero: a pinned variable does not move.
    /// </returns>
    internal static double[,] CrammerSingerJacobianInverse(
        double[,] kernel, double[,] alpha, int[] labels, int classes, double cost, double activeTolerance)
    {
        int n = labels.Length, size = n * classes;
        var free = new bool[size];
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < classes; k++)
            {
                double upper = k == labels[i] ? cost : 0.0;
                free[i * classes + k] = upper - alpha[i, k] > activeTolerance;
            }
        }

        // [Q_ff  A_f'] [dalpha_f]   [-dr_f]
        // [A_f   0   ] [dnu     ] = [ 0   ]  with Q the block-diagonal kernel and A the per-example sums.
        var freeIndices = new List<int>();
        for (int i = 0; i < size; i++) if (free[i]) freeIndices.Add(i);
        int f = freeIndices.Count;
        var system = new double[f + n, f + n];
        for (int a = 0; a < f; a++)
        {
            int rowExample = freeIndices[a] / classes, rowClass = freeIndices[a] % classes;
            for (int b = 0; b < f; b++)
            {
                int columnExample = freeIndices[b] / classes, columnClass = freeIndices[b] % classes;
                if (rowClass == columnClass) system[a, b] = kernel[rowExample, columnExample];
            }

            system[a, f + rowExample] = 1.0;
            system[f + rowExample, a] = 1.0;
        }

        // A pinned example (every entry at a bound) leaves its multiplier unconstrained; anchor it.
        for (int i = 0; i < n; i++)
        {
            bool anyFree = false;
            for (int k = 0; k < classes && !anyFree; k++) anyFree = free[i * classes + k];
            if (!anyFree) system[f + i, f + i] = 1.0;
        }

        var rightHand = new double[f + n, f];
        for (int a = 0; a < f; a++) rightHand[a, a] = 1.0;
        var solved = Solve(system, rightHand);

        var inverse = new double[size, size];
        for (int a = 0; a < f; a++)
            for (int b = 0; b < f; b++) inverse[freeIndices[a], freeIndices[b]] = solved[a, b];
        return inverse;
    }

    /// <summary>
    /// Multi-class logistic regression with L2 regularization, by Newton's method on the primal weights.
    /// </summary>
    /// <param name="features">The support embeddings, <c>[n, d]</c>.</param>
    /// <param name="oneHot">The one-hot support labels, <c>[n, classes]</c>.</param>
    /// <param name="lambda">The L2 regularization.</param>
    /// <param name="maxIterations">Newton steps.</param>
    /// <param name="tolerance">Stops when the largest weight change falls below this.</param>
    /// <returns>The weights <c>[classes, d]</c> and the inverse Hessian of the objective at them.</returns>
    internal static (double[,] Weights, double[,] HessianInverse) SolveLogistic(
        double[,] features, double[,] oneHot, double lambda, int maxIterations, double tolerance)
    {
        int n = features.GetLength(0), d = features.GetLength(1), classes = oneHot.GetLength(1);
        int size = classes * d;
        var weights = new double[classes, d];
        double[,] hessian = new double[size, size];

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            var probabilities = Probabilities(features, weights);
            var gradient = new double[size, 1];
            for (int c = 0; c < classes; c++)
                for (int j = 0; j < d; j++)
                {
                    double sum = lambda * weights[c, j];
                    for (int i = 0; i < n; i++) sum += (probabilities[i, c] - oneHot[i, c]) * features[i, j] / n;
                    gradient[c * d + j, 0] = sum;
                }

            hessian = LogisticHessian(features, probabilities, lambda, classes, d);
            var step = Solve(hessian, gradient);
            double largest = 0;
            for (int c = 0; c < classes; c++)
                for (int j = 0; j < d; j++)
                {
                    double delta = step[c * d + j, 0];
                    weights[c, j] -= delta;
                    largest = Math.Max(largest, Math.Abs(delta));
                }

            if (largest < tolerance) break;
        }

        hessian = LogisticHessian(features, Probabilities(features, weights), lambda, classes, d);
        return (weights, Invert(hessian));
    }

    private static double[,] Probabilities(double[,] features, double[,] weights)
    {
        int n = features.GetLength(0), d = features.GetLength(1), classes = weights.GetLength(0);
        var probabilities = new double[n, classes];
        for (int i = 0; i < n; i++)
        {
            double largest = double.NegativeInfinity;
            for (int c = 0; c < classes; c++)
            {
                double logit = 0;
                for (int j = 0; j < d; j++) logit += features[i, j] * weights[c, j];
                probabilities[i, c] = logit;
                largest = Math.Max(largest, logit);
            }

            double total = 0;
            for (int c = 0; c < classes; c++)
            {
                probabilities[i, c] = Math.Exp(probabilities[i, c] - largest);
                total += probabilities[i, c];
            }

            for (int c = 0; c < classes; c++) probabilities[i, c] /= total;
        }

        return probabilities;
    }

    private static double[,] LogisticHessian(double[,] features, double[,] probabilities, double lambda, int classes, int d)
    {
        int n = features.GetLength(0), size = classes * d;
        var hessian = new double[size, size];
        for (int c = 0; c < classes; c++)
            for (int e = 0; e < classes; e++)
            {
                for (int i = 0; i < n; i++)
                {
                    double weight = probabilities[i, c] * ((c == e ? 1.0 : 0.0) - probabilities[i, e]) / n;
                    if (weight == 0) continue;
                    for (int j = 0; j < d; j++)
                    {
                        double scaled = weight * features[i, j];
                        for (int l = 0; l < d; l++) hessian[c * d + j, e * d + l] += scaled * features[i, l];
                    }
                }
            }

        for (int i = 0; i < size; i++) hessian[i, i] += lambda;
        return hessian;
    }
}
