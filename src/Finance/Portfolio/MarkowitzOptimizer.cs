using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// Closed-form Markowitz mean-variance portfolio optimization: the global minimum-variance portfolio,
/// the tangency (maximum-Sharpe) portfolio, and the efficient-frontier target-return portfolio.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet's portfolio optimizers are <i>neural</i> (they learn weights from market data through a
/// trained network). The classic analytic solutions — the ones you can write down with a covariance
/// inverse and no training — were missing. This fills that gap with the standard unconstrained
/// (long/short-allowed, fully-invested) closed forms. Each is just a linear solve against the covariance
/// matrix, reusing AiDotNet's <see cref="Matrix{T}.Inverse"/> rather than hand-rolling elimination.
/// </para>
/// <para><b>For Beginners:</b> "Mean-variance" optimization picks how much to put in each asset by
/// trading off expected return against risk (variance), using how the assets move together (the
/// covariance matrix). The <i>minimum-variance</i> portfolio is the lowest-risk fully-invested mix and
/// ignores returns entirely. The <i>tangency</i> portfolio is the mix with the best risk-adjusted return
/// (highest Sharpe ratio) given a risk-free rate. <i>Target-return</i> finds the lowest-risk mix that
/// hits a return you specify. All three allow negative weights (short selling) and sum to 100%.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
public static class MarkowitzOptimizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Global minimum-variance portfolio: w = (Σ⁻¹·1) / (1ᵀ·Σ⁻¹·1). The fully-invested (weights sum to 1)
    /// mix with the lowest possible variance; expected returns are not used.
    /// </summary>
    /// <param name="covariance">The N×N asset return covariance matrix Σ (symmetric, positive-definite).</param>
    /// <returns>The N optimal weights, summing to 1.</returns>
    public static Vector<T> MinimumVariance(Matrix<T> covariance)
    {
        var n = ValidateSquare(covariance);
        var inv = covariance.Inverse();

        // z = Σ⁻¹·1  (the row sums of the inverse), then normalize so the weights sum to 1.
        var z = RowSums(inv, n);
        return Normalize(z);
    }

    /// <summary>
    /// Tangency (maximum-Sharpe) portfolio: w ∝ Σ⁻¹·(μ − rf·1), normalized to sum to 1. The fully-invested
    /// mix that maximizes the Sharpe ratio for the given risk-free rate.
    /// </summary>
    /// <param name="expectedReturns">The N expected asset returns μ.</param>
    /// <param name="covariance">The N×N asset return covariance matrix Σ.</param>
    /// <param name="riskFreeRate">The risk-free rate rf (in the same units as the returns).</param>
    /// <returns>The N optimal weights, summing to 1.</returns>
    public static Vector<T> Tangency(Vector<T> expectedReturns, Matrix<T> covariance, T riskFreeRate)
    {
        var n = ValidateSquare(covariance);
        if (expectedReturns is null)
        {
            throw new ArgumentNullException(nameof(expectedReturns));
        }

        if (expectedReturns.Length != n)
        {
            throw new ArgumentException(
                $"expectedReturns length ({expectedReturns.Length}) must match covariance dimension ({n}).",
                nameof(expectedReturns));
        }

        // excess = μ − rf·1
        var excess = new Vector<T>(n);
        for (var i = 0; i < n; i++)
        {
            excess[i] = NumOps.Subtract(expectedReturns[i], riskFreeRate);
        }

        // z = Σ⁻¹·(μ − rf·1), then normalize so the weights sum to 1.
        var z = covariance.Inverse().Multiply(excess);
        return Normalize(z);
    }

    /// <summary>
    /// Efficient-frontier portfolio for a given <paramref name="targetReturn"/>, via the standard
    /// two-fund (Merton) Lagrangian: the minimum-variance fully-invested portfolio whose expected return
    /// equals the target. Weights are a blend of two characteristic portfolios and sum to 1.
    /// </summary>
    /// <param name="expectedReturns">The N expected asset returns μ.</param>
    /// <param name="covariance">The N×N asset return covariance matrix Σ.</param>
    /// <param name="targetReturn">The desired portfolio expected return.</param>
    /// <returns>The N optimal weights, summing to 1.</returns>
    public static Vector<T> TargetReturn(Vector<T> expectedReturns, Matrix<T> covariance, T targetReturn)
    {
        var n = ValidateSquare(covariance);
        if (expectedReturns is null)
        {
            throw new ArgumentNullException(nameof(expectedReturns));
        }

        if (expectedReturns.Length != n)
        {
            throw new ArgumentException(
                $"expectedReturns length ({expectedReturns.Length}) must match covariance dimension ({n}).",
                nameof(expectedReturns));
        }

        var inv = covariance.Inverse();

        // Two characteristic portfolios: a = Σ⁻¹·1, b = Σ⁻¹·μ.
        var a = RowSums(inv, n);            // Σ⁻¹·1
        var b = inv.Multiply(expectedReturns); // Σ⁻¹·μ

        // Scalar invariants of the frontier (Merton's notation):
        //   A = 1ᵀΣ⁻¹μ,  B = μᵀΣ⁻¹μ,  C = 1ᵀΣ⁻¹1,  D = B·C − A².
        var capA = NumOps.Zero;
        var capB = NumOps.Zero;
        var capC = NumOps.Zero;
        for (var i = 0; i < n; i++)
        {
            capA = NumOps.Add(capA, b[i]);                               // sum of Σ⁻¹·μ  = 1ᵀΣ⁻¹μ
            capB = NumOps.Add(capB, NumOps.Multiply(expectedReturns[i], b[i])); // μᵀΣ⁻¹μ
            capC = NumOps.Add(capC, a[i]);                               // sum of Σ⁻¹·1  = 1ᵀΣ⁻¹1
        }

        var capD = NumOps.Subtract(NumOps.Multiply(capB, capC), NumOps.Multiply(capA, capA));
        if (!IsUsable(capD))
        {
            throw new InvalidOperationException(
                "Degenerate efficient frontier (B·C − A² is non-positive); expected returns may be collinear.");
        }

        // Lagrange multipliers for the equality-constrained QP:
        //   λ = (C·target − A) / D,  γ = (B − A·target) / D.
        // Weights:  w = λ·(Σ⁻¹·μ) + γ·(Σ⁻¹·1) = λ·b + γ·a.
        var lambda = NumOps.Divide(NumOps.Subtract(NumOps.Multiply(capC, targetReturn), capA), capD);
        var gamma = NumOps.Divide(NumOps.Subtract(capB, NumOps.Multiply(capA, targetReturn)), capD);

        var w = new Vector<T>(n);
        for (var i = 0; i < n; i++)
        {
            w[i] = NumOps.Add(NumOps.Multiply(lambda, b[i]), NumOps.Multiply(gamma, a[i]));
        }

        // By construction these already sum to 1; renormalize to clean up floating-point drift.
        return Normalize(w);
    }

    /// <summary>Σ⁻¹·1 — the row sums of the inverse covariance (equivalently Σ⁻¹ times the all-ones vector).</summary>
    private static Vector<T> RowSums(Matrix<T> inverse, int n)
    {
        var result = new Vector<T>(n);
        for (var i = 0; i < n; i++)
        {
            var sum = NumOps.Zero;
            for (var j = 0; j < n; j++)
            {
                sum = NumOps.Add(sum, inverse[i, j]);
            }

            result[i] = sum;
        }

        return result;
    }

    /// <summary>Scales a weight vector so its entries sum to 1 (fully invested).</summary>
    private static Vector<T> Normalize(Vector<T> weights)
    {
        var total = NumOps.Zero;
        for (var i = 0; i < weights.Length; i++)
        {
            total = NumOps.Add(total, weights[i]);
        }

        if (!IsUsable(NumOps.Abs(total)))
        {
            throw new InvalidOperationException(
                "Portfolio weights sum to (near) zero; the covariance matrix may be singular or ill-conditioned.");
        }

        var result = new Vector<T>(weights.Length);
        for (var i = 0; i < weights.Length; i++)
        {
            result[i] = NumOps.Divide(weights[i], total);
        }

        return result;
    }

    /// <summary>Validates that the covariance is square and non-empty; returns its dimension N.</summary>
    private static int ValidateSquare(Matrix<T> covariance)
    {
        if (covariance is null)
        {
            throw new ArgumentNullException(nameof(covariance));
        }

        if (covariance.Rows == 0 || covariance.Rows != covariance.Columns)
        {
            throw new ArgumentException(
                $"Covariance must be a non-empty square matrix; got {covariance.Rows}x{covariance.Columns}.",
                nameof(covariance));
        }

        return covariance.Rows;
    }

    /// <summary>True if a value is strictly positive and finite (net471-safe: no double.IsFinite).</summary>
    private static bool IsUsable(T value)
    {
        if (!NumOps.GreaterThan(value, NumOps.Zero))
        {
            return false;
        }

        var d = NumOps.ToDouble(value);
        return !double.IsNaN(d) && !double.IsInfinity(d);
    }
}
