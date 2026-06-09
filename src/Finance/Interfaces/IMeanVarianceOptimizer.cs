using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// A closed-form mean-variance portfolio optimizer: the analytic (training-free) weight solutions.
/// </summary>
/// <remarks>
/// <para>
/// This is a customization point, not a trainable model — unlike <see cref="IPortfolioOptimizer{T}"/>
/// (the neural, data-fit family), this exposes the classic closed-form Markowitz solutions a portfolio
/// model can use directly as a baseline or warm start. Consumers default to
/// <see cref="AiDotNet.Finance.Portfolio.MarkowitzOptimizer{T}"/> but can substitute their own analytic
/// solver (shrinkage covariance, constrained QP, …).
/// </para>
/// <para><b>For Beginners:</b> "Mean-variance" optimization divides money across assets by trading off
/// expected return against risk. These are the textbook closed-form answers; the default implementation
/// is the standard Markowitz solver.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
public interface IMeanVarianceOptimizer<T>
{
    /// <summary>Global minimum-variance fully-invested weights (expected returns ignored).</summary>
    Vector<T> MinimumVariance(Matrix<T> covariance);

    /// <summary>Tangency (maximum-Sharpe) fully-invested weights for the given risk-free rate.</summary>
    Vector<T> Tangency(Vector<T> expectedReturns, Matrix<T> covariance, T riskFreeRate);

    /// <summary>Minimum-variance fully-invested weights achieving a target expected return.</summary>
    Vector<T> TargetReturn(Vector<T> expectedReturns, Matrix<T> covariance, T targetReturn);
}
