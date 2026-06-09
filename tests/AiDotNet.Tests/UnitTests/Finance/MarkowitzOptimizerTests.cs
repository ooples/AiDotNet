using System;
using AiDotNet.Finance.Portfolio;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>Verifies closed-form Markowitz mean-variance optimization against hand-computed values.</summary>
public class MarkowitzOptimizerTests
{
    private static double Sum(Vector<double> v)
    {
        var s = 0.0;
        for (var i = 0; i < v.Length; i++)
        {
            s += v[i];
        }

        return s;
    }

    [Fact]
    public void MinimumVariance_weights_sum_to_one()
    {
        var cov = new Matrix<double>(new[,]
        {
            { 0.04, 0.01, 0.00 },
            { 0.01, 0.09, 0.02 },
            { 0.00, 0.02, 0.16 },
        });

        var w = MarkowitzOptimizer<double>.MinimumVariance(cov);

        Assert.Equal(1.0, Sum(w), 9);
    }

    [Fact]
    public void MinimumVariance_diagonal_cov_is_inverse_variance_weighted()
    {
        // For a diagonal Σ, w_i ∝ 1/σ_i². Variances 0.01, 0.04, 0.25 → inverses 100, 25, 4 (sum 129).
        var cov = new Matrix<double>(new[,]
        {
            { 0.01, 0.00, 0.00 },
            { 0.00, 0.04, 0.00 },
            { 0.00, 0.00, 0.25 },
        });

        var w = MarkowitzOptimizer<double>.MinimumVariance(cov);

        Assert.Equal(100.0 / 129.0, w[0], 9);
        Assert.Equal(25.0 / 129.0, w[1], 9);
        Assert.Equal(4.0 / 129.0, w[2], 9);
        Assert.Equal(1.0, Sum(w), 9);
    }

    [Fact]
    public void MinimumVariance_two_asset_known_case()
    {
        // Uncorrelated 2-asset: σ₁²=0.02, σ₂²=0.08 → w ∝ (1/0.02, 1/0.08) = (50, 12.5), sum 62.5.
        // w = (0.8, 0.2).
        var cov = new Matrix<double>(new[,]
        {
            { 0.02, 0.00 },
            { 0.00, 0.08 },
        });

        var w = MarkowitzOptimizer<double>.MinimumVariance(cov);

        Assert.Equal(0.8, w[0], 9);
        Assert.Equal(0.2, w[1], 9);
    }

    [Fact]
    public void MinimumVariance_correlated_two_asset_closed_form()
    {
        // Two correlated assets: w₁ = (σ₂² − σ₁₂) / (σ₁² + σ₂² − 2σ₁₂).
        // σ₁²=0.04, σ₂²=0.09, σ₁₂=0.012 → w₁ = (0.09 − 0.012)/(0.04+0.09−0.024) = 0.078/0.106.
        var cov = new Matrix<double>(new[,]
        {
            { 0.04, 0.012 },
            { 0.012, 0.09 },
        });

        var w = MarkowitzOptimizer<double>.MinimumVariance(cov);

        var expectedW1 = 0.078 / 0.106;
        Assert.Equal(expectedW1, w[0], 9);
        Assert.Equal(1.0 - expectedW1, w[1], 9);
    }

    [Fact]
    public void Tangency_weights_sum_to_one()
    {
        var cov = new Matrix<double>(new[,]
        {
            { 0.04, 0.01, 0.00 },
            { 0.01, 0.09, 0.02 },
            { 0.00, 0.02, 0.16 },
        });
        var mu = new Vector<double>(new[] { 0.10, 0.12, 0.14 });

        var w = MarkowitzOptimizer<double>.Tangency(mu, cov, 0.02);

        Assert.Equal(1.0, Sum(w), 9);
    }

    [Fact]
    public void Tangency_diagonal_cov_known_case()
    {
        // Diagonal Σ → w_i ∝ (μ_i − rf)/σ_i².
        // excess (μ−rf): 0.08, 0.06; variances 0.01, 0.04 → raw 8, 1.5; sum 9.5.
        var cov = new Matrix<double>(new[,]
        {
            { 0.01, 0.00 },
            { 0.00, 0.04 },
        });
        var mu = new Vector<double>(new[] { 0.10, 0.08 });

        var w = MarkowitzOptimizer<double>.Tangency(mu, cov, 0.02);

        Assert.Equal(8.0 / 9.5, w[0], 9);
        Assert.Equal(1.5 / 9.5, w[1], 9);
    }

    [Fact]
    public void TargetReturn_hits_target_and_sums_to_one()
    {
        var cov = new Matrix<double>(new[,]
        {
            { 0.04, 0.01, 0.00 },
            { 0.01, 0.09, 0.02 },
            { 0.00, 0.02, 0.16 },
        });
        var mu = new Vector<double>(new[] { 0.10, 0.12, 0.14 });

        var w = MarkowitzOptimizer<double>.TargetReturn(mu, cov, 0.13);

        // Weights sum to 1...
        Assert.Equal(1.0, Sum(w), 9);

        // ...and the realized portfolio return equals the target.
        var realized = 0.0;
        for (var i = 0; i < w.Length; i++)
        {
            realized += w[i] * mu[i];
        }

        Assert.Equal(0.13, realized, 9);
    }

    [Fact]
    public void TargetReturn_at_minvar_return_matches_minimum_variance()
    {
        var cov = new Matrix<double>(new[,]
        {
            { 0.04, 0.01 },
            { 0.01, 0.09 },
        });
        var mu = new Vector<double>(new[] { 0.10, 0.15 });

        var mvw = MarkowitzOptimizer<double>.MinimumVariance(cov);
        var minVarReturn = mvw[0] * mu[0] + mvw[1] * mu[1];

        // Targeting the min-variance portfolio's own return reproduces that portfolio.
        var w = MarkowitzOptimizer<double>.TargetReturn(mu, cov, minVarReturn);

        Assert.Equal(mvw[0], w[0], 9);
        Assert.Equal(mvw[1], w[1], 9);
    }

    [Fact]
    public void MinimumVariance_rejects_non_square()
    {
        var cov = new Matrix<double>(2, 3);
        Assert.Throws<ArgumentException>(() => MarkowitzOptimizer<double>.MinimumVariance(cov));
    }

    [Fact]
    public void MinimumVariance_rejects_null_covariance()
    {
        Assert.Throws<ArgumentNullException>(() => MarkowitzOptimizer<double>.MinimumVariance(null!));
    }

    [Fact]
    public void Tangency_rejects_null_arguments()
    {
        var cov = new Matrix<double>(new[,] { { 0.04, 0.01 }, { 0.01, 0.09 } });
        var mu = new Vector<double>(new[] { 0.10, 0.15 });
        Assert.Throws<ArgumentNullException>(() => MarkowitzOptimizer<double>.Tangency(mu, null!, 0.02));
        Assert.Throws<ArgumentNullException>(() => MarkowitzOptimizer<double>.Tangency(null!, cov, 0.02));
    }

    [Fact]
    public void Tangency_rejects_dimension_mismatch()
    {
        var cov = new Matrix<double>(new[,] { { 0.04, 0.01 }, { 0.01, 0.09 } });
        var mu = new Vector<double>(new[] { 0.10, 0.15, 0.20 }); // length 3 vs 2x2 covariance
        Assert.Throws<ArgumentException>(() => MarkowitzOptimizer<double>.Tangency(mu, cov, 0.02));
    }

    [Fact]
    public void TargetReturn_rejects_null_arguments()
    {
        var cov = new Matrix<double>(new[,] { { 0.04, 0.01 }, { 0.01, 0.09 } });
        var mu = new Vector<double>(new[] { 0.10, 0.15 });
        Assert.Throws<ArgumentNullException>(() => MarkowitzOptimizer<double>.TargetReturn(mu, null!, 0.12));
        Assert.Throws<ArgumentNullException>(() => MarkowitzOptimizer<double>.TargetReturn(null!, cov, 0.12));
    }
}
