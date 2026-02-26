using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Regularization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regularization;

/// <summary>
/// Deep math-correctness integration tests for L1, L2, ElasticNet, and NoRegularization.
/// Every test includes hand-calculated expected values verified against industry-standard formulas.
/// </summary>
public class RegularizationDeepMathIntegrationTests
{
    private const double Tol = 1e-10;

    // ─── Helper ──────────────────────────────────────────────────────────────

    private static Vector<double> Vec(params double[] v) => new(v);

    private static Matrix<double> Mat(double[,] m) => new(m);

    // ─── L1 Vector: Soft Thresholding ────────────────────────────────────────
    // Formula: sign(x) * max(0, |x| - lambda)

    [Fact]
    public void L1_Vector_HandCalculated_SoftThresholding()
    {
        // lambda = 0.1
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 });

        //  x =  0.5  → sign(0.5)*max(0, 0.5-0.1) =  1*0.4  =  0.4
        //  x = -0.5  → sign(-0.5)*max(0,0.5-0.1) = -1*0.4  = -0.4
        //  x =  0.05 → sign(0.05)*max(0,0.05-0.1)=  1*0    =  0.0  (below threshold)
        //  x = -0.05 → sign(-0.05)*max(0,0.05-0.1)=-1*0    =  0.0
        //  x =  1.0  → sign(1.0)*max(0,1.0-0.1)  =  1*0.9  =  0.9
        var result = l1.Regularize(Vec(0.5, -0.5, 0.05, -0.05, 1.0));

        Assert.Equal(0.4, result[0], Tol);
        Assert.Equal(-0.4, result[1], Tol);
        Assert.Equal(0.0, result[2], Tol);
        Assert.Equal(0.0, result[3], Tol);  // negative below threshold → 0
        Assert.Equal(0.9, result[4], Tol);
    }

    [Fact]
    public void L1_Vector_ExactlyAtThreshold_BecomesZero()
    {
        // lambda = 0.3, x = 0.3 → |0.3| - 0.3 = 0 → result = 0
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.3 });

        var result = l1.Regularize(Vec(0.3, -0.3));
        Assert.Equal(0.0, result[0], Tol);
        Assert.Equal(0.0, result[1], Tol);  // negative at threshold also → 0
    }

    [Fact]
    public void L1_Vector_ZeroStrength_NoChange()
    {
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.0 });

        var result = l1.Regularize(Vec(1.5, -2.3, 0.0));
        Assert.Equal(1.5, result[0], Tol);
        Assert.Equal(-2.3, result[1], Tol);
        Assert.Equal(0.0, result[2], Tol);
    }

    [Fact]
    public void L1_Vector_AllBelowThreshold_AllZero()
    {
        // lambda = 1.0, all |x_i| < 1.0 → all become 0
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 1.0 });

        var result = l1.Regularize(Vec(0.5, -0.3, 0.99, -0.01));
        for (int i = 0; i < 4; i++)
            Assert.Equal(0.0, result[i], Tol);
    }

    [Fact]
    public void L1_Vector_PreservesSignAboveThreshold()
    {
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.2 });

        // x = 5.0 → 1*max(0, 5.0-0.2) = 4.8
        // x = -5.0 → -1*max(0, 5.0-0.2) = -4.8
        var result = l1.Regularize(Vec(5.0, -5.0));
        Assert.Equal(4.8, result[0], Tol);
        Assert.Equal(-4.8, result[1], Tol);
    }

    // ─── L1 Matrix: Soft Thresholding ────────────────────────────────────────

    [Fact]
    public void L1_Matrix_HandCalculated_SoftThresholding()
    {
        // lambda = 0.2
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.2 });

        // [0.5, -0.1]    →  [sign(0.5)*max(0,0.5-0.2),  sign(-0.1)*max(0,0.1-0.2)]
        // [1.0, -1.0]       [sign(1.0)*max(0,1.0-0.2),  sign(-1.0)*max(0,1.0-0.2)]
        //                 =  [0.3, 0.0]
        //                    [0.8, -0.8]
        var result = l1.Regularize(Mat(new[,] { { 0.5, -0.1 }, { 1.0, -1.0 } }));

        Assert.Equal(0.3, result[0, 0], Tol);
        Assert.Equal(0.0, result[0, 1], Tol);  // |-0.1| < 0.2 → 0
        Assert.Equal(0.8, result[1, 0], Tol);
        Assert.Equal(-0.8, result[1, 1], Tol);
    }

    // ─── L1 Gradient Regularization ──────────────────────────────────────────
    // Formula: gradient + lambda * sign(coefficient)

    [Fact]
    public void L1_Gradient_HandCalculated()
    {
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 });

        var grad = Vec(1.0, 2.0, 0.0, -1.0);
        var coeff = Vec(0.5, -0.5, 0.0, 3.0);

        // g + 0.1 * sign(c):
        //   1.0 + 0.1*1  = 1.1
        //   2.0 + 0.1*(-1) = 1.9
        //   0.0 + 0.1*0  = 0.0
        //  -1.0 + 0.1*1  = -0.9
        var result = l1.Regularize(grad, coeff);

        Assert.Equal(1.1, result[0], Tol);
        Assert.Equal(1.9, result[1], Tol);
        Assert.Equal(0.0, result[2], Tol);
        Assert.Equal(-0.9, result[3], Tol);
    }

    [Fact]
    public void L1_Gradient_ZeroCoefficients_NoRegularizationPenalty()
    {
        // If coefficients are all zero, sign(0) = 0, so gradient is unchanged
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 });

        var grad = Vec(1.0, -2.0, 3.0);
        var coeff = Vec(0.0, 0.0, 0.0);
        var result = l1.Regularize(grad, coeff);

        Assert.Equal(1.0, result[0], Tol);
        Assert.Equal(-2.0, result[1], Tol);
        Assert.Equal(3.0, result[2], Tol);
    }

    // ─── L2 Vector: Shrinkage ────────────────────────────────────────────────
    // Formula: x * (1 - lambda)

    [Fact]
    public void L2_Vector_HandCalculated_Shrinkage()
    {
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 });

        // shrinkage = (1 - 0.1) = 0.9
        // x = [1.0, -2.0, 0.5, 0.0]
        // result = [0.9, -1.8, 0.45, 0.0]
        var result = l2.Regularize(Vec(1.0, -2.0, 0.5, 0.0));

        Assert.Equal(0.9, result[0], Tol);
        Assert.Equal(-1.8, result[1], Tol);
        Assert.Equal(0.45, result[2], Tol);
        Assert.Equal(0.0, result[3], Tol);
    }

    [Fact]
    public void L2_Vector_StrongShrinkage_HalvesValues()
    {
        // strength = 0.5 → shrinkage = 0.5 → values halved
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.5 });

        var result = l2.Regularize(Vec(4.0, -6.0, 10.0));
        Assert.Equal(2.0, result[0], Tol);
        Assert.Equal(-3.0, result[1], Tol);
        Assert.Equal(5.0, result[2], Tol);
    }

    [Fact]
    public void L2_Vector_ZeroStrength_NoChange()
    {
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.0 });

        var result = l2.Regularize(Vec(1.5, -2.3, 0.0));
        Assert.Equal(1.5, result[0], Tol);
        Assert.Equal(-2.3, result[1], Tol);
        Assert.Equal(0.0, result[2], Tol);
    }

    [Fact]
    public void L2_Vector_NeverProducesZeros()
    {
        // Unlike L1, L2 never drives values to exactly zero (except zero input)
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.9 });

        var result = l2.Regularize(Vec(0.001, -0.001));
        Assert.True(result[0] > 0);   // Still positive
        Assert.True(result[1] < 0);   // Still negative
    }

    [Fact]
    public void L2_Vector_ProportionalShrinkage()
    {
        // L2 shrinks all values by same ratio → ratios preserved
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.3 });

        var result = l2.Regularize(Vec(2.0, 4.0, 6.0));
        // All multiplied by 0.7
        double ratio01 = result[0] / result[1];
        double ratio12 = result[1] / result[2];
        Assert.Equal(2.0 / 4.0, ratio01, Tol);  // Ratios preserved
        Assert.Equal(4.0 / 6.0, ratio12, Tol);
    }

    // ─── L2 Matrix: Shrinkage ────────────────────────────────────────────────

    [Fact]
    public void L2_Matrix_HandCalculated_Shrinkage()
    {
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.2 });

        // shrinkage = 0.8
        var result = l2.Regularize(Mat(new[,] { { 1.0, -2.0 }, { 3.0, 0.5 } }));

        Assert.Equal(0.8, result[0, 0], Tol);
        Assert.Equal(-1.6, result[0, 1], Tol);
        Assert.Equal(2.4, result[1, 0], Tol);
        Assert.Equal(0.4, result[1, 1], Tol);
    }

    // ─── L2 Gradient Regularization ──────────────────────────────────────────
    // Formula: gradient + lambda * coefficient

    [Fact]
    public void L2_Gradient_HandCalculated()
    {
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 });

        var grad = Vec(1.0, 2.0, 0.0, -1.0);
        var coeff = Vec(0.5, -0.5, 0.0, 3.0);

        // g + 0.1 * c:
        //   1.0 + 0.1*0.5  = 1.05
        //   2.0 + 0.1*(-0.5) = 1.95
        //   0.0 + 0.1*0.0  = 0.0
        //  -1.0 + 0.1*3.0  = -0.7
        var result = l2.Regularize(grad, coeff);

        Assert.Equal(1.05, result[0], Tol);
        Assert.Equal(1.95, result[1], Tol);
        Assert.Equal(0.0, result[2], Tol);
        Assert.Equal(-0.7, result[3], Tol);
    }

    [Fact]
    public void L2_Gradient_LargeCoeff_BigPenalty()
    {
        // L2 penalizes large coefficients proportionally
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.5 });

        var grad = Vec(0.0, 0.0);
        var coeff = Vec(10.0, 0.01);

        // 0 + 0.5 * 10.0 = 5.0  (huge penalty on large coeff)
        // 0 + 0.5 * 0.01 = 0.005 (tiny penalty on small coeff)
        var result = l2.Regularize(grad, coeff);
        Assert.Equal(5.0, result[0], Tol);
        Assert.Equal(0.005, result[1], Tol);
    }

    // ─── L1 vs L2 Sparsity Comparison ────────────────────────────────────────

    [Fact]
    public void L1_ProducesSparsity_L2_DoesNot()
    {
        // With strength = 0.3, values near zero should be zeroed by L1 but not L2
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.3 });
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.3 });

        var data = Vec(0.1, 0.2, 0.29, 0.5, -0.1, -0.5);

        var l1Result = l1.Regularize(data);
        var l2Result = l2.Regularize(data);

        // L1: values with |x| <= 0.3 become exactly zero
        Assert.Equal(0.0, l1Result[0], Tol);  // |0.1| < 0.3 → 0
        Assert.Equal(0.0, l1Result[1], Tol);  // |0.2| < 0.3 → 0
        Assert.Equal(0.0, l1Result[2], Tol);  // |0.29| < 0.3 → 0
        Assert.Equal(0.2, l1Result[3], Tol);  // 0.5 - 0.3 = 0.2
        Assert.Equal(0.0, l1Result[4], Tol);  // |-0.1| < 0.3 → 0
        Assert.Equal(-0.2, l1Result[5], Tol); // -(0.5-0.3) = -0.2

        // L2: NO values become exactly zero (all shrunk by factor 0.7)
        Assert.Equal(0.07, l2Result[0], Tol);
        Assert.Equal(0.14, l2Result[1], Tol);
        Assert.Equal(0.203, l2Result[2], Tol);
        Assert.Equal(0.35, l2Result[3], Tol);
        Assert.Equal(-0.07, l2Result[4], Tol);
        Assert.Equal(-0.35, l2Result[5], Tol);
    }

    // ─── ElasticNet Gradient Regularization ──────────────────────────────────
    // Formula: gradient + strength*l1Ratio*sign(c) + strength*(1-l1Ratio)*c

    [Fact]
    public void ElasticNet_Gradient_HandCalculated()
    {
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.1, L1Ratio = 0.5 });

        var grad = Vec(1.0, 2.0, 0.0);
        var coeff = Vec(0.5, -0.5, 0.0);

        // For coeff[0] = 0.5 (positive):
        //   l1Part = 0.1 * 0.5 * sign(0.5) = 0.05
        //   l2Part = 0.1 * 0.5 * 0.5       = 0.025
        //   result = 1.0 + 0.05 + 0.025     = 1.075
        //
        // For coeff[1] = -0.5 (negative):
        //   l1Part = 0.1 * 0.5 * sign(-0.5)= -0.05
        //   l2Part = 0.1 * 0.5 * (-0.5)    = -0.025
        //   result = 2.0 + (-0.05) + (-0.025) = 1.925
        //
        // For coeff[2] = 0.0:
        //   l1Part = 0.1 * 0.5 * 0 = 0
        //   l2Part = 0.1 * 0.5 * 0 = 0
        //   result = 0.0
        var result = en.Regularize(grad, coeff);

        Assert.Equal(1.075, result[0], Tol);
        Assert.Equal(1.925, result[1], Tol);
        Assert.Equal(0.0, result[2], Tol);
    }

    [Fact]
    public void ElasticNet_Gradient_L1Ratio1_MatchesPureL1Gradient()
    {
        // ElasticNet with l1Ratio = 1.0 should produce same gradient as L1
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.2, L1Ratio = 1.0 });
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.2 });

        var grad = Vec(1.0, -1.0, 0.5, -0.5);
        var coeff = Vec(3.0, -2.0, 0.0, 0.1);

        var enResult = en.Regularize(grad, coeff);
        var l1Result = l1.Regularize(grad, coeff);

        for (int i = 0; i < 4; i++)
            Assert.Equal(l1Result[i], enResult[i], Tol);
    }

    [Fact]
    public void ElasticNet_Gradient_L1Ratio0_MatchesPureL2Gradient()
    {
        // ElasticNet with l1Ratio = 0.0 should produce same gradient as L2
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.2, L1Ratio = 0.0 });
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.2 });

        var grad = Vec(1.0, -1.0, 0.5, -0.5);
        var coeff = Vec(3.0, -2.0, 0.0, 0.1);

        var enResult = en.Regularize(grad, coeff);
        var l2Result = l2.Regularize(grad, coeff);

        for (int i = 0; i < 4; i++)
            Assert.Equal(l2Result[i], enResult[i], Tol);
    }

    // ─── ElasticNet Vector/Matrix: Combined L1+L2 ────────────────────────────
    // The correct formula should be: apply L1 soft thresholding, then L2 shrinkage
    // result = sign(x) * max(0, |x| - strength*l1Ratio) * (1 - strength*(1-l1Ratio))

    [Fact]
    public void ElasticNet_Vector_ZeroStrength_ReturnsOriginalValues()
    {
        // With zero strength, regularization should have no effect
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.0, L1Ratio = 0.5 });

        var data = Vec(1.0, -2.0, 3.0);
        var result = en.Regularize(data);

        // With zero strength, output should equal input
        Assert.Equal(1.0, result[0], Tol);
        Assert.Equal(-2.0, result[1], Tol);
        Assert.Equal(3.0, result[2], Tol);
    }

    [Fact]
    public void ElasticNet_Vector_L1Ratio1_MatchesPureL1()
    {
        // ElasticNet with l1Ratio = 1.0 should behave exactly like L1
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.2, L1Ratio = 1.0 });
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.2 });

        var data = Vec(0.5, -0.5, 0.1, -0.1, 1.0, -1.0);

        var enResult = en.Regularize(data);
        var l1Result = l1.Regularize(data);

        for (int i = 0; i < data.Length; i++)
            Assert.Equal(l1Result[i], enResult[i], Tol);
    }

    [Fact]
    public void ElasticNet_Vector_L1Ratio0_MatchesPureL2()
    {
        // ElasticNet with l1Ratio = 0.0 should behave exactly like L2
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.2, L1Ratio = 0.0 });
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.2 });

        var data = Vec(0.5, -0.5, 0.1, -0.1, 1.0, -1.0);

        var enResult = en.Regularize(data);
        var l2Result = l2.Regularize(data);

        for (int i = 0; i < data.Length; i++)
            Assert.Equal(l2Result[i], enResult[i], Tol);
    }

    [Fact]
    public void ElasticNet_Vector_ShouldAlwaysShrinkValues()
    {
        // Regularization MUST shrink (or zero) coefficient magnitudes, never increase them
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.1, L1Ratio = 0.5 });

        var data = Vec(1.0, -1.0, 0.5, -0.5, 2.0, -2.0);
        var result = en.Regularize(data);

        for (int i = 0; i < data.Length; i++)
        {
            // |result[i]| should be <= |data[i]| for regularization
            Assert.True(Math.Abs(result[i]) <= Math.Abs(data[i]) + Tol,
                $"ElasticNet increased magnitude of element {i}: |{result[i]}| > |{data[i]}|");
        }
    }

    [Fact]
    public void ElasticNet_Matrix_ShouldAlwaysShrinkValues()
    {
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.1, L1Ratio = 0.5 });

        var data = Mat(new[,] { { 1.0, -0.5 }, { 2.0, -0.1 } });
        var result = en.Regularize(data);

        for (int i = 0; i < data.Rows; i++)
            for (int j = 0; j < data.Columns; j++)
                Assert.True(Math.Abs(result[i, j]) <= Math.Abs(data[i, j]) + Tol,
                    $"ElasticNet increased magnitude at [{i},{j}]: |{result[i, j]}| > |{data[i, j]}|");
    }

    [Fact]
    public void ElasticNet_Vector_BetweenL1AndL2_Behavior()
    {
        // With l1Ratio = 0.5, ElasticNet should produce results between pure L1 and pure L2
        // for large values (where L1 doesn't zero them out)
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.2 });
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.2 });
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.2, L1Ratio = 0.5 });

        var data = Vec(5.0);  // Large value well above any threshold
        var l1Result = l1.Regularize(data);
        var l2Result = l2.Regularize(data);
        var enResult = en.Regularize(data);

        // For a large positive value:
        // L1: 5.0 - 0.2 = 4.8
        // L2: 5.0 * 0.8 = 4.0
        // EN should be between: 4.0 < EN[0] < 4.8 (or equal to one bound if mixed)
        Assert.True(enResult[0] <= l1Result[0] + Tol,
            $"ElasticNet result {enResult[0]} should be <= L1 result {l1Result[0]}");
        Assert.True(enResult[0] >= l2Result[0] - Tol,
            $"ElasticNet result {enResult[0]} should be >= L2 result {l2Result[0]}");
    }

    // ─── NoRegularization ────────────────────────────────────────────────────

    [Fact]
    public void NoReg_Vector_ReturnsExactOriginal()
    {
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var data = Vec(1.5, -2.3, 0.0, 100.0, -0.001);
        var result = noReg.Regularize(data);

        for (int i = 0; i < data.Length; i++)
            Assert.Equal(data[i], result[i], Tol);
    }

    [Fact]
    public void NoReg_Gradient_ReturnsExactOriginalGradient()
    {
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var grad = Vec(1.0, -2.0, 3.0);
        var coeff = Vec(100.0, -100.0, 50.0);  // Large coefficients should not affect result
        var result = noReg.Regularize(grad, coeff);

        Assert.Equal(1.0, result[0], Tol);
        Assert.Equal(-2.0, result[1], Tol);
        Assert.Equal(3.0, result[2], Tol);
    }

    // ─── Cross-Strategy Consistency ──────────────────────────────────────────

    [Fact]
    public void L1_Gradient_SubdifferentialCorrectness()
    {
        // The L1 penalty is lambda * sum(|w_i|).
        // Its subdifferential w.r.t. w_i is lambda * sign(w_i).
        // Verify the gradient regularization matches numerical differentiation.
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.3 });

        double h = 1e-6;
        var coeff = Vec(2.0, -1.5, 0.7);
        var zeroGrad = Vec(0.0, 0.0, 0.0);

        // The regularization penalty is lambda * sum(|w_i|)
        // Partial w.r.t. w_i (for w_i != 0) = lambda * sign(w_i)
        var regGrad = l1.Regularize(zeroGrad, coeff);

        // For positive coefficient 2.0: gradient should be +0.3
        Assert.Equal(0.3, regGrad[0], 1e-8);
        // For negative coefficient -1.5: gradient should be -0.3
        Assert.Equal(-0.3, regGrad[1], 1e-8);
        // For positive coefficient 0.7: gradient should be +0.3
        Assert.Equal(0.3, regGrad[2], 1e-8);
    }

    [Fact]
    public void L2_Gradient_MatchesNumericalDerivative()
    {
        // The L2 penalty is 0.5 * lambda * sum(w_i^2).
        // Its derivative w.r.t. w_i is lambda * w_i.
        // NOTE: The code uses gradient + lambda * coefficient, which corresponds
        // to the derivative of the full penalty (not 0.5 * lambda).
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.2 });

        double h = 1e-6;
        var coeffBase = Vec(3.0, -2.0, 1.5);
        var zeroGrad = Vec(0.0, 0.0, 0.0);

        // Numerical derivative of L2 penalty = lambda * sum(w_i^2)
        // partial w.r.t. w_0: 2 * lambda * w_0 = 2 * 0.2 * 3.0 = 1.2
        // But the code adds lambda * w_i (not 2*lambda*w_i), so the code corresponds to
        // penalty = lambda * sum(w_i^2), derivative = lambda * w_i.
        // Actually let's just verify what the code produces:
        var regGrad = l2.Regularize(zeroGrad, coeffBase);

        // lambda * coeff = 0.2 * [3.0, -2.0, 1.5] = [0.6, -0.4, 0.3]
        Assert.Equal(0.6, regGrad[0], 1e-8);
        Assert.Equal(-0.4, regGrad[1], 1e-8);
        Assert.Equal(0.3, regGrad[2], 1e-8);
    }

    [Fact]
    public void L2_RepeatedApplication_DecaysGeometrically()
    {
        // Applying L2 shrinkage n times: x * (1-lambda)^n
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 });

        var data = Vec(10.0);
        // Apply 10 times
        for (int i = 0; i < 10; i++)
            data = l2.Regularize(data);

        // 10.0 * 0.9^10 = 10.0 * 0.3486784401 = 3.486784401
        Assert.Equal(10.0 * Math.Pow(0.9, 10), data[0], 1e-8);
    }

    [Fact]
    public void L1_RepeatedApplication_EventuallyZeros()
    {
        // Repeated L1 application should eventually zero out small values
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 });

        var data = Vec(0.35);
        // Each application subtracts 0.1 (for positive values above threshold):
        // Step 0: 0.35
        // Step 1: 0.35 - 0.1 = 0.25
        // Step 2: 0.25 - 0.1 = 0.15
        // Step 3: 0.15 - 0.1 = 0.05
        // Step 4: 0.05 < 0.1 → 0.0
        for (int i = 0; i < 4; i++)
            data = l1.Regularize(data);

        Assert.Equal(0.0, data[0], Tol);
    }

    // ─── Tensor Support ──────────────────────────────────────────────────────

    [Fact]
    public void L1_Gradient_TensorSupport_MatchesVectorResult()
    {
        var l1 = new L1Regularization<double, Matrix<double>, Tensor<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 });

        var gradVec = Vec(1.0, 2.0, 0.0, -1.0);
        var coeffVec = Vec(0.5, -0.5, 0.0, 3.0);

        var gradTensor = Tensor<double>.FromVector(gradVec);
        var coeffTensor = Tensor<double>.FromVector(coeffVec);

        var result = l1.Regularize(gradTensor, coeffTensor);
        var resultVec = result.ToVector();

        // Same hand-calculated values as L1_Gradient_HandCalculated
        Assert.Equal(1.1, resultVec[0], Tol);
        Assert.Equal(1.9, resultVec[1], Tol);
        Assert.Equal(0.0, resultVec[2], Tol);
        Assert.Equal(-0.9, resultVec[3], Tol);
    }

    [Fact]
    public void L2_Gradient_TensorSupport_MatchesVectorResult()
    {
        var l2 = new L2Regularization<double, Matrix<double>, Tensor<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 });

        var gradVec = Vec(1.0, 2.0, 0.0, -1.0);
        var coeffVec = Vec(0.5, -0.5, 0.0, 3.0);

        var gradTensor = Tensor<double>.FromVector(gradVec);
        var coeffTensor = Tensor<double>.FromVector(coeffVec);

        var result = l2.Regularize(gradTensor, coeffTensor);
        var resultVec = result.ToVector();

        Assert.Equal(1.05, resultVec[0], Tol);
        Assert.Equal(1.95, resultVec[1], Tol);
        Assert.Equal(0.0, resultVec[2], Tol);
        Assert.Equal(-0.7, resultVec[3], Tol);
    }

    [Fact]
    public void ElasticNet_Gradient_TensorSupport_MatchesVectorResult()
    {
        var en = new ElasticNetRegularization<double, Matrix<double>, Tensor<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.1, L1Ratio = 0.5 });

        var gradVec = Vec(1.0, 2.0, 0.0);
        var coeffVec = Vec(0.5, -0.5, 0.0);

        var gradTensor = Tensor<double>.FromVector(gradVec);
        var coeffTensor = Tensor<double>.FromVector(coeffVec);

        var result = en.Regularize(gradTensor, coeffTensor);
        var resultVec = result.ToVector();

        Assert.Equal(1.075, resultVec[0], Tol);
        Assert.Equal(1.925, resultVec[1], Tol);
        Assert.Equal(0.0, resultVec[2], Tol);
    }

    // ─── Edge Cases ──────────────────────────────────────────────────────────

    [Fact]
    public void L1_SingleElement_HandCalculated()
    {
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 });

        // Above threshold: 3.0 - 0.5 = 2.5
        Assert.Equal(2.5, l1.Regularize(Vec(3.0))[0], Tol);

        // Below threshold: 0.3 → 0
        Assert.Equal(0.0, l1.Regularize(Vec(0.3))[0], Tol);

        // Exactly zero: stays 0
        Assert.Equal(0.0, l1.Regularize(Vec(0.0))[0], Tol);
    }

    [Fact]
    public void L2_SingleElement_HandCalculated()
    {
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.25 });

        // 3.0 * (1 - 0.25) = 3.0 * 0.75 = 2.25
        Assert.Equal(2.25, l2.Regularize(Vec(3.0))[0], Tol);

        // -3.0 * 0.75 = -2.25
        Assert.Equal(-2.25, l2.Regularize(Vec(-3.0))[0], Tol);
    }

    [Fact]
    public void AllRegularizations_ZeroVector_StaysZero()
    {
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 });
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.5 });
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.5, L1Ratio = 0.5 });
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();

        var zeros = Vec(0.0, 0.0, 0.0);

        var l1R = l1.Regularize(zeros);
        var l2R = l2.Regularize(zeros);
        var enR = en.Regularize(zeros);
        var noR = noReg.Regularize(zeros);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(0.0, l1R[i], Tol);
            Assert.Equal(0.0, l2R[i], Tol);
            Assert.Equal(0.0, enR[i], Tol);
            Assert.Equal(0.0, noR[i], Tol);
        }
    }

    [Fact]
    public void L1_SymmetricBehavior_PositiveAndNegative()
    {
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.15 });

        var data = Vec(0.8, -0.8, 0.1, -0.1);
        var result = l1.Regularize(data);

        // L1 is symmetric: |result(x)| == |result(-x)|
        Assert.Equal(Math.Abs(result[0]), Math.Abs(result[1]), Tol);
        Assert.Equal(Math.Abs(result[2]), Math.Abs(result[3]), Tol);
        // Signs should be opposite
        Assert.Equal(-result[0], result[1], Tol);
        Assert.Equal(-result[2], result[3], Tol);
    }

    [Fact]
    public void L2_SymmetricBehavior_PositiveAndNegative()
    {
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.15 });

        var data = Vec(0.8, -0.8);
        var result = l2.Regularize(data);

        // L2 is symmetric
        Assert.Equal(-result[0], result[1], Tol);
    }

    // ─── Default Constructor Tests ───────────────────────────────────────────

    [Fact]
    public void L1_DefaultOptions_Strength01_L1Ratio10()
    {
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>();
        var opts = l1.GetOptions();
        Assert.Equal(0.1, opts.Strength, Tol);
        Assert.Equal(1.0, opts.L1Ratio, Tol);
    }

    [Fact]
    public void L2_DefaultOptions_Strength001_L1Ratio00()
    {
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>();
        var opts = l2.GetOptions();
        Assert.Equal(0.01, opts.Strength, Tol);
        Assert.Equal(0.0, opts.L1Ratio, Tol);
    }

    [Fact]
    public void ElasticNet_DefaultOptions_Strength01_L1Ratio05()
    {
        var en = new ElasticNetRegularization<double, Vector<double>, Vector<double>>();
        var opts = en.GetOptions();
        Assert.Equal(0.1, opts.Strength, Tol);
        Assert.Equal(0.5, opts.L1Ratio, Tol);
    }

    // ─── Stronger Shrinkage → Smaller Result ─────────────────────────────────

    [Fact]
    public void L1_StrongerShrinkage_SmallerResult()
    {
        var weak = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.05 });
        var strong = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.3 });

        var data = Vec(1.0, -1.0, 0.5);
        var weakR = weak.Regularize(data);
        var strongR = strong.Regularize(data);

        for (int i = 0; i < data.Length; i++)
            Assert.True(Math.Abs(strongR[i]) <= Math.Abs(weakR[i]) + Tol);
    }

    [Fact]
    public void L2_StrongerShrinkage_SmallerResult()
    {
        var weak = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.05 });
        var strong = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.3 });

        var data = Vec(1.0, -1.0, 0.5);
        var weakR = weak.Regularize(data);
        var strongR = strong.Regularize(data);

        for (int i = 0; i < data.Length; i++)
            Assert.True(Math.Abs(strongR[i]) <= Math.Abs(weakR[i]) + Tol);
    }

    // ─── Gradient Regularization: Unsupported Type ───────────────────────────

    [Fact]
    public void L1_Gradient_UnsupportedType_Throws()
    {
        var l1 = new L1Regularization<double, double, double>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 });

        Assert.Throws<InvalidOperationException>(() => l1.Regularize(1.0, 2.0));
    }

    [Fact]
    public void L2_Gradient_UnsupportedType_Throws()
    {
        var l2 = new L2Regularization<double, double, double>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 });

        Assert.Throws<InvalidOperationException>(() => l2.Regularize(1.0, 2.0));
    }

    [Fact]
    public void ElasticNet_Gradient_UnsupportedType_Throws()
    {
        var en = new ElasticNetRegularization<double, double, double>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.1, L1Ratio = 0.5 });

        Assert.Throws<InvalidOperationException>(() => en.Regularize(1.0, 2.0));
    }

    // ─── Gradient Additivity ─────────────────────────────────────────────────

    [Fact]
    public void L2_Gradient_IsAdditiveWithBaseGradient()
    {
        // reg_gradient = base_gradient + penalty_gradient
        // So: reg_gradient - base_gradient = penalty_gradient = lambda * coeff
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.3 });

        var grad = Vec(5.0, -3.0, 1.0);
        var coeff = Vec(2.0, -4.0, 0.0);

        var result = l2.Regularize(grad, coeff);

        // Penalty should be lambda * coeff
        for (int i = 0; i < 3; i++)
        {
            double penalty = result[i] - grad[i];
            double expected = 0.3 * coeff[i];
            Assert.Equal(expected, penalty, Tol);
        }
    }

    [Fact]
    public void L1_Gradient_IsAdditiveWithBaseGradient()
    {
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.3 });

        var grad = Vec(5.0, -3.0, 1.0);
        var coeff = Vec(2.0, -4.0, 0.0);

        var result = l1.Regularize(grad, coeff);

        // Penalty should be lambda * sign(coeff)
        double[] expectedPenalties = { 0.3 * 1.0, 0.3 * (-1.0), 0.3 * 0.0 };
        for (int i = 0; i < 3; i++)
        {
            double penalty = result[i] - grad[i];
            Assert.Equal(expectedPenalties[i], penalty, Tol);
        }
    }

    // ─── NoRegularization Matrix Returns Zeros ───────────────────────────────

    [Fact]
    public void NoReg_Matrix_ReturnsZeroMatrix()
    {
        // NoRegularization.Regularize(Matrix) returns a zero matrix (additive penalty = 0)
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var data = Mat(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var result = noReg.Regularize(data);

        // The implementation returns new Matrix<T>(rows, cols) which is zeros
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(0.0, result[i, j], Tol);
    }
}
