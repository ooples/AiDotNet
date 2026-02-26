using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LossFunctions;

/// <summary>
/// Deep mathematical correctness tests for loss functions NOT covered in the first file.
/// Each test verifies hand-calculated expected values against the implementation.
/// Covers: Hinge, SquaredHinge, MAE, RMSE, ModifiedHuber, MeanBiasError, Wasserstein,
/// Jaccard, Margin, Exponential, WeightedCrossEntropy, OrdinalRegression.
/// </summary>
public class LossFunctionDeepMathIntegrationTests2
{
    private const double Tolerance = 1e-8;

    private static Vector<double> V(params double[] values) => new(values);

    #region Hinge Loss

    [Fact]
    public void Hinge_HandCalculated_CorrectClassification()
    {
        // y=1, f(x)=2 => margin = 1 - 1*2 = -1 => max(0, -1) = 0
        // y=1, f(x)=0.5 => margin = 1 - 1*0.5 = 0.5 => max(0, 0.5) = 0.5
        // loss = (0 + 0.5) / 2 = 0.25
        var loss = new HingeLoss<double>();
        double result = loss.CalculateLoss(V(2.0, 0.5), V(1.0, 1.0));
        Assert.Equal(0.25, result, Tolerance);
    }

    [Fact]
    public void Hinge_PerfectMargin_IsZero()
    {
        // y=1, f(x)=1 => margin = 1-1 = 0 => max(0,0) = 0
        // y=-1, f(x)=-1 => margin = 1-(-1)(-1) = 1-1 = 0 => max(0,0) = 0
        var loss = new HingeLoss<double>();
        double result = loss.CalculateLoss(V(1.0, -1.0), V(1.0, -1.0));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Hinge_Misclassification_HighLoss()
    {
        // y=1, f(x)=-2 => margin = 1-(-2) = 3 => max(0,3) = 3
        // y=-1, f(x)=3 => margin = 1-(-1)(3) = 1+3 = 4 => max(0,4) = 4
        // loss = (3+4)/2 = 3.5
        var loss = new HingeLoss<double>();
        double result = loss.CalculateLoss(V(-2.0, 3.0), V(1.0, -1.0));
        Assert.Equal(3.5, result, Tolerance);
    }

    [Fact]
    public void Hinge_ExactBoundary_IsZero()
    {
        // y=1, f(x)=1 => margin = 0 => loss = 0
        // Boundary case: exactly at margin
        var loss = new HingeLoss<double>();
        double result = loss.CalculateLoss(V(1.0), V(1.0));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Hinge_Derivative_ActiveAndInactiveRegions()
    {
        // y=1, f(x)=0.5 => margin=0.5>0 => deriv = -y/n = -1/2
        // y=1, f(x)=2.0 => margin=-1<=0 => deriv = 0
        var loss = new HingeLoss<double>();
        var deriv = loss.CalculateDerivative(V(0.5, 2.0), V(1.0, 1.0));
        Assert.Equal(-0.5, deriv[0], Tolerance);
        Assert.Equal(0.0, deriv[1], Tolerance);
    }

    [Fact]
    public void Hinge_Derivative_NegativeLabel()
    {
        // y=-1, f(x)=-0.5 => margin = 1-(-1)(-0.5) = 1-0.5 = 0.5 > 0
        // deriv = -(-1)/1 = 1/1 = 1.0
        var loss = new HingeLoss<double>();
        var deriv = loss.CalculateDerivative(V(-0.5), V(-1.0));
        Assert.Equal(1.0, deriv[0], Tolerance);
    }

    #endregion

    #region Squared Hinge Loss

    [Fact]
    public void SquaredHinge_HandCalculated()
    {
        // y=1, f(x)=0.5 => margin = 1-0.5 = 0.5 => max(0,0.5)^2 = 0.25
        // y=-1, f(x)=-0.5 => margin = 1-(-1)(-0.5) = 1-0.5 = 0.5 => max(0,0.5)^2 = 0.25
        // loss = (0.25+0.25)/2 = 0.25
        var loss = new SquaredHingeLoss<double>();
        double result = loss.CalculateLoss(V(0.5, -0.5), V(1.0, -1.0));
        Assert.Equal(0.25, result, Tolerance);
    }

    [Fact]
    public void SquaredHinge_PerfectClassification_IsZero()
    {
        // y=1, f(x)=2 => margin = 1-2 = -1 => max(0,-1) = 0 => 0^2 = 0
        var loss = new SquaredHingeLoss<double>();
        double result = loss.CalculateLoss(V(2.0, -2.0), V(1.0, -1.0));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void SquaredHinge_LargerThanHinge()
    {
        // For margin > 1: squared hinge >> hinge
        // y=1, f(x)=-1 => margin = 2 => hinge = 2, sqHinge = 4
        var hinge = new HingeLoss<double>();
        var sqHinge = new SquaredHingeLoss<double>();
        double h = hinge.CalculateLoss(V(-1.0), V(1.0));
        double sh = sqHinge.CalculateLoss(V(-1.0), V(1.0));
        Assert.Equal(2.0, h, Tolerance);
        Assert.Equal(4.0, sh, Tolerance);
        Assert.True(sh > h);
    }

    [Fact]
    public void SquaredHinge_Derivative_HandCalculated()
    {
        // y=1, f(x)=0.5 => margin=0.5 > 0 => deriv = -2*margin*y/n = -2*0.5*1/1 = -1.0
        var loss = new SquaredHingeLoss<double>();
        var deriv = loss.CalculateDerivative(V(0.5), V(1.0));
        Assert.Equal(-1.0, deriv[0], Tolerance);
    }

    [Fact]
    public void SquaredHinge_Derivative_InactiveRegion()
    {
        // y=1, f(x)=2 => margin=-1 <= 0 => deriv = 0
        var loss = new SquaredHingeLoss<double>();
        var deriv = loss.CalculateDerivative(V(2.0), V(1.0));
        Assert.Equal(0.0, deriv[0], Tolerance);
    }

    #endregion

    #region Mean Absolute Error (MAE)

    [Fact]
    public void MAE_HandCalculated()
    {
        // predicted=[1, 3, 5], actual=[2, 5, 4]
        // |1-2| = 1, |3-5| = 2, |5-4| = 1
        // MAE = (1+2+1)/3 = 4/3
        var loss = new MeanAbsoluteErrorLoss<double>();
        double result = loss.CalculateLoss(V(1, 3, 5), V(2, 5, 4));
        Assert.Equal(4.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void MAE_PerfectPrediction_IsZero()
    {
        var loss = new MeanAbsoluteErrorLoss<double>();
        double result = loss.CalculateLoss(V(1, 2, 3), V(1, 2, 3));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MAE_Symmetric()
    {
        // MAE(p, a) = MAE(a, p) since |p-a| = |a-p|
        var loss = new MeanAbsoluteErrorLoss<double>();
        double l1 = loss.CalculateLoss(V(1, 3), V(5, 7));
        double l2 = loss.CalculateLoss(V(5, 7), V(1, 3));
        Assert.Equal(l1, l2, Tolerance);
    }

    [Fact]
    public void MAE_NotSensitiveToOutliers_ComparedToMSE()
    {
        // MAE treats all errors linearly; MSE emphasizes large errors
        // predicted=[0, 0], actual=[1, 100]
        // MAE = (1 + 100)/2 = 50.5
        // MSE = (1 + 10000)/2 = 5000.5
        // MSE/MAE ratio shows MSE is much more sensitive to the outlier
        var mae = new MeanAbsoluteErrorLoss<double>();
        var mse = new MeanSquaredErrorLoss<double>();
        double maeVal = mae.CalculateLoss(V(0, 0), V(1, 100));
        double mseVal = mse.CalculateLoss(V(0, 0), V(1, 100));
        Assert.Equal(50.5, maeVal, Tolerance);
        Assert.Equal(5000.5, mseVal, Tolerance);
    }

    [Fact]
    public void MAE_Derivative_SignFunction()
    {
        // deriv = sign(predicted - actual) / n
        // predicted=[3, 1, 2], actual=[2, 2, 2]
        // signs: [1, -1, 0], /3 = [1/3, -1/3, 0]
        var loss = new MeanAbsoluteErrorLoss<double>();
        var deriv = loss.CalculateDerivative(V(3, 1, 2), V(2, 2, 2));
        Assert.Equal(1.0 / 3.0, deriv[0], Tolerance);
        Assert.Equal(-1.0 / 3.0, deriv[1], Tolerance);
        Assert.Equal(0.0, deriv[2], Tolerance);
    }

    #endregion

    #region Root Mean Squared Error (RMSE)

    [Fact]
    public void RMSE_HandCalculated()
    {
        // predicted=[1, 3], actual=[2, 5]
        // errors=[−1, −2], squared=[1, 4], mean=2.5
        // RMSE = sqrt(2.5)
        var loss = new RootMeanSquaredErrorLoss<double>();
        double result = loss.CalculateLoss(V(1, 3), V(2, 5));
        Assert.Equal(Math.Sqrt(2.5), result, Tolerance);
    }

    [Fact]
    public void RMSE_PerfectPrediction_IsZero()
    {
        var loss = new RootMeanSquaredErrorLoss<double>();
        double result = loss.CalculateLoss(V(1, 2, 3), V(1, 2, 3));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void RMSE_GreaterThanOrEqualMAE()
    {
        // By Jensen's inequality: RMSE >= MAE always
        var rmse = new RootMeanSquaredErrorLoss<double>();
        var mae = new MeanAbsoluteErrorLoss<double>();
        var pred = V(1, 3, 5, 7);
        var actual = V(2, 4, 3, 10);
        double rmseVal = rmse.CalculateLoss(pred, actual);
        double maeVal = mae.CalculateLoss(pred, actual);
        Assert.True(rmseVal >= maeVal - 1e-10);
    }

    [Fact]
    public void RMSE_EqualsMAE_ForUniformErrors()
    {
        // When all errors have the same magnitude, RMSE = MAE
        // predicted=[0, 0, 0], actual=[2, 2, 2]
        // MAE = 2, RMSE = sqrt(4) = 2
        var rmse = new RootMeanSquaredErrorLoss<double>();
        var mae = new MeanAbsoluteErrorLoss<double>();
        double rmseVal = rmse.CalculateLoss(V(0, 0, 0), V(2, 2, 2));
        double maeVal = mae.CalculateLoss(V(0, 0, 0), V(2, 2, 2));
        Assert.Equal(maeVal, rmseVal, Tolerance);
    }

    [Fact]
    public void RMSE_Derivative_HandCalculated()
    {
        // RMSE derivative: (predicted-actual) / (n * RMSE)
        // predicted=[1, 3], actual=[2, 5], diff=[-1, -2], RMSE=sqrt(2.5)
        // deriv[0] = -1 / (2 * sqrt(2.5)), deriv[1] = -2 / (2 * sqrt(2.5))
        var loss = new RootMeanSquaredErrorLoss<double>();
        var deriv = loss.CalculateDerivative(V(1, 3), V(2, 5));
        double rmse = Math.Sqrt(2.5);
        Assert.Equal(-1.0 / (2.0 * rmse), deriv[0], Tolerance);
        Assert.Equal(-2.0 / (2.0 * rmse), deriv[1], Tolerance);
    }

    #endregion

    #region Modified Huber Loss

    [Fact]
    public void ModifiedHuber_QuadraticRegion_HandCalculated()
    {
        // z = y*f(x) = 1*0.5 = 0.5, z >= -1
        // max(0, 1-0.5)^2 = max(0, 0.5)^2 = 0.25
        // loss = 0.25/1 = 0.25
        var loss = new ModifiedHuberLoss<double>();
        double result = loss.CalculateLoss(V(0.5), V(1.0));
        Assert.Equal(0.25, result, Tolerance);
    }

    [Fact]
    public void ModifiedHuber_LinearRegion_HandCalculated()
    {
        // z = y*f(x) = 1*(-2) = -2, z < -1
        // -4*z = -4*(-2) = 8
        // loss = 8/1 = 8
        var loss = new ModifiedHuberLoss<double>();
        double result = loss.CalculateLoss(V(-2.0), V(1.0));
        Assert.Equal(8.0, result, Tolerance);
    }

    [Fact]
    public void ModifiedHuber_BoundaryZ_NegativeOne()
    {
        // z = y*f(x) = 1*(-1) = -1, z >= -1 (boundary)
        // max(0, 1-(-1))^2 = max(0, 2)^2 = 4
        // loss = 4/1 = 4
        var loss = new ModifiedHuberLoss<double>();
        double result = loss.CalculateLoss(V(-1.0), V(1.0));
        Assert.Equal(4.0, result, Tolerance);
    }

    [Fact]
    public void ModifiedHuber_PerfectClassification_IsZero()
    {
        // z = 1*2 = 2, z >= 1 => max(0, 1-2)^2 = max(0, -1)^2 = 0
        var loss = new ModifiedHuberLoss<double>();
        double result = loss.CalculateLoss(V(2.0), V(1.0));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ModifiedHuber_ContinuityAtBoundary()
    {
        // At z = -1: quadratic gives max(0, 1-(-1))^2 = 4, linear gives -4*(-1) = 4
        // They should agree at the boundary
        var loss = new ModifiedHuberLoss<double>();
        // z = -1 exactly (quadratic side)
        double atBoundary = loss.CalculateLoss(V(-1.0), V(1.0));
        // z = -1 - epsilon (linear side)
        double justBelow = loss.CalculateLoss(V(-1.0 - 1e-10), V(1.0));
        Assert.Equal(atBoundary, justBelow, 1e-5);
    }

    [Fact]
    public void ModifiedHuber_Derivative_QuadraticRegion()
    {
        // z = 1*0.5 = 0.5, -1 <= z < 1
        // deriv = -2*y*(1-z)/n = -2*1*(1-0.5)/1 = -1.0
        var loss = new ModifiedHuberLoss<double>();
        var deriv = loss.CalculateDerivative(V(0.5), V(1.0));
        Assert.Equal(-1.0, deriv[0], Tolerance);
    }

    [Fact]
    public void ModifiedHuber_Derivative_LinearRegion()
    {
        // z = 1*(-2) = -2 < -1
        // deriv = -4*y/n = -4*1/1 = -4.0
        var loss = new ModifiedHuberLoss<double>();
        var deriv = loss.CalculateDerivative(V(-2.0), V(1.0));
        Assert.Equal(-4.0, deriv[0], Tolerance);
    }

    [Fact]
    public void ModifiedHuber_Derivative_ZeroWhenCorrect()
    {
        // z = 1*2 = 2 >= 1 => deriv = 0
        var loss = new ModifiedHuberLoss<double>();
        var deriv = loss.CalculateDerivative(V(2.0), V(1.0));
        Assert.Equal(0.0, deriv[0], Tolerance);
    }

    #endregion

    #region Mean Bias Error (MBE)

    [Fact]
    public void MBE_HandCalculated_UnderPrediction()
    {
        // MBE = mean(actual - predicted)
        // actual=[5, 6], predicted=[3, 4]
        // diffs=[2, 2], MBE = 4/2 = 2.0 (positive = under-prediction)
        var loss = new MeanBiasErrorLoss<double>();
        double result = loss.CalculateLoss(V(3, 4), V(5, 6));
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void MBE_HandCalculated_OverPrediction()
    {
        // actual=[1, 2], predicted=[5, 6]
        // diffs=[-4, -4], MBE = -8/2 = -4.0 (negative = over-prediction)
        var loss = new MeanBiasErrorLoss<double>();
        double result = loss.CalculateLoss(V(5, 6), V(1, 2));
        Assert.Equal(-4.0, result, Tolerance);
    }

    [Fact]
    public void MBE_ErrorsCancelOut()
    {
        // actual=[1, 5], predicted=[3, 3]
        // diffs=[-2, 2], MBE = 0 (errors cancel!)
        var loss = new MeanBiasErrorLoss<double>();
        double result = loss.CalculateLoss(V(3, 3), V(1, 5));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MBE_PerfectPrediction_IsZero()
    {
        var loss = new MeanBiasErrorLoss<double>();
        double result = loss.CalculateLoss(V(1, 2, 3), V(1, 2, 3));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MBE_Derivative_ConstantNegativeOneOverN()
    {
        // d(MBE)/d(predicted_i) = -1/n for all i
        var loss = new MeanBiasErrorLoss<double>();
        var deriv = loss.CalculateDerivative(V(1, 2, 3, 4), V(5, 6, 7, 8));
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(-0.25, deriv[i], Tolerance); // -1/4
        }
    }

    [Fact]
    public void MBE_Antisymmetric()
    {
        // MBE(p, a) = -MBE(a, p)
        var loss = new MeanBiasErrorLoss<double>();
        double l1 = loss.CalculateLoss(V(1, 3), V(5, 7));
        double l2 = loss.CalculateLoss(V(5, 7), V(1, 3));
        Assert.Equal(l1, -l2, Tolerance);
    }

    #endregion

    #region Wasserstein Loss

    [Fact]
    public void Wasserstein_HandCalculated_AllReal()
    {
        // Loss = -mean(predicted * actual)
        // predicted=[3, 5], actual=[1, 1] (real samples)
        // products=[3, 5], mean=4, loss = -4
        var loss = new WassersteinLoss<double>();
        double result = loss.CalculateLoss(V(3, 5), V(1, 1));
        Assert.Equal(-4.0, result, Tolerance);
    }

    [Fact]
    public void Wasserstein_HandCalculated_AllFake()
    {
        // predicted=[3, 5], actual=[-1, -1] (fake samples)
        // products=[-3, -5], mean=-4, loss = -(-4) = 4
        var loss = new WassersteinLoss<double>();
        double result = loss.CalculateLoss(V(3, 5), V(-1, -1));
        Assert.Equal(4.0, result, Tolerance);
    }

    [Fact]
    public void Wasserstein_MixedRealFake()
    {
        // predicted=[5, -3], actual=[1, -1]
        // products=[5, 3], mean=4, loss=-4
        var loss = new WassersteinLoss<double>();
        double result = loss.CalculateLoss(V(5, -3), V(1, -1));
        Assert.Equal(-4.0, result, Tolerance);
    }

    [Fact]
    public void Wasserstein_GoodCritic_NegativeLoss()
    {
        // A good critic gives high scores to real and low scores to fake
        // predicted=[10, -10], actual=[1, -1]
        // products=[10, 10], mean=10, loss=-10
        var loss = new WassersteinLoss<double>();
        double result = loss.CalculateLoss(V(10, -10), V(1, -1));
        Assert.Equal(-10.0, result, Tolerance);
    }

    [Fact]
    public void Wasserstein_Derivative_HandCalculated()
    {
        // deriv[i] = -actual[i] / n
        // actual=[1, -1], n=2
        // deriv = [-0.5, 0.5]
        var loss = new WassersteinLoss<double>();
        var deriv = loss.CalculateDerivative(V(3, -3), V(1, -1));
        Assert.Equal(-0.5, deriv[0], Tolerance);
        Assert.Equal(0.5, deriv[1], Tolerance);
    }

    #endregion

    #region Jaccard Loss

    [Fact]
    public void Jaccard_PerfectOverlap_IsZero()
    {
        // predicted=[1, 1, 0], actual=[1, 1, 0]
        // intersection = min(1,1)+min(1,1)+min(0,0) = 2
        // union = max(1,1)+max(1,1)+max(0,0) = 2
        // Jaccard = 1 - 2/2 = 0
        var loss = new JaccardLoss<double>();
        double result = loss.CalculateLoss(V(1, 1, 0), V(1, 1, 0));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Jaccard_NoOverlap_IsOne()
    {
        // predicted=[1, 1, 0], actual=[0, 0, 1]
        // intersection = min(1,0)+min(1,0)+min(0,1) = 0
        // union = max(1,0)+max(1,0)+max(0,1) = 3
        // Jaccard = 1 - 0/3 = 1
        var loss = new JaccardLoss<double>();
        double result = loss.CalculateLoss(V(1, 1, 0), V(0, 0, 1));
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void Jaccard_PartialOverlap_HandCalculated()
    {
        // predicted=[0.8, 0.2, 0.6], actual=[1.0, 0.0, 0.5]
        // intersection = min(0.8,1)+min(0.2,0)+min(0.6,0.5) = 0.8+0+0.5 = 1.3
        // union = max(0.8,1)+max(0.2,0)+max(0.6,0.5) = 1.0+0.2+0.6 = 1.8
        // Jaccard loss = 1 - 1.3/1.8 = 1 - 0.7222... = 0.2777...
        var loss = new JaccardLoss<double>();
        double result = loss.CalculateLoss(V(0.8, 0.2, 0.6), V(1.0, 0.0, 0.5));
        Assert.Equal(1.0 - 1.3 / 1.8, result, Tolerance);
    }

    [Fact]
    public void Jaccard_Symmetric()
    {
        // Jaccard(A,B) = Jaccard(B,A)
        var loss = new JaccardLoss<double>();
        double l1 = loss.CalculateLoss(V(0.3, 0.7, 0.5), V(0.6, 0.2, 0.8));
        double l2 = loss.CalculateLoss(V(0.6, 0.2, 0.8), V(0.3, 0.7, 0.5));
        Assert.Equal(l1, l2, Tolerance);
    }

    [Fact]
    public void Jaccard_RelationToDice()
    {
        // For binary: Jaccard = Dice / (2 - Dice) or Dice = 2*Jaccard/(1+Jaccard)
        // This is approximate for soft values, but exact relationship for binary
        var jaccard = new JaccardLoss<double>();
        var dice = new DiceLoss<double>();
        var pred = V(1, 0, 1, 1);
        var actual = V(1, 1, 1, 0);
        double j = 1.0 - jaccard.CalculateLoss(pred, actual); // Jaccard index
        double d = 1.0 - dice.CalculateLoss(pred, actual);    // Dice coefficient
        // For binary: D = 2J/(1+J)
        Assert.Equal(d, 2.0 * j / (1.0 + j), 1e-6);
    }

    #endregion

    #region Margin Loss (Capsule Networks)

    [Fact]
    public void Margin_ClassPresent_BelowMPlus()
    {
        // y=1 (class present), v=0.7, m+=0.9, m-=0.1, lambda=0.5
        // term1 = y*(m+-v) = 1*(0.9-0.7) = 0.2 > 0 => 0.2^2 = 0.04
        // term2 = (1-y)*(v-m-) = 0*(0.7-0.1) = 0, not > 0 => 0
        // loss = (0.04 + 0.5*0)/1 = 0.04
        var loss = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        double result = loss.CalculateLoss(V(0.7), V(1.0));
        Assert.Equal(0.04, result, Tolerance);
    }

    [Fact]
    public void Margin_ClassPresent_AboveMPlus_IsZero()
    {
        // y=1, v=0.95, m+=0.9
        // term1 = 1*(0.9-0.95) = -0.05 <= 0 => 0
        // No loss when prediction exceeds m+
        var loss = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        double result = loss.CalculateLoss(V(0.95), V(1.0));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Margin_ClassAbsent_AboveMMinus()
    {
        // y=0 (class absent), v=0.3, m-=0.1, lambda=0.5
        // term1 = 0*(0.9-0.3) = 0
        // term2 = (1-0)*(0.3-0.1) = 0.2 > 0 => 0.2^2 = 0.04
        // loss = (0 + 0.5*0.04)/1 = 0.02
        var loss = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        double result = loss.CalculateLoss(V(0.3), V(0.0));
        Assert.Equal(0.02, result, Tolerance);
    }

    [Fact]
    public void Margin_ClassAbsent_BelowMMinus_IsZero()
    {
        // y=0, v=0.05, m-=0.1
        // term2 = 1*(0.05-0.1) = -0.05 <= 0 => 0
        var loss = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        double result = loss.CalculateLoss(V(0.05), V(0.0));
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Margin_MultipleClasses_HandCalculated()
    {
        // Class 0 present (y=1, v=0.8): term1 = (0.9-0.8)^2 = 0.01
        // Class 1 absent (y=0, v=0.2): term2 = 0.5*(0.2-0.1)^2 = 0.5*0.01 = 0.005
        // loss = (0.01 + 0.005)/2 = 0.0075
        var loss = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        double result = loss.CalculateLoss(V(0.8, 0.2), V(1.0, 0.0));
        Assert.Equal(0.0075, result, Tolerance);
    }

    [Fact]
    public void Margin_Derivative_ClassPresent()
    {
        // y=1, v=0.7, m+=0.9
        // term1 = 1*(0.9-0.7) = 0.2 > 0
        // deriv = -2*term1/n = -2*0.2/1 = -0.4
        var loss = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        var deriv = loss.CalculateDerivative(V(0.7), V(1.0));
        Assert.Equal(-0.4, deriv[0], Tolerance);
    }

    #endregion

    #region Exponential Loss

    [Fact]
    public void Exponential_CorrectPrediction_LowLoss()
    {
        // y=1, f(x)=2 => exp(-1*2) = exp(-2) = 0.13533...
        var loss = new ExponentialLoss<double>();
        double result = loss.CalculateLoss(V(2.0), V(1.0));
        Assert.Equal(Math.Exp(-2.0), result, Tolerance);
    }

    [Fact]
    public void Exponential_IncorrectPrediction_HighLoss()
    {
        // y=1, f(x)=-2 => exp(-1*(-2)) = exp(2) = 7.3890...
        var loss = new ExponentialLoss<double>();
        double result = loss.CalculateLoss(V(-2.0), V(1.0));
        Assert.Equal(Math.Exp(2.0), result, Tolerance);
    }

    [Fact]
    public void Exponential_ZeroPrediction()
    {
        // y=1, f(x)=0 => exp(0) = 1.0
        var loss = new ExponentialLoss<double>();
        double result = loss.CalculateLoss(V(0.0), V(1.0));
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void Exponential_HandCalculated_TwoElements()
    {
        // y=1, f=1: exp(-1) = 0.367879...
        // y=-1, f=-1: exp(-(-1)(-1)) = exp(-1) = 0.367879...
        // loss = (exp(-1) + exp(-1))/2 = exp(-1)
        var loss = new ExponentialLoss<double>();
        double result = loss.CalculateLoss(V(1.0, -1.0), V(1.0, -1.0));
        Assert.Equal(Math.Exp(-1.0), result, Tolerance);
    }

    [Fact]
    public void Exponential_Derivative_HandCalculated()
    {
        // deriv[i] = -y[i] * exp(-y[i]*f[i]) / n
        // y=1, f=2: -1*exp(-2)/1 = -exp(-2) = -0.13533...
        var loss = new ExponentialLoss<double>();
        var deriv = loss.CalculateDerivative(V(2.0), V(1.0));
        Assert.Equal(-Math.Exp(-2.0), deriv[0], Tolerance);
    }

    [Fact]
    public void Exponential_AlwaysPositive()
    {
        // exp(x) > 0 for all x, so loss is always positive
        var loss = new ExponentialLoss<double>();
        Assert.True(loss.CalculateLoss(V(-5, 3), V(1, -1)) > 0);
        Assert.True(loss.CalculateLoss(V(5, -3), V(1, -1)) > 0);
        Assert.True(loss.CalculateLoss(V(0, 0), V(1, -1)) > 0);
    }

    #endregion

    #region Weighted Cross Entropy

    [Fact]
    public void WeightedCE_UniformWeights_EqualsBCE()
    {
        // With all weights = 1, weighted CE should equal standard BCE
        var weights = V(1.0, 1.0, 1.0);
        var wce = new WeightedCrossEntropyLoss<double>(weights);
        var bce = new BinaryCrossEntropyLoss<double>();

        var pred = V(0.8, 0.3, 0.6);
        var actual = V(1.0, 0.0, 1.0);

        double wceVal = wce.CalculateLoss(pred, actual);
        double bceVal = bce.CalculateLoss(pred, actual);
        Assert.Equal(bceVal, wceVal, 1e-6);
    }

    [Fact]
    public void WeightedCE_HandCalculated_WithWeights()
    {
        // predicted=[0.9], actual=[1.0], weight=[2.0]
        // loss = -weight * [y*log(p) + (1-y)*log(1-p)] / n
        // = -2 * [1*log(0.9) + 0*log(0.1)] / 1
        // = -2 * log(0.9)
        // = -2 * (-0.10536...) = 0.21072...
        var weights = V(2.0);
        var loss = new WeightedCrossEntropyLoss<double>(weights);
        double result = loss.CalculateLoss(V(0.9), V(1.0));
        Assert.Equal(-2.0 * Math.Log(0.9), result, 1e-6);
    }

    [Fact]
    public void WeightedCE_HigherWeight_HigherLoss()
    {
        // Same prediction/actual but different weights
        var pred = V(0.6);
        var actual = V(1.0);

        var wce1 = new WeightedCrossEntropyLoss<double>(V(1.0));
        var wce2 = new WeightedCrossEntropyLoss<double>(V(5.0));

        double loss1 = wce1.CalculateLoss(pred, actual);
        double loss2 = wce2.CalculateLoss(pred, actual);

        Assert.True(loss2 > loss1);
        Assert.Equal(5.0 * loss1, loss2, 1e-6);
    }

    [Fact]
    public void WeightedCE_ZeroWeight_ZeroContribution()
    {
        // Zero-weighted samples don't contribute to loss
        var pred = V(0.1, 0.9);
        var actual = V(1.0, 1.0);

        // First sample has huge error but zero weight
        var weights = V(0.0, 1.0);
        var loss = new WeightedCrossEntropyLoss<double>(weights);
        double result = loss.CalculateLoss(pred, actual);

        // Only second sample contributes: -1*log(0.9)/2
        double expected = -Math.Log(0.9) / 2.0;
        Assert.Equal(expected, result, 1e-6);
    }

    #endregion

    #region Ordinal Regression Loss

    [Fact]
    public void OrdinalRegression_HandCalculated_3Classes()
    {
        // numClasses=3 => 2 thresholds (j=0, j=1)
        // actual=2, predicted=1
        // j=0: indicator = (2>0) = 1, exp(-1*1) = exp(-1), log(1+exp(-1))
        // j=1: indicator = (2>1) = 1, exp(-1*1) = exp(-1), log(1+exp(-1))
        // total = 2 * log(1+exp(-1))
        var loss = new OrdinalRegressionLoss<double>(numClasses: 3);
        double result = loss.CalculateLoss(V(1.0), V(2.0));
        double expected = 2.0 * Math.Log(1.0 + Math.Exp(-1.0));
        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void OrdinalRegression_LowestClass()
    {
        // numClasses=3, actual=0, predicted=1
        // j=0: indicator = (0>0) = 0, exp(-0*1) = exp(0) = 1, log(1+1) = log(2)
        // j=1: indicator = (0>1) = 0, exp(-0*1) = 1, log(1+1) = log(2)
        // total = 2*log(2)
        var loss = new OrdinalRegressionLoss<double>(numClasses: 3);
        double result = loss.CalculateLoss(V(1.0), V(0.0));
        double expected = 2.0 * Math.Log(2.0);
        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void OrdinalRegression_NotAveraged()
    {
        // Note: OrdinalRegressionLoss does NOT divide by n (unlike most losses)
        // With 2 samples, loss should be sum, not mean
        var loss = new OrdinalRegressionLoss<double>(numClasses: 3);
        double single1 = loss.CalculateLoss(V(1.0), V(2.0));
        double single2 = loss.CalculateLoss(V(0.5), V(1.0));
        double combined = loss.CalculateLoss(V(1.0, 0.5), V(2.0, 1.0));
        Assert.Equal(single1 + single2, combined, Tolerance);
    }

    [Fact]
    public void OrdinalRegression_InvalidNumClasses_Throws()
    {
        Assert.Throws<ArgumentException>(() => new OrdinalRegressionLoss<double>(numClasses: 1));
    }

    #endregion

    #region Cross-Loss Mathematical Relationships

    [Fact]
    public void HingeLoss_UpperBoundsOnSquaredHinge()
    {
        // For margin in [0,1]: hinge = margin, sqHinge = margin^2
        // Since margin <= 1 in this range: sqHinge <= hinge
        var hinge = new HingeLoss<double>();
        var sqHinge = new SquaredHingeLoss<double>();

        // y=1, f=0.5 => margin = 0.5
        double h = hinge.CalculateLoss(V(0.5), V(1.0));
        double sh = sqHinge.CalculateLoss(V(0.5), V(1.0));
        Assert.True(sh <= h + 1e-10); // 0.25 <= 0.5
    }

    [Fact]
    public void MAE_EqualsQuantile05()
    {
        // Quantile loss with q=0.5 equals 0.5 * MAE
        var mae = new MeanAbsoluteErrorLoss<double>();
        var quantile = new QuantileLoss<double>(0.5);
        var pred = V(1, 4, 2);
        var actual = V(3, 2, 5);

        double maeVal = mae.CalculateLoss(pred, actual);
        double qVal = quantile.CalculateLoss(pred, actual);
        // QuantileLoss(0.5): underest penalty = 0.5*diff, overest penalty = 0.5*|diff|
        // So QuantileLoss(0.5) = 0.5 * MAE
        Assert.Equal(0.5 * maeVal, qVal, Tolerance);
    }

    [Fact]
    public void RMSE_SquaredEqualsMSE()
    {
        // RMSE^2 = MSE by definition
        var rmse = new RootMeanSquaredErrorLoss<double>();
        var mse = new MeanSquaredErrorLoss<double>();
        var pred = V(1, 3, 5);
        var actual = V(2, 4, 8);

        double rmseVal = rmse.CalculateLoss(pred, actual);
        double mseVal = mse.CalculateLoss(pred, actual);
        Assert.Equal(mseVal, rmseVal * rmseVal, Tolerance);
    }

    [Fact]
    public void Huber_Delta1_LinearRegionEqualsMAE_Minus_HalfDelta()
    {
        // For |error| > delta=1: Huber = delta*(|error| - 0.5*delta) = |error| - 0.5
        // predicted=[0], actual=[3] => error=3, Huber = 1*(3-0.5) = 2.5
        var huber = new HuberLoss<double>(delta: 1.0);
        double result = huber.CalculateLoss(V(0), V(3));
        Assert.Equal(2.5, result, Tolerance);
    }

    [Fact]
    public void ModifiedHuber_ReducesToSquaredHinge_InQuadraticRegion()
    {
        // For z >= -1: ModifiedHuber = max(0, 1-z)^2 = SquaredHinge
        // y=1, f=0.5, z=0.5
        var modHuber = new ModifiedHuberLoss<double>();
        var sqHinge = new SquaredHingeLoss<double>();
        double mh = modHuber.CalculateLoss(V(0.5), V(1.0));
        double sh = sqHinge.CalculateLoss(V(0.5), V(1.0));
        Assert.Equal(sh, mh, Tolerance);
    }

    #endregion

    #region Gradient Consistency Checks

    [Fact]
    public void Hinge_GradientNumericalCheck()
    {
        var loss = new HingeLoss<double>();
        var pred = V(0.3);
        var actual = V(1.0);
        double eps = 1e-5;

        double fPlus = loss.CalculateLoss(V(0.3 + eps), actual);
        double fMinus = loss.CalculateLoss(V(0.3 - eps), actual);
        double numericalGrad = (fPlus - fMinus) / (2 * eps);

        var analyticalGrad = loss.CalculateDerivative(pred, actual);
        Assert.Equal(numericalGrad, analyticalGrad[0], 1e-4);
    }

    [Fact]
    public void SquaredHinge_GradientNumericalCheck()
    {
        var loss = new SquaredHingeLoss<double>();
        var pred = V(0.3);
        var actual = V(1.0);
        double eps = 1e-5;

        double fPlus = loss.CalculateLoss(V(0.3 + eps), actual);
        double fMinus = loss.CalculateLoss(V(0.3 - eps), actual);
        double numericalGrad = (fPlus - fMinus) / (2 * eps);

        var analyticalGrad = loss.CalculateDerivative(pred, actual);
        Assert.Equal(numericalGrad, analyticalGrad[0], 1e-4);
    }

    [Fact]
    public void ModifiedHuber_GradientNumericalCheck_QuadraticRegion()
    {
        // z = 1*0.5 = 0.5, in quadratic region
        var loss = new ModifiedHuberLoss<double>();
        var actual = V(1.0);
        double eps = 1e-5;

        double fPlus = loss.CalculateLoss(V(0.5 + eps), actual);
        double fMinus = loss.CalculateLoss(V(0.5 - eps), actual);
        double numericalGrad = (fPlus - fMinus) / (2 * eps);

        var analyticalGrad = loss.CalculateDerivative(V(0.5), actual);
        Assert.Equal(numericalGrad, analyticalGrad[0], 1e-4);
    }

    [Fact]
    public void ModifiedHuber_GradientNumericalCheck_LinearRegion()
    {
        // z = 1*(-2) = -2, in linear region
        var loss = new ModifiedHuberLoss<double>();
        var actual = V(1.0);
        double eps = 1e-5;

        double fPlus = loss.CalculateLoss(V(-2.0 + eps), actual);
        double fMinus = loss.CalculateLoss(V(-2.0 - eps), actual);
        double numericalGrad = (fPlus - fMinus) / (2 * eps);

        var analyticalGrad = loss.CalculateDerivative(V(-2.0), actual);
        Assert.Equal(numericalGrad, analyticalGrad[0], 1e-4);
    }

    [Fact]
    public void Exponential_GradientNumericalCheck()
    {
        var loss = new ExponentialLoss<double>();
        var actual = V(1.0);
        double eps = 1e-5;

        double fPlus = loss.CalculateLoss(V(1.5 + eps), actual);
        double fMinus = loss.CalculateLoss(V(1.5 - eps), actual);
        double numericalGrad = (fPlus - fMinus) / (2 * eps);

        var analyticalGrad = loss.CalculateDerivative(V(1.5), actual);
        Assert.Equal(numericalGrad, analyticalGrad[0], 1e-4);
    }

    [Fact]
    public void Wasserstein_GradientNumericalCheck()
    {
        var loss = new WassersteinLoss<double>();
        var actual = V(1.0);
        double eps = 1e-5;

        double fPlus = loss.CalculateLoss(V(3.0 + eps), actual);
        double fMinus = loss.CalculateLoss(V(3.0 - eps), actual);
        double numericalGrad = (fPlus - fMinus) / (2 * eps);

        var analyticalGrad = loss.CalculateDerivative(V(3.0), actual);
        Assert.Equal(numericalGrad, analyticalGrad[0], 1e-4);
    }

    [Fact]
    public void Margin_GradientNumericalCheck_ClassPresent()
    {
        var loss = new MarginLoss<double>(mPlus: 0.9, mMinus: 0.1, lambda: 0.5);
        var actual = V(1.0);
        double eps = 1e-5;

        double fPlus = loss.CalculateLoss(V(0.7 + eps), actual);
        double fMinus = loss.CalculateLoss(V(0.7 - eps), actual);
        double numericalGrad = (fPlus - fMinus) / (2 * eps);

        var analyticalGrad = loss.CalculateDerivative(V(0.7), actual);
        Assert.Equal(numericalGrad, analyticalGrad[0], 1e-4);
    }

    [Fact]
    public void RMSE_GradientNumericalCheck()
    {
        var loss = new RootMeanSquaredErrorLoss<double>();
        var pred = V(1.0, 3.0);
        var actual = V(2.0, 5.0);
        double eps = 1e-5;

        // Perturb first element
        double fPlus = loss.CalculateLoss(V(1.0 + eps, 3.0), actual);
        double fMinus = loss.CalculateLoss(V(1.0 - eps, 3.0), actual);
        double numericalGrad0 = (fPlus - fMinus) / (2 * eps);

        var analyticalGrad = loss.CalculateDerivative(pred, actual);
        Assert.Equal(numericalGrad0, analyticalGrad[0], 1e-4);
    }

    #endregion
}
