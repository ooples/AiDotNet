#nullable disable
using AiDotNet.GaussianProcesses;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.GaussianProcesses;

/// <summary>
/// Deep math-correctness integration tests for GP Likelihoods and MeanFunctions.
/// Verifies analytical formulas, gradient consistency, Hessian correctness,
/// and known-value outputs against hand-calculated results.
/// </summary>
public class GPLikelihoodMeanDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double NumericalEpsilon = 1e-5;
    private const double GradientTolerance = 1e-4;

    #region Helpers

    private static Vector<double> MakeVector(params double[] vals)
    {
        var v = new Vector<double>(vals.Length);
        for (int i = 0; i < vals.Length; i++) v[i] = vals[i];
        return v;
    }

    private static Matrix<double> MakeMatrix(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var m = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = data[i, j];
        return m;
    }

    /// <summary>
    /// Computes numerical gradient of a scalar function of a vector using central differences.
    /// </summary>
    private static Vector<double> NumericalGradient(
        Vector<double> f, Func<Vector<double>, double> scalarFunc, double eps = NumericalEpsilon)
    {
        var grad = new Vector<double>(f.Length);
        for (int i = 0; i < f.Length; i++)
        {
            var fPlus = new Vector<double>(f.Length);
            var fMinus = new Vector<double>(f.Length);
            for (int j = 0; j < f.Length; j++)
            {
                fPlus[j] = f[j];
                fMinus[j] = f[j];
            }
            fPlus[i] += eps;
            fMinus[i] -= eps;
            grad[i] = (scalarFunc(fPlus) - scalarFunc(fMinus)) / (2 * eps);
        }
        return grad;
    }

    /// <summary>
    /// Computes numerical Hessian diagonal using central differences of the gradient.
    /// </summary>
    private static Vector<double> NumericalHessianDiag(
        Vector<double> y, Vector<double> f,
        Func<Vector<double>, Vector<double>, Vector<double>> gradientFunc,
        double eps = NumericalEpsilon)
    {
        var hess = new Vector<double>(f.Length);
        for (int i = 0; i < f.Length; i++)
        {
            var fPlus = new Vector<double>(f.Length);
            var fMinus = new Vector<double>(f.Length);
            for (int j = 0; j < f.Length; j++)
            {
                fPlus[j] = f[j];
                fMinus[j] = f[j];
            }
            fPlus[i] += eps;
            fMinus[i] -= eps;
            var gradPlus = gradientFunc(y, fPlus);
            var gradMinus = gradientFunc(y, fMinus);
            hess[i] = (gradPlus[i] - gradMinus[i]) / (2 * eps);
        }
        return hess;
    }

    private static void AssertClose(double expected, double actual, double tol, string msg = "")
    {
        Assert.True(Math.Abs(expected - actual) < tol,
            $"{msg}Expected {expected}, got {actual}, diff={Math.Abs(expected - actual)}");
    }

    private static void AssertVectorsClose(Vector<double> expected, Vector<double> actual, double tol, string msg = "")
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) < tol,
                $"{msg}Element [{i}]: expected {expected[i]}, got {actual[i]}, diff={Math.Abs(expected[i] - actual[i])}");
        }
    }

    #endregion

    // ==================== Gaussian Likelihood Tests ====================

    #region Gaussian Likelihood

    [Fact]
    public void GaussianLikelihood_LogLikelihood_HandCalculated()
    {
        // p(y|f) = N(y; f, σ²) for each element
        // log p(y|f) = -0.5 * Σ[(yi-fi)²/σ² + log(2πσ²)]
        double sigma2 = 0.1;
        var likelihood = new GaussianLikelihood<double>(sigma2);

        var y = MakeVector(1.0, 2.0, 3.0);
        var f = MakeVector(1.1, 1.9, 3.2);

        // Hand calculate: residuals = [-0.1, 0.1, -0.2]
        // Each term: -0.5 * [r²/0.1 + log(2π*0.1)]
        double logNorm = -0.5 * (Math.Log(2 * Math.PI) + Math.Log(sigma2));
        double expected = 0;
        double[] residuals = { -0.1, 0.1, -0.2 };
        foreach (double r in residuals)
            expected += logNorm - 0.5 * r * r / sigma2;

        double actual = likelihood.LogLikelihood(y, f);
        AssertClose(expected, actual, Tolerance, "Gaussian LogLikelihood: ");
    }

    [Fact]
    public void GaussianLikelihood_LogLikelihood_PerfectPrediction()
    {
        // When y == f, residuals are zero, so loglik = n * logNorm
        double sigma2 = 0.5;
        var likelihood = new GaussianLikelihood<double>(sigma2);

        var y = MakeVector(1.0, 2.0, 3.0);
        var f = MakeVector(1.0, 2.0, 3.0);

        double logNorm = -0.5 * (Math.Log(2 * Math.PI) + Math.Log(sigma2));
        double expected = 3 * logNorm; // 3 elements, all zero residual

        double actual = likelihood.LogLikelihood(y, f);
        AssertClose(expected, actual, Tolerance, "Gaussian LogLikelihood perfect: ");
    }

    [Fact]
    public void GaussianLikelihood_Gradient_MatchesNumerical()
    {
        double sigma2 = 0.2;
        var likelihood = new GaussianLikelihood<double>(sigma2);

        var y = MakeVector(1.0, 2.5, -0.5);
        var f = MakeVector(0.8, 2.3, -0.7);

        var analyticalGrad = likelihood.LogLikelihoodGradient(y, f);
        var numericalGrad = NumericalGradient(f,
            fv => likelihood.LogLikelihood(y, fv));

        AssertVectorsClose(numericalGrad, analyticalGrad, GradientTolerance, "Gaussian gradient: ");
    }

    [Fact]
    public void GaussianLikelihood_Gradient_HandCalculated()
    {
        // ∂log p/∂fi = (yi - fi) / σ²
        double sigma2 = 0.5;
        var likelihood = new GaussianLikelihood<double>(sigma2);

        var y = MakeVector(3.0, 1.0);
        var f = MakeVector(2.0, 1.5);

        var grad = likelihood.LogLikelihoodGradient(y, f);

        // (3-2)/0.5 = 2.0, (1-1.5)/0.5 = -1.0
        AssertClose(2.0, grad[0], Tolerance, "Gaussian grad[0]: ");
        AssertClose(-1.0, grad[1], Tolerance, "Gaussian grad[1]: ");
    }

    [Fact]
    public void GaussianLikelihood_HessianDiag_IsConstant()
    {
        // Hessian = -1/σ² for all elements (independent of y and f values)
        double sigma2 = 0.25;
        var likelihood = new GaussianLikelihood<double>(sigma2);

        var y = MakeVector(10.0, -5.0, 0.0);
        var f = MakeVector(0.0, 0.0, 0.0);

        var hess = likelihood.LogLikelihoodHessianDiag(y, f);

        double expected = -1.0 / sigma2; // -4.0
        for (int i = 0; i < 3; i++)
            AssertClose(expected, hess[i], Tolerance, $"Gaussian hessian[{i}]: ");
    }

    [Fact]
    public void GaussianLikelihood_HessianDiag_MatchesNumerical()
    {
        double sigma2 = 0.3;
        var likelihood = new GaussianLikelihood<double>(sigma2);

        var y = MakeVector(1.5, -0.5, 2.0);
        var f = MakeVector(1.0, 0.0, 1.5);

        var analyticalHess = likelihood.LogLikelihoodHessianDiag(y, f);
        var numericalHess = NumericalHessianDiag(y, f,
            (yv, fv) => likelihood.LogLikelihoodGradient(yv, fv));

        AssertVectorsClose(numericalHess, analyticalHess, GradientTolerance, "Gaussian hessian: ");
    }

    [Fact]
    public void GaussianLikelihood_PredictiveVariance_AddsNoiseVariance()
    {
        double sigma2 = 0.3;
        var likelihood = new GaussianLikelihood<double>(sigma2);

        double fVariance = 1.5;
        double result = likelihood.PredictiveVariance(0.0, fVariance);

        // Should be fVariance + noiseVariance
        AssertClose(fVariance + sigma2, result, Tolerance, "Gaussian PredictiveVariance: ");
    }

    [Fact]
    public void GaussianLikelihood_TransformMean_IsIdentity()
    {
        var likelihood = new GaussianLikelihood<double>(0.1);

        double f = 3.14;
        double result = likelihood.TransformMean(f);
        AssertClose(f, result, Tolerance, "Gaussian TransformMean: ");
    }

    #endregion

    // ==================== Bernoulli Likelihood Tests ====================

    #region Bernoulli Likelihood

    [Fact]
    public void BernoulliLikelihood_LogLikelihood_HandCalculated()
    {
        // log p(y|f) = y*f - log(1 + exp(f))
        var likelihood = new BernoulliLikelihood<double>();

        // Single observation: y=1, f=2.0
        // log p = 1*2 - log(1 + exp(2)) = 2 - log(1 + 7.389...) = 2 - log(8.389..) = 2 - 2.1269...
        var y = MakeVector(1.0);
        var f = MakeVector(2.0);

        double expected = 1.0 * 2.0 - Math.Log(1 + Math.Exp(2.0));
        double actual = likelihood.LogLikelihood(y, f);

        AssertClose(expected, actual, Tolerance, "Bernoulli LogLikelihood y=1: ");
    }

    [Fact]
    public void BernoulliLikelihood_LogLikelihood_NegativeClass()
    {
        // y=0, f=-1.0
        // log p = 0*(-1) - log(1 + exp(-1)) = 0 - log(1 + 0.3679...) = -log(1.3679..)
        var likelihood = new BernoulliLikelihood<double>();

        var y = MakeVector(0.0);
        var f = MakeVector(-1.0);

        double expected = 0.0 * (-1.0) - Math.Log(1 + Math.Exp(-1.0));
        double actual = likelihood.LogLikelihood(y, f);

        AssertClose(expected, actual, Tolerance, "Bernoulli LogLikelihood y=0: ");
    }

    [Fact]
    public void BernoulliLikelihood_LogLikelihood_MultipleObservations()
    {
        var likelihood = new BernoulliLikelihood<double>();

        var y = MakeVector(1.0, 0.0, 1.0);
        var f = MakeVector(2.0, -1.0, 0.5);

        // Sum: y[i]*f[i] - log(1+exp(f[i]))
        double expected = 0;
        double[] ys = { 1.0, 0.0, 1.0 };
        double[] fs = { 2.0, -1.0, 0.5 };
        for (int i = 0; i < 3; i++)
            expected += ys[i] * fs[i] - Math.Log(1 + Math.Exp(fs[i]));

        double actual = likelihood.LogLikelihood(y, f);
        AssertClose(expected, actual, Tolerance, "Bernoulli LogLikelihood multi: ");
    }

    [Fact]
    public void BernoulliLikelihood_Gradient_MatchesNumerical()
    {
        var likelihood = new BernoulliLikelihood<double>();

        var y = MakeVector(1.0, 0.0, 1.0, 0.0);
        var f = MakeVector(0.5, -0.5, 1.5, -2.0);

        var analyticalGrad = likelihood.LogLikelihoodGradient(y, f);
        var numericalGrad = NumericalGradient(f,
            fv => likelihood.LogLikelihood(y, fv));

        AssertVectorsClose(numericalGrad, analyticalGrad, GradientTolerance, "Bernoulli gradient: ");
    }

    [Fact]
    public void BernoulliLikelihood_Gradient_HandCalculated()
    {
        // ∂log p/∂fi = yi - σ(fi)
        var likelihood = new BernoulliLikelihood<double>();

        var y = MakeVector(1.0, 0.0);
        var f = MakeVector(0.0, 0.0);

        var grad = likelihood.LogLikelihoodGradient(y, f);

        // σ(0) = 0.5
        // grad[0] = 1 - 0.5 = 0.5
        // grad[1] = 0 - 0.5 = -0.5
        AssertClose(0.5, grad[0], Tolerance, "Bernoulli grad[0]: ");
        AssertClose(-0.5, grad[1], Tolerance, "Bernoulli grad[1]: ");
    }

    [Fact]
    public void BernoulliLikelihood_HessianDiag_HandCalculated()
    {
        // Hessian = -σ(f) * (1 - σ(f))
        var likelihood = new BernoulliLikelihood<double>();

        var y = MakeVector(1.0);
        var f = MakeVector(0.0);

        var hess = likelihood.LogLikelihoodHessianDiag(y, f);

        // σ(0) = 0.5, hessian = -0.5 * 0.5 = -0.25
        AssertClose(-0.25, hess[0], Tolerance, "Bernoulli hessian at f=0: ");
    }

    [Fact]
    public void BernoulliLikelihood_HessianDiag_MatchesNumerical()
    {
        var likelihood = new BernoulliLikelihood<double>();

        var y = MakeVector(1.0, 0.0, 1.0);
        var f = MakeVector(1.0, -1.0, 0.5);

        var analyticalHess = likelihood.LogLikelihoodHessianDiag(y, f);
        var numericalHess = NumericalHessianDiag(y, f,
            (yv, fv) => likelihood.LogLikelihoodGradient(yv, fv));

        AssertVectorsClose(numericalHess, analyticalHess, GradientTolerance, "Bernoulli hessian: ");
    }

    [Fact]
    public void BernoulliLikelihood_HessianDiag_AlwaysNegative()
    {
        // -σ(f)*(1-σ(f)) is always negative
        var likelihood = new BernoulliLikelihood<double>();

        var y = MakeVector(1.0, 0.0, 1.0);
        var f = MakeVector(5.0, -5.0, 0.0);

        var hess = likelihood.LogLikelihoodHessianDiag(y, f);

        for (int i = 0; i < 3; i++)
            Assert.True(hess[i] < 0, $"Bernoulli hessian[{i}] should be negative, got {hess[i]}");
    }

    [Fact]
    public void BernoulliLikelihood_HessianDiag_MaxMagnitudeAtZero()
    {
        // |Hessian| is maximized at f=0 (σ=0.5, so -0.5*0.5=-0.25)
        var likelihood = new BernoulliLikelihood<double>();

        var y = MakeVector(1.0, 1.0, 1.0);
        var fZero = MakeVector(0.0, 0.0, 0.0);
        var fLarge = MakeVector(3.0, 3.0, 3.0);

        var hessZero = likelihood.LogLikelihoodHessianDiag(y, fZero);
        var hessLarge = likelihood.LogLikelihoodHessianDiag(y, fLarge);

        Assert.True(Math.Abs(hessZero[0]) > Math.Abs(hessLarge[0]),
            $"|H(0)|={Math.Abs(hessZero[0])} should be > |H(3)|={Math.Abs(hessLarge[0])}");
    }

    [Fact]
    public void BernoulliLikelihood_TransformMean_IsSigmoid()
    {
        var likelihood = new BernoulliLikelihood<double>();

        // σ(0) = 0.5
        AssertClose(0.5, likelihood.TransformMean(0.0), Tolerance, "sigmoid(0): ");

        // σ(large) ≈ 1
        Assert.True(likelihood.TransformMean(10.0) > 0.999, "sigmoid(10) should be near 1");

        // σ(-large) ≈ 0
        Assert.True(likelihood.TransformMean(-10.0) < 0.001, "sigmoid(-10) should be near 0");
    }

    [Fact]
    public void BernoulliLikelihood_LogLikelihood_ConvertsNegativeLabels()
    {
        // Labels -1 should be converted to 0
        var likelihood = new BernoulliLikelihood<double>();

        var yNeg = MakeVector(-1.0);
        var yZero = MakeVector(0.0);
        var f = MakeVector(1.0);

        double llNeg = likelihood.LogLikelihood(yNeg, f);
        double llZero = likelihood.LogLikelihood(yZero, f);

        AssertClose(llZero, llNeg, Tolerance, "Negative label conversion: ");
    }

    #endregion

    // ==================== Poisson Likelihood Tests ====================

    #region Poisson Likelihood

    [Fact]
    public void PoissonLikelihood_LogLikelihood_HandCalculated()
    {
        // log p(y|f) = y*f - exp(f) - log(y!)
        var likelihood = new PoissonLikelihood<double>();

        // y=3, f=1.0
        // log p = 3*1 - exp(1) - log(3!) = 3 - 2.71828... - log(6) = 3 - 2.71828 - 1.79176 = -1.51004
        var y = MakeVector(3.0);
        var f = MakeVector(1.0);

        double expected = 3.0 * 1.0 - Math.Exp(1.0) - Math.Log(6.0); // log(3!) = log(6)
        double actual = likelihood.LogLikelihood(y, f);

        AssertClose(expected, actual, Tolerance, "Poisson LogLikelihood: ");
    }

    [Fact]
    public void PoissonLikelihood_LogLikelihood_ZeroCount()
    {
        // y=0, f=0.5
        // log p = 0*0.5 - exp(0.5) - log(0!) = 0 - 1.6487... - 0 = -1.6487...
        var likelihood = new PoissonLikelihood<double>();

        var y = MakeVector(0.0);
        var f = MakeVector(0.5);

        double expected = 0.0 * 0.5 - Math.Exp(0.5) - 0.0; // log(0!) = 0
        double actual = likelihood.LogLikelihood(y, f);

        AssertClose(expected, actual, Tolerance, "Poisson LogLikelihood y=0: ");
    }

    [Fact]
    public void PoissonLikelihood_Gradient_MatchesNumerical()
    {
        var likelihood = new PoissonLikelihood<double>();

        var y = MakeVector(2.0, 5.0, 0.0);
        var f = MakeVector(0.5, 1.5, -0.5);

        var analyticalGrad = likelihood.LogLikelihoodGradient(y, f);
        var numericalGrad = NumericalGradient(f,
            fv => likelihood.LogLikelihood(y, fv));

        AssertVectorsClose(numericalGrad, analyticalGrad, GradientTolerance, "Poisson gradient: ");
    }

    [Fact]
    public void PoissonLikelihood_Gradient_HandCalculated()
    {
        // ∂log p/∂fi = yi - exp(fi)
        var likelihood = new PoissonLikelihood<double>();

        var y = MakeVector(3.0, 1.0);
        var f = MakeVector(1.0, 0.0);

        var grad = likelihood.LogLikelihoodGradient(y, f);

        // grad[0] = 3 - exp(1) = 3 - 2.71828... = 0.28172
        // grad[1] = 1 - exp(0) = 1 - 1 = 0
        AssertClose(3.0 - Math.Exp(1.0), grad[0], Tolerance, "Poisson grad[0]: ");
        AssertClose(0.0, grad[1], Tolerance, "Poisson grad[1]: ");
    }

    [Fact]
    public void PoissonLikelihood_HessianDiag_HandCalculated()
    {
        // Hessian = -exp(fi)
        var likelihood = new PoissonLikelihood<double>();

        var y = MakeVector(1.0, 2.0);
        var f = MakeVector(0.0, 1.0);

        var hess = likelihood.LogLikelihoodHessianDiag(y, f);

        AssertClose(-Math.Exp(0.0), hess[0], Tolerance, "Poisson hessian[0]: ");
        AssertClose(-Math.Exp(1.0), hess[1], Tolerance, "Poisson hessian[1]: ");
    }

    [Fact]
    public void PoissonLikelihood_HessianDiag_MatchesNumerical()
    {
        var likelihood = new PoissonLikelihood<double>();

        var y = MakeVector(3.0, 1.0, 0.0);
        var f = MakeVector(0.5, 1.0, -0.5);

        var analyticalHess = likelihood.LogLikelihoodHessianDiag(y, f);
        var numericalHess = NumericalHessianDiag(y, f,
            (yv, fv) => likelihood.LogLikelihoodGradient(yv, fv));

        AssertVectorsClose(numericalHess, analyticalHess, GradientTolerance, "Poisson hessian: ");
    }

    [Fact]
    public void PoissonLikelihood_TransformMean_IsExp()
    {
        var likelihood = new PoissonLikelihood<double>();

        // exp(0) = 1
        AssertClose(1.0, likelihood.TransformMean(0.0), Tolerance, "Poisson TransformMean(0): ");

        // exp(1) = 2.71828...
        AssertClose(Math.Exp(1.0), likelihood.TransformMean(1.0), Tolerance, "Poisson TransformMean(1): ");

        // exp(-1) = 0.36788...
        AssertClose(Math.Exp(-1.0), likelihood.TransformMean(-1.0), Tolerance, "Poisson TransformMean(-1): ");
    }

    [Fact]
    public void PoissonLikelihood_GradientIsZero_WhenCountEqualsRate()
    {
        // When y == exp(f), gradient = y - exp(f) = 0
        var likelihood = new PoissonLikelihood<double>();

        // y=exp(1)≈2.718, f=1.0 => grad = exp(1) - exp(1) = 0
        var y = MakeVector(Math.Exp(1.0));
        var f = MakeVector(1.0);

        var grad = likelihood.LogLikelihoodGradient(y, f);
        AssertClose(0.0, grad[0], Tolerance, "Poisson gradient at equilibrium: ");
    }

    #endregion

    // ==================== Student-t Likelihood Tests ====================

    #region Student-t Likelihood

    [Fact]
    public void StudentTLikelihood_LogLikelihood_HandCalculated()
    {
        // log p(y|f) = logΓ((ν+1)/2) - logΓ(ν/2) - 0.5*log(νπσ²) - (ν+1)/2 * log(1 + (y-f)²/(νσ²))
        double nu = 4.0;
        double sigma = 0.5;
        var likelihood = new StudentTLikelihood<double>(sigma, nu);

        var y = MakeVector(1.0);
        var f = MakeVector(0.8);

        double diff = 1.0 - 0.8;
        double sigma2 = sigma * sigma;
        double z2 = diff * diff / (nu * sigma2);

        // Use .NET LogGamma equivalent through the formula
        double logNorm = LogGamma(0.5 * (nu + 1)) - LogGamma(0.5 * nu) - 0.5 * Math.Log(nu * Math.PI * sigma2);
        double expected = logNorm - 0.5 * (nu + 1) * Math.Log(1 + z2);

        double actual = likelihood.LogLikelihood(y, f);
        AssertClose(expected, actual, 1e-3, "StudentT LogLikelihood: ");
    }

    [Fact]
    public void StudentTLikelihood_Gradient_MatchesNumerical()
    {
        var likelihood = new StudentTLikelihood<double>(0.5, 4.0);

        var y = MakeVector(1.0, -0.5, 2.0);
        var f = MakeVector(0.8, -0.3, 1.5);

        var analyticalGrad = likelihood.LogLikelihoodGradient(y, f);
        var numericalGrad = NumericalGradient(f,
            fv => likelihood.LogLikelihood(y, fv));

        AssertVectorsClose(numericalGrad, analyticalGrad, GradientTolerance, "StudentT gradient: ");
    }

    [Fact]
    public void StudentTLikelihood_Gradient_HandCalculated()
    {
        // ∂log p/∂fi = (ν+1) * (yi-fi) / (νσ² + (yi-fi)²)
        double nu = 4.0;
        double sigma = 0.5;
        var likelihood = new StudentTLikelihood<double>(sigma, nu);

        var y = MakeVector(1.0);
        var f = MakeVector(0.0);

        var grad = likelihood.LogLikelihoodGradient(y, f);

        double diff = 1.0;
        double sigma2 = sigma * sigma;
        double expected = (nu + 1) * diff / (nu * sigma2 + diff * diff);

        AssertClose(expected, grad[0], Tolerance, "StudentT gradient hand: ");
    }

    [Fact]
    public void StudentTLikelihood_HessianDiag_MatchesNumerical()
    {
        var likelihood = new StudentTLikelihood<double>(0.5, 4.0);

        var y = MakeVector(1.0, -0.5, 2.0);
        var f = MakeVector(0.8, -0.3, 1.5);

        var analyticalHess = likelihood.LogLikelihoodHessianDiag(y, f);
        var numericalHess = NumericalHessianDiag(y, f,
            (yv, fv) => likelihood.LogLikelihoodGradient(yv, fv));

        AssertVectorsClose(numericalHess, analyticalHess, GradientTolerance, "StudentT hessian: ");
    }

    [Fact]
    public void StudentTLikelihood_Gradient_BoundedForOutliers()
    {
        // Student-t gradient is bounded: |(ν+1)*r/(νσ²+r²)| ≤ (ν+1)/(2√(νσ²))
        // This means outliers have limited influence (unlike Gaussian)
        var likelihood = new StudentTLikelihood<double>(0.5, 4.0);

        var yNormal = MakeVector(1.0);
        var yOutlier = MakeVector(100.0);
        var f = MakeVector(0.0);

        var gradNormal = likelihood.LogLikelihoodGradient(yNormal, f);
        var gradOutlier = likelihood.LogLikelihoodGradient(yOutlier, f);

        // Outlier gradient should be small relative to residual
        // |grad for outlier 100| << 100 (the residual itself)
        double outlierGradMagnitude = Math.Abs(gradOutlier[0]);
        Assert.True(outlierGradMagnitude < 1.0,
            $"StudentT outlier gradient should be bounded, got {outlierGradMagnitude}");
    }

    [Fact]
    public void StudentTLikelihood_PredictiveVariance_CorrectFormula()
    {
        // For ν>2: noise variance = νσ²/(ν-2)
        double nu = 6.0;
        double sigma = 0.5;
        var likelihood = new StudentTLikelihood<double>(sigma, nu);

        double fVariance = 1.0;
        double result = likelihood.PredictiveVariance(0.0, fVariance);

        double noiseVar = nu * sigma * sigma / (nu - 2);
        double expected = fVariance + noiseVar;

        AssertClose(expected, result, Tolerance, "StudentT PredictiveVariance: ");
    }

    [Fact]
    public void StudentTLikelihood_HessianDiag_HandCalculated()
    {
        // ∂²/∂f² = -(ν+1) * (νσ² - (y-f)²) / (νσ² + (y-f)²)²
        double nu = 4.0;
        double sigma = 0.5;
        var likelihood = new StudentTLikelihood<double>(sigma, nu);

        var y = MakeVector(1.0);
        var f = MakeVector(0.5);

        var hess = likelihood.LogLikelihoodHessianDiag(y, f);

        double diff = 0.5;
        double diff2 = diff * diff;
        double nuSigma2 = nu * sigma * sigma;
        double denom = nuSigma2 + diff2;
        double expected = -(nu + 1) * (nuSigma2 - diff2) / (denom * denom);

        AssertClose(expected, hess[0], Tolerance, "StudentT hessian hand: ");
    }

    #endregion

    // ==================== Zero Mean Tests ====================

    #region ZeroMean

    [Fact]
    public void ZeroMean_EvaluateSingle_ReturnsZero()
    {
        var mean = new ZeroMean<double>();
        var x = MakeVector(1.0, 2.0, 3.0);

        double result = mean.Evaluate(x);
        AssertClose(0.0, result, Tolerance, "ZeroMean single: ");
    }

    [Fact]
    public void ZeroMean_EvaluateMatrix_ReturnsZeros()
    {
        var mean = new ZeroMean<double>();
        var X = MakeMatrix(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } });

        var result = mean.Evaluate(X);

        Assert.Equal(3, result.Length);
        for (int i = 0; i < 3; i++)
            AssertClose(0.0, result[i], Tolerance, $"ZeroMean matrix[{i}]: ");
    }

    #endregion

    // ==================== Constant Mean Tests ====================

    #region ConstantMean

    [Fact]
    public void ConstantMean_EvaluateSingle_ReturnsConstant()
    {
        var mean = new ConstantMean<double>(5.0);
        var x = MakeVector(1.0, 2.0, 3.0);

        double result = mean.Evaluate(x);
        AssertClose(5.0, result, Tolerance, "ConstantMean single: ");
    }

    [Fact]
    public void ConstantMean_EvaluateMatrix_ReturnsSameConstant()
    {
        var mean = new ConstantMean<double>(3.14);
        var X = MakeMatrix(new double[,] { { 1.0 }, { 100.0 }, { -50.0 } });

        var result = mean.Evaluate(X);

        Assert.Equal(3, result.Length);
        for (int i = 0; i < 3; i++)
            AssertClose(3.14, result[i], Tolerance, $"ConstantMean matrix[{i}]: ");
    }

    [Fact]
    public void ConstantMean_NegativeConstant()
    {
        var mean = new ConstantMean<double>(-2.5);
        var x = MakeVector(999.0);

        double result = mean.Evaluate(x);
        AssertClose(-2.5, result, Tolerance, "ConstantMean negative: ");
    }

    #endregion

    // ==================== Linear Mean Tests ====================

    #region LinearMean

    [Fact]
    public void LinearMean_EvaluateSingle_HandCalculated()
    {
        // m(x) = w^T * x + b = 2*1 + 3*2 + 1 = 9
        var mean = new LinearMean<double>(new double[] { 2.0, 3.0 }, 1.0);
        var x = MakeVector(1.0, 2.0);

        double result = mean.Evaluate(x);
        AssertClose(9.0, result, Tolerance, "LinearMean single: ");
    }

    [Fact]
    public void LinearMean_EvaluateMatrix_HandCalculated()
    {
        var mean = new LinearMean<double>(new double[] { 1.0, -1.0 }, 0.5);
        var X = MakeMatrix(new double[,]
        {
            { 2.0, 1.0 },   // 1*2 + (-1)*1 + 0.5 = 1.5
            { 0.0, 3.0 },   // 1*0 + (-1)*3 + 0.5 = -2.5
            { 1.0, 1.0 },   // 1*1 + (-1)*1 + 0.5 = 0.5
        });

        var result = mean.Evaluate(X);

        Assert.Equal(3, result.Length);
        AssertClose(1.5, result[0], Tolerance, "LinearMean row 0: ");
        AssertClose(-2.5, result[1], Tolerance, "LinearMean row 1: ");
        AssertClose(0.5, result[2], Tolerance, "LinearMean row 2: ");
    }

    [Fact]
    public void LinearMean_ZeroWeights_ReturnsBias()
    {
        var mean = new LinearMean<double>(new double[] { 0.0, 0.0 }, 7.0);
        var x = MakeVector(100.0, -50.0);

        double result = mean.Evaluate(x);
        AssertClose(7.0, result, Tolerance, "LinearMean zero weights: ");
    }

    [Fact]
    public void LinearMean_DimensionMismatch_Throws()
    {
        var mean = new LinearMean<double>(new double[] { 1.0, 2.0 }, 0.0);
        var x = MakeVector(1.0, 2.0, 3.0); // 3D input but 2D weights

        Assert.Throws<ArgumentException>(() => mean.Evaluate(x));
    }

    [Fact]
    public void LinearMean_FromData_FitsLinearTrend()
    {
        // y = 2*x + 1 (perfect linear data)
        var X = MakeMatrix(new double[,]
        {
            { 0.0 },
            { 1.0 },
            { 2.0 },
            { 3.0 },
            { 4.0 },
        });
        var y = MakeVector(1.0, 3.0, 5.0, 7.0, 9.0);

        var mean = LinearMean<double>.FromData(X, y);

        // Should recover w ≈ 2.0, b ≈ 1.0
        var weights = mean.Weights;
        AssertClose(2.0, weights[0], 0.01, "LinearMean.FromData weight: ");
        AssertClose(1.0, mean.Bias, 0.01, "LinearMean.FromData bias: ");
    }

    [Fact]
    public void LinearMean_FromData_2D_FitsCorrectly()
    {
        // y = 1*x1 + 2*x2 + 3
        var X = MakeMatrix(new double[,]
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 },
            { 1.0, 1.0 },
            { 2.0, 1.0 },
            { 1.0, 2.0 },
        });
        var y = MakeVector(4.0, 5.0, 6.0, 7.0, 8.0);

        var mean = LinearMean<double>.FromData(X, y);

        var weights = mean.Weights;
        AssertClose(1.0, weights[0], 0.05, "LinearMean.FromData 2D w1: ");
        AssertClose(2.0, weights[1], 0.05, "LinearMean.FromData 2D w2: ");
        AssertClose(3.0, mean.Bias, 0.05, "LinearMean.FromData 2D bias: ");
    }

    #endregion

    // ==================== Polynomial Mean Tests ====================

    #region PolynomialMean

    [Fact]
    public void PolynomialMean_Constant_ReturnsCoeff0()
    {
        // m(x) = 5.0
        var mean = new PolynomialMean<double>(new double[] { 5.0 });
        var x = MakeVector(100.0);

        double result = mean.Evaluate(x);
        AssertClose(5.0, result, Tolerance, "PolynomialMean constant: ");
    }

    [Fact]
    public void PolynomialMean_Linear_HandCalculated()
    {
        // m(x) = 2 + 3x
        var mean = new PolynomialMean<double>(new double[] { 2.0, 3.0 });
        var x = MakeVector(4.0);

        double result = mean.Evaluate(x);
        // 2 + 3*4 = 14
        AssertClose(14.0, result, Tolerance, "PolynomialMean linear: ");
    }

    [Fact]
    public void PolynomialMean_Quadratic_HandCalculated()
    {
        // m(x) = 1 + 0*x + 2*x²
        var mean = new PolynomialMean<double>(new double[] { 1.0, 0.0, 2.0 });
        var x = MakeVector(3.0);

        double result = mean.Evaluate(x);
        // 1 + 0 + 2*9 = 19
        AssertClose(19.0, result, Tolerance, "PolynomialMean quadratic: ");
    }

    [Fact]
    public void PolynomialMean_Cubic_HandCalculated()
    {
        // m(x) = 1 + 2x + 3x² + 4x³
        var mean = new PolynomialMean<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var x = MakeVector(2.0);

        double result = mean.Evaluate(x);
        // 1 + 2*2 + 3*4 + 4*8 = 1 + 4 + 12 + 32 = 49
        AssertClose(49.0, result, Tolerance, "PolynomialMean cubic: ");
    }

    [Fact]
    public void PolynomialMean_AtZero_ReturnsConstant()
    {
        var mean = new PolynomialMean<double>(new double[] { 7.0, 3.0, 5.0 });
        var x = MakeVector(0.0);

        double result = mean.Evaluate(x);
        // At x=0: only a_0 survives = 7.0
        AssertClose(7.0, result, Tolerance, "PolynomialMean at zero: ");
    }

    [Fact]
    public void PolynomialMean_NegativeInput_HandCalculated()
    {
        // m(x) = 1 + 2x + 3x², x = -1
        var mean = new PolynomialMean<double>(new double[] { 1.0, 2.0, 3.0 });
        var x = MakeVector(-1.0);

        double result = mean.Evaluate(x);
        // 1 + 2*(-1) + 3*1 = 1 - 2 + 3 = 2
        AssertClose(2.0, result, Tolerance, "PolynomialMean negative input: ");
    }

    [Fact]
    public void PolynomialMean_EvaluateMatrix_HandCalculated()
    {
        // m(x) = 1 + 2x
        var mean = new PolynomialMean<double>(new double[] { 1.0, 2.0 });
        var X = MakeMatrix(new double[,] { { 0.0 }, { 1.0 }, { -1.0 }, { 2.5 } });

        var result = mean.Evaluate(X);

        Assert.Equal(4, result.Length);
        AssertClose(1.0, result[0], Tolerance, "PolyMean row 0: ");     // 1+2*0=1
        AssertClose(3.0, result[1], Tolerance, "PolyMean row 1: ");     // 1+2*1=3
        AssertClose(-1.0, result[2], Tolerance, "PolyMean row 2: ");    // 1+2*(-1)=-1
        AssertClose(6.0, result[3], Tolerance, "PolyMean row 3: ");     // 1+2*2.5=6
    }

    #endregion

    // ==================== BetaLikelihood Tests ====================

    #region BetaLikelihood

    [Fact]
    public void BetaLikelihood_GetMeans_IsSigmoid()
    {
        var likelihood = new BetaLikelihood<double>(10.0);

        var f = MakeVector(0.0, 2.0, -2.0);
        var mu = likelihood.GetMeans(f);

        // σ(0) = 0.5
        AssertClose(0.5, mu[0], Tolerance, "BetaLikelihood sigmoid(0): ");

        // σ(2) ≈ 0.8808
        double expected = 1.0 / (1.0 + Math.Exp(-2.0));
        AssertClose(expected, mu[1], Tolerance, "BetaLikelihood sigmoid(2): ");

        // σ(-2) ≈ 0.1192
        expected = 1.0 / (1.0 + Math.Exp(2.0));
        AssertClose(expected, mu[2], Tolerance, "BetaLikelihood sigmoid(-2): ");
    }

    [Fact]
    public void BetaLikelihood_GetBetaParameters_HandCalculated()
    {
        var likelihood = new BetaLikelihood<double>(10.0);

        // mu = 0.7
        var (alpha, beta) = likelihood.GetBetaParameters(0.7);

        // α = μ*ν = 0.7*10 = 7
        // β = (1-μ)*ν = 0.3*10 = 3
        AssertClose(7.0, alpha, Tolerance, "Beta alpha: ");
        AssertClose(3.0, beta, Tolerance, "Beta beta: ");
    }

    [Fact]
    public void BetaLikelihood_GetBetaParameters_Symmetry()
    {
        var likelihood = new BetaLikelihood<double>(10.0);

        // mu = 0.5 → α = β = 5
        var (alpha, beta) = likelihood.GetBetaParameters(0.5);

        AssertClose(5.0, alpha, Tolerance, "Beta symmetric alpha: ");
        AssertClose(5.0, beta, Tolerance, "Beta symmetric beta: ");
    }

    [Fact]
    public void BetaLikelihood_GetMeans_OutputInRange()
    {
        var likelihood = new BetaLikelihood<double>(10.0);

        var f = MakeVector(-10.0, -5.0, 0.0, 5.0, 10.0);
        var mu = likelihood.GetMeans(f);

        for (int i = 0; i < mu.Length; i++)
        {
            Assert.True(mu[i] > 0.0 && mu[i] < 1.0,
                $"BetaLikelihood mean[{i}]={mu[i]} should be in (0,1)");
        }
    }

    #endregion

    // ==================== Cross-likelihood Consistency Tests ====================

    #region Cross-Likelihood Consistency

    [Fact]
    public void GaussianLikelihood_LogLikelihood_IncreasesWithBetterFit()
    {
        var likelihood = new GaussianLikelihood<double>(0.1);

        var y = MakeVector(1.0, 2.0, 3.0);
        var fGood = MakeVector(1.0, 2.0, 3.0);  // Perfect fit
        var fBad = MakeVector(0.0, 0.0, 0.0);    // Poor fit

        double llGood = likelihood.LogLikelihood(y, fGood);
        double llBad = likelihood.LogLikelihood(y, fBad);

        Assert.True(llGood > llBad,
            $"Good fit LogLik ({llGood}) should be > bad fit ({llBad})");
    }

    [Fact]
    public void PoissonLikelihood_LogLikelihood_IncreasesWithBetterFit()
    {
        var likelihood = new PoissonLikelihood<double>();

        var y = MakeVector(3.0, 5.0, 1.0);
        // Perfect fit: f = log(y), so exp(f) = y
        var fGood = MakeVector(Math.Log(3.0), Math.Log(5.0), Math.Log(1.0));
        var fBad = MakeVector(0.0, 0.0, 0.0);

        double llGood = likelihood.LogLikelihood(y, fGood);
        double llBad = likelihood.LogLikelihood(y, fBad);

        Assert.True(llGood > llBad,
            $"Good fit LogLik ({llGood}) should be > bad fit ({llBad})");
    }

    [Fact]
    public void BernoulliLikelihood_LogLikelihood_IncreasesWithBetterFit()
    {
        var likelihood = new BernoulliLikelihood<double>();

        var y = MakeVector(1.0, 0.0, 1.0);
        var fGood = MakeVector(3.0, -3.0, 3.0);    // Correct and confident
        var fBad = MakeVector(-3.0, 3.0, -3.0);     // Opposite

        double llGood = likelihood.LogLikelihood(y, fGood);
        double llBad = likelihood.LogLikelihood(y, fBad);

        Assert.True(llGood > llBad,
            $"Good fit LogLik ({llGood}) should be > bad fit ({llBad})");
    }

    [Fact]
    public void StudentTLikelihood_MoreRobust_ThanGaussian()
    {
        // With outliers, Student-t log-likelihood should degrade less than Gaussian
        double sigma2 = 1.0;
        var gaussian = new GaussianLikelihood<double>(sigma2);
        var studentT = new StudentTLikelihood<double>(1.0, 4.0);

        var y = MakeVector(1.0, 2.0, 100.0); // Last point is outlier
        var f = MakeVector(1.0, 2.0, 3.0);

        double gaussianLL = gaussian.LogLikelihood(y, f);
        double studentLL = studentT.LogLikelihood(y, f);

        // Student-t should have a less extreme negative log-likelihood with outliers
        // The ratio |Gaussian LL| / |StudentT LL| should be > 1 because Gaussian is more penalized
        Assert.True(Math.Abs(gaussianLL) > Math.Abs(studentLL),
            $"|Gaussian LL| ({Math.Abs(gaussianLL)}) should be > |StudentT LL| ({Math.Abs(studentLL)}) with outliers");
    }

    [Fact]
    public void AllLikelihoods_GradientZero_AtOptimalPrediction()
    {
        // Gaussian: grad=0 when y==f
        var gaussian = new GaussianLikelihood<double>(0.5);
        var y = MakeVector(2.0);
        var f = MakeVector(2.0);
        var grad = gaussian.LogLikelihoodGradient(y, f);
        AssertClose(0.0, grad[0], Tolerance, "Gaussian grad at optimal: ");

        // Poisson: grad=0 when y==exp(f) → f=log(y)
        var poisson = new PoissonLikelihood<double>();
        var yp = MakeVector(3.0);
        var fp = MakeVector(Math.Log(3.0));
        var gradP = poisson.LogLikelihoodGradient(yp, fp);
        AssertClose(0.0, gradP[0], Tolerance, "Poisson grad at optimal: ");
    }

    #endregion

    // ==================== Edge Cases and Numerical Stability ====================

    #region Edge Cases

    [Fact]
    public void GaussianLikelihood_SmallVariance_HighPenaltyForMismatch()
    {
        // With σ²=0.001 and residual=1.0: penalty = 1.0²/(2*0.001) = 500
        // Even the positive normalization term can't offset this
        var smallVar = new GaussianLikelihood<double>(0.001);

        var y = MakeVector(1.0);
        var f = MakeVector(2.0); // Residual of 1.0 with tiny variance → huge penalty

        double ll = smallVar.LogLikelihood(y, f);
        Assert.True(ll < -100.0, $"Small variance with large residual should give very negative LL, got {ll}");
    }

    [Fact]
    public void BernoulliLikelihood_ExtremeF_Stable()
    {
        var likelihood = new BernoulliLikelihood<double>();

        // Very large f values shouldn't cause overflow
        var y = MakeVector(1.0, 0.0);
        var f = MakeVector(30.0, -30.0);

        double ll = likelihood.LogLikelihood(y, f);
        Assert.False(double.IsNaN(ll), "LogLikelihood should not be NaN for extreme f");
        Assert.False(double.IsInfinity(ll), "LogLikelihood should not be Infinity for extreme f");
    }

    [Fact]
    public void PoissonLikelihood_ClipsLargeF()
    {
        var likelihood = new PoissonLikelihood<double>();

        // f=100 would give exp(100)=overflow, should be clipped
        var y = MakeVector(1.0);
        var f = MakeVector(100.0);

        double ll = likelihood.LogLikelihood(y, f);
        Assert.False(double.IsNaN(ll), "Poisson LogLikelihood should not be NaN");
        Assert.False(double.IsPositiveInfinity(ll), "Poisson LogLikelihood should not be +Inf");
    }

    [Fact]
    public void StudentTLikelihood_ZeroResidual_GradientIsZero()
    {
        var likelihood = new StudentTLikelihood<double>(0.5, 4.0);

        var y = MakeVector(2.0);
        var f = MakeVector(2.0);

        var grad = likelihood.LogLikelihoodGradient(y, f);
        AssertClose(0.0, grad[0], Tolerance, "StudentT zero-residual gradient: ");
    }

    [Fact]
    public void GaussianLikelihood_LargeVariance_MildPenalty()
    {
        var largeVar = new GaussianLikelihood<double>(100.0);

        var y = MakeVector(1.0);
        var f = MakeVector(5.0); // Large residual but large variance → mild penalty

        double ll = largeVar.LogLikelihood(y, f);
        // With σ²=100, residual=4, penalty = 4²/(2*100)=0.08, total ≈ logNorm - 0.08
        Assert.True(ll > -5.0, $"Large variance should give mild penalty, got {ll}");
    }

    [Fact]
    public void PolynomialMean_HighDegree_HornersMethod()
    {
        // Test that Horner's method works correctly for degree 5
        // m(x) = 1 + x + x² + x³ + x⁴ + x⁵
        var mean = new PolynomialMean<double>(new double[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });
        var x = MakeVector(2.0);

        double result = mean.Evaluate(x);
        // 1 + 2 + 4 + 8 + 16 + 32 = 63
        AssertClose(63.0, result, Tolerance, "PolynomialMean degree 5: ");
    }

    [Fact]
    public void LinearMean_FromData_Overdetermined_LeastSquares()
    {
        // Noisy data: y ≈ 3*x + 2 with noise
        var X = MakeMatrix(new double[,]
        {
            { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 },
            { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 },
        });
        var y = MakeVector(2.1, 5.2, 7.9, 11.1, 14.0, 17.2, 19.8, 23.1, 25.9, 29.0);

        var mean = LinearMean<double>.FromData(X, y);

        // Should recover approximately w ≈ 3.0, b ≈ 2.0
        var weights = mean.Weights;
        AssertClose(3.0, weights[0], 0.2, "LinearMean.FromData noisy w: ");
        AssertClose(2.0, mean.Bias, 0.5, "LinearMean.FromData noisy b: ");
    }

    #endregion

    // ==================== LogGamma Helper (for StudentT verification) ====================

    /// <summary>
    /// LogGamma using Stirling's approximation with recurrence for small values.
    /// </summary>
    private static double LogGamma(double x)
    {
        if (x <= 0) return 0;

        if (x > 10)
        {
            return (x - 0.5) * Math.Log(x) - x + 0.5 * Math.Log(2 * Math.PI) + 1.0 / (12.0 * x);
        }

        double result = 0;
        while (x < 10)
        {
            result -= Math.Log(x);
            x += 1;
        }

        return result + (x - 0.5) * Math.Log(x) - x + 0.5 * Math.Log(2 * Math.PI) + 1.0 / (12.0 * x);
    }
}
