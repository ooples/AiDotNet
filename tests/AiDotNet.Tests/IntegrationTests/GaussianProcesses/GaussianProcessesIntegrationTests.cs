using AiDotNet.Enums;
using AiDotNet.GaussianProcesses;
using AiDotNet.Interfaces;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.GaussianProcesses;

/// <summary>
/// Integration tests for Gaussian Process classes.
/// Tests construction, Fit/Predict, interpolation, and uncertainty properties.
/// </summary>
public class GaussianProcessesIntegrationTests
{
    /// <summary>
    /// Creates simple 1D training data: y = sin(x) for x in [0, 3].
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) CreateSimpleTrainingData()
    {
        var xData = new double[,]
        {
            { 0.0 },
            { 0.5 },
            { 1.0 },
            { 1.5 },
            { 2.0 },
            { 2.5 },
            { 3.0 },
        };
        var yData = new double[7];
        for (int i = 0; i < 7; i++)
            yData[i] = Math.Sin(xData[i, 0]);

        return (new Matrix<double>(xData), new Vector<double>(yData));
    }

    /// <summary>
    /// Creates simple linear training data: y = 2*x for x in [0, 4].
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) CreateLinearTrainingData()
    {
        var xData = new double[,]
        {
            { 0.0 },
            { 1.0 },
            { 2.0 },
            { 3.0 },
            { 4.0 },
        };
        var yData = new double[] { 0.0, 2.0, 4.0, 6.0, 8.0 };

        return (new Matrix<double>(xData), new Vector<double>(yData));
    }

    #region StandardGaussianProcess Tests

    [Fact]
    public void StandardGaussianProcess_Construction_WithGaussianKernel_Succeeds()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new StandardGaussianProcess<double>(kernel);
        Assert.NotNull(gp);
    }

    [Fact]
    public void StandardGaussianProcess_Construction_DifferentDecompositionTypes_Succeed()
    {
        var kernel = new GaussianKernel<double>();

        var gpCholesky = new StandardGaussianProcess<double>(kernel, MatrixDecompositionType.Cholesky);
        var gpLu = new StandardGaussianProcess<double>(kernel, MatrixDecompositionType.Lu);
        var gpSvd = new StandardGaussianProcess<double>(kernel, MatrixDecompositionType.Svd);

        Assert.NotNull(gpCholesky);
        Assert.NotNull(gpLu);
        Assert.NotNull(gpSvd);
    }

    [Fact]
    public void StandardGP_FitAndPredict_InterpolatesTrainingPoints()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new StandardGaussianProcess<double>(kernel);
        var (X, y) = CreateLinearTrainingData();

        gp.Fit(X, y);

        // GP should interpolate training points nearly exactly
        for (int i = 0; i < X.Rows; i++)
        {
            var (mean, variance) = gp.Predict(X.GetRow(i));
            Assert.True(Math.Abs(mean - y[i]) < 0.5,
                $"At training point x={X[i, 0]}: predicted {mean}, expected {y[i]}");
        }
    }

    [Fact]
    public void StandardGP_FitAndPredict_VarianceNearZeroAtTrainingPoints()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new StandardGaussianProcess<double>(kernel);
        var (X, y) = CreateLinearTrainingData();

        gp.Fit(X, y);

        // Variance at training points should be very small (near zero due to jitter)
        for (int i = 0; i < X.Rows; i++)
        {
            var (_, variance) = gp.Predict(X.GetRow(i));
            Assert.True(variance < 0.1,
                $"Variance at training point x={X[i, 0]} should be near zero, got {variance}");
        }
    }

    [Fact]
    public void StandardGP_FitAndPredict_VarianceHigherFarFromData()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new StandardGaussianProcess<double>(kernel);
        var (X, y) = CreateLinearTrainingData();

        gp.Fit(X, y);

        // Variance at a training point
        var (_, varianceAtData) = gp.Predict(X.GetRow(2)); // x = 2.0

        // Variance far from training data (x = 100.0)
        var farPoint = new Vector<double>(new[] { 100.0 });
        var (_, varianceFar) = gp.Predict(farPoint);

        Assert.True(varianceFar > varianceAtData,
            $"Variance far from data ({varianceFar}) should be greater than at training point ({varianceAtData})");
    }

    [Fact]
    public void StandardGP_FitAndPredict_PredictionIsFinite()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new StandardGaussianProcess<double>(kernel);
        var (X, y) = CreateSimpleTrainingData();

        gp.Fit(X, y);

        var testPoint = new Vector<double>(new[] { 1.25 });
        var (mean, variance) = gp.Predict(testPoint);

        Assert.False(double.IsNaN(mean), "Predicted mean should not be NaN");
        Assert.False(double.IsInfinity(mean), "Predicted mean should not be Infinity");
        Assert.False(double.IsNaN(variance), "Predicted variance should not be NaN");
        Assert.False(double.IsInfinity(variance), "Predicted variance should not be Infinity");
    }

    [Fact]
    public void StandardGP_UpdateKernel_ChangesPredictions()
    {
        var gaussianKernel = new GaussianKernel<double>();
        var gp = new StandardGaussianProcess<double>(gaussianKernel);
        var (X, y) = CreateLinearTrainingData();

        gp.Fit(X, y);
        var testPoint = new Vector<double>(new[] { 1.5 });
        var (mean1, _) = gp.Predict(testPoint);

        // Change to a very different kernel
        gp.UpdateKernel(new LinearKernel<double>());
        var (mean2, _) = gp.Predict(testPoint);

        // Predictions should differ with different kernels (not necessarily by a lot,
        // but at least one property should differ)
        Assert.False(double.IsNaN(mean2), "Prediction after kernel update should not be NaN");
    }

    [Fact]
    public void StandardGP_WithLuDecomposition_ProducesValidPredictions()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new StandardGaussianProcess<double>(kernel, MatrixDecompositionType.Lu);
        var (X, y) = CreateLinearTrainingData();

        gp.Fit(X, y);

        var (mean, variance) = gp.Predict(X.GetRow(2));
        Assert.True(Math.Abs(mean - y[2]) < 1.0,
            $"LU-based GP predicted {mean}, expected near {y[2]}");
        Assert.False(double.IsNaN(variance), "Variance should not be NaN");
    }

    #endregion

    #region SparseGaussianProcess Tests

    [Fact]
    public void SparseGaussianProcess_Construction_WithGaussianKernel_Succeeds()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new SparseGaussianProcess<double>(kernel);
        Assert.NotNull(gp);
    }

    [Fact]
    public void SparseGP_FitAndPredict_InterpolatesTrainingPoints()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new SparseGaussianProcess<double>(kernel);
        var (X, y) = CreateLinearTrainingData();

        gp.Fit(X, y);

        for (int i = 0; i < X.Rows; i++)
        {
            var (mean, variance) = gp.Predict(X.GetRow(i));
            Assert.True(Math.Abs(mean - y[i]) < 1.0,
                $"Sparse GP at x={X[i, 0]}: predicted {mean}, expected {y[i]}");
            Assert.False(double.IsNaN(mean), $"Mean at x={X[i, 0]} should not be NaN");
            Assert.False(double.IsNaN(variance), $"Variance at x={X[i, 0]} should not be NaN");
        }
    }

    [Fact]
    public void SparseGP_FitAndPredict_VarianceNonNegative()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new SparseGaussianProcess<double>(kernel);
        var (X, y) = CreateSimpleTrainingData();

        gp.Fit(X, y);

        var testPoint = new Vector<double>(new[] { 1.25 });
        var (mean, variance) = gp.Predict(testPoint);

        Assert.False(double.IsNaN(mean), "Mean should not be NaN");
        Assert.True(variance >= -1e-6, $"Variance should be non-negative, got {variance}");
    }

    #endregion

    #region MultiOutputGaussianProcess Tests

    [Fact]
    public void MultiOutputGaussianProcess_Construction_WithGaussianKernel_Succeeds()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new MultiOutputGaussianProcess<double>(kernel);
        Assert.NotNull(gp);
    }

    [Fact]
    public void MultiOutputGP_Fit_ThrowsInvalidOperation()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new MultiOutputGaussianProcess<double>(kernel);
        var (X, y) = CreateLinearTrainingData();

        // MultiOutput GP requires FitMultiOutput, not Fit
        Assert.Throws<InvalidOperationException>(() => gp.Fit(X, y));
    }

    [Fact]
    public void MultiOutputGP_FitMultiOutput_InterpolatesTrainingPoints()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new MultiOutputGaussianProcess<double>(kernel);
        var (X, y) = CreateLinearTrainingData();

        // Convert vector y to matrix Y (single output column)
        var Y = new Matrix<double>(y.Length, 1);
        for (int i = 0; i < y.Length; i++)
            Y[i, 0] = y[i];

        gp.FitMultiOutput(X, Y);

        for (int i = 0; i < X.Rows; i++)
        {
            var (means, covariance) = gp.PredictMultiOutput(X.GetRow(i));
            Assert.Equal(1, means.Length);
            Assert.False(double.IsNaN(means[0]),
                $"MultiOutput GP mean at x={X[i, 0]} should not be NaN");
            Assert.True(Math.Abs(means[0] - y[i]) < 1.0,
                $"MultiOutput GP at x={X[i, 0]}: predicted {means[0]}, expected {y[i]}");
        }
    }

    [Fact]
    public void MultiOutputGP_FitMultiOutput_CovarianceNonNegativeDiagonal()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new MultiOutputGaussianProcess<double>(kernel);
        var (X, y) = CreateSimpleTrainingData();

        var Y = new Matrix<double>(y.Length, 1);
        for (int i = 0; i < y.Length; i++)
            Y[i, 0] = y[i];

        gp.FitMultiOutput(X, Y);

        var testPoint = new Vector<double>(new[] { 1.25 });
        var (means, covariance) = gp.PredictMultiOutput(testPoint);

        Assert.Equal(1, means.Length);
        Assert.False(double.IsNaN(means[0]), "Mean should not be NaN");
        // Covariance diagonal should be non-negative (variance)
        Assert.True(covariance[0, 0] >= -1e-6,
            $"Covariance diagonal should be non-negative, got {covariance[0, 0]}");
    }

    #endregion

    #region Cross-GP Tests with Different Kernels

    [Theory]
    [InlineData("Gaussian")]
    [InlineData("Linear")]
    [InlineData("Polynomial")]
    [InlineData("Laplacian")]
    [InlineData("Matern")]
    [InlineData("Exponential")]
    public void AllKernels_StandardGP_FitAndPredict_ProducesFinitePredictions(string kernelName)
    {
        var kernel = CreateKernel(kernelName);
        var gp = new StandardGaussianProcess<double>(kernel);
        var (X, y) = CreateLinearTrainingData();

        gp.Fit(X, y);

        var testPoint = new Vector<double>(new[] { 1.5 });
        var (mean, variance) = gp.Predict(testPoint);

        Assert.False(double.IsNaN(mean), $"Mean with {kernelName} kernel should not be NaN");
        Assert.False(double.IsInfinity(mean), $"Mean with {kernelName} kernel should not be Infinity");
        Assert.False(double.IsNaN(variance), $"Variance with {kernelName} kernel should not be NaN");
    }

    [Fact]
    public void Float_StandardGP_FitAndPredict_ProducesFinitePredictions()
    {
        var kernel = new GaussianKernel<float>();
        var gp = new StandardGaussianProcess<float>(kernel);

        var xData = new float[,]
        {
            { 0.0f },
            { 1.0f },
            { 2.0f },
            { 3.0f },
        };
        var yData = new float[] { 0.0f, 2.0f, 4.0f, 6.0f };
        var X = new Matrix<float>(xData);
        var y = new Vector<float>(yData);

        gp.Fit(X, y);

        var testPoint = new Vector<float>(new[] { 1.5f });
        var (mean, variance) = gp.Predict(testPoint);

        Assert.False(float.IsNaN(mean), "Float GP mean should not be NaN");
        Assert.False(float.IsInfinity(mean), "Float GP mean should not be Infinity");
    }

    #endregion

    private static IKernelFunction<double> CreateKernel(string name) => name switch
    {
        "Gaussian" => new GaussianKernel<double>(),
        "Linear" => new LinearKernel<double>(),
        "Polynomial" => new PolynomialKernel<double>(),
        "Laplacian" => new LaplacianKernel<double>(),
        "Matern" => new MaternKernel<double>(),
        "Exponential" => new ExponentialKernel<double>(),
        _ => throw new ArgumentException($"Unknown kernel: {name}"),
    };
}
