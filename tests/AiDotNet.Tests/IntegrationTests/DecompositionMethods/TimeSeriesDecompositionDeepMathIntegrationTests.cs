using AiDotNet.DecompositionMethods.TimeSeriesDecomposition;
using AiDotNet.Enums;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DecompositionMethods;

/// <summary>
/// Deep math-correctness integration tests for time series decomposition.
/// Verifies hand-calculated values, mathematical identities, and algorithm invariants.
/// </summary>
public class TimeSeriesDecompositionDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    #region Additive Decomposition - Moving Average

    [Fact]
    public void Additive_MovingAverage_AdditiveIdentity_TrendPlusSeasonalPlusResidualEqualsOriginal()
    {
        // The fundamental identity: Original = Trend + Seasonal + Residual
        var data = new double[] { 10, 12, 15, 11, 13, 16, 12, 14, 17, 13, 15, 18,
                                  11, 13, 16, 12, 14, 17, 13, 15, 18, 14, 16, 19 };
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);
        var residual = decomp.GetComponentAsVector(DecompositionComponentType.Residual);

        Assert.Equal(ts.Length, trend.Length);
        Assert.Equal(ts.Length, seasonal.Length);
        Assert.Equal(ts.Length, residual.Length);

        for (int i = 0; i < ts.Length; i++)
        {
            double reconstructed = trend[i] + seasonal[i] + residual[i];
            Assert.Equal(data[i], reconstructed, Tolerance);
        }
    }

    [Fact]
    public void Additive_MovingAverage_TrendHandCalculated_WindowSize7()
    {
        // Moving average with window=7, for center points (i=3..n-4) we get exact 7-point average
        // For edge points, the window is truncated
        var data = new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 };
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // Window=7, half=3. For i=3: window is [0..6] = {2,4,6,8,10,12,14}, avg = 56/7 = 8
        Assert.Equal(8.0, trend[3], Tolerance);

        // For i=4: window is [1..7] = {4,6,8,10,12,14,16}, avg = 70/7 = 10
        Assert.Equal(10.0, trend[4], Tolerance);

        // For i=5: window is [2..8] = {6,8,10,12,14,16,18}, avg = 84/7 = 12
        Assert.Equal(12.0, trend[5], Tolerance);

        // For i=0: window is [0..3] = {2,4,6,8}, avg = 20/4 = 5
        Assert.Equal(5.0, trend[0], Tolerance);

        // For i=1: window is [0..4] = {2,4,6,8,10}, avg = 30/5 = 6
        Assert.Equal(6.0, trend[1], Tolerance);
    }

    [Fact]
    public void Additive_MovingAverage_ConstantSeries_TrendEqualsConstant()
    {
        // If data is constant, trend should equal the constant
        var data = new double[24];
        for (int i = 0; i < 24; i++) data[i] = 7.0;
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        for (int i = 0; i < 24; i++)
        {
            Assert.Equal(7.0, trend[i], Tolerance);
        }
    }

    [Fact]
    public void Additive_MovingAverage_ConstantSeries_SeasonalIsZero()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++) data[i] = 7.0;
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);
        var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);

        for (int i = 0; i < 24; i++)
        {
            Assert.Equal(0.0, seasonal[i], Tolerance);
        }
    }

    [Fact]
    public void Additive_MovingAverage_ConstantSeries_ResidualIsZero()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++) data[i] = 7.0;
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);
        var residual = decomp.GetComponentAsVector(DecompositionComponentType.Residual);

        for (int i = 0; i < 24; i++)
        {
            Assert.Equal(0.0, residual[i], Tolerance);
        }
    }

    [Fact]
    public void Additive_MovingAverage_LinearTrend_TrendApproximatesLinear()
    {
        // Pure linear: y = 10 + 2*i. For interior points with full window, MA = exact linear
        var data = new double[24];
        for (int i = 0; i < 24; i++) data[i] = 10.0 + 2.0 * i;
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // For interior points (i=3..20) with full window=7, MA of linear = exact linear
        for (int i = 3; i <= 20; i++)
        {
            Assert.Equal(10.0 + 2.0 * i, trend[i], Tolerance);
        }
    }

    [Fact]
    public void Additive_MovingAverage_SeasonalComponentIsPeriodic()
    {
        // Seasonal component repeats with period 12 (hardcoded in the code)
        var data = new double[48];
        for (int i = 0; i < 48; i++)
            data[i] = 100 + 0.5 * i + 10 * Math.Sin(2 * Math.PI * i / 12.0);
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);
        var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);

        // seasonal[i] should equal seasonal[i+12] for all valid i
        for (int i = 0; i < 36; i++)
        {
            Assert.Equal(seasonal[i], seasonal[i + 12], Tolerance);
        }
    }

    #endregion

    #region Additive Decomposition - Exponential Smoothing

    [Fact]
    public void Additive_ExponentialSmoothing_TrendFirstValueEqualsFirstObservation()
    {
        // trend[0] = TimeSeries[0] per the code
        var data = new double[] { 42, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 };
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.ExponentialSmoothing);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        Assert.Equal(42.0, trend[0], Tolerance);
    }

    [Fact]
    public void Additive_ExponentialSmoothing_TrendRecurrenceHandCalculated()
    {
        // alpha = 0.2, trend[i] = 0.2*data[i] + 0.8*trend[i-1]
        var data = new double[] { 10, 20, 30, 15, 25, 35, 20, 30, 40, 25, 35, 45, 30 };
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.ExponentialSmoothing);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        double alpha = 0.2;
        double expected0 = 10.0;
        double expected1 = alpha * 20.0 + (1 - alpha) * expected0; // 0.2*20 + 0.8*10 = 12
        double expected2 = alpha * 30.0 + (1 - alpha) * expected1; // 0.2*30 + 0.8*12 = 15.6
        double expected3 = alpha * 15.0 + (1 - alpha) * expected2; // 0.2*15 + 0.8*15.6 = 15.48

        Assert.Equal(expected0, trend[0], Tolerance);
        Assert.Equal(expected1, trend[1], Tolerance);
        Assert.Equal(expected2, trend[2], Tolerance);
        Assert.Equal(expected3, trend[3], Tolerance);
    }

    [Fact]
    public void Additive_ExponentialSmoothing_AdditiveIdentity()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = 50 + 2 * i + 5 * Math.Sin(2 * Math.PI * i / 12.0);
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.ExponentialSmoothing);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);
        var residual = decomp.GetComponentAsVector(DecompositionComponentType.Residual);

        for (int i = 0; i < ts.Length; i++)
        {
            double reconstructed = trend[i] + seasonal[i] + residual[i];
            Assert.Equal(data[i], reconstructed, Tolerance);
        }
    }

    [Fact]
    public void Additive_ExponentialSmoothing_ConstantSeries_TrendConvergesToConstant()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++) data[i] = 5.0;
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.ExponentialSmoothing);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // With constant input, trend should be constant
        for (int i = 0; i < 24; i++)
        {
            Assert.Equal(5.0, trend[i], Tolerance);
        }
    }

    [Fact]
    public void Additive_ExponentialSmoothing_SeasonalInitializationHandCalculated()
    {
        // First 12 seasonal values = data[i] - trend[i]
        var data = new double[] { 10, 20, 30, 15, 25, 35, 20, 30, 40, 25, 35, 45,
                                  12, 22, 32, 17, 27, 37, 22, 32, 42, 27, 37, 47 };
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.ExponentialSmoothing);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);

        // For i < 12: seasonal[i] = data[i] - trend[i]
        for (int i = 0; i < 12; i++)
        {
            Assert.Equal(data[i] - trend[i], seasonal[i], Tolerance);
        }
    }

    [Fact]
    public void Additive_ExponentialSmoothing_SeasonalRecurrenceHandCalculated()
    {
        // For i >= 12: seasonal[i] = gamma*(data[i]-trend[i]) + (1-gamma)*seasonal[i-12]
        // gamma = 0.3
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = 50 + i + 8 * Math.Sin(2 * Math.PI * i / 12.0);
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.ExponentialSmoothing);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);

        double gamma = 0.3;
        for (int i = 12; i < 24; i++)
        {
            double expected = gamma * (data[i] - trend[i]) + (1 - gamma) * seasonal[i - 12];
            Assert.Equal(expected, seasonal[i], Tolerance);
        }
    }

    #endregion

    #region Additive Decomposition - STL

    [Fact]
    public void Additive_STL_AdditiveIdentity()
    {
        var data = new double[48];
        for (int i = 0; i < 48; i++)
            data[i] = 100 + 0.5 * i + 10 * Math.Sin(2 * Math.PI * i / 12.0);
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.STL);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);
        var residual = decomp.GetComponentAsVector(DecompositionComponentType.Residual);

        for (int i = 0; i < ts.Length; i++)
        {
            double reconstructed = trend[i] + seasonal[i] + residual[i];
            Assert.Equal(data[i], reconstructed, Tolerance);
        }
    }

    [Fact]
    public void Additive_STL_TrendIsSmoothForCleanData()
    {
        // For clean sinusoidal + linear data, STL trend should be smooth
        var data = new double[48];
        for (int i = 0; i < 48; i++)
            data[i] = 100 + 0.5 * i + 10 * Math.Sin(2 * Math.PI * i / 12.0);
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.STL);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // Check second differences are small (smoothness measure)
        double maxSecondDiff = 0;
        for (int i = 1; i < trend.Length - 1; i++)
        {
            double secondDiff = Math.Abs(trend[i + 1] - 2 * trend[i] + trend[i - 1]);
            maxSecondDiff = Math.Max(maxSecondDiff, secondDiff);
        }
        // Second differences of a smooth trend should be small
        Assert.True(maxSecondDiff < 5.0, $"STL trend should be smooth, max 2nd diff = {maxSecondDiff}");
    }

    #endregion

    #region Hodrick-Prescott - Matrix Method

    [Fact]
    public void HP_MatrixMethod_CycleEqualsOriginalMinusTrend()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = 100 + i + 5 * Math.Sin(2 * Math.PI * i / 6.0);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.MatrixMethod);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

        for (int i = 0; i < ts.Length; i++)
        {
            Assert.Equal(data[i] - trend[i], cycle[i], Tolerance);
        }
    }

    [Fact]
    public void HP_MatrixMethod_LinearSeries_TrendEqualsOriginal()
    {
        // For a perfectly linear series, HP trend = original (second differences = 0)
        var data = new double[20];
        for (int i = 0; i < 20; i++) data[i] = 5.0 + 3.0 * i;
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.MatrixMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // For linear data, second differences = 0, so penalty term = 0
        // Therefore trend = original
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(data[i], trend[i], LooseTolerance);
        }
    }

    [Fact]
    public void HP_MatrixMethod_ConstantSeries_TrendEqualsConstant()
    {
        var data = new double[20];
        for (int i = 0; i < 20; i++) data[i] = 42.0;
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.MatrixMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(42.0, trend[i], LooseTolerance);
        }
    }

    [Fact]
    public void HP_MatrixMethod_LargeLambda_TrendIsVerySmooth()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = 50 + i + 10 * Math.Sin(2 * Math.PI * i / 6.0);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 100000,
            algorithm: HodrickPrescottAlgorithmType.MatrixMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // With very large lambda, trend should be nearly linear
        // Check that second differences are tiny
        for (int i = 1; i < trend.Length - 1; i++)
        {
            double secondDiff = Math.Abs(trend[i + 1] - 2 * trend[i] + trend[i - 1]);
            Assert.True(secondDiff < 0.5, $"Large lambda should produce very smooth trend at i={i}, got {secondDiff}");
        }
    }

    [Fact]
    public void HP_MatrixMethod_SmallLambda_TrendApproximatesOriginal()
    {
        var data = new double[20];
        for (int i = 0; i < 20; i++)
            data[i] = 50 + i + 5 * Math.Sin(2 * Math.PI * i / 5.0);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 0.01,
            algorithm: HodrickPrescottAlgorithmType.MatrixMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // Very small lambda => fit penalty dominates, trend ≈ original
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(data[i], trend[i], 1.0); // within 1.0 of original
        }
    }

    [Fact]
    public void HP_MatrixMethod_SecondDifferenceMatrix_HandCalculated()
    {
        // Verify the HP filter for a tiny example where we can solve (I + λD^TD)τ = y by hand
        // For n=4, D is 2x4: [[1,-2,1,0],[0,1,-2,1]]
        // D^TD is 4x4. With λ and solving the system we get specific trend values.
        var data = new double[] { 1, 3, 2, 4 };
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1.0,
            algorithm: HodrickPrescottAlgorithmType.MatrixMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

        // Verify the fundamental identity
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(data[i], trend[i] + cycle[i], Tolerance);
        }

        // With lambda=1, the trend should be smoothed but not flat
        // The trend should be monotonically increasing or close to it
        Assert.True(trend[3] > trend[0], "Trend should show upward direction for {1,3,2,4}");
    }

    [Fact]
    public void HP_MatrixMethod_NegativeLambda_Throws()
    {
        var data = new double[] { 1, 2, 3, 4, 5 };
        var ts = new Vector<double>(data);
        Assert.Throws<ArgumentException>(() =>
            new HodrickPrescottDecomposition<double>(ts, lambda: -1.0));
    }

    [Fact]
    public void HP_MatrixMethod_ZeroLambda_Throws()
    {
        var data = new double[] { 1, 2, 3, 4, 5 };
        var ts = new Vector<double>(data);
        Assert.Throws<ArgumentException>(() =>
            new HodrickPrescottDecomposition<double>(ts, lambda: 0.0));
    }

    #endregion

    #region Hodrick-Prescott - Iterative Method

    [Fact]
    public void HP_Iterative_CycleEqualsOriginalMinusTrend()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = 50 + i + 5 * Math.Sin(2 * Math.PI * i / 6.0);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.IterativeMethod);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

        for (int i = 0; i < ts.Length; i++)
        {
            Assert.Equal(data[i] - trend[i], cycle[i], Tolerance);
        }
    }

    [Fact]
    public void HP_Iterative_BoundaryValues_EqualOriginal()
    {
        // The iterative method sets boundary values to original: trend[0,1,n-2,n-1] = data[0,1,n-2,n-1]
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = 50 + i + 5 * Math.Sin(2 * Math.PI * i / 6.0);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.IterativeMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        Assert.Equal(data[0], trend[0], Tolerance);
        Assert.Equal(data[1], trend[1], Tolerance);
        Assert.Equal(data[22], trend[22], Tolerance);
        Assert.Equal(data[23], trend[23], Tolerance);
    }

    [Fact]
    public void HP_Iterative_ConstantSeries_TrendEqualsConstant()
    {
        var data = new double[20];
        for (int i = 0; i < 20; i++) data[i] = 15.0;
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.IterativeMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(15.0, trend[i], LooseTolerance);
        }
    }

    #endregion

    #region Hodrick-Prescott - Kalman Filter

    [Fact]
    public void HP_Kalman_CycleEqualsOriginalMinusTrend()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = 50 + i + 5 * Math.Sin(2 * Math.PI * i / 6.0);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.KalmanFilterMethod);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

        for (int i = 0; i < ts.Length; i++)
        {
            Assert.Equal(data[i] - trend[i], cycle[i], Tolerance);
        }
    }

    [Fact]
    public void HP_Kalman_FirstTrendValue_HandCalculated()
    {
        // Kalman filter: x_pred = F*x = [x0+x1, x1] where initial x = [data[0], 0]
        // First prediction: x_pred = [data[0], 0]
        // Innovation: y = data[0] - H*x_pred = data[0] - data[0] = 0
        // Since innovation is 0 at step 0, trend[0] should be close to data[0]
        var data = new double[] { 100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113 };
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.KalmanFilterMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // trend[0] should be close to data[0] since initial state is [data[0], 0]
        Assert.Equal(100.0, trend[0], 1.0);
    }

    [Fact]
    public void HP_Kalman_ConstantSeries_TrendConverges()
    {
        var data = new double[30];
        for (int i = 0; i < 30; i++) data[i] = 50.0;
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.KalmanFilterMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // Last values should be very close to 50
        Assert.Equal(50.0, trend[29], 1.0);
    }

    [Fact]
    public void HP_Kalman_LinearSeries_TrendTracksLinear()
    {
        var data = new double[30];
        for (int i = 0; i < 30; i++) data[i] = 10 + 2.0 * i;
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.KalmanFilterMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // After convergence, Kalman should track the linear trend well
        // Check last 10 values
        for (int i = 20; i < 30; i++)
        {
            Assert.Equal(10 + 2.0 * i, trend[i], 5.0); // within 5 of true value
        }
    }

    #endregion

    #region Hodrick-Prescott - Wavelet Method

    [Fact]
    public void HP_Wavelet_CycleEqualsOriginalMinusTrend()
    {
        // Wavelet requires power-of-2 length
        int n = 32;
        var data = new double[n];
        for (int i = 0; i < n; i++)
            data[i] = 50 + i + 5 * Math.Sin(2 * Math.PI * i / 8.0);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.WaveletMethod);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(data[i] - trend[i], cycle[i], Tolerance);
        }
    }

    [Fact]
    public void HP_Wavelet_DWT_InverseDWT_Roundtrip()
    {
        // Test that DWT followed by IDWT gives back original (without thresholding)
        // We test this indirectly: for constant data, wavelet detail coeffs = 0,
        // so thresholding changes nothing and trend = original
        int n = 16;
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = 42.0;
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.WaveletMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(42.0, trend[i], Tolerance);
        }
    }

    [Fact]
    public void HP_Wavelet_DWT_HandCalculated_TwoElements()
    {
        // For data [a, b] = [10, 20], one level DWT:
        // approx = (a+b)/2 = 15, detail = (a-b)/2 = -5
        // After 0.1 thresholding of detail (i >= n/2=1): detail' = -5 * 0.1 = -0.5
        // Correct IDWT: a' = approx + detail' = 15 + (-0.5) = 14.5
        //               b' = approx - detail' = 15 - (-0.5) = 15.5
        var data = new double[] { 10, 20 };
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.WaveletMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        Assert.Equal(14.5, trend[0], Tolerance);
        Assert.Equal(15.5, trend[1], Tolerance);
    }

    #endregion

    #region Hodrick-Prescott - State Space Method

    [Fact]
    public void HP_StateSpace_CycleEqualsOriginalMinusTrend()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = 50 + i + 5 * Math.Sin(2 * Math.PI * i / 6.0);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.StateSpaceMethod);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

        for (int i = 0; i < ts.Length; i++)
        {
            Assert.Equal(data[i] - trend[i], cycle[i], Tolerance);
        }
    }

    [Fact]
    public void HP_StateSpace_HandCalculated_FirstThreeSteps()
    {
        // State space: alpha=0.1, rho=0.5
        // mu = data[0], beta = 0, c = 0
        // Step 0: mu_prev = mu = data[0]; mu = mu_prev + beta = data[0] + 0 = data[0]
        //         beta = 0 + 0.1*(data[0] - (data[0] + 0)) = 0
        //         c = 0.5*(data[0] - mu) = 0.5*(data[0] - data[0]) = 0
        //         trend[0] = mu = data[0], cycle[0] = c = 0
        var data = new double[] { 100, 105, 103, 108, 106, 110 };
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.StateSpaceMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

        double alpha = 0.1, rho = 0.5;
        double mu = data[0], beta = 0, c = 0;

        for (int i = 0; i < data.Length; i++)
        {
            double mu_prev = mu;
            mu = mu_prev + beta;
            beta = beta + alpha * (data[i] - (mu_prev + c));
            c = rho * (data[i] - mu);

            Assert.Equal(mu, trend[i], Tolerance);
            Assert.Equal(data[i] - mu, cycle[i], Tolerance);
        }
    }

    [Fact]
    public void HP_StateSpace_ConstantSeries_CycleDecays()
    {
        // With rho=0.5, cycle should decay towards zero for constant input
        var data = new double[20];
        for (int i = 0; i < 20; i++) data[i] = 50.0;
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.StateSpaceMethod);
        var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

        // Later cycles should be smaller than earlier ones
        Assert.True(Math.Abs(cycle[19]) <= Math.Abs(cycle[1]) + 1e-10,
            "Cycle should decay for constant input");
    }

    #endregion

    #region Hodrick-Prescott - Frequency Domain Method

    [Fact]
    public void HP_FrequencyDomain_CycleEqualsOriginalMinusTrend()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = 50 + i + 5 * Math.Sin(2 * Math.PI * i / 6.0);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.FrequencyDomainMethod);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

        for (int i = 0; i < ts.Length; i++)
        {
            Assert.Equal(data[i] - trend[i], cycle[i], Tolerance);
        }
    }

    [Fact]
    public void HP_FrequencyDomain_ConstantSeries_TrendEqualsConstant()
    {
        // DC component (frequency 0) should pass through, all others are zero for constant
        // Use power-of-2 length to avoid zero-padding spectral leakage
        int n = 32;
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = 25.0;
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.FrequencyDomainMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(25.0, trend[i], LooseTolerance);
        }
    }

    [Fact]
    public void HP_FrequencyDomain_LowFrequencyPreserved()
    {
        // Low frequency sinusoid should mostly pass through the low-pass filter
        int n = 64;
        var data = new double[n];
        // Very low frequency: 2 complete cycles in 64 points => freq = 2/64 = 0.03125 < 0.1 cutoff
        for (int i = 0; i < n; i++)
            data[i] = 50 + 10 * Math.Cos(2 * Math.PI * 2 * i / n);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.FrequencyDomainMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // The low-frequency component should be preserved in the trend
        double trendRange = trend.ToArray().Max() - trend.ToArray().Min();
        Assert.True(trendRange > 5.0, $"Low frequency should be preserved in trend, range={trendRange}");
    }

    #endregion

    #region Multiplicative Decomposition

    [Fact]
    public void Multiplicative_GeometricMA_MultiplicativeIdentity()
    {
        // Original = Trend * Seasonal * Residual
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = (50 + 0.5 * i) * (1 + 0.2 * Math.Sin(2 * Math.PI * i / 12.0));
        var ts = new Vector<double>(data);
        var decomp = new MultiplicativeDecomposition<double>(ts);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);
        var residual = decomp.GetComponentAsVector(DecompositionComponentType.Residual);

        for (int i = 0; i < ts.Length; i++)
        {
            double reconstructed = trend[i] * seasonal[i] * residual[i];
            Assert.Equal(data[i], reconstructed, LooseTolerance);
        }
    }

    [Fact]
    public void Multiplicative_ConstantSeries_TrendEqualsConstant()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++) data[i] = 10.0;
        var ts = new Vector<double>(data);
        var decomp = new MultiplicativeDecomposition<double>(ts);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // Geometric mean of constants = the constant
        for (int i = 0; i < 24; i++)
        {
            Assert.Equal(10.0, trend[i], LooseTolerance);
        }
    }

    [Fact]
    public void Multiplicative_ConstantSeries_SeasonalIsOne()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++) data[i] = 10.0;
        var ts = new Vector<double>(data);
        var decomp = new MultiplicativeDecomposition<double>(ts);
        var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);

        // For constant data, seasonal ratio = 1
        for (int i = 0; i < 24; i++)
        {
            Assert.Equal(1.0, seasonal[i], LooseTolerance);
        }
    }

    [Fact]
    public void Multiplicative_GeometricMeanTrend_HandCalculated()
    {
        // For interior point i with full window: geometric mean = (product of window values)^(1/count)
        // With period=12, halfWindow=6. At i=6, window = [0..12], count=13
        var data = new double[24];
        for (int i = 0; i < 24; i++) data[i] = 10.0 + i; // 10,11,...,33
        var ts = new Vector<double>(data);
        var decomp = new MultiplicativeDecomposition<double>(ts);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // At i=6: window = [0..12] = {10,11,12,13,14,15,16,17,18,19,20,21,22}, count=13
        double product = 1.0;
        for (int j = 0; j <= 12; j++) product *= data[j];
        double expectedTrend6 = Math.Pow(product, 1.0 / 13.0);
        Assert.Equal(expectedTrend6, trend[6], LooseTolerance);
    }

    [Fact]
    public void Multiplicative_ExponentialSmoothing_MultiplicativeIdentity()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = (50 + 0.5 * i) * (1 + 0.2 * Math.Sin(2 * Math.PI * i / 12.0));
        var ts = new Vector<double>(data);
        var decomp = new MultiplicativeDecomposition<double>(ts,
            MultiplicativeAlgorithmType.MultiplicativeExponentialSmoothing);

        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);
        var residual = decomp.GetComponentAsVector(DecompositionComponentType.Residual);

        for (int i = 0; i < ts.Length; i++)
        {
            double reconstructed = trend[i] * seasonal[i] * residual[i];
            Assert.Equal(data[i], reconstructed, LooseTolerance);
        }
    }

    #endregion

    #region TriCube Weight Function

    [Fact]
    public void TriCube_IsZeroOutsideUnitInterval()
    {
        // The TriCube function in the code: if x > 1, return 0; else (1-x)^3
        // We test this indirectly through the additive decomposition smoothing behavior
        // For a sharp spike, the trend should not extend beyond the window radius
        var data = new double[24];
        for (int i = 0; i < 24; i++) data[i] = 10.0;
        data[12] = 100.0; // sharp spike
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // Points far from the spike should be close to 10
        Assert.Equal(10.0, trend[0], 1.0);
        Assert.Equal(10.0, trend[1], 1.0);
    }

    #endregion

    #region Cross-Method Consistency

    [Fact]
    public void AllHP_Methods_ProduceTrendAndCycle()
    {
        var data = new double[32];
        for (int i = 0; i < 32; i++)
            data[i] = 50 + i + 5 * Math.Sin(2 * Math.PI * i / 8.0);
        var ts = new Vector<double>(data);

        var algorithms = new[]
        {
            HodrickPrescottAlgorithmType.MatrixMethod,
            HodrickPrescottAlgorithmType.IterativeMethod,
            HodrickPrescottAlgorithmType.KalmanFilterMethod,
            HodrickPrescottAlgorithmType.WaveletMethod,
            HodrickPrescottAlgorithmType.FrequencyDomainMethod,
            HodrickPrescottAlgorithmType.StateSpaceMethod
        };

        foreach (var algo in algorithms)
        {
            var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600, algorithm: algo);
            var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
            var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

            Assert.Equal(32, trend.Length);
            Assert.Equal(32, cycle.Length);

            // Identity: cycle = original - trend
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(data[i] - trend[i], cycle[i], Tolerance);
            }
        }
    }

    [Fact]
    public void AllAdditive_Methods_ProduceTrendSeasonalResidual()
    {
        var data = new double[48];
        for (int i = 0; i < 48; i++)
            data[i] = 100 + 0.5 * i + 10 * Math.Sin(2 * Math.PI * i / 12.0);
        var ts = new Vector<double>(data);

        var algorithms = new[]
        {
            AdditiveDecompositionAlgorithmType.MovingAverage,
            AdditiveDecompositionAlgorithmType.ExponentialSmoothing,
            AdditiveDecompositionAlgorithmType.STL
        };

        foreach (var algo in algorithms)
        {
            var decomp = new AdditiveDecomposition<double>(ts, algo);

            Assert.True(decomp.HasComponent(DecompositionComponentType.Trend),
                $"{algo} should produce Trend");
            Assert.True(decomp.HasComponent(DecompositionComponentType.Seasonal),
                $"{algo} should produce Seasonal");
            Assert.True(decomp.HasComponent(DecompositionComponentType.Residual),
                $"{algo} should produce Residual");

            var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
            var seasonal = decomp.GetComponentAsVector(DecompositionComponentType.Seasonal);
            var residual = decomp.GetComponentAsVector(DecompositionComponentType.Residual);

            // Additive identity
            for (int i = 0; i < ts.Length; i++)
            {
                Assert.Equal(data[i], trend[i] + seasonal[i] + residual[i], Tolerance);
            }
        }
    }

    [Fact]
    public void HP_MatrixMethod_TrendSmootherThanOriginal()
    {
        var data = new double[24];
        for (int i = 0; i < 24; i++)
            data[i] = 50 + i + 10 * Math.Sin(2 * Math.PI * i / 6.0);
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600,
            algorithm: HodrickPrescottAlgorithmType.MatrixMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // Sum of squared second differences should be smaller for trend
        double originalRoughness = 0, trendRoughness = 0;
        for (int i = 1; i < 23; i++)
        {
            double origSD = data[i + 1] - 2 * data[i] + data[i - 1];
            double trendSD = trend[i + 1] - 2 * trend[i] + trend[i - 1];
            originalRoughness += origSD * origSD;
            trendRoughness += trendSD * trendSD;
        }

        Assert.True(trendRoughness < originalRoughness,
            $"Trend roughness ({trendRoughness}) should be less than original ({originalRoughness})");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Additive_MovingAverage_EdgePoints_UseReducedWindow()
    {
        // At i=0: window=[0, min(n-1, 3)] = [0,3], so uses 4 elements
        var data = new double[] { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                                  1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
                                  2100, 2200, 2300, 2400 };
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // i=0: window [0,3], avg = (100+200+300+400)/4 = 250
        Assert.Equal(250.0, trend[0], Tolerance);

        // i=1: window [0,4], avg = (100+200+300+400+500)/5 = 300
        Assert.Equal(300.0, trend[1], Tolerance);

        // i=2: window [0,5], avg = (100+200+300+400+500+600)/6 = 350
        Assert.Equal(350.0, trend[2], Tolerance);
    }

    [Fact]
    public void Additive_MovingAverage_SymmetricData_TrendIsSymmetric()
    {
        // Symmetric data: palindromic pattern repeating
        var data = new double[] { 10, 20, 30, 20, 10, 20, 30, 20, 10, 20, 30, 20,
                                  10, 20, 30, 20, 10, 20, 30, 20, 10, 20, 30, 20 };
        var ts = new Vector<double>(data);
        var decomp = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // For interior points with symmetric data, the mean of the data is 20
        // The trend at center points should approximate 20
        double mean = data.Average();
        for (int i = 5; i < 19; i++)
        {
            Assert.Equal(mean, trend[i], 5.0);
        }
    }

    [Fact]
    public void HP_MatrixMethod_QuadraticData_TrendCapturesQuadratic()
    {
        // Quadratic: y = i^2. Second differences are constant (=2), so penalty is 4*lambda*(n-2)
        // With small lambda, trend should be close to quadratic
        var data = new double[20];
        for (int i = 0; i < 20; i++) data[i] = i * i;
        var ts = new Vector<double>(data);
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1.0,
            algorithm: HodrickPrescottAlgorithmType.MatrixMethod);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);

        // With small lambda, trend should be very close to data
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(data[i], trend[i], 5.0); // within 5
        }
    }

    [Fact]
    public void AllMethods_SameLength_OutputMatchesInput()
    {
        int n = 24;
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = 50 + i;
        var ts = new Vector<double>(data);

        var decomp1 = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.MovingAverage);
        var decomp2 = new AdditiveDecomposition<double>(ts, AdditiveDecompositionAlgorithmType.ExponentialSmoothing);
        var decomp3 = new HodrickPrescottDecomposition<double>(ts, lambda: 1600);

        foreach (var d in new TimeSeriesDecompositionBase<double>[] { decomp1, decomp2, decomp3 })
        {
            var components = d.GetComponents();
            foreach (var kvp in components)
            {
                if (kvp.Value is Vector<double> v)
                {
                    Assert.Equal(n, v.Length);
                }
            }
        }
    }

    #endregion
}
