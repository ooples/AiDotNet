using AiDotNet.LinearAlgebra;
using AiDotNet.DecompositionMethods.TimeSeriesDecomposition;
using AiDotNet.RadialBasisFunctions;
using AiDotNet.Interpolation;
using AiDotNet.WindowFunctions;
using AiDotNet.WaveletFunctions;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive coverage benchmarks for additional AiDotNet features
/// Covers interpolation, wavelets, window functions, RBF, and time series decomposition
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class ComprehensiveCoverageBenchmarks
{
    [Params(100, 500, 1000)]
    public int DataSize { get; set; }

    private Vector<double> _xData = null!;
    private Vector<double> _yData = null!;
    private Vector<double> _timeSeriesData = null!;
    private Vector<double> _signalData = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize interpolation data
        _xData = new Vector<double>(DataSize);
        _yData = new Vector<double>(DataSize);

        for (int i = 0; i < DataSize; i++)
        {
            _xData[i] = i;
            _yData[i] = Math.Sin(2 * Math.PI * i / 50) + random.NextDouble() * 0.1;
        }

        // Initialize time series data with trend and seasonality
        _timeSeriesData = new Vector<double>(DataSize);
        for (int i = 0; i < DataSize; i++)
        {
            double trend = i * 0.05;
            double seasonal = Math.Sin(2 * Math.PI * i / 12) * 10;
            double noise = random.NextDouble() * 2;
            _timeSeriesData[i] = trend + seasonal + noise + 50;
        }

        // Initialize signal data for window and wavelet functions
        _signalData = new Vector<double>(DataSize);
        for (int i = 0; i < DataSize; i++)
        {
            _signalData[i] = Math.Sin(2 * Math.PI * i / 20) + Math.Cos(2 * Math.PI * i / 40) * 0.5;
        }
    }

    #region Interpolation Methods

    [Benchmark(Baseline = true)]
    public double Interpolation_Linear()
    {
        var interp = new LinearInterpolation<double>();
        interp.Fit(_xData, _yData);
        return interp.Interpolate(DataSize / 2.5);
    }

    [Benchmark]
    public double Interpolation_Polynomial()
    {
        var interp = new PolynomialInterpolation<double>(degree: 3);
        interp.Fit(_xData, _yData);
        return interp.Interpolate(DataSize / 2.5);
    }

    [Benchmark]
    public double Interpolation_Spline()
    {
        var interp = new SplineInterpolation<double>();
        interp.Fit(_xData, _yData);
        return interp.Interpolate(DataSize / 2.5);
    }

    [Benchmark]
    public double Interpolation_CubicSpline()
    {
        var interp = new CubicSplineInterpolation<double>();
        interp.Fit(_xData, _yData);
        return interp.Interpolate(DataSize / 2.5);
    }

    [Benchmark]
    public double Interpolation_Hermite()
    {
        var interp = new HermiteInterpolation<double>();
        interp.Fit(_xData, _yData);
        return interp.Interpolate(DataSize / 2.5);
    }

    [Benchmark]
    public double Interpolation_Akima()
    {
        var interp = new AkimaInterpolation<double>();
        interp.Fit(_xData, _yData);
        return interp.Interpolate(DataSize / 2.5);
    }

    #endregion

    #region Time Series Decomposition

    [Benchmark]
    public (Vector<double> trend, Vector<double> seasonal, Vector<double> residual) TimeSeriesDecomposition_Additive()
    {
        var decomp = new AdditiveDecomposition<double>(seasonalPeriod: 12);
        return decomp.Decompose(_timeSeriesData);
    }

    [Benchmark]
    public (Vector<double> trend, Vector<double> seasonal, Vector<double> residual) TimeSeriesDecomposition_Multiplicative()
    {
        var decomp = new MultiplicativeDecomposition<double>(seasonalPeriod: 12);
        return decomp.Decompose(_timeSeriesData);
    }

    [Benchmark]
    public (Vector<double> trend, Vector<double> seasonal, Vector<double> residual) TimeSeriesDecomposition_STL()
    {
        var decomp = new STLTimeSeriesDecomposition<double>(seasonalPeriod: 12);
        return decomp.Decompose(_timeSeriesData);
    }

    [Benchmark]
    public (Vector<double> trend, Vector<double> cycle) TimeSeriesDecomposition_HodrickPrescott()
    {
        var decomp = new HodrickPrescottDecomposition<double>(lambda: 1600);
        return decomp.Decompose(_timeSeriesData);
    }

    #endregion

    #region Radial Basis Functions

    [Benchmark]
    public double RBF_Gaussian()
    {
        var rbf = new GaussianRBF<double>(epsilon: 1.0);
        return rbf.Evaluate(_xData[0], _xData[DataSize / 2]);
    }

    [Benchmark]
    public double RBF_Multiquadric()
    {
        var rbf = new MultiquadricRBF<double>(epsilon: 1.0);
        return rbf.Evaluate(_xData[0], _xData[DataSize / 2]);
    }

    [Benchmark]
    public double RBF_InverseMultiquadric()
    {
        var rbf = new InverseMultiquadricRBF<double>(epsilon: 1.0);
        return rbf.Evaluate(_xData[0], _xData[DataSize / 2]);
    }

    [Benchmark]
    public double RBF_ThinPlateSpline()
    {
        var rbf = new ThinPlateSplineRBF<double>();
        return rbf.Evaluate(_xData[0], _xData[DataSize / 2]);
    }

    #endregion

    #region Window Functions

    [Benchmark]
    public Vector<double> WindowFunction_Hamming()
    {
        var window = new HammingWindow<double>();
        return window.Apply(_signalData);
    }

    [Benchmark]
    public Vector<double> WindowFunction_Hanning()
    {
        var window = new HanningWindow<double>();
        return window.Apply(_signalData);
    }

    [Benchmark]
    public Vector<double> WindowFunction_Blackman()
    {
        var window = new BlackmanWindow<double>();
        return window.Apply(_signalData);
    }

    [Benchmark]
    public Vector<double> WindowFunction_Kaiser()
    {
        var window = new KaiserWindow<double>(alpha: 3.0);
        return window.Apply(_signalData);
    }

    [Benchmark]
    public Vector<double> WindowFunction_Bartlett()
    {
        var window = new BartlettWindow<double>();
        return window.Apply(_signalData);
    }

    [Benchmark]
    public Vector<double> WindowFunction_Tukey()
    {
        var window = new TukeyWindow<double>(alpha: 0.5);
        return window.Apply(_signalData);
    }

    #endregion

    #region Wavelet Functions

    [Benchmark]
    public (Vector<double> approximation, Vector<double> detail) Wavelet_Haar()
    {
        var wavelet = new HaarWavelet<double>();
        return wavelet.Transform(_signalData);
    }

    [Benchmark]
    public (Vector<double> approximation, Vector<double> detail) Wavelet_Daubechies()
    {
        var wavelet = new DaubechiesWavelet<double>(order: 4);
        return wavelet.Transform(_signalData);
    }

    [Benchmark]
    public (Vector<double> approximation, Vector<double> detail) Wavelet_Symlet()
    {
        var wavelet = new SymletWavelet<double>(order: 4);
        return wavelet.Transform(_signalData);
    }

    [Benchmark]
    public (Vector<double> approximation, Vector<double> detail) Wavelet_Coiflet()
    {
        var wavelet = new CoifletWavelet<double>(order: 2);
        return wavelet.Transform(_signalData);
    }

    [Benchmark]
    public (Vector<double> approximation, Vector<double> detail) Wavelet_Morlet()
    {
        var wavelet = new MorletWavelet<double>();
        return wavelet.Transform(_signalData);
    }

    #endregion

    #region RBF Interpolation

    [Benchmark]
    public double RBFInterpolation_Gaussian()
    {
        var interp = new RBFInterpolation<double>(new GaussianRBF<double>(epsilon: 1.0));
        interp.Fit(_xData, _yData);
        return interp.Interpolate(DataSize / 2.5);
    }

    [Benchmark]
    public double RBFInterpolation_Multiquadric()
    {
        var interp = new RBFInterpolation<double>(new MultiquadricRBF<double>(epsilon: 1.0));
        interp.Fit(_xData, _yData);
        return interp.Interpolate(DataSize / 2.5);
    }

    #endregion

    #region Batch Interpolation

    [Benchmark]
    public Vector<double> Interpolation_BatchLinear()
    {
        var interp = new LinearInterpolation<double>();
        interp.Fit(_xData, _yData);

        var queryPoints = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
        {
            queryPoints[i] = i * (DataSize / 50.0);
        }

        var results = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
        {
            results[i] = interp.Interpolate(queryPoints[i]);
        }
        return results;
    }

    [Benchmark]
    public Vector<double> Interpolation_BatchSpline()
    {
        var interp = new SplineInterpolation<double>();
        interp.Fit(_xData, _yData);

        var queryPoints = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
        {
            queryPoints[i] = i * (DataSize / 50.0);
        }

        var results = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
        {
            results[i] = interp.Interpolate(queryPoints[i]);
        }
        return results;
    }

    #endregion
}
