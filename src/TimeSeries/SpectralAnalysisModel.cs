
namespace AiDotNet.TimeSeries;

public class SpectralAnalysisModel<T> : TimeSeriesModelBase<T>
{
    private SpectralAnalysisOptions<T> _spectralOptions;
    private Vector<T> _frequencies;
    private Vector<T> _periodogram;

    public SpectralAnalysisModel(SpectralAnalysisOptions<T>? options = null) : base(options ?? new SpectralAnalysisOptions<T>())
    {
        _spectralOptions = (SpectralAnalysisOptions<T>)_options;
        _frequencies = new Vector<T>(1);
        _periodogram = new Vector<T>(1);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = y.Length;
        int nfft = _spectralOptions.NFFT;

        // Apply window function if specified
        Vector<T> windowedSignal = _spectralOptions.UseWindowFunction ? ApplyWindowFunction(y) : y;

        // Compute FFT
        Vector<Complex<T>> fft = ComputeFFT(windowedSignal, nfft);

        // Compute periodogram
        _periodogram = new Vector<T>(nfft / 2 + 1);
        for (int i = 0; i < _periodogram.Length; i++)
        {
            T magnitude = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(fft[i].Real, fft[i].Real), NumOps.Multiply(fft[i].Imaginary, fft[i].Imaginary)));
            _periodogram[i] = NumOps.Divide(NumOps.Multiply(magnitude, magnitude), NumOps.FromDouble(n));
        }

        // Compute frequencies
        _frequencies = new Vector<T>(nfft / 2 + 1);
        for (int i = 0; i < _frequencies.Length; i++)
        {
            _frequencies[i] = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(nfft));
        }
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        // Spectral analysis doesn't typically make predictions, but we can return the periodogram
        return _periodogram;
    }

    public Vector<T> GetFrequencies()
    {
        return _frequencies;
    }

    public Vector<T> GetPeriodogram()
    {
        return _periodogram;
    }

    private Vector<T> ApplyWindowFunction(Vector<T> signal)
    {
        var window = _spectralOptions.WindowFunction.Create(signal.Length);
        Vector<T> windowedSignal = new Vector<T>(signal.Length);
        for (int i = 0; i < signal.Length; i++)
        {
            windowedSignal[i] = NumOps.Multiply(signal[i], window[i]);
        }

        return windowedSignal;
    }

    private Vector<Complex<T>> ComputeFFT(Vector<T> signal, int nfft)
    {
        Vector<Complex<T>> padded = new Vector<Complex<T>>(nfft);
        for (int i = 0; i < signal.Length; i++)
        {
            padded[i] = new Complex<T>(signal[i], NumOps.Zero);
        }

        return FFT(padded);
    }

    private Vector<Complex<T>> FFT(Vector<Complex<T>> x)
    {
        int n = x.Length;
        if (n <= 1) return x;

        Vector<Complex<T>> even = new Vector<Complex<T>>(n / 2);
        Vector<Complex<T>> odd = new Vector<Complex<T>>(n / 2);

        for (int i = 0; i < n / 2; i++)
        {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        Vector<Complex<T>> evenFFT = FFT(even);
        Vector<Complex<T>> oddFFT = FFT(odd);

        Vector<Complex<T>> result = new Vector<Complex<T>>(n);
        for (int k = 0; k < n / 2; k++)
        {
            T angle = NumOps.Multiply(NumOps.FromDouble(-2 * Math.PI * k), NumOps.Divide(NumOps.One, NumOps.FromDouble(n)));
            Complex<T> t = Complex<T>.FromPolarCoordinates(NumOps.One, angle) * oddFFT[k];
            result[k] = evenFFT[k] + t;
            result[k + n / 2] = evenFFT[k] - t;
        }

        return result;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Serialize SpectralAnalysisOptions
        writer.Write(_spectralOptions.NFFT);
        writer.Write(_spectralOptions.UseWindowFunction);
        writer.Write((int)_spectralOptions.WindowFunction.GetWindowFunctionType());
        writer.Write(_spectralOptions.OverlapPercentage);

        // Serialize frequencies
        writer.Write(_frequencies.Length);
        for (int i = 0; i < _frequencies.Length; i++)
        {
            writer.Write(Convert.ToDouble(_frequencies[i]));
        }

        // Serialize periodogram
        writer.Write(_periodogram.Length);
        for (int i = 0; i < _periodogram.Length; i++)
        {
            writer.Write(Convert.ToDouble(_periodogram[i]));
        }
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Deserialize SpectralAnalysisOptions
        int nfft = reader.ReadInt32();
        bool useWindowFunction = reader.ReadBoolean();
        WindowFunctionType windowFunctionType = (WindowFunctionType)reader.ReadInt32();
        int overlapPercentage = reader.ReadInt32();

        _spectralOptions = new SpectralAnalysisOptions<T>
        {
            NFFT = nfft,
            UseWindowFunction = useWindowFunction,
            WindowFunction = WindowFunctionFactory.CreateWindowFunction<T>(windowFunctionType),
            OverlapPercentage = overlapPercentage
        };

        // Deserialize frequencies
        int frequenciesLength = reader.ReadInt32();
        _frequencies = new Vector<T>(frequenciesLength);
        for (int i = 0; i < frequenciesLength; i++)
        {
            _frequencies[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Deserialize periodogram
        int periodogramLength = reader.ReadInt32();
        _periodogram = new Vector<T>(periodogramLength);
        for (int i = 0; i < periodogramLength; i++)
        {
            _periodogram[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        var results = new Dictionary<string, T>();

        // Compute spectral analysis for the test data
        var testSpectralAnalysis = new SpectralAnalysisModel<T>(_spectralOptions);
        testSpectralAnalysis.Train(xTest, yTest);

        // Calculate Mean Squared Error (MSE) between training and test periodograms
        T mse = StatisticsHelper<T>.CalculateMeanSquaredError(_periodogram, testSpectralAnalysis._periodogram);
        results["MSE"] = mse;

        // Calculate Root Mean Squared Error (RMSE)
        T rmse = StatisticsHelper<T>.CalculateRootMeanSquaredError(_periodogram, testSpectralAnalysis._periodogram);
        results["RMSE"] = rmse;

        // Calculate Mean Absolute Error (MAE)
        T mae = StatisticsHelper<T>.CalculateMeanAbsoluteError(_periodogram, testSpectralAnalysis._periodogram);
        results["MAE"] = mae;

        // Calculate R-squared
        T r2 = StatisticsHelper<T>.CalculateR2(_periodogram, testSpectralAnalysis._periodogram);
        results["R2"] = r2;

        // Calculate peak frequency difference
        T peakFreqDiff = StatisticsHelper<T>.CalculatePeakDifference(_frequencies, _periodogram, 
                                                                     testSpectralAnalysis._frequencies, 
                                                                     testSpectralAnalysis._periodogram);
        results["PeakFrequencyDifference"] = peakFreqDiff;

        return results;
    }
}