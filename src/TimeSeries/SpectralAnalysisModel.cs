using System;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements spectral analysis for time series data, which transforms time domain signals into the frequency domain.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Spectral analysis is a technique used to analyze the frequency content of time series data. It helps identify
/// periodic patterns and dominant frequencies in the data by transforming it from the time domain to the frequency domain.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Spectral analysis is like breaking down a song into its individual notes. Just as a song is made up of different
/// notes played at different times, time series data can contain different patterns that repeat at different frequencies.
/// 
/// For example, if you analyze temperature data over several years, spectral analysis might reveal:
/// - A strong yearly cycle (frequency = 1/365 days) due to seasonal changes
/// - A daily cycle (frequency = 1/24 hours) due to day/night temperature differences
/// - Other cycles you might not notice just by looking at the raw data
/// 
/// This model uses the Fast Fourier Transform (FFT) algorithm to convert time data into frequency information,
/// showing you how strong each frequency component is in your data.
/// </para>
/// </remarks>
public class SpectralAnalysisModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Configuration options for the spectral analysis model.
    /// </summary>
    private SpectralAnalysisOptions<T> _spectralOptions;

    /// <summary>
    /// The frequency values corresponding to each point in the periodogram.
    /// </summary>
    private Vector<T> _frequencies;

    /// <summary>
    /// The power spectral density (periodogram) values for each frequency.
    /// </summary>
    private Vector<T> _periodogram;

    /// <summary>
    /// Initializes a new instance of the SpectralAnalysisModel class with optional configuration options.
    /// </summary>
    /// <param name="options">The configuration options for spectral analysis. If null, default options are used.</param>
    public SpectralAnalysisModel(SpectralAnalysisOptions<T>? options = null) : base(options ?? new SpectralAnalysisOptions<T>())
    {
        _spectralOptions = (SpectralAnalysisOptions<T>)_options;
        _frequencies = new Vector<T>(1);
        _periodogram = new Vector<T>(1);
    }

    /// <summary>
    /// Trains the spectral analysis model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input features matrix (not used in spectral analysis).</param>
    /// <param name="y">The time series data to analyze.</param>
    /// <remarks>
    /// <para>
    /// The training process involves:
    /// 1. Optionally applying a window function to the signal
    /// 2. Computing the Fast Fourier Transform (FFT)
    /// 3. Calculating the periodogram (power spectral density)
    /// 4. Computing the corresponding frequencies
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Training this model means analyzing your time series data to find repeating patterns. The process:
    /// 1. First, we might apply a "window function" to your data (this reduces artifacts in the analysis)
    /// 2. Then we use a mathematical technique called the Fast Fourier Transform (FFT) to convert your
    ///    time data into frequency information
    /// 3. Next, we calculate how strong each frequency is in your data (the periodogram)
    /// 4. Finally, we calculate what actual frequencies these values correspond to
    /// 
    /// After training, you can see which frequencies (or repeating patterns) are strongest in your data.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Returns the periodogram (power spectral density) as the prediction result.
    /// </summary>
    /// <param name="input">The input features matrix (not used in spectral analysis).</param>
    /// <returns>The periodogram (power spectral density) of the trained model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Unlike most prediction models, spectral analysis doesn't predict future values.
    /// Instead, it analyzes the frequency content of your data. This method simply returns
    /// the periodogram (the strength of each frequency component) that was calculated during training.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // Spectral analysis doesn't typically make predictions, but we can return the periodogram
        return _periodogram;
    }

    /// <summary>
    /// Gets the frequency values corresponding to each point in the periodogram.
    /// </summary>
    /// <returns>A vector of frequency values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method returns the frequencies that were analyzed in your data.
    /// Each value represents a frequency (how often a pattern repeats).
    /// For example, a frequency of 0.5 means the pattern repeats every 2 time units.
    /// </para>
    /// </remarks>
    public Vector<T> GetFrequencies()
    {
        return _frequencies;
    }

    /// <summary>
    /// Gets the periodogram (power spectral density) values for each frequency.
    /// </summary>
    /// <returns>A vector of periodogram values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method returns the "strength" of each frequency in your data.
    /// Higher values mean that frequency is more prominent in your data.
    /// By looking at which frequencies have the highest values, you can identify
    /// the most important repeating patterns in your time series.
    /// </para>
    /// </remarks>
    public Vector<T> GetPeriodogram()
    {
        return _periodogram;
    }

    /// <summary>
    /// Applies a window function to the input signal to reduce spectral leakage.
    /// </summary>
    /// <param name="signal">The original time series signal.</param>
    /// <returns>The windowed signal.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Window functions help improve the accuracy of spectral analysis by reducing "spectral leakage."
    /// 
    /// Imagine you're taking a photo through a window frame - the frame might block part of the view.
    /// Similarly, when we analyze only a portion of a time series, we get distortions at the edges.
    /// 
    /// Window functions gently reduce the signal at the edges (like fading in and out), which
    /// minimizes these distortions and gives more accurate frequency information.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Computes the Fast Fourier Transform (FFT) of the input signal.
    /// </summary>
    /// <param name="signal">The time series signal.</param>
    /// <param name="nfft">The number of points to use in the FFT calculation.</param>
    /// <returns>The complex FFT result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The Fast Fourier Transform (FFT) is a mathematical technique that converts time-based data
    /// into frequency-based data. It's like translating a piece of music from sheet music (which shows
    /// notes over time) into a graph showing which notes are played most often.
    /// 
    /// This method pads the signal with zeros if needed to reach the desired length (nfft),
    /// then converts each value to a complex number before performing the FFT calculation.
    /// </para>
    /// </remarks>
    private Vector<Complex<T>> ComputeFFT(Vector<T> signal, int nfft)
    {
        Vector<Complex<T>> padded = new Vector<Complex<T>>(nfft);
        for (int i = 0; i < signal.Length; i++)
        {
            padded[i] = new Complex<T>(signal[i], NumOps.Zero);
        }

        return FFT(padded);
    }

    /// <summary>
    /// Recursively computes the Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm.
    /// </summary>
    /// <param name="x">The complex input vector.</param>
    /// <returns>The complex FFT result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is the core FFT algorithm that efficiently converts time data to frequency data.
    /// It uses a "divide and conquer" approach by splitting the problem into smaller parts:
    /// 1. It divides the signal into even and odd indexed elements
    /// 2. It recursively computes the FFT of these smaller parts
    /// 3. It combines the results to get the FFT of the original signal
    /// 
    /// This approach is much faster than calculating the Fourier transform directly,
    /// especially for large datasets.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Serializes the model's core parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization is the process of converting the model's state into a format that can be saved to disk.
    /// This allows you to save a trained model and load it later without having to retrain it.
    /// 
    /// This method saves:
    /// - The spectral analysis options (like FFT size and window function settings)
    /// - The calculated frequencies
    /// - The calculated periodogram (power spectral density)
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Deserializes the model's core parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the process of loading a previously saved model from disk.
    /// This method reads the model's parameters from a file and reconstructs the model
    /// exactly as it was when it was saved.
    /// 
    /// This allows you to train a model once and then use it many times without retraining.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Evaluates the performance of the trained model on test data.
    /// </summary>
    /// <param name="xTest">The input features matrix for testing (not used directly).</param>
    /// <param name="yTest">The actual time series data for testing.</param>
    /// <returns>A dictionary containing evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method compares the spectral analysis of your training data with the spectral analysis
    /// of your test data to see how similar they are. It calculates several metrics:
    /// 
    /// - MSE (Mean Squared Error): The average squared difference between the training and test periodograms
    /// - RMSE (Root Mean Squared Error): The square root of MSE, giving an error in the same units
    /// - MAE (Mean Absolute Error): The average absolute difference between the periodograms
    /// - R2 (R-squared): How well the training periodogram explains the variation in the test periodogram
    /// - Peak Frequency Difference: The difference between the dominant frequencies in both datasets
    /// 
    /// Lower values for MSE, RMSE, MAE, and Peak Frequency Difference indicate better model performance.
    /// Higher values for R2 (closer to 1) indicate better model performance.
    /// </para>
    /// </remarks>
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