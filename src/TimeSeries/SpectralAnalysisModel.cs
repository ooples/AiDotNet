using AiDotNet.Autodiff;

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
        _spectralOptions = (SpectralAnalysisOptions<T>)Options;
        _frequencies = new Vector<T>(1);
        _periodogram = new Vector<T>(1);
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
        return (Vector<T>)Engine.Multiply(signal, window);
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

    /// <summary>
    /// Core implementation of the training logic for the spectral analysis model.
    /// </summary>
    /// <param name="x">The input features matrix (not used in spectral analysis).</param>
    /// <param name="y">The time series data to analyze.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method contains the actual implementation of the spectral analysis process.
    /// It takes your time series data and breaks it down into its frequency components,
    /// similar to how a prism breaks light into different colors. This helps identify
    /// which patterns (frequencies) are most prominent in your data.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Validate input
        if (y == null || y.Length == 0)
        {
            throw new ArgumentException("Time series data cannot be null or empty", nameof(y));
        }

        int n = y.Length;
        int nfft = _spectralOptions.NFFT;

        // If NFFT wasn't specified or is invalid, use the next power of 2 >= n
        if (nfft <= 0)
        {
            nfft = 1;
            while (nfft < n)
            {
                nfft *= 2;
            }
            _spectralOptions.NFFT = nfft;
        }

        // Apply window function if specified
        Vector<T> windowedSignal = _spectralOptions.UseWindowFunction
            ? ApplyWindowFunction(y)
            : new Vector<T>(y); // Create a copy to avoid modifying the input

        // Compute FFT
        Vector<Complex<T>> fft = ComputeFFT(windowedSignal, nfft);

        // Compute periodogram
        _periodogram = new Vector<T>(nfft / 2 + 1);
        for (int i = 0; i < _periodogram.Length; i++)
        {
            T magnitude = NumOps.Sqrt(NumOps.Add(
                NumOps.Multiply(fft[i].Real, fft[i].Real),
                NumOps.Multiply(fft[i].Imaginary, fft[i].Imaginary)
            ));

            // Scale by 1/n for proper normalization
            _periodogram[i] = NumOps.Divide(NumOps.Multiply(magnitude, magnitude), NumOps.FromDouble(n));
        }

        // Compute frequencies
        _frequencies = new Vector<T>(nfft / 2 + 1);
        for (int i = 0; i < _frequencies.Length; i++)
        {
            // Normalize frequency by sampling rate if provided, otherwise use normalized frequency
            if (NumOps.GreaterThan(NumOps.FromDouble(_spectralOptions.SamplingRate), NumOps.Zero))
            {
                _frequencies[i] = NumOps.Multiply(
                    NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(nfft)),
                    NumOps.FromDouble(_spectralOptions.SamplingRate)
                );
            }
            else
            {
                _frequencies[i] = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(nfft));
            }
        }
    }

    /// <summary>
    /// Predicts a single value based on the dominant frequency component.
    /// </summary>
    /// <param name="input">The input vector containing a time index.</param>
    /// <returns>A predicted value based on the dominant frequency component.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Unlike traditional forecasting models, spectral analysis doesn't really predict future values.
    /// Instead, it identifies patterns in the data. This method tries to provide a prediction by:
    /// 1. Finding the strongest frequency component in your data
    /// 2. Using that frequency to generate a value at the time index you specified
    /// 
    /// This is primarily useful for synthetic data generation or simplified predictions
    /// based on the dominant cycle in your data.
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Check if model has been trained
        if (_periodogram == null || _periodogram.Length <= 1 || _frequencies == null || _frequencies.Length <= 1)
        {
            throw new InvalidOperationException("Model must be trained before making predictions");
        }

        // Validate input
        if (input == null || input.Length < 1)
        {
            throw new ArgumentException("Input vector must contain at least one element (time index)", nameof(input));
        }

        // Find the dominant frequency (excluding DC component at index 0)
        int dominantIndex = 1;
        T maxPower = _periodogram[1];

        for (int i = 2; i < _periodogram.Length; i++)
        {
            if (NumOps.GreaterThan(_periodogram[i], maxPower))
            {
                maxPower = _periodogram[i];
                dominantIndex = i;
            }
        }

        // Get the dominant frequency
        T dominantFreq = _frequencies[dominantIndex];

        // Generate a sinusoidal value at the specified time index using the dominant frequency
        T timeIndex = input[0];
        T amplitude = NumOps.Sqrt(maxPower);

        // Calculate sin(2p * frequency * timeIndex)
        T angle = NumOps.Multiply(
            NumOps.Multiply(NumOps.FromDouble(2 * Math.PI), dominantFreq),
            timeIndex
        );

        // Convert angle to a value between 0 and 2p
        while (NumOps.GreaterThan(angle, NumOps.FromDouble(2 * Math.PI)))
        {
            angle = NumOps.Subtract(angle, NumOps.FromDouble(2 * Math.PI));
        }

        while (NumOps.LessThan(angle, NumOps.Zero))
        {
            angle = NumOps.Add(angle, NumOps.FromDouble(2 * Math.PI));
        }

        // Calculate sine using an approximation
        T sinValue;
        if (NumOps.LessThan(angle, NumOps.FromDouble(Math.PI)))
        {
            // Use sin(x) ≈ x - x²/6 for small x
            if (NumOps.LessThan(angle, NumOps.FromDouble(Math.PI / 2)))
            {
                T xSquared = NumOps.Multiply(angle, angle);
                T xCubed = NumOps.Multiply(xSquared, angle);
                sinValue = NumOps.Subtract(angle, NumOps.Divide(xCubed, NumOps.FromDouble(6)));
            }
            // For x near p/2, use sin(x) ≈ 1 - (x - p/2)²/2
            else
            {
                T diff = NumOps.Subtract(angle, NumOps.FromDouble(Math.PI / 2));
                T diffSquared = NumOps.Multiply(diff, diff);
                sinValue = NumOps.Subtract(NumOps.One, NumOps.Divide(diffSquared, NumOps.FromDouble(2)));
            }
        }
        else
        {
            // For p to 2p, use sin(x) = -sin(x - p)
            T reducedAngle = NumOps.Subtract(angle, NumOps.FromDouble(Math.PI));

            // Calculate sin for reduced angle between 0 and p
            T reducedSin;
            if (NumOps.LessThan(reducedAngle, NumOps.FromDouble(Math.PI / 2)))
            {
                T xSquared = NumOps.Multiply(reducedAngle, reducedAngle);
                T xCubed = NumOps.Multiply(xSquared, reducedAngle);
                reducedSin = NumOps.Subtract(reducedAngle, NumOps.Divide(xCubed, NumOps.FromDouble(6)));
            }
            else
            {
                T diff = NumOps.Subtract(reducedAngle, NumOps.FromDouble(Math.PI / 2));
                T diffSquared = NumOps.Multiply(diff, diff);
                reducedSin = NumOps.Subtract(NumOps.One, NumOps.Divide(diffSquared, NumOps.FromDouble(2)));
            }

            // Negate for the full range
            sinValue = NumOps.Negate(reducedSin);
        }

        // Scale by amplitude
        return NumOps.Multiply(amplitude, sinValue);
    }

    /// <summary>
    /// Gets metadata about the model, including its type, parameters, and characteristics of the spectral analysis.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides a summary of the spectral analysis, including:
    /// - The configuration settings used (FFT size, window function type, etc.)
    /// - Statistics about the frequency analysis (dominant frequencies, frequency range, etc.)
    /// - Overall metrics that describe the spectrum
    /// 
    /// This information is useful for comparing different analyses, documenting your process,
    /// or sharing your findings with others.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Create a new metadata object
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.SpectralAnalysisModel,
            AdditionalInfo = new Dictionary<string, object>()
        };

        // Add configuration parameters
        metadata.AdditionalInfo["NFFT"] = _spectralOptions.NFFT;
        metadata.AdditionalInfo["UseWindowFunction"] = _spectralOptions.UseWindowFunction;
        metadata.AdditionalInfo["WindowFunctionType"] = _spectralOptions.WindowFunction?.GetWindowFunctionType().ToString() ?? "None";
        metadata.AdditionalInfo["OverlapPercentage"] = _spectralOptions.OverlapPercentage;

        // Only include sampling rate if it was specified
        if (NumOps.GreaterThan(NumOps.FromDouble(_spectralOptions.SamplingRate), NumOps.Zero))
        {
            metadata.AdditionalInfo["SamplingRate"] = Convert.ToDouble(_spectralOptions.SamplingRate);
        }

        // Add metadata about the frequencies and periodogram
        if (_frequencies != null && _frequencies.Length > 0 && _periodogram != null && _periodogram.Length > 0)
        {
            // Find dominant frequency (excluding DC component at index 0)
            int dominantIndex = 0;
            T maxPower = NumOps.Zero;

            for (int i = 1; i < _periodogram.Length; i++)
            {
                if (NumOps.GreaterThan(_periodogram[i], maxPower))
                {
                    maxPower = _periodogram[i];
                    dominantIndex = i;
                }
            }

            // Add frequency statistics
            metadata.AdditionalInfo["FrequencyCount"] = _frequencies.Length;
            metadata.AdditionalInfo["MinFrequency"] = Convert.ToDouble(_frequencies[0]);
            metadata.AdditionalInfo["MaxFrequency"] = Convert.ToDouble(_frequencies[_frequencies.Length - 1]);
            metadata.AdditionalInfo["DominantFrequency"] = Convert.ToDouble(_frequencies[dominantIndex]);
            metadata.AdditionalInfo["DominantFrequencyPower"] = Convert.ToDouble(maxPower);

            // Calculate the total power (sum of periodogram)
            T totalPower = NumOps.Zero;
            for (int i = 0; i < _periodogram.Length; i++)
            {
                totalPower = NumOps.Add(totalPower, _periodogram[i]);
            }
            metadata.AdditionalInfo["TotalPower"] = Convert.ToDouble(totalPower);

            // Calculate the spectral entropy as a measure of randomness
            T logSum = NumOps.Zero;
            for (int i = 0; i < _periodogram.Length; i++)
            {
                if (NumOps.GreaterThan(_periodogram[i], NumOps.Zero))
                {
                    T normalizedPower = NumOps.Divide(_periodogram[i], totalPower);
                    T logPower = NumOps.Log(normalizedPower);
                    logSum = NumOps.Subtract(logSum, NumOps.Multiply(normalizedPower, logPower));
                }
            }
            metadata.AdditionalInfo["SpectralEntropy"] = Convert.ToDouble(logSum);
        }

        // Add the serialized model data
        metadata.ModelData = this.Serialize();

        return metadata;
    }

    /// <summary>
    /// Creates a new instance of the SpectralAnalysisModel with the same options.
    /// </summary>
    /// <returns>A new instance of the SpectralAnalysisModel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method creates a fresh copy of the spectral analysis model with the same
    /// configuration settings but no trained data. It's useful when you want to:
    /// - Analyze a different dataset with the same settings
    /// - Create multiple models with similar configurations
    /// - Reset the model to start fresh with the same options
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a deep copy of the options
        var newOptions = new SpectralAnalysisOptions<T>
        {
            NFFT = _spectralOptions.NFFT,
            UseWindowFunction = _spectralOptions.UseWindowFunction,
            OverlapPercentage = _spectralOptions.OverlapPercentage,
            SamplingRate = _spectralOptions.SamplingRate
        };

        // Copy the window function if one is specified
        if (_spectralOptions.WindowFunction != null)
        {
            newOptions.WindowFunction = WindowFunctionFactory.CreateWindowFunction<T>(
                _spectralOptions.WindowFunction.GetWindowFunctionType());
        }

        // Create a new instance with the copied options
        return new SpectralAnalysisModel<T>(newOptions);
    }

    /// <summary>
    /// Gets whether this model supports JIT compilation.
    /// </summary>
    /// <value>
    /// Returns <c>true</c> when the model has computed a periodogram.
    /// Spectral analysis prediction simply returns the precomputed periodogram,
    /// which can be efficiently exported as a constant tensor.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> JIT compilation for spectral analysis is straightforward
    /// because the "prediction" is just the precomputed periodogram. The FFT analysis
    /// is done during training, and prediction just returns the result.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => _periodogram != null && _periodogram.Length > 1;

    /// <summary>
    /// Exports the Spectral Analysis Model as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">A list to which input nodes will be added (not used for spectral analysis).</param>
    /// <returns>The output computation node containing the periodogram.</returns>
    /// <remarks>
    /// <para>
    /// Since spectral analysis doesn't make traditional predictions but returns the computed
    /// periodogram, the computation graph simply returns the precomputed periodogram as a constant.
    /// </para>
    /// <para><b>For Beginners:</b> Unlike other models that compute predictions from input,
    /// spectral analysis just returns the frequency content it found in your data during training.
    /// The JIT-compiled version returns this precomputed result efficiently.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
        {
            throw new ArgumentNullException(nameof(inputNodes), "Input nodes list cannot be null.");
        }

        if (_periodogram == null || _periodogram.Length <= 1)
        {
            throw new InvalidOperationException("Cannot export computation graph: Periodogram has not been computed.");
        }

        // Provide a consistent API by including an input node (even though the output is precomputed).
        var inputTensor = new Tensor<T>(new[] { Math.Max(1, _spectralOptions.NFFT) });
        var inputNode = TensorOperations<T>.Variable(inputTensor, "spectral_input", requiresGradient: false);
        inputNodes.Add(inputNode);

        // For spectral analysis, prediction just returns the periodogram (precomputed during training).
        var periodogramData = new T[_periodogram.Length];
        for (int i = 0; i < _periodogram.Length; i++)
        {
            periodogramData[i] = _periodogram[i];
        }
        var periodogramTensor = new Tensor<T>(new[] { _periodogram.Length }, new Vector<T>(periodogramData));
        var outputNode = TensorOperations<T>.Constant(periodogramTensor, "periodogram");

        return outputNode;
    }
}
