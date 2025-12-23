global using AiDotNet.Interpolation;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements the Empirical Mode Decomposition (EMD) method for time series decomposition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> EMD breaks down a complex signal (like stock prices or temperature readings over time) 
/// into simpler components called Intrinsic Mode Functions (IMFs). Think of it like separating 
/// different instruments in a song - you can hear the whole song, but EMD helps you identify 
/// the individual instruments playing together.
/// </remarks>
public class EMDDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly int _maxImf;
    private readonly double _threshold;
    private readonly EMDAlgorithmType _algorithm;
    private readonly IInterpolation<T>? _interpolation;
    private readonly double _residualEnergyThreshold = 1e-6;

    /// <summary>
    /// Initializes a new instance of the EMD decomposition algorithm.
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose.</param>
    /// <param name="interpolation">The interpolation method to use for envelope calculation (optional).</param>
    /// <param name="maxImf">Maximum number of Intrinsic Mode Functions to extract.</param>
    /// <param name="threshold">Convergence threshold for the sifting process.</param>
    /// <param name="algorithm">The specific EMD algorithm variant to use.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor sets up the EMD algorithm with your data and preferences.
    /// - timeSeries: Your data points over time (e.g., daily temperatures)
    /// - maxImf: How many components to break your data into (default: 10)
    /// - threshold: How precise you want the decomposition to be (smaller = more precise)
    /// - algorithm: Which version of EMD to use (standard is the basic version)
    /// </remarks>
    public EMDDecomposition(Vector<T> timeSeries, IInterpolation<T>? interpolation = null, int maxImf = 10, double threshold = 0.05, EMDAlgorithmType algorithm = EMDAlgorithmType.Standard)
        : base(timeSeries)
    {
        _maxImf = maxImf;
        _threshold = threshold;
        _algorithm = algorithm;
        _interpolation = interpolation;
        Decompose();
    }

    /// <summary>
    /// Performs the time series decomposition using the selected EMD algorithm.
    /// </summary>
    /// <remarks>
    /// This method selects and executes the appropriate EMD algorithm variant based on the configuration.
    /// </remarks>
    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case EMDAlgorithmType.Standard:
                DecomposeStandard();
                break;
            case EMDAlgorithmType.Ensemble:
                DecomposeEnsemble();
                break;
            case EMDAlgorithmType.CompleteEnsemble:
                DecomposeCompleteEnsemble();
                break;
            case EMDAlgorithmType.Multivariate:
                DecomposeMultivariate();
                break;
            default:
                throw new NotSupportedException($"EMD decomposition algorithm {_algorithm} is not supported.");
        }
    }

    /// <summary>
    /// Implements the standard EMD algorithm to decompose the time series.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method extracts components from your data one by one, from fastest-changing 
    /// to slowest-changing patterns. It's like peeling layers from an onion, where each layer 
    /// represents a different pattern in your data.
    /// </remarks>
    private void DecomposeStandard()
    {
        Vector<T> residual = TimeSeries.Clone();
        List<Vector<T>> imfs = new List<Vector<T>>();

        for (int i = 0; i < _maxImf; i++)
        {
            Vector<T> imf = ExtractIMF(residual);
            if (imf == null || IsResidual(imf) || IsIMFNegligible(imf))
                break;

            imfs.Add(imf);
            residual = residual.Subtract(imf);
        }

        // Add IMFs and residual as components
        for (int i = 0; i < imfs.Count; i++)
        {
            AddComponent((DecompositionComponentType)i, imfs[i]);
        }

        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Implements the Ensemble EMD (EEMD) algorithm for more robust decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Standard EMD can sometimes mix different patterns together. EEMD solves this by:
    /// 1. Adding small random noise to your data multiple times
    /// 2. Decomposing each noisy version
    /// 3. Averaging the results to get cleaner components
    /// 
    /// It's like taking multiple slightly different photos of the same scene and averaging them
    /// to get a clearer picture with less random noise.
    /// </remarks>
    private void DecomposeEnsemble()
    {
        int ensembleSize = 100; // Number of ensemble trials
        double noiseAmplitude = 0.2; // Amplitude of added white noise

        List<List<Vector<T>>> allImfs = new List<List<Vector<T>>>();

        for (int i = 0; i < ensembleSize; i++)
        {
            Vector<T> noisySignal = AddWhiteNoise(TimeSeries, noiseAmplitude);
            List<Vector<T>> imfs = DecomposeSignal(noisySignal);
            allImfs.Add(imfs);
        }

        // Average IMFs across ensemble trials
        List<Vector<T>> averagedImfs = AverageImfs(allImfs);

        // Add averaged IMFs as components
        for (int i = 0; i < averagedImfs.Count; i++)
        {
            AddComponent((DecompositionComponentType)i, averagedImfs[i]);
        }

        // Calculate and add residual
        Vector<T> residual = CalculateResidual(TimeSeries, averagedImfs);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Implements the Complete Ensemble EMD (CEEMD) algorithm for enhanced decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> CEEMD is an improved version of EEMD that:
    /// 1. Adds noise in a more controlled way
    /// 2. Extracts one component at a time
    /// 3. Gradually reduces the noise level for each component
    /// 
    /// This results in even cleaner separation of patterns in your data, like using
    /// increasingly finer filters to separate different sized particles.
    /// </remarks>
    private void DecomposeCompleteEnsemble()
    {
        int ensembleSize = 100; // Number of ensemble trials
        double noiseAmplitude = 0.2; // Initial amplitude of added white noise

        List<Vector<T>> imfs = new List<Vector<T>>();
        Vector<T> residual = TimeSeries.Clone();

        while (!IsResidual(residual) && imfs.Count < _maxImf)
        {
            Vector<T> imf = ExtractCEEMDImf(residual, ensembleSize, noiseAmplitude);
            imfs.Add(imf);
            residual = residual.Subtract(imf);
            noiseAmplitude *= 0.9; // Reduce noise amplitude for each IMF
        }

        // Add IMFs as components
        for (int i = 0; i < imfs.Count; i++)
        {
            AddComponent((DecompositionComponentType)i, imfs[i]);
        }

        // Add residual
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Implements Multivariate EMD (MEMD) for decomposing multiple related time series simultaneously.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> While standard EMD works on a single data series (like temperature over time),
    /// MEMD can analyze multiple related series at once (like temperature, humidity, and pressure).
    /// 
    /// It finds patterns that exist across all your data series, similar to how you might identify
    /// common themes across different stories. This helps ensure the components from different
    /// series align with each other in a meaningful way.
    /// </remarks>
    private void DecomposeMultivariate()
    {
        if (!(TimeSeries is Matrix<T> multivariateSeries))
        {
            throw new InvalidOperationException("Multivariate EMD requires a matrix input.");
        }

        int channels = multivariateSeries.Columns;
        int dataPoints = multivariateSeries.Rows;
        int numDirections = 8; // Number of projection directions

        List<List<Vector<T>>> allImfs = new List<List<Vector<T>>>();

        for (int i = 0; i < channels; i++)
        {
            allImfs.Add(new List<Vector<T>>());
        }

        Matrix<T> residual = multivariateSeries.Clone();

        while (!IsMultivariateResidual(residual) && allImfs[0].Count < _maxImf)
        {
            List<Vector<T>> imfs = ExtractMultivariateIMF(residual, numDirections);

            for (int i = 0; i < channels; i++)
            {
                allImfs[i].Add(imfs[i]);
                residual.SetColumn(i, residual.GetColumn(i).Subtract(imfs[i]));
            }
        }

        // Add IMFs as components
        for (int i = 0; i < allImfs[0].Count; i++)
        {
            Matrix<T> imfMatrix = new Matrix<T>(dataPoints, channels);
            for (int j = 0; j < channels; j++)
            {
                imfMatrix.SetColumn(j, allImfs[j][i]);
            }

            AddComponent((DecompositionComponentType)i, imfMatrix);
        }

        // Add residual
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Adds random noise to a signal for ensemble methods.
    /// </summary>
    /// <param name="signal">The original signal.</param>
    /// <param name="amplitude">The amplitude of the noise to add.</param>
    /// <returns>A new signal with added white noise.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This adds small random fluctuations to your data, like adding static to a clear radio signal.
    /// This helps ensemble methods (EEMD, CEEMD) better separate mixed patterns in your data.
    /// </remarks>
    private Vector<T> AddWhiteNoise(Vector<T> signal, double amplitude)
    {
        Random random = RandomHelper.CreateSecureRandom();
        Vector<T> noisySignal = signal.Clone();
        for (int i = 0; i < signal.Length; i++)
        {
            double noise = (random.NextDouble() * 2 - 1) * amplitude;
            noisySignal[i] = NumOps.Add(noisySignal[i], NumOps.FromDouble(noise));
        }

        return noisySignal;
    }

    /// <summary>
    /// Decomposes a time series signal into its Intrinsic Mode Functions (IMFs).
    /// </summary>
    /// <param name="signal">The input time series signal to decompose.</param>
    /// <returns>A list of Intrinsic Mode Functions extracted from the signal.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method breaks down a complex signal into simpler components called IMFs.
    /// Think of it like separating different instruments in a music track - each IMF represents
    /// a different "frequency band" or pattern in your data.
    /// </remarks>
    private List<Vector<T>> DecomposeSignal(Vector<T> signal)
    {
        List<Vector<T>> imfs = new List<Vector<T>>();
        Vector<T> residual = signal.Clone();

        while (!IsResidual(residual) && imfs.Count < _maxImf)
        {
            Vector<T> imf = ExtractIMF(residual);
            imfs.Add(imf);
            residual = residual.Subtract(imf);
        }

        return imfs;
    }

    /// <summary>
    /// Averages multiple sets of Intrinsic Mode Functions (IMFs) into a single set.
    /// </summary>
    /// <param name="allImfs">A list of IMF sets to be averaged.</param>
    /// <returns>A single list of averaged IMFs.</returns>
    /// <remarks>
    /// Used in ensemble methods to combine results from multiple decompositions with noise.
    /// </remarks>
    private List<Vector<T>> AverageImfs(List<List<Vector<T>>> allImfs)
    {
        int maxImfCount = allImfs.Max(imfs => imfs.Count);
        List<Vector<T>> averagedImfs = new List<Vector<T>>();

        for (int i = 0; i < maxImfCount; i++)
        {
            Vector<T> sum = new Vector<T>(TimeSeries.Length);
            int count = 0;

            foreach (var imfs in allImfs)
            {
                if (i < imfs.Count)
                {
                    sum = sum.Add(imfs[i]);
                    count++;
                }
            }

            averagedImfs.Add(sum.Divide(NumOps.FromDouble(count)));
        }

        return averagedImfs;
    }

    /// <summary>
    /// Calculates the residual signal by subtracting all IMFs from the original signal.
    /// </summary>
    /// <param name="original">The original input signal.</param>
    /// <param name="imfs">The list of IMFs extracted from the signal.</param>
    /// <returns>The residual signal after removing all IMFs.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The residual is what remains after extracting all patterns (IMFs) from your data.
    /// It typically represents the overall trend of your data.
    /// </remarks>
    private static Vector<T> CalculateResidual(Vector<T> original, List<Vector<T>> imfs)
    {
        Vector<T> residual = original.Clone();
        foreach (var imf in imfs)
        {
            residual = residual.Subtract(imf);
        }

        return residual;
    }

    /// <summary>
    /// Extracts an Intrinsic Mode Function using the Complete Ensemble Empirical Mode Decomposition method.
    /// </summary>
    /// <param name="signal">The input signal to process.</param>
    /// <param name="ensembleSize">The number of noise-added copies to create.</param>
    /// <param name="noiseAmplitude">The amplitude of white noise to add.</param>
    /// <returns>An IMF extracted using the CEEMD method.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method improves decomposition accuracy by adding controlled noise to multiple
    /// copies of your signal and then averaging the results, which helps reduce artifacts.
    /// </remarks>
    private Vector<T> ExtractCEEMDImf(Vector<T> signal, int ensembleSize, double noiseAmplitude)
    {
        List<Vector<T>> ensembleImfs = new List<Vector<T>>();

        for (int i = 0; i < ensembleSize; i++)
        {
            Vector<T> noisySignal = AddWhiteNoise(signal, noiseAmplitude);
            Vector<T> imf = ExtractIMF(noisySignal);
            ensembleImfs.Add(imf);
        }

        return AverageVectors(ensembleImfs);
    }

    /// <summary>
    /// Calculates the average of multiple vectors.
    /// </summary>
    /// <param name="vectors">The list of vectors to average.</param>
    /// <returns>A vector containing the average values.</returns>
    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        Vector<T> sum = new Vector<T>(vectors[0].Length);
        foreach (var vector in vectors)
        {
            sum = sum.Add(vector);
        }

        return sum.Divide(NumOps.FromDouble(vectors.Count));
    }

    /// <summary>
    /// Extracts Intrinsic Mode Functions from a multivariate (multi-channel) signal.
    /// </summary>
    /// <param name="signal">The multivariate input signal as a matrix.</param>
    /// <param name="numDirections">The number of projection directions to use.</param>
    /// <returns>A list of IMFs, one for each channel of the input signal.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This handles decomposition of signals with multiple related measurements
    /// (like 3D motion data or multiple sensor readings) by considering how they relate to each other.
    /// </remarks>
    private List<Vector<T>> ExtractMultivariateIMF(Matrix<T> signal, int numDirections)
    {
        int channels = signal.Columns;
        int dataPoints = signal.Rows;
        Matrix<T> imf = signal.Clone();
        Matrix<T> prevImf;

        do
        {
            prevImf = imf.Clone();
            List<Vector<T>> projections = ProjectSignal(imf, numDirections);
            List<Vector<T>> envelopes = ComputeMultivariateEnvelopes(projections);
            Vector<T> mean = ComputeMeanEnvelope(envelopes);

            for (int i = 0; i < channels; i++)
            {
                imf.SetColumn(i, imf.GetColumn(i).Subtract(mean));
            }
        } while (!IsMultivariateMeanEnvelopeNearZero(imf, prevImf));

        List<Vector<T>> imfs = new List<Vector<T>>();
        for (int i = 0; i < channels; i++)
        {
            imfs.Add(imf.GetColumn(i));
        }

        return imfs;
    }

    /// <summary>
    /// Projects a multivariate signal onto different directions to analyze its components.
    /// </summary>
    /// <param name="signal">The multivariate signal to project.</param>
    /// <param name="numDirections">The number of projection directions to use.</param>
    /// <returns>A list of projected signals.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is like looking at a 3D object from different angles to understand
    /// its shape better. Each projection gives us a different view of the multivariate data.
    /// </remarks>
    private List<Vector<T>> ProjectSignal(Matrix<T> signal, int numDirections)
    {
        List<Vector<T>> projections = new List<Vector<T>>();
        int channels = signal.Columns;

        for (int i = 0; i < numDirections; i++)
        {
            double angle = 2 * Math.PI * i / numDirections;
            Vector<T> direction = new Vector<T>(channels);
            for (int j = 0; j < channels; j++)
            {
                direction[j] = NumOps.FromDouble(Math.Cos(angle + 2 * Math.PI * j / channels));
            }
            projections.Add(signal.Multiply(direction));
        }

        return projections;
    }

    /// <summary>
    /// Computes upper and lower envelopes for each projection of a multivariate signal.
    /// </summary>
    /// <param name="projections">The list of signal projections.</param>
    /// <returns>A list of envelope vectors.</returns>
    private List<Vector<T>> ComputeMultivariateEnvelopes(List<Vector<T>> projections)
    {
        List<Vector<T>> envelopes = new List<Vector<T>>();
        foreach (var projection in projections)
        {
            Vector<T> upperEnvelope = ComputeEnvelope(projection, EnvelopeType.Upper);
            Vector<T> lowerEnvelope = ComputeEnvelope(projection, EnvelopeType.Lower);
            envelopes.Add(upperEnvelope);
            envelopes.Add(lowerEnvelope);
        }

        return envelopes;
    }

    /// <summary>
    /// Computes the mean of multiple envelope vectors.
    /// </summary>
    /// <param name="envelopes">The list of envelope vectors.</param>
    /// <returns>The mean envelope vector.</returns>
    private Vector<T> ComputeMeanEnvelope(List<Vector<T>> envelopes)
    {
        Vector<T> sum = new Vector<T>(envelopes[0].Length);
        foreach (var envelope in envelopes)
        {
            sum = sum.Add(envelope);
        }

        return sum.Divide(NumOps.FromDouble(envelopes.Count));
    }

    /// <summary>
    /// Determines if the difference between two multivariate IMFs is small enough to consider the sifting process complete.
    /// </summary>
    /// <param name="imf">The current IMF.</param>
    /// <param name="prevImf">The previous iteration's IMF.</param>
    /// <returns>True if the mean squared difference is below the threshold, false otherwise.</returns>
    private bool IsMultivariateMeanEnvelopeNearZero(Matrix<T> imf, Matrix<T> prevImf)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < imf.Rows; i++)
        {
            for (int j = 0; j < imf.Columns; j++)
            {
                T diff = NumOps.Subtract(imf[i, j], prevImf[i, j]);
                sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
            }
        }

        T meanSquaredDiff = NumOps.Divide(sum, NumOps.FromDouble(imf.Rows * imf.Columns));

        return NumOps.LessThan(meanSquaredDiff, NumOps.FromDouble(_threshold));
    }

    /// <summary>
    /// Checks if a multivariate signal meets the criteria to be considered a residual.
    /// </summary>
    /// <param name="signal">The multivariate signal to check.</param>
    /// <returns>True if the signal is a residual, false otherwise.</returns>
    private bool IsMultivariateResidual(Matrix<T> signal)
    {
        for (int i = 0; i < signal.Columns; i++)
        {
            if (!IsResidual(signal.GetColumn(i)))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Determines if a signal meets the criteria to be considered a residual.
    /// </summary>
    /// <param name="signal">The signal to check.</param>
    /// <returns>True if the signal is a residual, false otherwise.</returns>
    /// <remarks>
    /// A signal is considered a residual if it:
    /// 1. Has at most one extremum (peak or valley)
    /// 2. Is monotonic (consistently increasing or decreasing)
    /// 3. Has energy below the residual energy threshold
    /// 
    /// <b>For Beginners:</b> This method checks if we've extracted all meaningful patterns from the data.
    /// When what's left is just a simple trend (always going up, always going down, or nearly flat),
    /// we can stop the decomposition process.
    /// </remarks>
    private bool IsResidual(Vector<T> signal)
    {
        // Check if the signal has at most one extremum
        int extremaCount = 0;
        for (int i = 1; i < signal.Length - 1; i++)
        {
            if (IsLocalMaximum(signal, i) || IsLocalMinimum(signal, i))
            {
                extremaCount++;
                if (extremaCount > 1)
                {
                    return false;
                }
            }
        }

        // Check if the signal is monotonic
        bool isIncreasing = true;
        bool isDecreasing = true;
        for (int i = 1; i < signal.Length; i++)
        {
            if (NumOps.LessThan(signal[i], signal[i - 1]))
            {
                isIncreasing = false;
            }
            if (NumOps.GreaterThan(signal[i], signal[i - 1]))
            {
                isDecreasing = false;
            }
            if (!isIncreasing && !isDecreasing)
            {
                return false;
            }
        }

        // Check if the signal's energy is below a threshold
        T energy = CalculateEnergy(signal);
        if (NumOps.GreaterThan(energy, NumOps.FromDouble(_residualEnergyThreshold)))
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Calculates the average energy of a signal.
    /// </summary>
    /// <param name="signal">The signal to calculate energy for.</param>
    /// <returns>The average energy value.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Energy here measures how much the signal varies or fluctuates.
    /// A high energy signal has large ups and downs, while a low energy signal is more flat.
    /// </remarks>
    private T CalculateEnergy(Vector<T> signal)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < signal.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(signal[i], signal[i]));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(signal.Length));
    }

    /// <summary>
    /// Extracts a single Intrinsic Mode Function from a signal using the sifting process.
    /// </summary>
    /// <param name="signal">The input signal to extract an IMF from.</param>
    /// <returns>The extracted IMF.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is the core process that isolates a single pattern from your data.
    /// It works by repeatedly subtracting the average of the upper and lower envelopes until
    /// the result meets the criteria of an IMF (roughly equal number of peaks and zero crossings).
    /// </remarks>
    private Vector<T> ExtractIMF(Vector<T> signal)
    {
        Vector<T> h = signal.Clone();
        Vector<T> prevH;

        do
        {
            prevH = h.Clone();
            Vector<T> upperEnvelope = ComputeEnvelope(h, EnvelopeType.Upper);
            Vector<T> lowerEnvelope = ComputeEnvelope(h, EnvelopeType.Lower);
            Vector<T> mean = upperEnvelope.Add(lowerEnvelope).Divide(NumOps.FromDouble(2));
            h = h.Subtract(mean);
        } while (!IsMeanEnvelopeNearZero(h, prevH));

        return h;
    }

    /// <summary>
    /// Computes an envelope (upper or lower) for a signal by interpolating between its extrema.
    /// </summary>
    /// <param name="signal">The signal to compute an envelope for.</param>
    /// <param name="envelopeType">The type of envelope to compute (Upper or Lower).</param>
    /// <returns>The computed envelope vector.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> An envelope is like drawing a smooth line that connects all the peaks (upper envelope)
    /// or valleys (lower envelope) in your data. It helps identify the overall shape or trend.
    /// </remarks>
    private Vector<T> ComputeEnvelope(Vector<T> signal, EnvelopeType envelopeType)
    {
        List<int> extremaIndices = FindExtrema(signal, envelopeType);

        if (extremaIndices.Count < 2)
        {
            // Not enough extrema to compute envelope, return original signal
            return signal.Clone();
        }

        // Ensure the envelope starts and ends with the signal
        if (extremaIndices[0] != 0)
            extremaIndices.Insert(0, 0);
        if (extremaIndices[extremaIndices.Count - 1] != signal.Length - 1)
            extremaIndices.Add(signal.Length - 1);

        Vector<T> x = new Vector<T>(extremaIndices.Count);
        Vector<T> y = new Vector<T>(extremaIndices.Count);

        for (int i = 0; i < extremaIndices.Count; i++)
        {
            x[i] = NumOps.FromDouble(extremaIndices[i]);
            y[i] = signal[extremaIndices[i]];
        }

        IInterpolation<T> spline = _interpolation ?? new CubicSplineInterpolation<T>(x, y);

        Vector<T> envelope = new Vector<T>(signal.Length);
        for (int i = 0; i < signal.Length; i++)
        {
            envelope[i] = spline.Interpolate(NumOps.FromDouble(i));
        }

        return envelope;
    }

    /// <summary>
    /// Finds the indices of local maxima or minima in a signal.
    /// </summary>
    /// <param name="signal">The signal to find extrema in.</param>
    /// <param name="envelopeType">The type of extrema to find (maxima for Upper, minima for Lower).</param>
    /// <returns>A list of indices where extrema occur.</returns>
    private List<int> FindExtrema(Vector<T> signal, EnvelopeType envelopeType)
    {
        List<int> extremaIndices = new List<int>();

        for (int i = 1; i < signal.Length - 1; i++)
        {
            if (envelopeType == EnvelopeType.Upper && IsLocalMaximum(signal, i))
            {
                extremaIndices.Add(i);
            }
            else if (envelopeType == EnvelopeType.Lower && IsLocalMinimum(signal, i))
            {
                extremaIndices.Add(i);
            }
        }

        return extremaIndices;
    }

    /// <summary>
    /// Determines if a point in a signal is a local maximum.
    /// </summary>
    /// <param name="signal">The signal to check.</param>
    /// <param name="index">The index of the point to check.</param>
    /// <returns>True if the point is a local maximum, false otherwise.</returns>
    private bool IsLocalMaximum(Vector<T> signal, int index)
    {
        return NumOps.GreaterThan(signal[index], signal[index - 1]) &&
               NumOps.GreaterThan(signal[index], signal[index + 1]);
    }

    /// <summary>
    /// Determines if a point in a signal is a local minimum.
    /// </summary>
    /// <param name="signal">The signal to check.</param>
    /// <param name="index">The index of the point to check.</param>
    /// <returns>True if the point is a local minimum, false otherwise.</returns>
    private bool IsLocalMinimum(Vector<T> signal, int index)
    {
        return NumOps.LessThan(signal[index], signal[index - 1]) &&
               NumOps.LessThan(signal[index], signal[index + 1]);
    }

    /// <summary>
    /// Determines if the difference between two IMFs is small enough to consider the sifting process complete.
    /// </summary>
    /// <param name="h">The current IMF candidate.</param>
    /// <param name="prevH">The previous iteration's IMF candidate.</param>
    /// <returns>True if the mean squared difference is below the threshold, false otherwise.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This checks if further processing would make meaningful changes to our extracted pattern.
    /// When consecutive iterations produce nearly identical results, we can stop the extraction process.
    /// </remarks>
    private bool IsMeanEnvelopeNearZero(Vector<T> h, Vector<T> prevH)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < h.Length; i++)
        {
            T diff = NumOps.Subtract(h[i], prevH[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        T meanSquaredDiff = NumOps.Divide(sum, NumOps.FromDouble(h.Length));

        return NumOps.LessThan(meanSquaredDiff, NumOps.FromDouble(_threshold));
    }

    /// <summary>
    /// Determines if an Intrinsic Mode Function has negligible energy.
    /// </summary>
    /// <param name="imf">The IMF to check.</param>
    /// <returns>True if the IMF's energy is below the threshold, false otherwise.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method checks if an extracted pattern is significant enough to keep.
    /// If an IMF has very low energy (small fluctuations), it might just be noise or computational artifacts
    /// rather than a meaningful pattern in your data.
    /// </remarks>
    private bool IsIMFNegligible(Vector<T> imf)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < imf.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(imf[i], imf[i]));
        }
        T energy = NumOps.Divide(sum, NumOps.FromDouble(imf.Length));

        return NumOps.LessThan(energy, NumOps.FromDouble(_threshold));
    }
}
