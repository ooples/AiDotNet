global using AiDotNet.Interpolation;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class EMDDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly int _maxImf;
    private readonly double _threshold;
    private readonly EMDAlgorithmType _algorithm;
    private readonly IInterpolation<T>? _interpolation;
    private readonly double _residualEnergyThreshold = 1e-6;

    public EMDDecomposition(Vector<T> timeSeries, IInterpolation<T>? interpolation = null, int maxImf = 10, double threshold = 0.05, EMDAlgorithmType algorithm = EMDAlgorithmType.Standard)
        : base(timeSeries)
    {
        _maxImf = maxImf;
        _threshold = threshold;
        _algorithm = algorithm;
        _interpolation = interpolation;
        Decompose();
    }

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
                throw new NotImplementedException($"EMD decomposition algorithm {_algorithm} is not implemented.");
        }
    }

    private void DecomposeStandard()
    {
        // Implement standard EMD decomposition
        Vector<T> residual = TimeSeries.Copy();
        List<Vector<T>> imfs = new List<Vector<T>>();

        for (int i = 0; i < _maxImf; i++)
        {
            Vector<T> imf = ExtractIMF(residual);
            if (imf == null || IsResidual(imf))
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

    private void DecomposeCompleteEnsemble()
    {
        int ensembleSize = 100; // Number of ensemble trials
        double noiseAmplitude = 0.2; // Initial amplitude of added white noise

        List<Vector<T>> imfs = new List<Vector<T>>();
        Vector<T> residual = TimeSeries.Copy();

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

        Matrix<T> residual = multivariateSeries.Copy();

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

    private Vector<T> AddWhiteNoise(Vector<T> signal, double amplitude)
    {
        Random random = new Random();
        Vector<T> noisySignal = signal.Copy();
        for (int i = 0; i < signal.Length; i++)
        {
            double noise = (random.NextDouble() * 2 - 1) * amplitude;
            noisySignal[i] = NumOps.Add(noisySignal[i], NumOps.FromDouble(noise));
        }

        return noisySignal;
    }

    private List<Vector<T>> DecomposeSignal(Vector<T> signal)
    {
        List<Vector<T>> imfs = new List<Vector<T>>();
        Vector<T> residual = signal.Copy();

        while (!IsResidual(residual) && imfs.Count < _maxImf)
        {
            Vector<T> imf = ExtractIMF(residual);
            imfs.Add(imf);
            residual = residual.Subtract(imf);
        }

        return imfs;
    }

    private List<Vector<T>> AverageImfs(List<List<Vector<T>>> allImfs)
    {
        int maxImfCount = allImfs.Max(imfs => imfs.Count);
        List<Vector<T>> averagedImfs = new List<Vector<T>>();

        for (int i = 0; i < maxImfCount; i++)
        {
            Vector<T> sum = new Vector<T>(TimeSeries.Length, NumOps);
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

    private Vector<T> CalculateResidual(Vector<T> original, List<Vector<T>> imfs)
    {
        Vector<T> residual = original.Copy();
        foreach (var imf in imfs)
        {
            residual = residual.Subtract(imf);
        }

        return residual;
    }

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

    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        Vector<T> sum = new Vector<T>(vectors[0].Length, NumOps);
        foreach (var vector in vectors)
        {
            sum = sum.Add(vector);
        }

        return sum.Divide(NumOps.FromDouble(vectors.Count));
    }

    private List<Vector<T>> ExtractMultivariateIMF(Matrix<T> signal, int numDirections)
    {
        int channels = signal.Columns;
        int dataPoints = signal.Rows;
        Matrix<T> imf = signal.Copy();
        Matrix<T> prevImf;

        do
        {
            prevImf = imf.Copy();
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

    private List<Vector<T>> ProjectSignal(Matrix<T> signal, int numDirections)
    {
        List<Vector<T>> projections = new List<Vector<T>>();
        int channels = signal.Columns;

        for (int i = 0; i < numDirections; i++)
        {
            double angle = 2 * Math.PI * i / numDirections;
            Vector<T> direction = new Vector<T>(channels, NumOps);
            for (int j = 0; j < channels; j++)
            {
                direction[j] = NumOps.FromDouble(Math.Cos(angle + 2 * Math.PI * j / channels));
            }
            projections.Add(signal.Multiply(direction));
        }

        return projections;
    }

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

    private Vector<T> ComputeMeanEnvelope(List<Vector<T>> envelopes)
    {
        Vector<T> sum = new Vector<T>(envelopes[0].Length, NumOps);
        foreach (var envelope in envelopes)
        {
            sum = sum.Add(envelope);
        }

        return sum.Divide(NumOps.FromDouble(envelopes.Count));
    }

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

    private T CalculateEnergy(Vector<T> signal)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < signal.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(signal[i], signal[i]));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(signal.Length));
    }

    private Vector<T> ExtractIMF(Vector<T> signal)
    {
        Vector<T> h = signal.Copy();
        Vector<T> prevH;

        do
        {
            prevH = h.Copy();
            Vector<T> upperEnvelope = ComputeEnvelope(h, EnvelopeType.Upper);
            Vector<T> lowerEnvelope = ComputeEnvelope(h, EnvelopeType.Lower);
            Vector<T> mean = upperEnvelope.Add(lowerEnvelope).Divide(NumOps.FromDouble(2));
            h = h.Subtract(mean);
        } while (!IsMeanEnvelopeNearZero(h, prevH));

        return h;
    }

    private Vector<T> ComputeEnvelope(Vector<T> signal, EnvelopeType envelopeType)
    {
        List<int> extremaIndices = FindExtrema(signal, envelopeType);
        
        if (extremaIndices.Count < 2)
        {
            // Not enough extrema to compute envelope, return original signal
            return signal.Copy();
        }

        // Ensure the envelope starts and ends with the signal
        if (extremaIndices[0] != 0)
            extremaIndices.Insert(0, 0);
        if (extremaIndices[extremaIndices.Count - 1] != signal.Length - 1)
            extremaIndices.Add(signal.Length - 1);

        Vector<T> x = new Vector<T>(extremaIndices.Count, NumOps);
        Vector<T> y = new Vector<T>(extremaIndices.Count, NumOps);

        for (int i = 0; i < extremaIndices.Count; i++)
        {
            x[i] = NumOps.FromDouble(extremaIndices[i]);
            y[i] = signal[extremaIndices[i]];
        }

        IInterpolation<T> spline = _interpolation ?? new CubicSplineInterpolation<T>(x, y);

        Vector<T> envelope = new Vector<T>(signal.Length, NumOps);
        for (int i = 0; i < signal.Length; i++)
        {
            envelope[i] = spline.Interpolate(NumOps.FromDouble(i));
        }

        return envelope;
    }

    private List<int> FindExtrema(Vector<T> signal, EnvelopeType envelopeType)
    {
        List<int> extremaIndices = [];

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

    private bool IsLocalMaximum(Vector<T> signal, int index)
    {
        return NumOps.GreaterThan(signal[index], signal[index - 1]) &&
               NumOps.GreaterThan(signal[index], signal[index + 1]);
    }

    private bool IsLocalMinimum(Vector<T> signal, int index)
    {
        return NumOps.LessThan(signal[index], signal[index - 1]) &&
               NumOps.LessThan(signal[index], signal[index + 1]);
    }

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