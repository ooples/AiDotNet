global using AiDotNet.Interpolation;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class EMDDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly int _maxImf;
    private readonly double _threshold;
    private readonly IInterpolation<T>? _interpolation;

    public EMDDecomposition(Vector<T> timeSeries, IInterpolation<T>? interpolation = null, int maxImf = 10, double threshold = 0.05)
        : base(timeSeries)
    {
        _maxImf = maxImf;
        _threshold = threshold;
        _interpolation = interpolation;
    }

    public void Decompose()
    {
        Vector<T> residual = TimeSeries.Copy();
        List<Vector<T>> imfs = new List<Vector<T>>();

        for (int i = 0; i < _maxImf; i++)
        {
            Vector<T> imf = ExtractIMF(residual);
            if (IsIMFNegligible(imf))
                break;

            imfs.Add(imf);
            residual = residual.Subtract(imf);

            AddComponent(DecompositionComponentType.IMF, imf);
        }

        AddComponent(DecompositionComponentType.Residual, residual);
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