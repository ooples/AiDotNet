namespace AiDotNet.WaveletFunctions;

public class ReverseBiorthogonalWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Vector<T> _decompositionLowPass;
    private readonly Vector<T> _decompositionHighPass;
    private readonly Vector<T> _reconstructionLowPass;
    private readonly Vector<T> _reconstructionHighPass;
    private readonly BoundaryHandlingMethod _boundaryMethod;
    private readonly int _chunkSize;
    private readonly WaveletType _waveletType;

   public ReverseBiorthogonalWavelet(
        INumericOperations<T> numOps, 
        WaveletType waveletType = WaveletType.ReverseBior22, 
        BoundaryHandlingMethod boundaryMethod = BoundaryHandlingMethod.Periodic, 
        int chunkSize = 1024)
    {
        _numOps = numOps;
        _boundaryMethod = boundaryMethod;
        _chunkSize = chunkSize;
        _waveletType = waveletType;
        (_decompositionLowPass, _decompositionHighPass, _reconstructionLowPass, _reconstructionHighPass) = 
            GetReverseBiorthogonalCoefficients(_waveletType);
    }

    public T Calculate(T x)
    {
        T result = _numOps.Zero;
        int centerIndex = _reconstructionLowPass.Length / 2;

        for (int k = 0; k < _reconstructionLowPass.Length; k++)
        {
            T shiftedX = _numOps.Subtract(x, _numOps.FromDouble(k - centerIndex));
            T phiValue = DiscreteCascadeAlgorithm(shiftedX);
            result = _numOps.Add(result, _numOps.Multiply(_reconstructionLowPass[k], phiValue));
        }

        return result;
    }

    private T DiscreteCascadeAlgorithm(T x)
    {
        const int resolution = 1024;
        const int iterations = 7;
        var values = new T[resolution];

        // Initialize with the scaling function
        for (int i = 0; i < resolution; i++)
        {
            T xValue = _numOps.Divide(_numOps.FromDouble(i), _numOps.FromDouble(resolution - 1));
            values[i] = ScalingFunction(xValue);
        }

        // Perform iterations
        for (int iter = 0; iter < iterations; iter++)
        {
            var newValues = new T[resolution];
            for (int i = 0; i < resolution; i++)
            {
                T sum = _numOps.Zero;
                for (int k = 0; k < _reconstructionLowPass.Length; k++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(_reconstructionLowPass[k], values[(2 * i - k + resolution) % resolution]));
                }

                newValues[i] = sum;
            }

            values = newValues;
        }

        // Interpolate to find the value at x
        int index = (int)(Convert.ToDouble(x) * (resolution - 1));
        index = Math.Max(0, Math.Min(resolution - 2, index));
        T t = _numOps.Subtract(x, _numOps.Divide(_numOps.FromDouble(index), _numOps.FromDouble(resolution - 1)));
        return _numOps.Add(
            _numOps.Multiply(_numOps.Subtract(_numOps.One, t), values[index]),
            _numOps.Multiply(t, values[index + 1])
        );
    }

    private T ScalingFunction(T x)
    {
        T absX = _numOps.Abs(x);
        T result = _numOps.Zero;

        if (_numOps.LessThan(absX, _numOps.One))
        {
            result = _numOps.Subtract(_numOps.One, absX);
        }
        else if (_numOps.LessThan(absX, _numOps.FromDouble(2)))
        {
            T temp = _numOps.Subtract(_numOps.FromDouble(2), absX);
            result = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Multiply(temp, temp));
        }

        return result;
    }

    private Vector<T> GetDecompositionCoefficients(int order)
    {
        return order switch
        {
            1 => new Vector<T>(
                            [
                                _numOps.FromDouble(0),
                    _numOps.FromDouble(-0.1767766952966369),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(1.0606601717798214),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(-0.1767766952966369)
                            ]),
            2 => new Vector<T>(new T[]
                            {
                    _numOps.FromDouble(0),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(-0.7071067811865476),
                    _numOps.FromDouble(0.3535533905932738)
                            }),
            _ => throw new ArgumentException($"Decomposition order {order} is not supported."),
        };
    }

    private Vector<T> GetReconstructionCoefficients(int order)
    {
        return order switch
        {
            1 => new Vector<T>(
                            [
                    _numOps.FromDouble(0),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(-0.7071067811865476),
                    _numOps.FromDouble(0.3535533905932738)
                            ]),
            2 => new Vector<T>(
                [
                    _numOps.FromDouble(0),
                    _numOps.FromDouble(-0.1767766952966369),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(1.0606601717798214),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(-0.1767766952966369)
                ]),
            _ => throw new ArgumentException($"Reconstruction order {order} is not supported."),
        };
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int n = input.Length;
        var approximation = new Vector<T>((n + 1) / 2);
        var detail = new Vector<T>((n + 1) / 2);

        for (int i = 0; i < n; i += _chunkSize)
        {
            int chunkEnd = Math.Min(i + _chunkSize, n);
            DecomposeChunk(input, approximation, detail, i, chunkEnd);
        }

        return (approximation, detail);
    }

    private void DecomposeChunk(Vector<T> input, Vector<T> approximation, Vector<T> detail, int start, int end)
    {
        for (int i = start; i < end; i += 2)
        {
            T approx = _numOps.Zero;
            T det = _numOps.Zero;

            for (int j = 0; j < _decompositionLowPass.Length; j++)
            {
                int index = GetExtendedIndex(i + j - _decompositionLowPass.Length / 2 + 1, input.Length);
                approx = _numOps.Add(approx, _numOps.Multiply(_decompositionLowPass[j], input[index]));
                det = _numOps.Add(det, _numOps.Multiply(_decompositionHighPass[j], input[index]));
            }

            approximation[i / 2] = approx;
            detail[i / 2] = det;
        }
    }

    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        int n = (approximation.Length + detail.Length) * 2;
        var reconstructed = new Vector<T>(n);

        for (int i = 0; i < n; i += _chunkSize)
        {
            int chunkEnd = Math.Min(i + _chunkSize, n);
            ReconstructChunk(approximation, detail, reconstructed, i, chunkEnd);
        }

        return reconstructed;
    }

    private void ReconstructChunk(Vector<T> approximation, Vector<T> detail, Vector<T> reconstructed, int start, int end)
    {
        for (int i = start; i < end; i++)
        {
            T value = _numOps.Zero;

            for (int j = 0; j < _reconstructionLowPass.Length; j++)
            {
                int index = (i / 2 - j + _reconstructionLowPass.Length) % approximation.Length;
                if ((i - j) % 2 == 0)
                {
                    value = _numOps.Add(value, _numOps.Multiply(_reconstructionLowPass[j], approximation[index]));
                    value = _numOps.Add(value, _numOps.Multiply(_reconstructionHighPass[j], detail[index]));
                }
            }

            reconstructed[i] = value;
        }
    }

    public (Vector<T> approximation, List<Vector<T>> details) DecomposeMultiLevel(Vector<T> input, int levels)
    {
        var details = new List<Vector<T>>();
        var currentApproximation = input;

        for (int i = 0; i < levels; i++)
        {
            var (newApproximation, detail) = Decompose(currentApproximation);
            details.Add(detail);
            currentApproximation = newApproximation;

            if (currentApproximation.Length <= _decompositionLowPass.Length)
            {
                break;
            }
        }

        return (currentApproximation, details);
    }

    public Vector<T> ReconstructMultiLevel(Vector<T> approximation, List<Vector<T>> details)
    {
        var currentApproximation = approximation;

        for (int i = details.Count - 1; i >= 0; i--)
        {
            currentApproximation = Reconstruct(currentApproximation, details[i]);
        }

        return currentApproximation;
    }

    public Vector<T> GetScalingCoefficients()
    {
        return _reconstructionLowPass;
    }

    public Vector<T> GetWaveletCoefficients()
    {
        return _reconstructionHighPass;
    }

    private int GetExtendedIndex(int index, int length)
    {
        switch (_boundaryMethod)
        {
            case BoundaryHandlingMethod.Periodic:
                return (index % length + length) % length;
            case BoundaryHandlingMethod.Symmetric:
                if (index < 0)
                    return -index - 1;
                if (index >= length)
                    return 2 * length - index - 1;
                return index;
            case BoundaryHandlingMethod.ZeroPadding:
                if (index < 0 || index >= length)
                    return -1;
                return index;
            default:
                throw new ArgumentException("Invalid boundary handling method");
        }
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBiorthogonalCoefficients(WaveletType waveletType)
    {
        return waveletType switch
        {
            WaveletType.ReverseBior22 => GetReverseBior22Coefficients(),
            WaveletType.ReverseBior11 => GetReverseBior11Coefficients(),
            WaveletType.ReverseBior13 => GetReverseBior13Coefficients(),
            WaveletType.ReverseBior24 => GetReverseBior24Coefficients(),
            WaveletType.ReverseBior26 => GetReverseBior26Coefficients(),
            WaveletType.ReverseBior28 => GetReverseBior28Coefficients(),
            WaveletType.ReverseBior31 => GetReverseBior31Coefficients(),
            WaveletType.ReverseBior33 => GetReverseBior33Coefficients(),
            WaveletType.ReverseBior35 => GetReverseBior35Coefficients(),
            WaveletType.ReverseBior37 => GetReverseBior37Coefficients(),
            WaveletType.ReverseBior39 => GetReverseBior39Coefficients(),
            WaveletType.ReverseBior44 => GetReverseBior44Coefficients(),
            WaveletType.ReverseBior46 => GetReverseBior46Coefficients(),
            WaveletType.ReverseBior48 => GetReverseBior48Coefficients(),
            WaveletType.ReverseBior55 => GetReverseBior55Coefficients(),
            WaveletType.ReverseBior68 => GetReverseBior68Coefficients(),
            _ => throw new NotImplementedException($"Wavelet type {waveletType} is not implemented."),
        };
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior22Coefficients()
    {
        var decompositionLowPass = new Vector<T>(6);
        var decompositionHighPass = new Vector<T>(6);
        var reconstructionLowPass = new Vector<T>(6);
        var reconstructionHighPass = new Vector<T>(6);

        decompositionLowPass[0] = _numOps.FromDouble(0);
        decompositionLowPass[1] = _numOps.FromDouble(-0.1767766952966369);
        decompositionLowPass[2] = _numOps.FromDouble(0.3535533905932738);
        decompositionLowPass[3] = _numOps.FromDouble(1.0606601717798214);
        decompositionLowPass[4] = _numOps.FromDouble(0.3535533905932738);
        decompositionLowPass[5] = _numOps.FromDouble(-0.1767766952966369);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(0.3535533905932738);
        decompositionHighPass[2] = _numOps.FromDouble(-0.7071067811865476);
        decompositionHighPass[3] = _numOps.FromDouble(0.3535533905932738);
        decompositionHighPass[4] = _numOps.FromDouble(0);
        decompositionHighPass[5] = _numOps.FromDouble(0);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(0.3535533905932738);
        reconstructionLowPass[2] = _numOps.FromDouble(0.7071067811865476);
        reconstructionLowPass[3] = _numOps.FromDouble(0.3535533905932738);
        reconstructionLowPass[4] = _numOps.FromDouble(0);
        reconstructionLowPass[5] = _numOps.FromDouble(0);

        reconstructionHighPass[0] = _numOps.FromDouble(0);
        reconstructionHighPass[1] = _numOps.FromDouble(-0.1767766952966369);
        reconstructionHighPass[2] = _numOps.FromDouble(0.3535533905932738);
        reconstructionHighPass[3] = _numOps.FromDouble(-1.0606601717798214);
        reconstructionHighPass[4] = _numOps.FromDouble(0.3535533905932738);
        reconstructionHighPass[5] = _numOps.FromDouble(-0.1767766952966369);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior11Coefficients()
    {
        var decompositionLowPass = new Vector<T>(2);
        var decompositionHighPass = new Vector<T>(2);
        var reconstructionLowPass = new Vector<T>(2);
        var reconstructionHighPass = new Vector<T>(2);

        // Decomposition low-pass filter
        decompositionLowPass[0] = _numOps.FromDouble(0.7071067811865476);
        decompositionLowPass[1] = _numOps.FromDouble(0.7071067811865476);

        // Decomposition high-pass filter
        decompositionHighPass[0] = _numOps.FromDouble(-0.7071067811865476);
        decompositionHighPass[1] = _numOps.FromDouble(0.7071067811865476);

        // Reconstruction low-pass filter
        reconstructionLowPass[0] = _numOps.FromDouble(0.7071067811865476);
        reconstructionLowPass[1] = _numOps.FromDouble(0.7071067811865476);

        // Reconstruction high-pass filter
        reconstructionHighPass[0] = _numOps.FromDouble(0.7071067811865476);
        reconstructionHighPass[1] = _numOps.FromDouble(-0.7071067811865476);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior13Coefficients()
    {
        var decompositionLowPass = new Vector<T>(6);
        var decompositionHighPass = new Vector<T>(2);
        var reconstructionLowPass = new Vector<T>(2);
        var reconstructionHighPass = new Vector<T>(6);

        // Decomposition low-pass filter
        decompositionLowPass[0] = _numOps.FromDouble(-0.0883883476483184);
        decompositionLowPass[1] = _numOps.FromDouble(0.0883883476483184);
        decompositionLowPass[2] = _numOps.FromDouble(0.7071067811865476);
        decompositionLowPass[3] = _numOps.FromDouble(0.7071067811865476);
        decompositionLowPass[4] = _numOps.FromDouble(0.0883883476483184);
        decompositionLowPass[5] = _numOps.FromDouble(-0.0883883476483184);

        // Decomposition high-pass filter
        decompositionHighPass[0] = _numOps.FromDouble(-0.7071067811865476);
        decompositionHighPass[1] = _numOps.FromDouble(0.7071067811865476);

        // Reconstruction low-pass filter
        reconstructionLowPass[0] = _numOps.FromDouble(0.7071067811865476);
        reconstructionLowPass[1] = _numOps.FromDouble(0.7071067811865476);

        // Reconstruction high-pass filter
        reconstructionHighPass[0] = _numOps.FromDouble(0.0883883476483184);
        reconstructionHighPass[1] = _numOps.FromDouble(0.0883883476483184);
        reconstructionHighPass[2] = _numOps.FromDouble(-0.7071067811865476);
        reconstructionHighPass[3] = _numOps.FromDouble(0.7071067811865476);
        reconstructionHighPass[4] = _numOps.FromDouble(-0.0883883476483184);
        reconstructionHighPass[5] = _numOps.FromDouble(-0.0883883476483184);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior24Coefficients()
    {
        var decompositionLowPass = new Vector<T>(10);
        var decompositionHighPass = new Vector<T>(6);
        var reconstructionLowPass = new Vector<T>(6);
        var reconstructionHighPass = new Vector<T>(10);

        decompositionLowPass[0] = _numOps.FromDouble(0);
        decompositionLowPass[1] = _numOps.FromDouble(-0.0331456303681194);
        decompositionLowPass[2] = _numOps.FromDouble(-0.0662912607362388);
        decompositionLowPass[3] = _numOps.FromDouble(0.1767766952966369);
        decompositionLowPass[4] = _numOps.FromDouble(0.4198446513295126);
        decompositionLowPass[5] = _numOps.FromDouble(0.9943689110435825);
        decompositionLowPass[6] = _numOps.FromDouble(0.4198446513295126);
        decompositionLowPass[7] = _numOps.FromDouble(0.1767766952966369);
        decompositionLowPass[8] = _numOps.FromDouble(-0.0662912607362388);
        decompositionLowPass[9] = _numOps.FromDouble(-0.0331456303681194);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(-0.1767766952966369);
        decompositionHighPass[2] = _numOps.FromDouble(0.3535533905932738);
        decompositionHighPass[3] = _numOps.FromDouble(-0.3535533905932738);
        decompositionHighPass[4] = _numOps.FromDouble(0.1767766952966369);
        decompositionHighPass[5] = _numOps.FromDouble(0);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(0.1767766952966369);
        reconstructionLowPass[2] = _numOps.FromDouble(0.3535533905932738);
        reconstructionLowPass[3] = _numOps.FromDouble(0.3535533905932738);
        reconstructionLowPass[4] = _numOps.FromDouble(0.1767766952966369);
        reconstructionLowPass[5] = _numOps.FromDouble(0);

        reconstructionHighPass[0] = _numOps.FromDouble(0);
        reconstructionHighPass[1] = _numOps.FromDouble(0.0331456303681194);
        reconstructionHighPass[2] = _numOps.FromDouble(-0.0662912607362388);
        reconstructionHighPass[3] = _numOps.FromDouble(-0.1767766952966369);
        reconstructionHighPass[4] = _numOps.FromDouble(0.4198446513295126);
        reconstructionHighPass[5] = _numOps.FromDouble(-0.9943689110435825);
        reconstructionHighPass[6] = _numOps.FromDouble(0.4198446513295126);
        reconstructionHighPass[7] = _numOps.FromDouble(-0.1767766952966369);
        reconstructionHighPass[8] = _numOps.FromDouble(-0.0662912607362388);
        reconstructionHighPass[9] = _numOps.FromDouble(0.0331456303681194);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior26Coefficients()
    {
        var decompositionLowPass = new Vector<T>(14);
        var decompositionHighPass = new Vector<T>(6);
        var reconstructionLowPass = new Vector<T>(6);
        var reconstructionHighPass = new Vector<T>(14);

        decompositionLowPass[0] = _numOps.FromDouble(0);
        decompositionLowPass[1] = _numOps.FromDouble(0.0069053396600248);
        decompositionLowPass[2] = _numOps.FromDouble(0.0138106793200496);
        decompositionLowPass[3] = _numOps.FromDouble(-0.0469563096881692);
        decompositionLowPass[4] = _numOps.FromDouble(-0.1077232986963880);
        decompositionLowPass[5] = _numOps.FromDouble(0.1697627774134332);
        decompositionLowPass[6] = _numOps.FromDouble(0.4474660099696121);
        decompositionLowPass[7] = _numOps.FromDouble(0.9667475524034829);
        decompositionLowPass[8] = _numOps.FromDouble(0.4474660099696121);
        decompositionLowPass[9] = _numOps.FromDouble(0.1697627774134332);
        decompositionLowPass[10] = _numOps.FromDouble(-0.1077232986963880);
        decompositionLowPass[11] = _numOps.FromDouble(-0.0469563096881692);
        decompositionLowPass[12] = _numOps.FromDouble(0.0138106793200496);
        decompositionLowPass[13] = _numOps.FromDouble(0.0069053396600248);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(0.1767766952966369);
        decompositionHighPass[2] = _numOps.FromDouble(-0.3535533905932738);
        decompositionHighPass[3] = _numOps.FromDouble(0.3535533905932738);
        decompositionHighPass[4] = _numOps.FromDouble(-0.1767766952966369);
        decompositionHighPass[5] = _numOps.FromDouble(0);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(-0.1767766952966369);
        reconstructionLowPass[2] = _numOps.FromDouble(-0.3535533905932738);
        reconstructionLowPass[3] = _numOps.FromDouble(0.3535533905932738);
        reconstructionLowPass[4] = _numOps.FromDouble(0.1767766952966369);
        reconstructionLowPass[5] = _numOps.FromDouble(0);

        reconstructionHighPass[0] = _numOps.FromDouble(0);
        reconstructionHighPass[1] = _numOps.FromDouble(-0.0069053396600248);
        reconstructionHighPass[2] = _numOps.FromDouble(0.0138106793200496);
        reconstructionHighPass[3] = _numOps.FromDouble(0.0469563096881692);
        reconstructionHighPass[4] = _numOps.FromDouble(-0.1077232986963880);
        reconstructionHighPass[5] = _numOps.FromDouble(-0.1697627774134332);
        reconstructionHighPass[6] = _numOps.FromDouble(0.4474660099696121);
        reconstructionHighPass[7] = _numOps.FromDouble(-0.9667475524034829);
        reconstructionHighPass[8] = _numOps.FromDouble(0.4474660099696121);
        reconstructionHighPass[9] = _numOps.FromDouble(-0.1697627774134332);
        reconstructionHighPass[10] = _numOps.FromDouble(-0.1077232986963880);
        reconstructionHighPass[11] = _numOps.FromDouble(0.0469563096881692);
        reconstructionHighPass[12] = _numOps.FromDouble(0.0138106793200496);
        reconstructionHighPass[13] = _numOps.FromDouble(-0.0069053396600248);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior28Coefficients()
    {
        var decompositionLowPass = new Vector<T>(16);
        var decompositionHighPass = new Vector<T>(6);
        var reconstructionLowPass = new Vector<T>(6);
        var reconstructionHighPass = new Vector<T>(16);

        decompositionLowPass[0] = _numOps.FromDouble(0);
        decompositionLowPass[1] = _numOps.FromDouble(0.0015105430506304422);
        decompositionLowPass[2] = _numOps.FromDouble(-0.0030210861012608843);
        decompositionLowPass[3] = _numOps.FromDouble(-0.012947511862546647);
        decompositionLowPass[4] = _numOps.FromDouble(0.02891610982635418);
        decompositionLowPass[5] = _numOps.FromDouble(0.052998481890690945);
        decompositionLowPass[6] = _numOps.FromDouble(-0.13491307360773608);
        decompositionLowPass[7] = _numOps.FromDouble(-0.16382918343409025);
        decompositionLowPass[8] = _numOps.FromDouble(0.4625714404759166);
        decompositionLowPass[9] = _numOps.FromDouble(0.9516421218971786);
        decompositionLowPass[10] = _numOps.FromDouble(0.4625714404759166);
        decompositionLowPass[11] = _numOps.FromDouble(-0.16382918343409025);
        decompositionLowPass[12] = _numOps.FromDouble(-0.13491307360773608);
        decompositionLowPass[13] = _numOps.FromDouble(0.052998481890690945);
        decompositionLowPass[14] = _numOps.FromDouble(0.02891610982635418);
        decompositionLowPass[15] = _numOps.FromDouble(-0.012947511862546647);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(0);
        decompositionHighPass[2] = _numOps.FromDouble(0.3535533905932738);
        decompositionHighPass[3] = _numOps.FromDouble(-0.7071067811865476);
        decompositionHighPass[4] = _numOps.FromDouble(0.3535533905932738);
        decompositionHighPass[5] = _numOps.FromDouble(0);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(0);
        reconstructionLowPass[2] = _numOps.FromDouble(-0.3535533905932738);
        reconstructionLowPass[3] = _numOps.FromDouble(-0.7071067811865476);
        reconstructionLowPass[4] = _numOps.FromDouble(-0.3535533905932738);
        reconstructionLowPass[5] = _numOps.FromDouble(0);

        reconstructionHighPass[0] = _numOps.FromDouble(0);
        reconstructionHighPass[1] = _numOps.FromDouble(-0.0015105430506304422);
        reconstructionHighPass[2] = _numOps.FromDouble(-0.0030210861012608843);
        reconstructionHighPass[3] = _numOps.FromDouble(0.012947511862546647);
        reconstructionHighPass[4] = _numOps.FromDouble(0.02891610982635418);
        reconstructionHighPass[5] = _numOps.FromDouble(-0.052998481890690945);
        reconstructionHighPass[6] = _numOps.FromDouble(-0.13491307360773608);
        reconstructionHighPass[7] = _numOps.FromDouble(0.16382918343409025);
        reconstructionHighPass[8] = _numOps.FromDouble(0.4625714404759166);
        reconstructionHighPass[9] = _numOps.FromDouble(-0.9516421218971786);
        reconstructionHighPass[10] = _numOps.FromDouble(0.4625714404759166);
        reconstructionHighPass[11] = _numOps.FromDouble(0.16382918343409025);
        reconstructionHighPass[12] = _numOps.FromDouble(-0.13491307360773608);
        reconstructionHighPass[13] = _numOps.FromDouble(-0.052998481890690945);
        reconstructionHighPass[14] = _numOps.FromDouble(0.02891610982635418);
        reconstructionHighPass[15] = _numOps.FromDouble(0.012947511862546647);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior31Coefficients()
    {
        var decompositionLowPass = new Vector<T>(4);
        var decompositionHighPass = new Vector<T>(8);
        var reconstructionLowPass = new Vector<T>(8);
        var reconstructionHighPass = new Vector<T>(4);

        decompositionLowPass[0] = _numOps.FromDouble(-0.3535533905932738);
        decompositionLowPass[1] = _numOps.FromDouble(1.0606601717798214);
        decompositionLowPass[2] = _numOps.FromDouble(1.0606601717798214);
        decompositionLowPass[3] = _numOps.FromDouble(-0.3535533905932738);

        decompositionHighPass[0] = _numOps.FromDouble(-0.0662912607362388);
        decompositionHighPass[1] = _numOps.FromDouble(0.1988737822087164);
        decompositionHighPass[2] = _numOps.FromDouble(-0.1546796083845572);
        decompositionHighPass[3] = _numOps.FromDouble(-0.9943689110435825);
        decompositionHighPass[4] = _numOps.FromDouble(0.9943689110435825);
        decompositionHighPass[5] = _numOps.FromDouble(0.1546796083845572);
        decompositionHighPass[6] = _numOps.FromDouble(-0.1988737822087164);
        decompositionHighPass[7] = _numOps.FromDouble(0.0662912607362388);

        reconstructionLowPass[0] = _numOps.FromDouble(0.0662912607362388);
        reconstructionLowPass[1] = _numOps.FromDouble(0.1988737822087164);
        reconstructionLowPass[2] = _numOps.FromDouble(0.1546796083845572);
        reconstructionLowPass[3] = _numOps.FromDouble(-0.9943689110435825);
        reconstructionLowPass[4] = _numOps.FromDouble(-0.9943689110435825);
        reconstructionLowPass[5] = _numOps.FromDouble(0.1546796083845572);
        reconstructionLowPass[6] = _numOps.FromDouble(0.1988737822087164);
        reconstructionLowPass[7] = _numOps.FromDouble(0.0662912607362388);

        reconstructionHighPass[0] = _numOps.FromDouble(-0.3535533905932738);
        reconstructionHighPass[1] = _numOps.FromDouble(-1.0606601717798214);
        reconstructionHighPass[2] = _numOps.FromDouble(1.0606601717798214);
        reconstructionHighPass[3] = _numOps.FromDouble(0.3535533905932738);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior33Coefficients()
    {
        var decompositionLowPass = new Vector<T>(8);
        var decompositionHighPass = new Vector<T>(8);
        var reconstructionLowPass = new Vector<T>(8);
        var reconstructionHighPass = new Vector<T>(8);

        decompositionLowPass[0] = _numOps.FromDouble(0.0352262918857095);
        decompositionLowPass[1] = _numOps.FromDouble(-0.0854412738820267);
        decompositionLowPass[2] = _numOps.FromDouble(-0.1350110200102546);
        decompositionLowPass[3] = _numOps.FromDouble(0.4598775021184914);
        decompositionLowPass[4] = _numOps.FromDouble(0.8068915093110924);
        decompositionLowPass[5] = _numOps.FromDouble(0.3326705529500825);
        decompositionLowPass[6] = _numOps.FromDouble(-0.0279837694168599);
        decompositionLowPass[7] = _numOps.FromDouble(-0.0105974017850690);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(0);
        decompositionHighPass[2] = _numOps.FromDouble(-0.1767766952966369);
        decompositionHighPass[3] = _numOps.FromDouble(0.5303300858899107);
        decompositionHighPass[4] = _numOps.FromDouble(-0.5303300858899107);
        decompositionHighPass[5] = _numOps.FromDouble(0.1767766952966369);
        decompositionHighPass[6] = _numOps.FromDouble(0);
        decompositionHighPass[7] = _numOps.FromDouble(0);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(0);
        reconstructionLowPass[2] = _numOps.FromDouble(0.1767766952966369);
        reconstructionLowPass[3] = _numOps.FromDouble(0.5303300858899107);
        reconstructionLowPass[4] = _numOps.FromDouble(0.5303300858899107);
        reconstructionLowPass[5] = _numOps.FromDouble(0.1767766952966369);
        reconstructionLowPass[6] = _numOps.FromDouble(0);
        reconstructionLowPass[7] = _numOps.FromDouble(0);

        reconstructionHighPass[0] = _numOps.FromDouble(-0.0105974017850690);
        reconstructionHighPass[1] = _numOps.FromDouble(0.0279837694168599);
        reconstructionHighPass[2] = _numOps.FromDouble(0.3326705529500825);
        reconstructionHighPass[3] = _numOps.FromDouble(-0.8068915093110924);
        reconstructionHighPass[4] = _numOps.FromDouble(0.4598775021184914);
        reconstructionHighPass[5] = _numOps.FromDouble(0.1350110200102546);
        reconstructionHighPass[6] = _numOps.FromDouble(-0.0854412738820267);
        reconstructionHighPass[7] = _numOps.FromDouble(-0.0352262918857095);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior35Coefficients()
    {
        var decompositionLowPass = new Vector<T>(12);
        var decompositionHighPass = new Vector<T>(12);
        var reconstructionLowPass = new Vector<T>(12);
        var reconstructionHighPass = new Vector<T>(12);

        decompositionLowPass[0] = _numOps.FromDouble(-0.0130514548443985);
        decompositionLowPass[1] = _numOps.FromDouble(0.0307358671058437);
        decompositionLowPass[2] = _numOps.FromDouble(0.0686539440891211);
        decompositionLowPass[3] = _numOps.FromDouble(-0.1485354424027703);
        decompositionLowPass[4] = _numOps.FromDouble(-0.2746482511903850);
        decompositionLowPass[5] = _numOps.FromDouble(0.2746482511903850);
        decompositionLowPass[6] = _numOps.FromDouble(0.7366601814282105);
        decompositionLowPass[7] = _numOps.FromDouble(0.4976186676320155);
        decompositionLowPass[8] = _numOps.FromDouble(0.0746831846544829);
        decompositionLowPass[9] = _numOps.FromDouble(-0.0305795375195906);
        decompositionLowPass[10] = _numOps.FromDouble(-0.0126815724766769);
        decompositionLowPass[11] = _numOps.FromDouble(0.0010131419871576);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(0);
        decompositionHighPass[2] = _numOps.FromDouble(0);
        decompositionHighPass[3] = _numOps.FromDouble(0.0662912607362388);
        decompositionHighPass[4] = _numOps.FromDouble(-0.1988737822087164);
        decompositionHighPass[5] = _numOps.FromDouble(0.1546796083845572);
        decompositionHighPass[6] = _numOps.FromDouble(0.9943689110435825);
        decompositionHighPass[7] = _numOps.FromDouble(-0.1546796083845572);
        decompositionHighPass[8] = _numOps.FromDouble(-0.1988737822087164);
        decompositionHighPass[9] = _numOps.FromDouble(0.0662912607362388);
        decompositionHighPass[10] = _numOps.FromDouble(0);
        decompositionHighPass[11] = _numOps.FromDouble(0);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(0);
        reconstructionLowPass[2] = _numOps.FromDouble(0);
        reconstructionLowPass[3] = _numOps.FromDouble(-0.0662912607362388);
        reconstructionLowPass[4] = _numOps.FromDouble(-0.1988737822087164);
        reconstructionLowPass[5] = _numOps.FromDouble(0.1546796083845572);
        reconstructionLowPass[6] = _numOps.FromDouble(0.9943689110435825);
        reconstructionLowPass[7] = _numOps.FromDouble(0.1546796083845572);
        reconstructionLowPass[8] = _numOps.FromDouble(-0.1988737822087164);
        reconstructionLowPass[9] = _numOps.FromDouble(-0.0662912607362388);
        reconstructionLowPass[10] = _numOps.FromDouble(0);
        reconstructionLowPass[11] = _numOps.FromDouble(0);

        reconstructionHighPass[0] = _numOps.FromDouble(0.0010131419871576);
        reconstructionHighPass[1] = _numOps.FromDouble(0.0126815724766769);
        reconstructionHighPass[2] = _numOps.FromDouble(-0.0305795375195906);
        reconstructionHighPass[3] = _numOps.FromDouble(-0.0746831846544829);
        reconstructionHighPass[4] = _numOps.FromDouble(0.4976186676320155);
        reconstructionHighPass[5] = _numOps.FromDouble(-0.7366601814282105);
        reconstructionHighPass[6] = _numOps.FromDouble(0.2746482511903850);
        reconstructionHighPass[7] = _numOps.FromDouble(0.2746482511903850);
        reconstructionHighPass[8] = _numOps.FromDouble(-0.1485354424027703);
        reconstructionHighPass[9] = _numOps.FromDouble(-0.0686539440891211);
        reconstructionHighPass[10] = _numOps.FromDouble(0.0307358671058437);
        reconstructionHighPass[11] = _numOps.FromDouble(0.0130514548443985);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior37Coefficients()
    {
        var decompositionLowPass = new Vector<T>(12);
        var decompositionHighPass = new Vector<T>(8);
        var reconstructionLowPass = new Vector<T>(8);
        var reconstructionHighPass = new Vector<T>(12);

        decompositionLowPass[0] = _numOps.FromDouble(0.0030210861012608843);
        decompositionLowPass[1] = _numOps.FromDouble(-0.009063258303782653);
        decompositionLowPass[2] = _numOps.FromDouble(-0.01683176542131064);
        decompositionLowPass[3] = _numOps.FromDouble(0.074663985074019);
        decompositionLowPass[4] = _numOps.FromDouble(0.03133297870736289);
        decompositionLowPass[5] = _numOps.FromDouble(-0.301159125922835);
        decompositionLowPass[6] = _numOps.FromDouble(-0.026499240945345472);
        decompositionLowPass[7] = _numOps.FromDouble(0.9516421218971786);
        decompositionLowPass[8] = _numOps.FromDouble(0.9516421218971786);
        decompositionLowPass[9] = _numOps.FromDouble(-0.026499240945345472);
        decompositionLowPass[10] = _numOps.FromDouble(-0.301159125922835);
        decompositionLowPass[11] = _numOps.FromDouble(0.03133297870736289);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(0);
        decompositionHighPass[2] = _numOps.FromDouble(-0.1767766952966369);
        decompositionHighPass[3] = _numOps.FromDouble(0.5303300858899107);
        decompositionHighPass[4] = _numOps.FromDouble(-0.5303300858899107);
        decompositionHighPass[5] = _numOps.FromDouble(0.1767766952966369);
        decompositionHighPass[6] = _numOps.FromDouble(0);
        decompositionHighPass[7] = _numOps.FromDouble(0);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(0);
        reconstructionLowPass[2] = _numOps.FromDouble(0.1767766952966369);
        reconstructionLowPass[3] = _numOps.FromDouble(0.5303300858899107);
        reconstructionLowPass[4] = _numOps.FromDouble(0.5303300858899107);
        reconstructionLowPass[5] = _numOps.FromDouble(0.1767766952966369);
        reconstructionLowPass[6] = _numOps.FromDouble(0);
        reconstructionLowPass[7] = _numOps.FromDouble(0);

        reconstructionHighPass[0] = _numOps.FromDouble(0.0030210861012608843);
        reconstructionHighPass[1] = _numOps.FromDouble(0.009063258303782653);
        reconstructionHighPass[2] = _numOps.FromDouble(-0.01683176542131064);
        reconstructionHighPass[3] = _numOps.FromDouble(-0.074663985074019);
        reconstructionHighPass[4] = _numOps.FromDouble(0.03133297870736289);
        reconstructionHighPass[5] = _numOps.FromDouble(0.301159125922835);
        reconstructionHighPass[6] = _numOps.FromDouble(-0.026499240945345472);
        reconstructionHighPass[7] = _numOps.FromDouble(-0.9516421218971786);
        reconstructionHighPass[8] = _numOps.FromDouble(0.9516421218971786);
        reconstructionHighPass[9] = _numOps.FromDouble(0.026499240945345472);
        reconstructionHighPass[10] = _numOps.FromDouble(-0.301159125922835);
        reconstructionHighPass[11] = _numOps.FromDouble(-0.03133297870736289);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior39Coefficients()
    {
        var decompositionLowPass = new Vector<T>(16);
        var decompositionHighPass = new Vector<T>(10);
        var reconstructionLowPass = new Vector<T>(10);
        var reconstructionHighPass = new Vector<T>(16);

        decompositionLowPass[0] = _numOps.FromDouble(-0.000679744372783699);
        decompositionLowPass[1] = _numOps.FromDouble(0.002039233118351097);
        decompositionLowPass[2] = _numOps.FromDouble(0.005060319219611981);
        decompositionLowPass[3] = _numOps.FromDouble(-0.020618912641105536);
        decompositionLowPass[4] = _numOps.FromDouble(-0.014112787930175846);
        decompositionLowPass[5] = _numOps.FromDouble(0.09913478249423216);
        decompositionLowPass[6] = _numOps.FromDouble(0.012300136269419315);
        decompositionLowPass[7] = _numOps.FromDouble(-0.32019196836077857);
        decompositionLowPass[8] = _numOps.FromDouble(0.0020500227115698858);
        decompositionLowPass[9] = _numOps.FromDouble(0.9421257006782068);
        decompositionLowPass[10] = _numOps.FromDouble(0.9421257006782068);
        decompositionLowPass[11] = _numOps.FromDouble(0.0020500227115698858);
        decompositionLowPass[12] = _numOps.FromDouble(-0.32019196836077857);
        decompositionLowPass[13] = _numOps.FromDouble(0.012300136269419315);
        decompositionLowPass[14] = _numOps.FromDouble(0.09913478249423216);
        decompositionLowPass[15] = _numOps.FromDouble(-0.014112787930175846);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(0);
        decompositionHighPass[2] = _numOps.FromDouble(0);
        decompositionHighPass[3] = _numOps.FromDouble(0.0662912607362388);
        decompositionHighPass[4] = _numOps.FromDouble(0.1988737822087164);
        decompositionHighPass[5] = _numOps.FromDouble(0.1988737822087164);
        decompositionHighPass[6] = _numOps.FromDouble(0.0662912607362388);
        decompositionHighPass[7] = _numOps.FromDouble(0);
        decompositionHighPass[8] = _numOps.FromDouble(0);
        decompositionHighPass[9] = _numOps.FromDouble(0);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(0);
        reconstructionLowPass[2] = _numOps.FromDouble(0);
        reconstructionLowPass[3] = _numOps.FromDouble(-0.0662912607362388);
        reconstructionLowPass[4] = _numOps.FromDouble(0.1988737822087164);
        reconstructionLowPass[5] = _numOps.FromDouble(-0.1988737822087164);
        reconstructionLowPass[6] = _numOps.FromDouble(0.0662912607362388);
        reconstructionLowPass[7] = _numOps.FromDouble(0);
        reconstructionLowPass[8] = _numOps.FromDouble(0);
        reconstructionLowPass[9] = _numOps.FromDouble(0);

        reconstructionHighPass[0] = _numOps.FromDouble(-0.014112787930175846);
        reconstructionHighPass[1] = _numOps.FromDouble(-0.09913478249423216);
        reconstructionHighPass[2] = _numOps.FromDouble(0.012300136269419315);
        reconstructionHighPass[3] = _numOps.FromDouble(0.32019196836077857);
        reconstructionHighPass[4] = _numOps.FromDouble(0.0020500227115698858);
        reconstructionHighPass[5] = _numOps.FromDouble(-0.9421257006782068);
        reconstructionHighPass[6] = _numOps.FromDouble(0.9421257006782068);
        reconstructionHighPass[7] = _numOps.FromDouble(-0.0020500227115698858);
        reconstructionHighPass[8] = _numOps.FromDouble(-0.32019196836077857);
        reconstructionHighPass[9] = _numOps.FromDouble(-0.012300136269419315);
        reconstructionHighPass[10] = _numOps.FromDouble(0.09913478249423216);
        reconstructionHighPass[11] = _numOps.FromDouble(0.014112787930175846);
        reconstructionHighPass[12] = _numOps.FromDouble(-0.020618912641105536);
        reconstructionHighPass[13] = _numOps.FromDouble(-0.005060319219611981);
        reconstructionHighPass[14] = _numOps.FromDouble(0.002039233118351097);
        reconstructionHighPass[15] = _numOps.FromDouble(0.000679744372783699);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior44Coefficients()
    {
        var decompositionLowPass = new Vector<T>(10);
        var decompositionHighPass = new Vector<T>(10);
        var reconstructionLowPass = new Vector<T>(10);
        var reconstructionHighPass = new Vector<T>(10);

        decompositionLowPass[0] = _numOps.FromDouble(0);
        decompositionLowPass[1] = _numOps.FromDouble(0.03782845550699535);
        decompositionLowPass[2] = _numOps.FromDouble(-0.023849465019380);
        decompositionLowPass[3] = _numOps.FromDouble(-0.11062440441842);
        decompositionLowPass[4] = _numOps.FromDouble(0.37740285561265);
        decompositionLowPass[5] = _numOps.FromDouble(0.85269867900940);
        decompositionLowPass[6] = _numOps.FromDouble(0.37740285561265);
        decompositionLowPass[7] = _numOps.FromDouble(-0.11062440441842);
        decompositionLowPass[8] = _numOps.FromDouble(-0.023849465019380);
        decompositionLowPass[9] = _numOps.FromDouble(0.03782845550699535);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(-0.06453888262893856);
        decompositionHighPass[2] = _numOps.FromDouble(0.04068941760955867);
        decompositionHighPass[3] = _numOps.FromDouble(0.41809227322161724);
        decompositionHighPass[4] = _numOps.FromDouble(-0.7884856164056651);
        decompositionHighPass[5] = _numOps.FromDouble(0.4180922732216172);
        decompositionHighPass[6] = _numOps.FromDouble(0.040689417609558675);
        decompositionHighPass[7] = _numOps.FromDouble(-0.06453888262893856);
        decompositionHighPass[8] = _numOps.FromDouble(0);
        decompositionHighPass[9] = _numOps.FromDouble(0);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(-0.06453888262893856);
        reconstructionLowPass[2] = _numOps.FromDouble(-0.04068941760955867);
        reconstructionLowPass[3] = _numOps.FromDouble(0.41809227322161724);
        reconstructionLowPass[4] = _numOps.FromDouble(0.7884856164056651);
        reconstructionLowPass[5] = _numOps.FromDouble(0.4180922732216172);
        reconstructionLowPass[6] = _numOps.FromDouble(-0.040689417609558675);
        reconstructionLowPass[7] = _numOps.FromDouble(-0.06453888262893856);
        reconstructionLowPass[8] = _numOps.FromDouble(0);
        reconstructionLowPass[9] = _numOps.FromDouble(0);

        reconstructionHighPass[0] = _numOps.FromDouble(0);
        reconstructionHighPass[1] = _numOps.FromDouble(0.03782845550699535);
        reconstructionHighPass[2] = _numOps.FromDouble(0.023849465019380);
        reconstructionHighPass[3] = _numOps.FromDouble(-0.11062440441842);
        reconstructionHighPass[4] = _numOps.FromDouble(-0.37740285561265);
        reconstructionHighPass[5] = _numOps.FromDouble(0.85269867900940);
        reconstructionHighPass[6] = _numOps.FromDouble(-0.37740285561265);
        reconstructionHighPass[7] = _numOps.FromDouble(-0.11062440441842);
        reconstructionHighPass[8] = _numOps.FromDouble(0.023849465019380);
        reconstructionHighPass[9] = _numOps.FromDouble(0.03782845550699535);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior46Coefficients()
    {
        var decompositionLowPass = new Vector<T>(14);
        var decompositionHighPass = new Vector<T>(10);
        var reconstructionLowPass = new Vector<T>(10);
        var reconstructionHighPass = new Vector<T>(14);

        decompositionLowPass[0] = _numOps.FromDouble(0.0019088317364812906);
        decompositionLowPass[1] = _numOps.FromDouble(-0.0019142861290887667);
        decompositionLowPass[2] = _numOps.FromDouble(-0.016990639867602342);
        decompositionLowPass[3] = _numOps.FromDouble(0.01193456527972926);
        decompositionLowPass[4] = _numOps.FromDouble(0.04973290349094079);
        decompositionLowPass[5] = _numOps.FromDouble(-0.07726317316720414);
        decompositionLowPass[6] = _numOps.FromDouble(-0.09405920349573646);
        decompositionLowPass[7] = _numOps.FromDouble(0.4207962846098268);
        decompositionLowPass[8] = _numOps.FromDouble(0.8259229974584023);
        decompositionLowPass[9] = _numOps.FromDouble(0.4207962846098268);
        decompositionLowPass[10] = _numOps.FromDouble(-0.09405920349573646);
        decompositionLowPass[11] = _numOps.FromDouble(-0.07726317316720414);
        decompositionLowPass[12] = _numOps.FromDouble(0.04973290349094079);
        decompositionLowPass[13] = _numOps.FromDouble(0.01193456527972926);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(0);
        decompositionHighPass[2] = _numOps.FromDouble(0.03782845550699535);
        decompositionHighPass[3] = _numOps.FromDouble(-0.023849465019380);
        decompositionHighPass[4] = _numOps.FromDouble(-0.11062440441842);
        decompositionHighPass[5] = _numOps.FromDouble(0.37740285561265);
        decompositionHighPass[6] = _numOps.FromDouble(-0.85269867900940);
        decompositionHighPass[7] = _numOps.FromDouble(0.37740285561265);
        decompositionHighPass[8] = _numOps.FromDouble(-0.11062440441842);
        decompositionHighPass[9] = _numOps.FromDouble(-0.023849465019380);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(0);
        reconstructionLowPass[2] = _numOps.FromDouble(0.03782845550699535);
        reconstructionLowPass[3] = _numOps.FromDouble(0.023849465019380);
        reconstructionLowPass[4] = _numOps.FromDouble(-0.11062440441842);
        reconstructionLowPass[5] = _numOps.FromDouble(-0.37740285561265);
        reconstructionLowPass[6] = _numOps.FromDouble(0.85269867900940);
        reconstructionLowPass[7] = _numOps.FromDouble(-0.37740285561265);
        reconstructionLowPass[8] = _numOps.FromDouble(-0.11062440441842);
        reconstructionLowPass[9] = _numOps.FromDouble(0.023849465019380);

        reconstructionHighPass[0] = _numOps.FromDouble(0.0019088317364812906);
        reconstructionHighPass[1] = _numOps.FromDouble(0.0019142861290887667);
        reconstructionHighPass[2] = _numOps.FromDouble(-0.016990639867602342);
        reconstructionHighPass[3] = _numOps.FromDouble(-0.01193456527972926);
        reconstructionHighPass[4] = _numOps.FromDouble(0.04973290349094079);
        reconstructionHighPass[5] = _numOps.FromDouble(0.07726317316720414);
        reconstructionHighPass[6] = _numOps.FromDouble(-0.09405920349573646);
        reconstructionHighPass[7] = _numOps.FromDouble(-0.4207962846098268);
        reconstructionHighPass[8] = _numOps.FromDouble(0.8259229974584023);
        reconstructionHighPass[9] = _numOps.FromDouble(-0.4207962846098268);
        reconstructionHighPass[10] = _numOps.FromDouble(-0.09405920349573646);
        reconstructionHighPass[11] = _numOps.FromDouble(0.07726317316720414);
        reconstructionHighPass[12] = _numOps.FromDouble(0.04973290349094079);
        reconstructionHighPass[13] = _numOps.FromDouble(-0.01193456527972926);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior48Coefficients()
    {
        var decompositionLowPass = new Vector<T>(18);
        var decompositionHighPass = new Vector<T>(10);
        var reconstructionLowPass = new Vector<T>(10);
        var reconstructionHighPass = new Vector<T>(18);

        decompositionLowPass[0] = _numOps.FromDouble(0);
        decompositionLowPass[1] = _numOps.FromDouble(-0.0001174767841);
        decompositionLowPass[2] = _numOps.FromDouble(-0.0002349535682);
        decompositionLowPass[3] = _numOps.FromDouble(0.0013925484327);
        decompositionLowPass[4] = _numOps.FromDouble(0.0030931751602);
        decompositionLowPass[5] = _numOps.FromDouble(-0.0138017446325);
        decompositionLowPass[6] = _numOps.FromDouble(-0.0457246565401);
        decompositionLowPass[7] = _numOps.FromDouble(0.0687510184021);
        decompositionLowPass[8] = _numOps.FromDouble(0.3848874557553);
        decompositionLowPass[9] = _numOps.FromDouble(0.8525720202122);
        decompositionLowPass[10] = _numOps.FromDouble(0.3848874557553);
        decompositionLowPass[11] = _numOps.FromDouble(0.0687510184021);
        decompositionLowPass[12] = _numOps.FromDouble(-0.0457246565401);
        decompositionLowPass[13] = _numOps.FromDouble(-0.0138017446325);
        decompositionLowPass[14] = _numOps.FromDouble(0.0030931751602);
        decompositionLowPass[15] = _numOps.FromDouble(0.0013925484327);
        decompositionLowPass[16] = _numOps.FromDouble(-0.0002349535682);
        decompositionLowPass[17] = _numOps.FromDouble(-0.0001174767841);

        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(-0.0645388826289);
        decompositionHighPass[2] = _numOps.FromDouble(0.0406894176091);
        decompositionHighPass[3] = _numOps.FromDouble(0.4180922732222);
        decompositionHighPass[4] = _numOps.FromDouble(-0.7884856164057);
        decompositionHighPass[5] = _numOps.FromDouble(0.4180922732222);
        decompositionHighPass[6] = _numOps.FromDouble(0.0406894176091);
        decompositionHighPass[7] = _numOps.FromDouble(-0.0645388826289);
        decompositionHighPass[8] = _numOps.FromDouble(0);
        decompositionHighPass[9] = _numOps.FromDouble(0);

        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(0.0645388826289);
        reconstructionLowPass[2] = _numOps.FromDouble(0.0406894176091);
        reconstructionLowPass[3] = _numOps.FromDouble(-0.4180922732222);
        reconstructionLowPass[4] = _numOps.FromDouble(-0.7884856164057);
        reconstructionLowPass[5] = _numOps.FromDouble(-0.4180922732222);
        reconstructionLowPass[6] = _numOps.FromDouble(0.0406894176091);
        reconstructionLowPass[7] = _numOps.FromDouble(0.0645388826289);
        reconstructionLowPass[8] = _numOps.FromDouble(0);
        reconstructionLowPass[9] = _numOps.FromDouble(0);

        reconstructionHighPass[0] = _numOps.FromDouble(0);
        reconstructionHighPass[1] = _numOps.FromDouble(-0.0001174767841);
        reconstructionHighPass[2] = _numOps.FromDouble(0.0002349535682);
        reconstructionHighPass[3] = _numOps.FromDouble(0.0013925484327);
        reconstructionHighPass[4] = _numOps.FromDouble(-0.0030931751602);
        reconstructionHighPass[5] = _numOps.FromDouble(-0.0138017446325);
        reconstructionHighPass[6] = _numOps.FromDouble(0.0457246565401);
        reconstructionHighPass[7] = _numOps.FromDouble(0.0687510184021);
        reconstructionHighPass[8] = _numOps.FromDouble(-0.3848874557553);
        reconstructionHighPass[9] = _numOps.FromDouble(0.8525720202122);
        reconstructionHighPass[10] = _numOps.FromDouble(-0.3848874557553);
        reconstructionHighPass[11] = _numOps.FromDouble(0.0687510184021);
        reconstructionHighPass[12] = _numOps.FromDouble(0.0457246565401);
        reconstructionHighPass[13] = _numOps.FromDouble(-0.0138017446325);
        reconstructionHighPass[14] = _numOps.FromDouble(-0.0030931751602);
        reconstructionHighPass[15] = _numOps.FromDouble(0.0013925484327);
        reconstructionHighPass[16] = _numOps.FromDouble(0.0002349535682);
        reconstructionHighPass[17] = _numOps.FromDouble(-0.0001174767841);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior55Coefficients()
    {
        var decompositionLowPass = new Vector<T>(12);
        var decompositionHighPass = new Vector<T>(12);
        var reconstructionLowPass = new Vector<T>(12);
        var reconstructionHighPass = new Vector<T>(12);

        decompositionLowPass[0] = _numOps.FromDouble(0.0);
        decompositionLowPass[1] = _numOps.FromDouble(0.013456709459118716);
        decompositionLowPass[2] = _numOps.FromDouble(-0.002694966880111507);
        decompositionLowPass[3] = _numOps.FromDouble(-0.13670658466432914);
        decompositionLowPass[4] = _numOps.FromDouble(-0.09350469740093886);
        decompositionLowPass[5] = _numOps.FromDouble(0.47680326579848425);
        decompositionLowPass[6] = _numOps.FromDouble(0.8995061097486484);
        decompositionLowPass[7] = _numOps.FromDouble(0.47680326579848425);
        decompositionLowPass[8] = _numOps.FromDouble(-0.09350469740093886);
        decompositionLowPass[9] = _numOps.FromDouble(-0.13670658466432914);
        decompositionLowPass[10] = _numOps.FromDouble(-0.002694966880111507);
        decompositionLowPass[11] = _numOps.FromDouble(0.013456709459118716);

        decompositionHighPass[0] = _numOps.FromDouble(0.013456709459118716);
        decompositionHighPass[1] = _numOps.FromDouble(-0.002694966880111507);
        decompositionHighPass[2] = _numOps.FromDouble(-0.13670658466432914);
        decompositionHighPass[3] = _numOps.FromDouble(-0.09350469740093886);
        decompositionHighPass[4] = _numOps.FromDouble(0.47680326579848425);
        decompositionHighPass[5] = _numOps.FromDouble(-0.8995061097486484);
        decompositionHighPass[6] = _numOps.FromDouble(0.47680326579848425);
        decompositionHighPass[7] = _numOps.FromDouble(-0.09350469740093886);
        decompositionHighPass[8] = _numOps.FromDouble(-0.13670658466432914);
        decompositionHighPass[9] = _numOps.FromDouble(-0.002694966880111507);
        decompositionHighPass[10] = _numOps.FromDouble(0.013456709459118716);
        decompositionHighPass[11] = _numOps.FromDouble(0.0);

        reconstructionLowPass[0] = _numOps.FromDouble(0.013456709459118716);
        reconstructionLowPass[1] = _numOps.FromDouble(0.002694966880111507);
        reconstructionLowPass[2] = _numOps.FromDouble(-0.13670658466432914);
        reconstructionLowPass[3] = _numOps.FromDouble(0.09350469740093886);
        reconstructionLowPass[4] = _numOps.FromDouble(0.47680326579848425);
        reconstructionLowPass[5] = _numOps.FromDouble(0.8995061097486484);
        reconstructionLowPass[6] = _numOps.FromDouble(0.47680326579848425);
        reconstructionLowPass[7] = _numOps.FromDouble(0.09350469740093886);
        reconstructionLowPass[8] = _numOps.FromDouble(-0.13670658466432914);
        reconstructionLowPass[9] = _numOps.FromDouble(0.002694966880111507);
        reconstructionLowPass[10] = _numOps.FromDouble(0.013456709459118716);
        reconstructionLowPass[11] = _numOps.FromDouble(0.0);

        reconstructionHighPass[0] = _numOps.FromDouble(0.0);
        reconstructionHighPass[1] = _numOps.FromDouble(0.013456709459118716);
        reconstructionHighPass[2] = _numOps.FromDouble(-0.002694966880111507);
        reconstructionHighPass[3] = _numOps.FromDouble(-0.13670658466432914);
        reconstructionHighPass[4] = _numOps.FromDouble(-0.09350469740093886);
        reconstructionHighPass[5] = _numOps.FromDouble(0.47680326579848425);
        reconstructionHighPass[6] = _numOps.FromDouble(-0.8995061097486484);
        reconstructionHighPass[7] = _numOps.FromDouble(0.47680326579848425);
        reconstructionHighPass[8] = _numOps.FromDouble(-0.09350469740093886);
        reconstructionHighPass[9] = _numOps.FromDouble(-0.13670658466432914);
        reconstructionHighPass[10] = _numOps.FromDouble(-0.002694966880111507);
        reconstructionHighPass[11] = _numOps.FromDouble(0.013456709459118716);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }

    private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior68Coefficients()
    {
        var decompositionLowPass = new Vector<T>(18);
        var decompositionHighPass = new Vector<T>(10);
        var reconstructionLowPass = new Vector<T>(10);
        var reconstructionHighPass = new Vector<T>(18);

        // Decomposition low-pass filter
        decompositionLowPass[0] = _numOps.FromDouble(0);
        decompositionLowPass[1] = _numOps.FromDouble(0.0001490583487665);
        decompositionLowPass[2] = _numOps.FromDouble(-0.0003179695108439);
        decompositionLowPass[3] = _numOps.FromDouble(-0.0018118519793764);
        decompositionLowPass[4] = _numOps.FromDouble(0.0047314665272548);
        decompositionLowPass[5] = _numOps.FromDouble(0.0087901063101452);
        decompositionLowPass[6] = _numOps.FromDouble(-0.0297451247861220);
        decompositionLowPass[7] = _numOps.FromDouble(-0.0736365678679802);
        decompositionLowPass[8] = _numOps.FromDouble(0.1485354836691763);
        decompositionLowPass[9] = _numOps.FromDouble(0.4675630700319812);
        decompositionLowPass[10] = _numOps.FromDouble(0.9667475524034829);
        decompositionLowPass[11] = _numOps.FromDouble(0.4675630700319812);
        decompositionLowPass[12] = _numOps.FromDouble(0.1485354836691763);
        decompositionLowPass[13] = _numOps.FromDouble(-0.0736365678679802);
        decompositionLowPass[14] = _numOps.FromDouble(-0.0297451247861220);
        decompositionLowPass[15] = _numOps.FromDouble(0.0087901063101452);
        decompositionLowPass[16] = _numOps.FromDouble(0.0047314665272548);
        decompositionLowPass[17] = _numOps.FromDouble(-0.0018118519793764);

        // Decomposition high-pass filter
        decompositionHighPass[0] = _numOps.FromDouble(0);
        decompositionHighPass[1] = _numOps.FromDouble(-0.0107148249460572);
        decompositionHighPass[2] = _numOps.FromDouble(0.0328921202609630);
        decompositionHighPass[3] = _numOps.FromDouble(0.0308560115845869);
        decompositionHighPass[4] = _numOps.FromDouble(-0.1870348117190931);
        decompositionHighPass[5] = _numOps.FromDouble(0.0279837694169839);
        decompositionHighPass[6] = _numOps.FromDouble(0.6308807679295904);
        decompositionHighPass[7] = _numOps.FromDouble(-0.7148465705525415);
        decompositionHighPass[8] = _numOps.FromDouble(0.2303778133088552);
        decompositionHighPass[9] = _numOps.FromDouble(0.0279837694169839);

        // Reconstruction low-pass filter
        reconstructionLowPass[0] = _numOps.FromDouble(0);
        reconstructionLowPass[1] = _numOps.FromDouble(0.0279837694169839);
        reconstructionLowPass[2] = _numOps.FromDouble(0.2303778133088552);
        reconstructionLowPass[3] = _numOps.FromDouble(0.7148465705525415);
        reconstructionLowPass[4] = _numOps.FromDouble(0.6308807679295904);
        reconstructionLowPass[5] = _numOps.FromDouble(0.0279837694169839);
        reconstructionLowPass[6] = _numOps.FromDouble(-0.1870348117190931);
        reconstructionLowPass[7] = _numOps.FromDouble(0.0308560115845869);
        reconstructionLowPass[8] = _numOps.FromDouble(0.0328921202609630);
        reconstructionLowPass[9] = _numOps.FromDouble(-0.0107148249460572);

        // Reconstruction high-pass filter
        reconstructionHighPass[0] = _numOps.FromDouble(0);
        reconstructionHighPass[1] = _numOps.FromDouble(-0.0018118519793764);
        reconstructionHighPass[2] = _numOps.FromDouble(0.0047314665272548);
        reconstructionHighPass[3] = _numOps.FromDouble(0.0087901063101452);
        reconstructionHighPass[4] = _numOps.FromDouble(-0.0297451247861220);
        reconstructionHighPass[5] = _numOps.FromDouble(-0.0736365678679802);
        reconstructionHighPass[6] = _numOps.FromDouble(0.1485354836691763);
        reconstructionHighPass[7] = _numOps.FromDouble(0.4675630700319812);
        reconstructionHighPass[8] = _numOps.FromDouble(-0.9667475524034829);
        reconstructionHighPass[9] = _numOps.FromDouble(0.4675630700319812);
        reconstructionHighPass[10] = _numOps.FromDouble(0.1485354836691763);
        reconstructionHighPass[11] = _numOps.FromDouble(-0.0736365678679802);
        reconstructionHighPass[12] = _numOps.FromDouble(-0.0297451247861220);
        reconstructionHighPass[13] = _numOps.FromDouble(0.0087901063101452);
        reconstructionHighPass[14] = _numOps.FromDouble(0.0047314665272548);
        reconstructionHighPass[15] = _numOps.FromDouble(-0.0018118519793764);
        reconstructionHighPass[16] = _numOps.FromDouble(-0.0003179695108439);
        reconstructionHighPass[17] = _numOps.FromDouble(0.0001490583487665);

        return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
    }
}