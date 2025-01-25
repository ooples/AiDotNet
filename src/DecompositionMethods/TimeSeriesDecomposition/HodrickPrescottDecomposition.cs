namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class HodrickPrescottDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly T _lambda;
    private readonly IMatrixDecomposition<T>? _decomposition;
    private readonly HodrickPrescottAlgorithm _algorithm;

    public HodrickPrescottDecomposition(Vector<T> timeSeries, double lambda = 1600, IMatrixDecomposition<T>? decomposition = null, 
        HodrickPrescottAlgorithm algorithm = HodrickPrescottAlgorithm.MatrixMethod) 
        : base(timeSeries)
    {
        if (lambda <= 0)
        {
            throw new ArgumentException("Lambda must be a positive value.", nameof(lambda));
        }

        _lambda = NumOps.FromDouble(lambda);
        _decomposition = decomposition;
        _algorithm = algorithm;
        Decompose();
    }

    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case HodrickPrescottAlgorithm.MatrixMethod:
                DecomposeMatrixMethod();
                break;
            case HodrickPrescottAlgorithm.IterativeMethod:
                DecomposeIterativeMethod();
                break;
            case HodrickPrescottAlgorithm.KalmanFilterMethod:
                DecomposeKalmanFilterMethod();
                break;
            case HodrickPrescottAlgorithm.WaveletMethod:
                DecomposeWaveletMethod();
                break;
            case HodrickPrescottAlgorithm.FrequencyDomainMethod:
                DecomposeFrequencyDomainMethod();
                break;
            case HodrickPrescottAlgorithm.StateSpaceMethod:
                DecomposeStateSpaceMethod();
                break;
            default:
                throw new ArgumentException("Unsupported algorithm", nameof(_algorithm));
        }
    }

    private void DecomposeKalmanFilterMethod()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n, NumOps);
        Vector<T> cycle = new Vector<T>(n, NumOps);

        // State transition matrix
        Matrix<T> F = new Matrix<T>(new T[,] { { NumOps.One, NumOps.One }, { NumOps.Zero, NumOps.One } }, NumOps);
    
        // Observation matrix
        Matrix<T> H = new Matrix<T>(new T[,] { { NumOps.One, NumOps.Zero } }, NumOps);

        // Initial state estimate
        Vector<T> x = new Vector<T>(new T[] { TimeSeries[0], NumOps.Zero }, NumOps);

        // Initial error covariance
        Matrix<T> P = new Matrix<T>(new T[,] { 
            { NumOps.FromDouble(1000), NumOps.Zero }, 
            { NumOps.Zero, NumOps.FromDouble(1000) } 
        }, NumOps);

        // Process noise covariance
        Matrix<T> Q = new Matrix<T>(new T[,] { 
            { NumOps.FromDouble(0.01), NumOps.Zero }, 
            { NumOps.Zero, NumOps.FromDouble(0.01) } 
        }, NumOps);

        // Measurement noise covariance
        T R = NumOps.FromDouble(1);

        for (int i = 0; i < n; i++)
        {
            // Predict
            Vector<T> x_pred = F * x;
            Matrix<T> P_pred = F * P * F.Transpose() + Q;

            // Update
            T y = NumOps.Subtract(TimeSeries[i], (H * x_pred)[0]);
            T S = NumOps.Add((H * P_pred * H.Transpose())[0, 0], R);
            Vector<T> K = (P_pred * H.Transpose()).GetColumn(0) * NumOps.Divide(NumOps.One, S);

            x = x_pred + K * y;
            P = P_pred - K.OuterProduct((H * P_pred).GetRow(0));

            trend[i] = x[0];
            cycle[i] = NumOps.Subtract(TimeSeries[i], trend[i]);
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    private void DecomposeWaveletMethod()
    {
        int n = TimeSeries.Length;
        int levels = (int)Math.Log(n, 2);

        Vector<T> trend = new Vector<T>(n, NumOps);
        Vector<T> cycle = new Vector<T>(n, NumOps);

        // Perform wavelet decomposition
        Vector<T> coeffs = DiscreteWaveletTransform(TimeSeries, levels);

        // Threshold detail coefficients
        for (int i = 0; i < n; i++)
        {
            if (i >= n / 2)  // Detail coefficients
            {
                coeffs[i] = NumOps.Multiply(coeffs[i], NumOps.FromDouble(0.1));  // Soft thresholding
            }
        }

        // Perform inverse wavelet transform
        trend = InverseDiscreteWaveletTransform(coeffs, levels);

        // Calculate cycle
        for (int i = 0; i < n; i++)
        {
            cycle[i] = NumOps.Subtract(TimeSeries[i], trend[i]);
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    private Vector<T> DiscreteWaveletTransform(Vector<T> data, int levels)
    {
        int n = data.Length;
        Vector<T> coeffs = data.Copy();

        for (int level = 0; level < levels; level++)
        {
            int levelSize = n >> level;
            int halfSize = levelSize >> 1;

            for (int i = 0; i < halfSize; i++)
            {
                T sum = NumOps.Add(coeffs[i * 2], coeffs[i * 2 + 1]);
                T difference = NumOps.Subtract(coeffs[i * 2], coeffs[i * 2 + 1]);
                coeffs[i] = NumOps.Multiply(sum, NumOps.FromDouble(0.5));
                coeffs[halfSize + i] = NumOps.Multiply(difference, NumOps.FromDouble(0.5));
            }
        }

        return coeffs;
    }

    private Vector<T> InverseDiscreteWaveletTransform(Vector<T> coeffs, int levels)
    {
        int n = coeffs.Length;
        Vector<T> data = coeffs.Copy();

        for (int level = levels - 1; level >= 0; level--)
        {
            int levelSize = n >> level;
            int halfSize = levelSize >> 1;

            for (int i = 0; i < halfSize; i++)
            {
                T sum = NumOps.Multiply(data[i], NumOps.FromDouble(2));
                T difference = NumOps.Multiply(data[halfSize + i], NumOps.FromDouble(2));
                data[i * 2] = NumOps.Add(sum, difference);
                data[i * 2 + 1] = NumOps.Subtract(sum, difference);
            }
        }

        return data;
    }

    private void DecomposeFrequencyDomainMethod()
    {
        int n = TimeSeries.Length;
    
        // Perform FFT
        FastFourierTransform<T> fft = new();
        Vector<Complex<T>> frequencyDomain = fft.Forward(TimeSeries);

        // Apply low-pass filter in frequency domain
        T cutoffFrequency = NumOps.FromDouble(0.1);  // Adjust as needed
        for (int i = 0; i < n; i++)
        {
            T frequency = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(n));
            if (NumOps.GreaterThan(frequency, cutoffFrequency) && NumOps.LessThan(frequency, NumOps.Subtract(NumOps.One, cutoffFrequency)))
            {
                frequencyDomain[i] = new Complex<T>(NumOps.Zero, NumOps.Zero);
            }
        }

        // Inverse FFT to get trend
        Vector<T> trend = fft.Inverse(frequencyDomain);

        // Calculate cycle
        Vector<T> cycle = new Vector<T>(n, NumOps);
        for (int i = 0; i < n; i++)
        {
            cycle[i] = NumOps.Subtract(TimeSeries[i], trend[i]);
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    private void DecomposeStateSpaceMethod()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n, NumOps);
        Vector<T> cycle = new Vector<T>(n, NumOps);

        // State space model parameters
        T alpha = NumOps.FromDouble(0.1);  // Trend smoothness parameter
        T rho = NumOps.FromDouble(0.5);    // Cycle persistence parameter

        // Initialize state variables
        T mu = TimeSeries[0];
        T beta = NumOps.Zero;
        T c = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            // Update state variables
            T mu_prev = mu;
            mu = NumOps.Add(mu_prev, beta);
            beta = NumOps.Add(beta, NumOps.Multiply(alpha, NumOps.Subtract(TimeSeries[i], NumOps.Add(mu_prev, c))));
            c = NumOps.Multiply(rho, NumOps.Subtract(TimeSeries[i], mu));

            trend[i] = mu;
            cycle[i] = c;
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    private void DecomposeMatrixMethod()
    {
        int n = TimeSeries.Length;

        Matrix<T> D = ConstructSecondDifferenceMatrix(n);
        Matrix<T> I = Matrix<T>.CreateIdentity(n, NumOps);
        Matrix<T> A = I.Add(D.Transpose().Multiply(D).Multiply(_lambda));

        var decomposition = _decomposition ?? new LuDecomposition<T>(A);
        Vector<T> trend = MatrixSolutionHelper.SolveLinearSystem(TimeSeries, decomposition);
        Vector<T> cycle = TimeSeries.Subtract(trend);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    private void DecomposeIterativeMethod()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(TimeSeries);
        Vector<T> cycle = new Vector<T>(n, NumOps);

        T two = NumOps.FromDouble(2);
        T lambda2 = NumOps.Multiply(_lambda, two);
        T lambda22 = NumOps.Multiply(lambda2, two);

        for (int iteration = 0; iteration < 100; iteration++)
        {
            for (int i = 2; i < n - 2; i++)
            {
                T numerator = NumOps.Multiply(lambda2, NumOps.Add(NumOps.Add(trend[i - 1], trend[i + 1]), 
                              NumOps.Multiply(lambda22, NumOps.Add(NumOps.Add(trend[i - 2], trend[i + 2]), TimeSeries[i]))));
                T denominator = NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(6), _lambda));
                trend[i] = NumOps.Divide(numerator, denominator);
            }

            // Handle boundary cases
            trend[0] = TimeSeries[0];
            trend[1] = TimeSeries[1];
            trend[n - 2] = TimeSeries[n - 2];
            trend[n - 1] = TimeSeries[n - 1];
        }

        for (int i = 0; i < n; i++)
        {
            cycle[i] = NumOps.Subtract(TimeSeries[i], trend[i]);
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    private Matrix<T> ConstructSecondDifferenceMatrix(int n)
    {
        Matrix<T> D = new Matrix<T>(n - 2, n, NumOps);

        for (int i = 0; i < n - 2; i++)
        {
            D[i, i] = NumOps.One;
            D[i, i + 1] = NumOps.Multiply(NumOps.FromDouble(-2), NumOps.One);
            D[i, i + 2] = NumOps.One;
        }

        return D;
    }
}