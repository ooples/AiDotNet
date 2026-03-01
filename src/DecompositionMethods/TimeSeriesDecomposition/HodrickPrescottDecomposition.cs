namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements the Hodrick-Prescott filter for decomposing time series data into trend and cyclical components.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> The Hodrick-Prescott filter is a mathematical tool that helps separate a time series 
/// (like stock prices or economic data over time) into two parts:
/// 1. A smooth trend component that shows the long-term direction
/// 2. A cyclical component that shows short-term fluctuations around the trend
/// 
/// Think of it like separating a bumpy road (your data) into the general path (trend) 
/// and the bumps along the way (cycles).
/// </remarks>
public class HodrickPrescottDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    /// <summary>
    /// The smoothing parameter that controls the balance between smoothness of the trend and fit to the data.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of lambda as a "smoothness knob" - higher values make the trend smoother 
    /// but potentially less accurate, while lower values make the trend follow the data more closely.
    /// Common values: 1600 for quarterly data, 100 for yearly data, 14400 for monthly data.
    /// </remarks>
    private readonly T _lambda;

    /// <summary>
    /// Optional matrix decomposition method used for solving the linear system in the matrix method.
    /// </summary>
    private readonly IMatrixDecomposition<T>? _decomposition;

    /// <summary>
    /// The algorithm type to use for the Hodrick-Prescott decomposition.
    /// </summary>
    private readonly HodrickPrescottAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the Hodrick-Prescott decomposition.
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose.</param>
    /// <param name="lambda">The smoothing parameter (default: 1600, suitable for quarterly data).</param>
    /// <param name="decomposition">Optional matrix decomposition method for solving linear systems.</param>
    /// <param name="algorithm">The algorithm type to use for decomposition (default: MatrixMethod).</param>
    /// <exception cref="ArgumentException">Thrown when lambda is not positive.</exception>
    public HodrickPrescottDecomposition(Vector<T> timeSeries, double lambda = 1600, IMatrixDecomposition<T>? decomposition = null,
        HodrickPrescottAlgorithmType algorithm = HodrickPrescottAlgorithmType.MatrixMethod)
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

    /// <summary>
    /// Performs the time series decomposition using the selected algorithm.
    /// </summary>
    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case HodrickPrescottAlgorithmType.MatrixMethod:
                DecomposeMatrixMethod();
                break;
            case HodrickPrescottAlgorithmType.IterativeMethod:
                DecomposeIterativeMethod();
                break;
            case HodrickPrescottAlgorithmType.KalmanFilterMethod:
                DecomposeKalmanFilterMethod();
                break;
            case HodrickPrescottAlgorithmType.WaveletMethod:
                DecomposeWaveletMethod();
                break;
            case HodrickPrescottAlgorithmType.FrequencyDomainMethod:
                DecomposeFrequencyDomainMethod();
                break;
            case HodrickPrescottAlgorithmType.StateSpaceMethod:
                DecomposeStateSpaceMethod();
                break;
            default:
                throw new ArgumentException("Unsupported algorithm", nameof(_algorithm));
        }
    }

    /// <summary>
    /// Decomposes the time series using a Kalman filter approach.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A Kalman filter is like a GPS system for your data - it continuously 
    /// updates its estimate of where the trend is, based on new observations and previous estimates.
    /// It's especially good at handling noisy data.
    /// </remarks>
    private void DecomposeKalmanFilterMethod()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n);
        Vector<T> cycle = new Vector<T>(n);

        // State transition matrix
        Matrix<T> F = new Matrix<T>(new T[,] { { NumOps.One, NumOps.One }, { NumOps.Zero, NumOps.One } });

        // Observation matrix
        Matrix<T> H = new Matrix<T>(new T[,] { { NumOps.One, NumOps.Zero } });

        // Initial state estimate
        Vector<T> x = new Vector<T>(new T[] { TimeSeries[0], NumOps.Zero });

        // Initial error covariance
        Matrix<T> P = new Matrix<T>(new T[,] {
            { NumOps.FromDouble(1000), NumOps.Zero },
            { NumOps.Zero, NumOps.FromDouble(1000) }
        });

        // Process noise covariance
        Matrix<T> Q = new Matrix<T>(new T[,] {
            { NumOps.FromDouble(0.01), NumOps.Zero },
            { NumOps.Zero, NumOps.FromDouble(0.01) }
        });

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

    /// <summary>
    /// Decomposes the time series using wavelet transform.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Wavelet decomposition is like breaking down a song into different instruments.
    /// It separates your data into different frequency components, allowing you to keep the low-frequency
    /// parts (trend) and filter out high-frequency parts (cycles/noise).
    /// </remarks>
    private void DecomposeWaveletMethod()
    {
        int n = TimeSeries.Length;
        int levels = (int)Math.Log(n, 2);

        Vector<T> trend = new Vector<T>(n);
        Vector<T> cycle = new Vector<T>(n);

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

    /// <summary>
    /// Performs a discrete wavelet transform on the input data.
    /// </summary>
    /// <param name="data">The input time series data.</param>
    /// <param name="levels">The number of decomposition levels.</param>
    /// <returns>The wavelet coefficients.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method transforms your data into a different representation where
    /// different scales (frequencies) of patterns are separated. It's like breaking down a complex
    /// sound wave into its component frequencies.
    /// </remarks>
    private Vector<T> DiscreteWaveletTransform(Vector<T> data, int levels)
    {
        int n = data.Length;
        Vector<T> coeffs = data.Clone();

        for (int level = 0; level < levels; level++)
        {
            int levelSize = n >> level;
            int halfSize = levelSize >> 1;

            // Use temporary arrays to avoid in-place overwrite corruption
            T[] approx = new T[halfSize];
            T[] detail = new T[halfSize];
            for (int i = 0; i < halfSize; i++)
            {
                T sum = NumOps.Add(coeffs[i * 2], coeffs[i * 2 + 1]);
                T difference = NumOps.Subtract(coeffs[i * 2], coeffs[i * 2 + 1]);
                approx[i] = NumOps.Multiply(sum, NumOps.FromDouble(0.5));
                detail[i] = NumOps.Multiply(difference, NumOps.FromDouble(0.5));
            }
            for (int i = 0; i < halfSize; i++)
            {
                coeffs[i] = approx[i];
                coeffs[halfSize + i] = detail[i];
            }
        }

        return coeffs;
    }

    /// <summary>
    /// Performs an inverse discrete wavelet transform to reconstruct the original signal.
    /// </summary>
    /// <param name="coeffs">The wavelet coefficients.</param>
    /// <param name="levels">The number of decomposition levels.</param>
    /// <returns>The reconstructed signal.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes the wavelet coefficients (which represent different frequency components)
    /// and recombines them to create a time series. It's like mixing different instrument tracks back
    /// together to form a complete song.
    /// </remarks>
    private Vector<T> InverseDiscreteWaveletTransform(Vector<T> coeffs, int levels)
    {
        int n = coeffs.Length;
        Vector<T> data = coeffs.Clone();

        for (int level = levels - 1; level >= 0; level--)
        {
            int levelSize = n >> level;
            int halfSize = levelSize >> 1;

            // Use temporary array to avoid in-place overwrite corruption
            // Inverse of DWT: approx = (a+b)/2, detail = (a-b)/2
            // => a = approx + detail, b = approx - detail (no factor of 2)
            T[] expanded = new T[levelSize];
            for (int i = 0; i < halfSize; i++)
            {
                T approx = data[i];
                T detail = data[halfSize + i];
                expanded[i * 2] = NumOps.Add(approx, detail);
                expanded[i * 2 + 1] = NumOps.Subtract(approx, detail);
            }
            for (int i = 0; i < levelSize; i++)
            {
                data[i] = expanded[i];
            }
        }

        return data;
    }

    /// <summary>
    /// Decomposes the time series using frequency domain analysis with Fast Fourier Transform.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method transforms your time series data from the time domain (values over time)
    /// to the frequency domain (showing which frequencies are present in your data). It then filters out
    /// high-frequency components (which represent rapid changes or noise) and keeps low-frequency components
    /// (which represent the trend). Think of it like adjusting the bass and treble on a music system - 
    /// we're keeping the "bass" (trend) and reducing the "treble" (cycles/noise).
    /// </remarks>
    private void DecomposeFrequencyDomainMethod()
    {
        int n = TimeSeries.Length;

        // Zero-pad to next power of 2 (FFT requires power-of-2 length)
        int nPadded = 1;
        while (nPadded < n) nPadded <<= 1;

        Vector<T> padded = new Vector<T>(nPadded);
        for (int i = 0; i < n; i++)
        {
            padded[i] = TimeSeries[i];
        }

        // Perform FFT on zero-padded data
        FastFourierTransform<T> fft = new();
        Vector<Complex<T>> frequencyDomain = fft.Forward(padded);

        // Apply low-pass filter in frequency domain
        T cutoffFrequency = NumOps.FromDouble(0.1);  // Adjust as needed
        for (int i = 0; i < nPadded; i++)
        {
            T frequency = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(nPadded));
            if (NumOps.GreaterThan(frequency, cutoffFrequency) && NumOps.LessThan(frequency, NumOps.Subtract(NumOps.One, cutoffFrequency)))
            {
                frequencyDomain[i] = new Complex<T>(NumOps.Zero, NumOps.Zero);
            }
        }

        // Inverse FFT to get trend, then truncate to original length
        Vector<T> paddedTrend = fft.Inverse(frequencyDomain);

        Vector<T> trend = new Vector<T>(n);
        Vector<T> cycle = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            trend[i] = paddedTrend[i];
            cycle[i] = NumOps.Subtract(TimeSeries[i], trend[i]);
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    /// <summary>
    /// Decomposes the time series using a state space modeling approach.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A state space model represents your data as a system with hidden states that evolve over time.
    /// In this case, we track three states: the current trend level (mu), the trend growth rate (beta), 
    /// and the cyclical component (c). As we process each data point, we update these states to best explain 
    /// the observed data. It's like tracking a moving object by continuously updating your estimate of its 
    /// position, velocity, and acceleration.
    /// </remarks>
    private void DecomposeStateSpaceMethod()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n);
        Vector<T> cycle = new Vector<T>(n);

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
            cycle[i] = NumOps.Subtract(TimeSeries[i], mu);
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    /// <summary>
    /// Decomposes the time series using the standard matrix-based Hodrick-Prescott filter.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the classic implementation of the Hodrick-Prescott filter. It works by setting up
    /// a mathematical problem that balances two goals: (1) making the trend close to the original data, and
    /// (2) making the trend smooth. This problem is solved using matrix operations, which is like solving
    /// a system of equations all at once. The result gives us the optimal trend component that balances
    /// these two goals according to the lambda parameter.
    /// </remarks>
    private void DecomposeMatrixMethod()
    {
        int n = TimeSeries.Length;

        Matrix<T> D = ConstructSecondDifferenceMatrix(n);
        Matrix<T> I = Matrix<T>.CreateIdentity(n);
        Matrix<T> A = I.Add(D.Transpose().Multiply(D).Multiply(_lambda));

        var decomposition = _decomposition ?? new LuDecomposition<T>(A);
        Vector<T> trend = MatrixSolutionHelper.SolveLinearSystem(TimeSeries, decomposition);
        Vector<T> cycle = TimeSeries.Subtract(trend);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    /// <summary>
    /// Decomposes the time series using an iterative approach to the Hodrick-Prescott filter.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Instead of solving the entire problem at once (as in the matrix method),
    /// this approach gradually improves the trend estimate through repeated passes over the data.
    /// In each iteration, we update each point in the trend based on its neighbors and the original data.
    /// It's like smoothing a curve by repeatedly running your hand over it until it reaches the desired smoothness.
    /// This method can be more efficient for very large datasets.
    /// </remarks>
    private void DecomposeIterativeMethod()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(TimeSeries);
        Vector<T> cycle = new Vector<T>(n);

        // Gauss-Seidel iteration for (I + λD^TD)τ = y
        // D^TD has pentadiagonal structure with coefficients [1, -4, 6, -4, 1] for interior rows
        // So: (1 + 6λ)τ[i] = y[i] + λ(4τ[i-1] - τ[i-2] + 4τ[i+1] - τ[i+2])
        T fourLambda = NumOps.Multiply(NumOps.FromDouble(4), _lambda);
        T denominator = NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(6), _lambda));

        for (int iteration = 0; iteration < 100; iteration++)
        {
            for (int i = 2; i < n - 2; i++)
            {
                // numerator = y[i] + λ*(4*τ[i-1] + 4*τ[i+1] - τ[i-2] - τ[i+2])
                T neighbors = NumOps.Subtract(
                    NumOps.Multiply(fourLambda, NumOps.Add(trend[i - 1], trend[i + 1])),
                    NumOps.Multiply(_lambda, NumOps.Add(trend[i - 2], trend[i + 2]))
                );
                T numerator = NumOps.Add(TimeSeries[i], neighbors);
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

    /// <summary>
    /// Constructs a second difference matrix used in the matrix-based Hodrick-Prescott filter.
    /// </summary>
    /// <param name="n">The size of the time series.</param>
    /// <returns>A matrix that represents the second difference operator.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This matrix is used to measure how "wiggly" or non-smooth the trend is.
    /// Each row of this matrix computes the difference between adjacent differences in the trend,
    /// which is a way to measure curvature or acceleration in the trend. The Hodrick-Prescott filter
    /// uses this matrix to penalize trends that change direction too quickly or too often.
    /// </remarks>
    private Matrix<T> ConstructSecondDifferenceMatrix(int n)
    {
        Matrix<T> D = new Matrix<T>(n - 2, n);

        for (int i = 0; i < n - 2; i++)
        {
            D[i, i] = NumOps.One;
            D[i, i + 1] = NumOps.Multiply(NumOps.FromDouble(-2), NumOps.One);
            D[i, i + 2] = NumOps.One;
        }

        return D;
    }
}
