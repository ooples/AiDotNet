namespace AiDotNet.Kernels;

/// <summary>
/// Linear Coregionalization Model (LCM) kernel for multi-output Gaussian Processes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Linear Coregionalization Model (LCM) is a powerful method
/// for multi-output Gaussian Processes. It models the correlations between multiple
/// outputs using a sum of products of input kernels and output covariance matrices.
///
/// The model assumes each output is a linear combination of latent functions:
/// y_d(x) = Σᵢ aᵢ_d × uᵢ(x)
///
/// Where:
/// - y_d(x) is output d at input x
/// - uᵢ(x) are independent latent functions, each with their own kernel kᵢ(x, x')
/// - aᵢ_d are mixing coefficients
///
/// The resulting multi-output kernel is:
/// k((x, d), (x', d')) = Σᵢ Bᵢ[d, d'] × kᵢ(x, x')
///
/// Where Bᵢ = aᵢ × aᵢᵀ is the coregionalization matrix for component i.
///
/// Why use LCM?
/// 1. **Flexible correlation structure**: Different components can capture different relationships
/// 2. **Interpretability**: Each component models a specific pattern shared across outputs
/// 3. **Efficiency**: Exploits Kronecker structure for faster computation
/// </para>
/// <para>
/// Applications:
/// - Multi-task learning (related prediction tasks)
/// - Sensor networks (multiple sensors measuring related quantities)
/// - Financial modeling (correlated asset returns)
/// - Geostatistics (co-kriging for multiple spatial variables)
/// </para>
/// </remarks>
public class LCMKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The input kernels for each component.
    /// </summary>
    private readonly IKernelFunction<T>[] _inputKernels;

    /// <summary>
    /// The coregionalization matrices for each component.
    /// </summary>
    private readonly double[][,] _coregMatrices;

    /// <summary>
    /// Number of components (latent functions).
    /// </summary>
    private readonly int _numComponents;

    /// <summary>
    /// Number of outputs.
    /// </summary>
    private readonly int _numOutputs;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new LCM kernel with specified kernels and coregionalization matrices.
    /// </summary>
    /// <param name="inputKernels">Input kernel for each component.</param>
    /// <param name="coregMatrices">Coregionalization matrix B_i for each component.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates an LCM kernel with explicit component specifications.
    ///
    /// Example for 2 outputs and 2 components:
    /// var rbf = new GaussianKernel&lt;double&gt;(lengthScale: 1.0);
    /// var matern = MaternKernel&lt;double&gt;.Matern52(lengthScale: 0.5);
    /// var kernels = new[] { rbf, matern };
    ///
    /// // Coregionalization matrices define output correlations
    /// var B1 = new double[,] { { 1.0, 0.8 }, { 0.8, 1.0 } };  // Strong correlation
    /// var B2 = new double[,] { { 0.5, 0.1 }, { 0.1, 0.5 } };  // Weak correlation
    /// var coregMatrices = new[] { B1, B2 };
    ///
    /// var lcm = new LCMKernel&lt;double&gt;(kernels, coregMatrices);
    /// </para>
    /// </remarks>
    public LCMKernel(IKernelFunction<T>[] inputKernels, double[][,] coregMatrices)
    {
        if (inputKernels is null) throw new ArgumentNullException(nameof(inputKernels));
        if (coregMatrices is null) throw new ArgumentNullException(nameof(coregMatrices));
        if (inputKernels.Length == 0)
            throw new ArgumentException("Must have at least one component.", nameof(inputKernels));
        if (inputKernels.Length != coregMatrices.Length)
            throw new ArgumentException("Number of kernels must match number of coregionalization matrices.");

        _numComponents = inputKernels.Length;

        // Verify all coregionalization matrices have the same dimensions
        int numOutputs = coregMatrices[0].GetLength(0);
        if (coregMatrices[0].GetLength(0) != coregMatrices[0].GetLength(1))
            throw new ArgumentException("Coregionalization matrices must be square.");

        foreach (var B in coregMatrices)
        {
            if (B.GetLength(0) != numOutputs || B.GetLength(1) != numOutputs)
                throw new ArgumentException("All coregionalization matrices must have the same dimensions.");
        }

        _numOutputs = numOutputs;
        _inputKernels = new IKernelFunction<T>[_numComponents];
        _coregMatrices = new double[_numComponents][,];

        for (int i = 0; i < _numComponents; i++)
        {
            _inputKernels[i] = inputKernels[i] ?? throw new ArgumentNullException($"inputKernels[{i}]");
            _coregMatrices[i] = (double[,])coregMatrices[i].Clone();
        }

        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Initializes a new LCM kernel with random coregionalization matrices.
    /// </summary>
    /// <param name="inputKernels">Input kernel for each component.</param>
    /// <param name="numOutputs">Number of outputs.</param>
    /// <param name="rank">Rank of each coregionalization matrix. If null, uses full rank.</param>
    /// <param name="seed">Random seed for initialization.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates an LCM kernel with randomly initialized correlations.
    ///
    /// The rank parameter controls the complexity:
    /// - Full rank: Can model any correlation pattern between outputs
    /// - Low rank: Assumes outputs share some underlying factors
    ///
    /// This is useful when you want to learn the output correlations from data.
    /// </para>
    /// </remarks>
    public LCMKernel(IKernelFunction<T>[] inputKernels, int numOutputs, int? rank = null, int? seed = null)
    {
        if (inputKernels is null) throw new ArgumentNullException(nameof(inputKernels));
        if (inputKernels.Length == 0)
            throw new ArgumentException("Must have at least one component.", nameof(inputKernels));
        if (numOutputs < 1)
            throw new ArgumentException("Must have at least one output.", nameof(numOutputs));

        _numComponents = inputKernels.Length;
        _numOutputs = numOutputs;
        int effectiveRank = rank ?? numOutputs;
        if (effectiveRank < 1 || effectiveRank > numOutputs)
            throw new ArgumentException("Rank must be between 1 and numOutputs.", nameof(rank));

        var rand = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        _inputKernels = new IKernelFunction<T>[_numComponents];
        _coregMatrices = new double[_numComponents][,];

        for (int i = 0; i < _numComponents; i++)
        {
            _inputKernels[i] = inputKernels[i] ?? throw new ArgumentNullException($"inputKernels[{i}]");

            // Generate random factor matrix L_i, then B_i = L_i × L_i^T
            var L = new double[numOutputs, effectiveRank];
            for (int d = 0; d < numOutputs; d++)
            {
                for (int r = 0; r < effectiveRank; r++)
                {
                    L[d, r] = (rand.NextDouble() - 0.5) * 2 / Math.Sqrt(effectiveRank);
                }
            }

            _coregMatrices[i] = new double[numOutputs, numOutputs];
            for (int d1 = 0; d1 < numOutputs; d1++)
            {
                for (int d2 = 0; d2 <= d1; d2++)
                {
                    double val = 0;
                    for (int r = 0; r < effectiveRank; r++)
                    {
                        val += L[d1, r] * L[d2, r];
                    }
                    // Add small diagonal for stability
                    if (d1 == d2) val += 0.1;
                    _coregMatrices[i][d1, d2] = val;
                    _coregMatrices[i][d2, d1] = val;
                }
            }
        }

        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the number of components (latent functions).
    /// </summary>
    public int NumComponents => _numComponents;

    /// <summary>
    /// Gets the number of outputs.
    /// </summary>
    public int NumOutputs => _numOutputs;

    /// <summary>
    /// Gets the input kernel for a specific component.
    /// </summary>
    /// <param name="componentIndex">Index of the component.</param>
    /// <returns>The input kernel for that component.</returns>
    public IKernelFunction<T> GetInputKernel(int componentIndex)
    {
        if (componentIndex < 0 || componentIndex >= _numComponents)
            throw new ArgumentOutOfRangeException(nameof(componentIndex));
        return _inputKernels[componentIndex];
    }

    /// <summary>
    /// Gets a copy of the coregionalization matrix for a specific component.
    /// </summary>
    /// <param name="componentIndex">Index of the component.</param>
    /// <returns>The coregionalization matrix B_i.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns the matrix that describes how outputs correlate
    /// under this component. B[d, d'] indicates how much outputs d and d' co-vary
    /// when driven by this latent function.
    /// </para>
    /// </remarks>
    public double[,] GetCoregMatrix(int componentIndex)
    {
        if (componentIndex < 0 || componentIndex >= _numComponents)
            throw new ArgumentOutOfRangeException(nameof(componentIndex));
        return (double[,])_coregMatrices[componentIndex].Clone();
    }

    /// <summary>
    /// Calculates the LCM kernel value between two output-input pairs.
    /// </summary>
    /// <param name="x1">First vector: [input..., outputIndex]</param>
    /// <param name="x2">Second vector: [input..., outputIndex]</param>
    /// <returns>The kernel value k((x1, d1), (x2, d2)).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The extended vector format includes the output index as the last element.
    ///
    /// The computation is:
    /// k((x, d), (x', d')) = Σᵢ B_i[d, d'] × k_i(x, x')
    ///
    /// This sums over all components, each contributing its coregionalization weight
    /// times its input kernel evaluation.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        // Expected format: [input_0, ..., input_{d-1}, outputIndex]
        int len1 = x1.Length;
        int len2 = x2.Length;

        if (len1 < 2 || len2 < 2)
            throw new ArgumentException("Extended vectors must have at least 2 elements.");

        int output1 = (int)Math.Round(_numOps.ToDouble(x1[len1 - 1]));
        int output2 = (int)Math.Round(_numOps.ToDouble(x2[len2 - 1]));

        if (output1 < 0 || output1 >= _numOutputs || output2 < 0 || output2 >= _numOutputs)
        {
            return _numOps.FromDouble(0);
        }

        // Extract input parts
        var input1 = ExtractInput(x1);
        var input2 = ExtractInput(x2);

        // Sum over components: k((x,d), (x',d')) = Σᵢ B_i[d,d'] × k_i(x, x')
        double result = 0;
        for (int i = 0; i < _numComponents; i++)
        {
            double kInput = _numOps.ToDouble(_inputKernels[i].Calculate(input1, input2));
            double bEntry = _coregMatrices[i][output1, output2];
            result += bEntry * kInput;
        }

        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Calculates the kernel value for explicit output indices.
    /// </summary>
    /// <param name="input1">First input vector.</param>
    /// <param name="input2">Second input vector.</param>
    /// <param name="output1">Index of first output.</param>
    /// <param name="output2">Index of second output.</param>
    /// <returns>The kernel value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Direct computation when you have separate input vectors and output indices.
    /// More efficient than creating extended vectors.
    /// </para>
    /// </remarks>
    public double CalculateForOutputs(Vector<T> input1, Vector<T> input2, int output1, int output2)
    {
        if (output1 < 0 || output1 >= _numOutputs)
            throw new ArgumentOutOfRangeException(nameof(output1));
        if (output2 < 0 || output2 >= _numOutputs)
            throw new ArgumentOutOfRangeException(nameof(output2));

        double result = 0;
        for (int i = 0; i < _numComponents; i++)
        {
            double kInput = _numOps.ToDouble(_inputKernels[i].Calculate(input1, input2));
            double bEntry = _coregMatrices[i][output1, output2];
            result += bEntry * kInput;
        }
        return result;
    }

    /// <summary>
    /// Extracts the input part from an extended vector.
    /// </summary>
    private Vector<T> ExtractInput(Vector<T> extended)
    {
        var result = new Vector<T>(extended.Length - 1);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = extended[i];
        }
        return result;
    }

    /// <summary>
    /// Creates an extended vector for a specific output.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="outputIndex">The output index.</param>
    /// <returns>Extended vector with output index appended.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to create the extended format for LCM kernel computation.
    /// </para>
    /// </remarks>
    public Vector<T> CreateExtendedVector(Vector<T> input, int outputIndex)
    {
        if (outputIndex < 0 || outputIndex >= _numOutputs)
            throw new ArgumentOutOfRangeException(nameof(outputIndex));

        var extended = new Vector<T>(input.Length + 1);
        for (int i = 0; i < input.Length; i++)
        {
            extended[i] = input[i];
        }
        extended[input.Length] = _numOps.FromDouble(outputIndex);
        return extended;
    }

    /// <summary>
    /// Gets the total correlation matrix across all components.
    /// </summary>
    /// <returns>Summed and normalized correlation matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns an overall view of how outputs correlate,
    /// combining information from all components. Useful for understanding
    /// the aggregate output relationships.
    /// </para>
    /// </remarks>
    public double[,] GetTotalCorrelation()
    {
        var total = new double[_numOutputs, _numOutputs];

        // Sum coregionalization matrices
        for (int i = 0; i < _numComponents; i++)
        {
            for (int d1 = 0; d1 < _numOutputs; d1++)
            {
                for (int d2 = 0; d2 < _numOutputs; d2++)
                {
                    total[d1, d2] += _coregMatrices[i][d1, d2];
                }
            }
        }

        // Convert to correlation
        var correlation = new double[_numOutputs, _numOutputs];
        for (int d1 = 0; d1 < _numOutputs; d1++)
        {
            for (int d2 = 0; d2 < _numOutputs; d2++)
            {
                double std1 = Math.Sqrt(Math.Max(total[d1, d1], 1e-10));
                double std2 = Math.Sqrt(Math.Max(total[d2, d2], 1e-10));
                correlation[d1, d2] = total[d1, d2] / (std1 * std2);
            }
        }

        return correlation;
    }

    /// <summary>
    /// Creates a simple LCM kernel with one component.
    /// </summary>
    /// <param name="inputKernel">The input kernel.</param>
    /// <param name="numOutputs">Number of outputs.</param>
    /// <param name="outputCorrelation">Correlation between all outputs.</param>
    /// <returns>An LCM kernel with uniform output correlation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates the simplest multi-output kernel:
    /// one latent function with uniform correlation between all outputs.
    ///
    /// This is known as the Intrinsic Coregionalization Model (ICM).
    /// </para>
    /// </remarks>
    public static LCMKernel<T> SingleComponent(
        IKernelFunction<T> inputKernel,
        int numOutputs,
        double outputCorrelation = 0.5)
    {
        if (outputCorrelation < -1.0 / (numOutputs - 1) || outputCorrelation > 1.0)
            throw new ArgumentException(
                $"Correlation must be between {-1.0 / (numOutputs - 1):F3} and 1.0.",
                nameof(outputCorrelation));

        var B = new double[numOutputs, numOutputs];
        for (int i = 0; i < numOutputs; i++)
        {
            for (int j = 0; j < numOutputs; j++)
            {
                B[i, j] = (i == j) ? 1.0 : outputCorrelation;
            }
        }

        return new LCMKernel<T>(
            new[] { inputKernel },
            new[] { B });
    }
}
