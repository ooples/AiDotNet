using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Deployment.Optimization.Quantization.Strategies;

/// <summary>
/// SpinQuant quantizer - uses learned rotation matrices to reduce outliers before quantization.
/// Applies orthogonal transformations to weight matrices to minimize quantization error.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SpinQuant "rotates" the data in a mathematical sense to spread
/// out outliers more evenly. This makes quantization more accurate because extreme values
/// cause the most problems during compression.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Analyze weight distribution to identify outliers</description></item>
/// <item><description>Learn rotation matrices (Cayley parameterization) to minimize outliers</description></item>
/// <item><description>Apply rotation: W' = R * W * R^T (orthogonal transformation)</description></item>
/// <item><description>Quantize the rotated weights (outliers are now spread out)</description></item>
/// <item><description>Store R for inference (apply inverse rotation to recover original space)</description></item>
/// </list>
///
/// <para><b>Key Features:</b></para>
/// <list type="bullet">
/// <item><description>Learned rotations via gradient descent on Cayley-parameterized matrices</description></item>
/// <item><description>Orthogonal transformations preserve weight magnitudes</description></item>
/// <item><description>Particularly effective for models with significant outliers</description></item>
/// <item><description>Can be combined with other quantization methods (GPTQ, AWQ)</description></item>
/// </list>
///
/// <para><b>Reference:</b> Liu et al., "SpinQuant: LLM quantization with learned rotations" (ICLR 2025)</para>
/// </remarks>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class SpinQuantQuantizer<T, TInput, TOutput> : IQuantizer<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly QuantizationConfiguration _config;
    private readonly Dictionary<string, T> _scaleFactors = new();
    private readonly Dictionary<string, int> _zeroPoints = new();
    private readonly Dictionary<string, Matrix<T>> _rotationMatrices = new();
    private readonly List<string> _calibrationWarnings = new();

    /// <summary>
    /// Gets whether the quantizer has been calibrated.
    /// </summary>
    public bool IsCalibrated { get; private set; }

    /// <summary>
    /// Gets any warnings generated during calibration.
    /// </summary>
    public IReadOnlyList<string> CalibrationWarnings => _calibrationWarnings;

    // SpinQuant hyperparameters
    private readonly int _numIterations;
    private readonly T _learningRate;
    private readonly int _blockSize;

    /// <inheritdoc/>
    public QuantizationMode Mode => _config.Mode;

    /// <inheritdoc/>
    public int BitWidth => _config.EffectiveBitWidth;

    /// <summary>
    /// Initializes a new instance of the SpinQuantQuantizer.
    /// </summary>
    /// <param name="config">Quantization configuration</param>
    /// <param name="numIterations">Number of optimization iterations for learning rotations</param>
    /// <param name="learningRate">Learning rate for rotation optimization</param>
    /// <param name="blockSize">Block size for block-diagonal rotations (0 for full rotation)</param>
    public SpinQuantQuantizer(
        QuantizationConfiguration? config = null,
        int numIterations = 100,
        double learningRate = 0.01,
        int blockSize = 0)
    {
        _config = config ?? new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            Strategy = QuantizationStrategy.SpinQuant,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = 128,
            UseSymmetricQuantization = true
        };

        if (_config.Strategy != QuantizationStrategy.SpinQuant)
        {
            throw new ArgumentException(
                $"SpinQuantQuantizer requires Strategy to be SpinQuant, but got {_config.Strategy}",
                nameof(config));
        }

        _numIterations = Math.Max(1, numIterations);
        _learningRate = NumOps.FromDouble(Math.Max(0.001, learningRate));
        _blockSize = Math.Max(0, blockSize);
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> Quantize(IFullModel<T, TInput, TOutput> model, QuantizationConfiguration config)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));

        var parameters = model.GetParameters();

        // Learn optimal rotation matrix using the provided config
        var rotation = LearnRotationMatrix(parameters, config);
        _rotationMatrices["global"] = rotation;

        // Apply rotation to weights: W' = R * W
        var rotatedParams = ApplyRotation(parameters, rotation);

        // Quantize the rotated weights
        var quantizedRotated = QuantizeSymmetric(rotatedParams, config);

        // Apply inverse rotation to recover original space: W_final = R^T * Q(W')
        // For orthogonal rotation matrices, R^-1 = R^T (transpose)
        var rotationTranspose = rotation.Transpose();
        var quantizedOriginalSpace = ApplyRotation(quantizedRotated, rotationTranspose);

        return model.WithParameters(quantizedOriginalSpace);
    }

    /// <inheritdoc/>
    public void Calibrate(IFullModel<T, TInput, TOutput> model, IEnumerable<TInput> calibrationData)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (calibrationData == null) throw new ArgumentNullException(nameof(calibrationData));

        var dataList = calibrationData.ToList();
        if (dataList.Count == 0)
        {
            throw new ArgumentException("Calibration data cannot be empty", nameof(calibrationData));
        }

        // Run calibration samples through model to gather activation statistics
        // SpinQuant uses these to inform the rotation matrix learning
        T activationScale = NumOps.One;
        int sampleCount = 0;
        foreach (var sample in dataList.Take(Math.Min(100, dataList.Count)))
        {
            try
            {
                var output = model.Predict(sample);
                // Collect output magnitude for activation-aware scaling
                if (output is Vector<T> vec)
                {
                    for (int i = 0; i < vec.Length; i++)
                    {
                        T absVal = NumOps.Abs(vec[i]);
                        if (NumOps.Compare(absVal, activationScale) > 0)
                        {
                            activationScale = absVal;
                        }
                    }
                }
                sampleCount++;
            }
            catch (Exception ex)
            {
                // Track calibration failures for diagnostics
                _calibrationWarnings.Add($"Calibration sample failed: {ex.Message}");
            }
        }

        // Store activation scale for use in quantization
        if (sampleCount > 0)
        {
            _scaleFactors["activation"] = activationScale;
        }

        // Compute scale factors from model parameters
        var parameters = model.GetParameters();
        ComputeScaleFactors(parameters, _config);

        IsCalibrated = true;
    }

    /// <inheritdoc/>
    public double GetScaleFactor(string layerName)
    {
        if (_scaleFactors.TryGetValue(layerName, out var scale))
            return NumOps.ToDouble(scale);
        if (_scaleFactors.TryGetValue("global", out var globalScale))
            return NumOps.ToDouble(globalScale);
        return 1.0;
    }

    /// <inheritdoc/>
    public int GetZeroPoint(string layerName)
    {
        return 0; // SpinQuant uses symmetric quantization
    }

    /// <summary>
    /// Learns optimal rotation matrix to minimize quantization error.
    /// Uses Cayley parameterization for gradient-based optimization on orthogonal matrices.
    /// </summary>
    private Matrix<T> LearnRotationMatrix(Vector<T> weights, QuantizationConfiguration config)
    {
        int n = weights.Length;

        // Determine rotation matrix size (use block-diagonal for large weights)
        int rotationSize = _blockSize > 0 ? Math.Min(_blockSize, n) : Math.Min(n, 256);

        // Initialize as identity matrix
        var rotation = Matrix<T>.CreateIdentity(rotationSize);

        // Initialize Cayley parameter (skew-symmetric matrix)
        var cayleyParam = new Matrix<T>(rotationSize, rotationSize);

        // Optimization loop
        for (int iter = 0; iter < _numIterations; iter++)
        {
            // Compute gradient of quantization error with respect to Cayley parameter
            var gradient = ComputeRotationGradient(weights, rotation, rotationSize, config);

            // Update Cayley parameter - only update upper triangular (i < j)
            // then enforce skew-symmetry after all updates to avoid overwriting
            T gradientThreshold = NumOps.FromDouble(1e-10);
            for (int i = 0; i < rotationSize; i++)
            {
                for (int j = i + 1; j < rotationSize; j++)
                {
                    // Skip negligible gradients to reduce floating point operations
                    T grad = gradient[i, j];
                    if (NumOps.Compare(NumOps.Abs(grad), gradientThreshold) >= 0)
                    {
                        cayleyParam[i, j] = NumOps.Subtract(cayleyParam[i, j], NumOps.Multiply(_learningRate, grad));
                    }
                }
            }

            // Enforce skew-symmetry after all updates: A[j,i] = -A[i,j]
            for (int i = 0; i < rotationSize; i++)
            {
                cayleyParam[i, i] = NumOps.Zero; // Diagonal must be zero
                for (int j = i + 1; j < rotationSize; j++)
                {
                    cayleyParam[j, i] = NumOps.Negate(cayleyParam[i, j]);
                }
            }

            // Convert Cayley parameter to rotation matrix: R = (I + A)^-1 * (I - A)
            rotation = CayleyToRotation(cayleyParam);
        }

        return rotation;
    }

    /// <summary>
    /// Computes gradient of quantization error with respect to rotation parameters.
    /// </summary>
    private Matrix<T> ComputeRotationGradient(Vector<T> weights, Matrix<T> currentRotation, int size, QuantizationConfiguration config)
    {
        var gradient = new Matrix<T>(size, size);

        // Extract a block of weights for rotation
        int blockCount = Math.Min(size, weights.Length);
        var weightBlock = new Vector<T>(blockCount);
        for (int i = 0; i < blockCount; i++)
        {
            weightBlock[i] = weights[i];
        }

        // Compute rotated weights
        var rotatedBlock = new Vector<T>(blockCount);
        for (int i = 0; i < blockCount; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < blockCount; j++)
            {
                int ri = Math.Min(i, size - 1);
                int rj = Math.Min(j, size - 1);
                sum = NumOps.Add(sum, NumOps.Multiply(currentRotation[ri, rj], weightBlock[j]));
            }
            rotatedBlock[i] = sum;
        }

        // Compute quantization error gradient
        // The gradient points in the direction that reduces quantization error
        T qMax = NumOps.FromDouble((1L << (config.EffectiveBitWidth - 1)) - 1);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                // Finite difference approximation of gradient with adaptive epsilon
                // Scale epsilon with rotation value magnitude for numerical stability
                T absRotation = NumOps.Abs(currentRotation[i, j]);
                T eps = NumOps.FromDouble(Math.Max(1e-5, NumOps.ToDouble(absRotation) * 1e-6));

                // Perturb rotation - clone the matrix
                var perturbedRotation = currentRotation.Clone();
                perturbedRotation[i, j] = NumOps.Add(perturbedRotation[i, j], eps);

                // Compute error with perturbation
                T errorPlus = ComputeQuantizationError(weightBlock, perturbedRotation, qMax);

                perturbedRotation[i, j] = NumOps.Subtract(currentRotation[i, j], eps);
                T errorMinus = ComputeQuantizationError(weightBlock, perturbedRotation, qMax);

                T two = NumOps.FromDouble(2.0);
                gradient[i, j] = NumOps.Divide(NumOps.Subtract(errorPlus, errorMinus), NumOps.Multiply(two, eps));
            }
        }

        return gradient;
    }

    /// <summary>
    /// Computes quantization error for a weight block with given rotation.
    /// </summary>
    private T ComputeQuantizationError(Vector<T> weights, Matrix<T> rotation, T qMax)
    {
        int n = weights.Length;
        int size = rotation.Rows;
        T totalError = NumOps.Zero;

        // Compute max abs for scale
        T maxAbs = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T rotated = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                int ri = Math.Min(i, size - 1);
                int rj = Math.Min(j, size - 1);
                rotated = NumOps.Add(rotated, NumOps.Multiply(rotation[ri, rj], weights[j]));
            }
            T absRotated = NumOps.Abs(rotated);
            if (NumOps.Compare(absRotated, maxAbs) > 0)
            {
                maxAbs = absRotated;
            }
        }

        T scale = NumOps.Compare(maxAbs, NumOps.Zero) > 0
            ? NumOps.Divide(maxAbs, qMax)
            : NumOps.One;

        // Compute quantization error
        for (int i = 0; i < n; i++)
        {
            T rotated = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                int ri = Math.Min(i, size - 1);
                int rj = Math.Min(j, size - 1);
                rotated = NumOps.Add(rotated, NumOps.Multiply(rotation[ri, rj], weights[j]));
            }

            T quantized = NumOps.Round(NumOps.Divide(rotated, scale));
            T negQMax = NumOps.Negate(qMax);

            // Clamp to quantization range
            if (NumOps.Compare(quantized, negQMax) < 0)
                quantized = negQMax;
            else if (NumOps.Compare(quantized, qMax) > 0)
                quantized = qMax;

            T dequantized = NumOps.Multiply(quantized, scale);

            T error = NumOps.Subtract(rotated, dequantized);
            totalError = NumOps.Add(totalError, NumOps.Multiply(error, error));
        }

        return totalError;
    }

    /// <summary>
    /// Converts Cayley parameter (skew-symmetric matrix) to rotation matrix.
    /// R = (I + A)^-1 * (I - A) where A is skew-symmetric
    /// </summary>
    private Matrix<T> CayleyToRotation(Matrix<T> cayleyParam)
    {
        int n = cayleyParam.Rows;

        // Compute (I + A) and (I - A)
        var iPlusA = new Matrix<T>(n, n);
        var iMinusA = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T identity = i == j ? NumOps.One : NumOps.Zero;
                iPlusA[i, j] = NumOps.Add(identity, cayleyParam[i, j]);
                iMinusA[i, j] = NumOps.Subtract(identity, cayleyParam[i, j]);
            }
        }

        // Compute inverse of (I + A) using Gaussian elimination
        var iPlusAInv = InvertMatrix(iPlusA);

        // Multiply: R = (I + A)^-1 * (I - A)
        var rotation = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(iPlusAInv[i, k], iMinusA[k, j]));
                }
                rotation[i, j] = sum;
            }
        }

        return rotation;
    }

    /// <summary>
    /// Inverts a matrix using Gaussian elimination with partial pivoting.
    /// </summary>
    private Matrix<T> InvertMatrix(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var augmented = new Matrix<T>(n, 2 * n);

        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = matrix[i, j];
            }
            augmented[i, n + i] = NumOps.One;
        }

        // Gaussian elimination
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (NumOps.Compare(NumOps.Abs(augmented[row, col]), NumOps.Abs(augmented[maxRow, col])) > 0)
                {
                    maxRow = row;
                }
            }

            // Swap rows
            for (int j = 0; j < 2 * n; j++)
            {
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
            }

            // Scale pivot row
            T pivot = augmented[col, col];
            T pivotThreshold = NumOps.FromDouble(1e-10);
            if (NumOps.Compare(NumOps.Abs(pivot), pivotThreshold) < 0)
            {
                // Matrix is singular - add warning and return identity as fallback
                _calibrationWarnings.Add($"Warning: Singular matrix detected at column {col} during Cayley transform, falling back to identity rotation.");
                return Matrix<T>.CreateIdentity(n);
            }

            for (int j = 0; j < 2 * n; j++)
            {
                augmented[col, j] = NumOps.Divide(augmented[col, j], pivot);
            }

            // Eliminate column
            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    T factor = augmented[row, col];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        augmented[row, j] = NumOps.Subtract(augmented[row, j], NumOps.Multiply(factor, augmented[col, j]));
                    }
                }
            }
        }

        // Extract inverse
        var inverse = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inverse[i, j] = augmented[i, n + j];
            }
        }

        return inverse;
    }

    /// <summary>
    /// Applies rotation matrix to weight vector.
    /// </summary>
    private Vector<T> ApplyRotation(Vector<T> weights, Matrix<T> rotation)
    {
        int n = weights.Length;
        int size = rotation.Rows;
        var result = new Vector<T>(n);

        // Apply rotation in blocks
        for (int blockStart = 0; blockStart < n; blockStart += size)
        {
            int blockEnd = Math.Min(blockStart + size, n);
            int blockLen = blockEnd - blockStart;

            for (int i = 0; i < blockLen; i++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < blockLen; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(rotation[i, j], weights[blockStart + j]));
                }
                result[blockStart + i] = sum;
            }
        }

        return result;
    }

    /// <summary>
    /// Computes scale factors from parameter statistics.
    /// </summary>
    private void ComputeScaleFactors(Vector<T> parameters, QuantizationConfiguration config)
    {
        T maxAbs = NumOps.Zero;
        for (int i = 0; i < parameters.Length; i++)
        {
            T absVal = NumOps.Abs(parameters[i]);
            if (NumOps.Compare(absVal, maxAbs) > 0)
            {
                maxAbs = absVal;
            }
        }

        T qMax = NumOps.FromDouble((1L << (config.EffectiveBitWidth - 1)) - 1);
        _scaleFactors["global"] = NumOps.Compare(maxAbs, NumOps.Zero) > 0
            ? NumOps.Divide(maxAbs, qMax)
            : NumOps.One;
    }

    /// <summary>
    /// Performs symmetric quantization on rotated parameters.
    /// </summary>
    private Vector<T> QuantizeSymmetric(Vector<T> parameters, QuantizationConfiguration config)
    {
        int n = parameters.Length;
        var result = new Vector<T>(n);

        T qMax = NumOps.FromDouble((1L << (config.EffectiveBitWidth - 1)) - 1);
        T qMin = NumOps.Negate(qMax);

        // Compute scale from rotated parameters
        T maxAbs = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T absVal = NumOps.Abs(parameters[i]);
            if (NumOps.Compare(absVal, maxAbs) > 0)
            {
                maxAbs = absVal;
            }
        }

        T scale = NumOps.Compare(maxAbs, NumOps.Zero) > 0
            ? NumOps.Divide(maxAbs, qMax)
            : NumOps.One;
        _scaleFactors["global"] = scale;

        // Quantize
        for (int i = 0; i < n; i++)
        {
            T value = parameters[i];
            T quantized = NumOps.Round(NumOps.Divide(value, scale));

            // Clamp to quantization range
            if (NumOps.Compare(quantized, qMin) < 0)
                quantized = qMin;
            else if (NumOps.Compare(quantized, qMax) > 0)
                quantized = qMax;

            T dequantized = NumOps.Multiply(quantized, scale);
            result[i] = dequantized;
        }

        return result;
    }
}
