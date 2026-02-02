using AiDotNet.Helpers;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements an 8-bit quantized Adam optimizer that reduces memory usage by storing optimizer states in 8-bit format.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// 8-bit Adam provides the same optimization algorithm as standard Adam but uses quantized 8-bit representations
/// for storing the first moment (m) and second moment (v) estimates. This reduces memory usage by approximately
/// 4x for optimizer states, which is particularly beneficial when training large models.
/// </para>
/// <para><b>For Beginners:</b> When training a neural network, the optimizer needs to remember information about
/// past gradients. Standard Adam stores two numbers per parameter (momentum and variance), which can use a lot of
/// memory for large models. 8-bit Adam compresses these numbers, similar to how images are compressed, reducing
/// memory usage while maintaining training quality.
/// </para>
/// <para><b>How It Works:</b>
/// <list type="bullet">
/// <item>Optimizer states are divided into blocks (default 2048 elements each)</item>
/// <item>Each block has its own scaling factor for accurate quantization</item>
/// <item>States are dequantized before computing updates, then requantized after</item>
/// <item>The actual parameter updates use full precision for accuracy</item>
/// </list>
/// </para>
/// <para><b>When to Use:</b>
/// <list type="bullet">
/// <item>Training large models where optimizer memory is a bottleneck</item>
/// <item>GPU training with limited VRAM</item>
/// <item>Distributed training where memory per GPU is constrained</item>
/// </list>
/// </para>
/// </remarks>
public class Adam8BitOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the 8-bit Adam optimizer.
    /// </summary>
    private Adam8BitOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Quantized first moment vector (moving average of gradients).
    /// </summary>
    private byte[]? _mQuantized;

    /// <summary>
    /// Quantized second moment vector (moving average of squared gradients).
    /// </summary>
    private byte[]? _vQuantized;

    /// <summary>
    /// Scaling factors for first moment quantization blocks.
    /// </summary>
    private double[]? _mScales;

    /// <summary>
    /// Scaling factors for second moment quantization blocks.
    /// </summary>
    private double[]? _vScales;

    /// <summary>
    /// Full-precision first moment vector (used when CompressBothMoments is false).
    /// </summary>
    private Vector<T>? _mFullPrecision;

    /// <summary>
    /// The current time step (iteration count).
    /// </summary>
    private int _t;

    /// <summary>
    /// The current value of beta1 (exponential decay rate for first moment estimates).
    /// </summary>
    private T _currentBeta1;

    /// <summary>
    /// The current value of beta2 (exponential decay rate for second moment estimates).
    /// </summary>
    private T _currentBeta2;

    /// <summary>
    /// Number of quantization blocks.
    /// </summary>
    private int _numBlocks;

    /// <summary>
    /// Length of the parameter vector.
    /// </summary>
    private int _parameterLength;

    /// <summary>
    /// Random number generator for stochastic rounding.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the Adam8BitOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the 8-bit Adam optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the 8-bit Adam optimizer with its initial configuration.
    /// The optimizer will use quantized storage for momentum and variance estimates, reducing memory usage.
    /// </para>
    /// </remarks>
    public Adam8BitOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        Adam8BitOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _t = 0;
        _options = options ?? new();
        _currentBeta1 = NumOps.Zero;
        _currentBeta2 = NumOps.Zero;
        _random = RandomHelper.CreateSeededRandom(42);

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the Adam optimizer.
    /// </summary>
    protected override void InitializeAdaptiveParameters()
    {
        _currentBeta1 = NumOps.FromDouble(_options.Beta1);
        _currentBeta2 = NumOps.FromDouble(_options.Beta2);
    }

    /// <summary>
    /// Initializes the quantized optimizer state buffers.
    /// </summary>
    /// <param name="length">The number of parameters to optimize.</param>
    private void InitializeQuantizedState(int length)
    {
        _parameterLength = length;
        _numBlocks = (length + _options.BlockSize - 1) / _options.BlockSize;

        // Initialize quantized second moment (always quantized, unsigned so 0 represents 0)
        _vQuantized = new byte[length];
        _vScales = new double[_numBlocks];

        // Initialize first moment (quantized or full precision based on options)
        if (_options.CompressBothMoments)
        {
            _mQuantized = new byte[length];
            _mScales = new double[_numBlocks];
            _mFullPrecision = null;

            // For signed quantization, 128 represents 0 (since we map [-127,127] to [1,255] with 128=0)
            for (int i = 0; i < length; i++)
            {
                _mQuantized[i] = 128;
            }
        }
        else
        {
            _mQuantized = null;
            _mScales = null;
            _mFullPrecision = new Vector<T>(length);
        }

        // Initialize scales (scale of 1.0 works with the zero-initialized state)
        for (int b = 0; b < _numBlocks; b++)
        {
            if (_mScales != null) _mScales[b] = 1.0;
            _vScales![b] = 1.0;
        }
    }

    /// <summary>
    /// Quantizes a full-precision vector to 8-bit representation.
    /// </summary>
    /// <param name="values">The full-precision values to quantize.</param>
    /// <param name="quantized">The output quantized byte array.</param>
    /// <param name="scales">The output scaling factors per block.</param>
    /// <param name="isSigned">Whether to use signed quantization (for m) or unsigned (for v).</param>
    private void Quantize(Vector<T> values, byte[] quantized, double[] scales, bool isSigned)
    {
        for (int b = 0; b < _numBlocks; b++)
        {
            int blockStart = b * _options.BlockSize;
            int blockEnd = Math.Min(blockStart + _options.BlockSize, _parameterLength);

            // Find the scale for this block
            double maxAbs = 0;
            if (_options.QuantizationPercentile >= 100)
            {
                // Use absolute maximum
                for (int i = blockStart; i < blockEnd; i++)
                {
                    double val = Math.Abs(NumOps.ToDouble(values[i]));
                    if (val > maxAbs) maxAbs = val;
                }
            }
            else
            {
                // Use percentile-based scale (collect values, sort, take percentile)
                var absValues = new List<double>(blockEnd - blockStart);
                for (int i = blockStart; i < blockEnd; i++)
                {
                    absValues.Add(Math.Abs(NumOps.ToDouble(values[i])));
                }
                absValues.Sort();
                int percentileIdx = (int)((absValues.Count - 1) * _options.QuantizationPercentile / 100.0);
                maxAbs = absValues[percentileIdx];
            }

            // Compute scale (with small epsilon to avoid division by zero)
            double scale = maxAbs / (isSigned ? 127.0 : 255.0);
            if (scale < 1e-10) scale = 1e-10;
            scales[b] = scale;

            // Quantize values in this block
            for (int i = blockStart; i < blockEnd; i++)
            {
                double val = NumOps.ToDouble(values[i]);
                double scaled = val / scale;

                // Apply rounding
                int quantizedVal;
                if (_options.UseStochasticRounding)
                {
                    double floor = Math.Floor(scaled);
                    double frac = scaled - floor;
                    quantizedVal = (int)(floor + (_random.NextDouble() < frac ? 1 : 0));
                }
                else
                {
                    quantizedVal = (int)Math.Round(scaled);
                }

                // Clamp to valid range
                if (isSigned)
                {
                    quantizedVal = MathHelper.Clamp(quantizedVal, -127, 127);
                    quantized[i] = (byte)(quantizedVal + 128); // Map [-127, 127] to [1, 255], 0 maps to 128
                }
                else
                {
                    quantizedVal = MathHelper.Clamp(quantizedVal, 0, 255);
                    quantized[i] = (byte)quantizedVal;
                }
            }
        }
    }

    /// <summary>
    /// Dequantizes an 8-bit representation back to full precision.
    /// </summary>
    /// <param name="quantized">The quantized byte array.</param>
    /// <param name="scales">The scaling factors per block.</param>
    /// <param name="isSigned">Whether the quantization used signed format.</param>
    /// <returns>The dequantized full-precision vector.</returns>
    private Vector<T> Dequantize(byte[] quantized, double[] scales, bool isSigned)
    {
        var result = new Vector<T>(_parameterLength);

        for (int b = 0; b < _numBlocks; b++)
        {
            int blockStart = b * _options.BlockSize;
            int blockEnd = Math.Min(blockStart + _options.BlockSize, _parameterLength);
            double scale = scales[b];

            for (int i = blockStart; i < blockEnd; i++)
            {
                double quantizedVal;
                if (isSigned)
                {
                    quantizedVal = (int)quantized[i] - 128; // Map [1, 255] back to [-127, 127]
                }
                else
                {
                    quantizedVal = quantized[i];
                }

                result[i] = NumOps.FromDouble(quantizedVal * scale);
            }
        }

        return result;
    }

    /// <summary>
    /// Performs the optimization process using the 8-bit Adam algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = currentSolution.GetParameters();

        InitializeQuantizedState(parameters.Length);
        _t = 0;

        InitializeAdaptiveParameters();

        var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);

            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                _t++;
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                var newSolution = UpdateSolution(currentSolution, gradient);
                currentSolution = newSolution;
            }

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(
                NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)),
                NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveLearningRate)
        {
            CurrentLearningRate = MathHelper.Max(NumOps.FromDouble(_options.MinLearningRate),
                MathHelper.Min(NumOps.FromDouble(_options.MaxLearningRate), CurrentLearningRate));
        }

        if (_options.UseAdaptiveBetas)
        {
            _currentBeta1 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta1),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta1), _currentBeta1));
            _currentBeta2 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta2),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta2), _currentBeta2));
        }
    }

    /// <summary>
    /// Updates the current solution using the 8-bit Adam update rule.
    /// </summary>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();

        if (_mQuantized == null && _mFullPrecision == null)
        {
            InitializeQuantizedState(parameters.Length);
        }

        // Dequantize current moment estimates
        Vector<T> m;
        if (_options.CompressBothMoments)
        {
            m = Dequantize(_mQuantized!, _mScales!, isSigned: true);
        }
        else
        {
            m = _mFullPrecision!;
        }
        var v = Dequantize(_vQuantized!, _vScales!, isSigned: false);

        // Compute Adam update using full precision
        T beta1 = _currentBeta1;
        T beta2 = _currentBeta2;
        T oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        T oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        T biasCorrection1 = NumOps.Subtract(NumOps.One, NumOps.Power(beta1, NumOps.FromDouble(_t)));
        T biasCorrection2 = NumOps.Subtract(NumOps.One, NumOps.Power(beta2, NumOps.FromDouble(_t)));
        T epsilon = NumOps.FromDouble(_options.Epsilon);

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(v, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Re-quantize the updated moments
        if (_options.CompressBothMoments)
        {
            Quantize(m, _mQuantized!, _mScales!, isSigned: true);
        }
        else
        {
            _mFullPrecision = m;
        }
        Quantize(v, _vQuantized!, _vScales!, isSigned: false);

        // Compute bias-corrected first moment: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(m, biasCorrection1);

        // Compute bias-corrected second moment: vHat = v / (1 - beta2^t)
        var vHat = (Vector<T>)Engine.Divide(v, biasCorrection2);

        // Compute update: update = learningRate * mHat / (sqrt(vHat) + epsilon)
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var updateDiv = (Vector<T>)Engine.Divide(mHat, denominator);
        var update = (Vector<T>)Engine.Multiply(updateDiv, CurrentLearningRate);

        // Apply update: parameters = parameters - update
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, update);

        return currentSolution.WithParameters(updatedParams);
    }

    /// <summary>
    /// Updates a vector of parameters using the 8-bit Adam optimization algorithm.
    /// </summary>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_mQuantized == null && _mFullPrecision == null || _parameterLength != parameters.Length)
        {
            InitializeQuantizedState(parameters.Length);
        }

        _t++;

        // Dequantize current moment estimates
        Vector<T> m;
        if (_options.CompressBothMoments)
        {
            m = Dequantize(_mQuantized!, _mScales!, isSigned: true);
        }
        else
        {
            m = _mFullPrecision!;
        }
        var v = Dequantize(_vQuantized!, _vScales!, isSigned: false);

        // Compute Adam update using full precision
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(v, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Re-quantize the updated moments
        if (_options.CompressBothMoments)
        {
            Quantize(m, _mQuantized!, _mScales!, isSigned: true);
        }
        else
        {
            _mFullPrecision = m;
        }
        Quantize(v, _vQuantized!, _vScales!, isSigned: false);

        // Compute bias-corrected moments
        var mHat = (Vector<T>)Engine.Divide(m, biasCorrection1);
        var vHat = (Vector<T>)Engine.Divide(v, biasCorrection2);

        // Compute update
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var update = (Vector<T>)Engine.Divide(mHat, denominator);
        var scaledUpdate = (Vector<T>)Engine.Multiply(update, CurrentLearningRate);

        // Apply update
        return (Vector<T>)Engine.Subtract(parameters, scaledUpdate);
    }

    /// <summary>
    /// Updates a matrix of parameters using the 8-bit Adam optimization algorithm.
    /// </summary>
    public override Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        int totalSize = parameters.Rows * parameters.Columns;

        // Flatten matrix to vector
        var paramVec = new Vector<T>(totalSize);
        var gradVec = new Vector<T>(totalSize);
        int idx = 0;
        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                paramVec[idx] = parameters[i, j];
                gradVec[idx] = gradient[i, j];
                idx++;
            }
        }

        // Update using vector method
        var updatedVec = UpdateParameters(paramVec, gradVec);

        // Unflatten vector back to matrix
        var updatedMatrix = new Matrix<T>(parameters.Rows, parameters.Columns);
        idx = 0;
        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                updatedMatrix[i, j] = updatedVec[idx];
                idx++;
            }
        }

        return updatedMatrix;
    }

    /// <summary>
    /// Resets the optimizer's internal state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _mQuantized = null;
        _vQuantized = null;
        _mScales = null;
        _vScales = null;
        _mFullPrecision = null;
        _t = 0;
        _parameterLength = 0;
        _numBlocks = 0;
    }

    /// <summary>
    /// Updates the optimizer's options.
    /// </summary>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is Adam8BitOptimizerOptions<T, TInput, TOutput> adamOptions)
        {
            _options = adamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected Adam8BitOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimizer options.
    /// </summary>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Gets the memory usage statistics for this optimizer.
    /// </summary>
    /// <returns>A dictionary containing memory usage information.</returns>
    public Dictionary<string, long> GetMemoryUsage()
    {
        var stats = new Dictionary<string, long>();

        // Quantized state memory
        long quantizedStateMemory = 0;
        if (_mQuantized != null) quantizedStateMemory += _mQuantized.Length;
        if (_vQuantized != null) quantizedStateMemory += _vQuantized.Length;

        // Scaling factors memory (8 bytes per double)
        long scalesMemory = 0;
        if (_mScales != null) scalesMemory += _mScales.Length * 8;
        if (_vScales != null) scalesMemory += _vScales.Length * 8;

        // Full precision state memory (if used)
        long fullPrecisionMemory = 0;
        if (_mFullPrecision != null)
        {
            // Assuming T is float (4 bytes) or double (8 bytes)
            int typeSize = typeof(T) == typeof(float) ? 4 : 8;
            fullPrecisionMemory += _mFullPrecision.Length * typeSize;
        }

        stats["QuantizedStateBytes"] = quantizedStateMemory;
        stats["ScalingFactorBytes"] = scalesMemory;
        stats["FullPrecisionStateBytes"] = fullPrecisionMemory;
        stats["TotalBytes"] = quantizedStateMemory + scalesMemory + fullPrecisionMemory;

        // Calculate savings compared to standard Adam
        int typeSize2 = typeof(T) == typeof(float) ? 4 : 8;
        long standardAdamMemory = _parameterLength * 2 * typeSize2; // m and v at full precision
        stats["StandardAdamBytes"] = standardAdamMemory;
        stats["MemorySavingsBytes"] = standardAdamMemory - stats["TotalBytes"];

        return stats;
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize options
            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            // Serialize 8-bit Adam-specific state
            writer.Write(_t);
            writer.Write(_parameterLength);
            writer.Write(_numBlocks);

            // Serialize quantized first moment (if used)
            writer.Write(_options.CompressBothMoments);
            if (_options.CompressBothMoments && _mQuantized != null)
            {
                writer.Write(_mQuantized.Length);
                writer.Write(_mQuantized);
                foreach (var scale in _mScales!)
                {
                    writer.Write(scale);
                }
            }
            else if (_mFullPrecision != null)
            {
                writer.Write(_mFullPrecision.Length);
                foreach (var value in _mFullPrecision)
                {
                    writer.Write(Convert.ToDouble(value));
                }
            }

            // Serialize quantized second moment
            if (_vQuantized != null)
            {
                writer.Write(_vQuantized.Length);
                writer.Write(_vQuantized);
                foreach (var scale in _vScales!)
                {
                    writer.Write(scale);
                }
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the optimizer's state from a byte array.
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize options
            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<Adam8BitOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize state
            _t = reader.ReadInt32();
            _parameterLength = reader.ReadInt32();
            _numBlocks = reader.ReadInt32();

            // Deserialize first moment
            bool compressBothMoments = reader.ReadBoolean();
            if (compressBothMoments)
            {
                int mLength = reader.ReadInt32();
                _mQuantized = reader.ReadBytes(mLength);
                _mScales = new double[_numBlocks];
                for (int i = 0; i < _numBlocks; i++)
                {
                    _mScales[i] = reader.ReadDouble();
                }
            }
            else
            {
                int mLength = reader.ReadInt32();
                _mFullPrecision = new Vector<T>(mLength);
                for (int i = 0; i < mLength; i++)
                {
                    _mFullPrecision[i] = NumOps.FromDouble(reader.ReadDouble());
                }
            }

            // Deserialize second moment
            int vLength = reader.ReadInt32();
            _vQuantized = reader.ReadBytes(vLength);
            _vScales = new double[_numBlocks];
            for (int i = 0; i < _numBlocks; i++)
            {
                _vScales[i] = reader.ReadDouble();
            }

            InitializeAdaptiveParameters();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Adam8Bit_{_options.InitialLearningRate}_{_options.MaxIterations}_{_options.BlockSize}";
    }
}
