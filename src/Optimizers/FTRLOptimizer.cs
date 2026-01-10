using Newtonsoft.Json;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a Follow The Regularized Leader (FTRL) optimizer for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// The FTRL optimizer is an online learning algorithm that adapts regularization in a per-coordinate fashion.
/// It's particularly effective for sparse datasets and is widely used in click-through rate (CTR) prediction
/// and other online learning scenarios.
/// </para>
/// <para><b>For Beginners:</b> FTRL is an advanced optimization technique that's good at handling large-scale,
/// sparse data. It's often used in online advertising and recommendation systems.
/// 
/// Think of FTRL like a smart learning system that:
/// - Adjusts its learning speed for each feature independently
/// - Can handle situations where most features are zero (sparse data)
/// - Is good at balancing between finding a good solution and not overfitting
/// 
/// For example, in an online advertising system, FTRL might:
/// - Quickly learn which ad categories are important for a user
/// - Ignore or learn slowly from features that rarely appear
/// - Automatically adjust how much it learns from each new piece of data
/// 
/// This makes FTRL particularly good for systems that need to learn and predict in real-time with lots of data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class FTRLOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the FTRL algorithm.
    /// </summary>
    private FTRLOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Auxiliary vector used in the FTRL update rule.
    /// </summary>
    private Vector<T>? _z;

    /// <summary>
    /// Vector of accumulated squared gradients.
    /// </summary>
    private Vector<T>? _n;

    /// <summary>
    /// The current time step or iteration count.
    /// </summary>
    private int _t;

    /// <summary>
    /// Stores the pre-update parameters for approximate reverse updates.
    /// </summary>
    private Vector<T>? _previousParameters;

    /// <summary>
    /// GPU buffer for z state (accumulated sum state).
    /// </summary>
    private IGpuBuffer? _gpuZ;

    /// <summary>
    /// GPU buffer for n state (accumulated squared gradient state).
    /// </summary>
    private IGpuBuffer? _gpuN;

    /// <summary>
    /// Initializes a new instance of the FTRLOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the FTRL optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings if you're unsure.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the FTRL algorithm.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    public FTRLOptimizer(
        IFullModel<T, TInput, TOutput> model,
        FTRLOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new FTRLOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the FTRL algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial learning rate and resets the time step.
    /// The learning rate determines how big the steps are when the algorithm is searching for the best solution.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();

        CurrentLearningRate = NumOps.FromDouble(_options.Alpha);
        _t = 0;
    }

    /// <summary>
    /// Performs the main optimization process using the FTRL algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the FTRL algorithm. It starts with a random solution
    /// and iteratively improves it. In each iteration, it:
    /// 1. Calculates how to improve the current solution (the gradient)
    /// 2. Updates the solution based on this calculation
    /// 3. Checks if this new solution is the best one found so far
    /// 4. Decides whether to stop or continue improving
    ///
    /// It's like a climber trying to find the highest peak, taking steps based on the slope they feel under their feet.
    /// </para>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for efficient batch processing.
    /// It creates a batcher using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.CreateBatcher"/>
    /// and notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var parameters = currentSolution.GetParameters();
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _z = new Vector<T>(parameters.Length);
        _n = new Vector<T>(parameters.Length);
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                _t++;
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                currentSolution = UpdateSolution(currentSolution, gradient);
            }

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution using the FTRL update rule.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the FTRL algorithm's specific way of updating the solution.
    /// It's like adjusting the recipe based on how well the last batch of cookies turned out, but in a very
    /// sophisticated way that considers the history of all previous batches.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>The updated solution.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        // === Partially Vectorized FTRL Update using IEngine (Phase B: US-GPU-015) ===
        // FTRL uses L1 thresholding which requires conditional logic per-element
        // Vectorized: gradient operations, sigma calculation, state updates
        // Element-wise: L1 thresholding conditional

        var parameters = currentSolution.GetParameters();

        // Save pre-update parameters for reverse updates (vectorized copy)
        _previousParameters = new Vector<T>(parameters);

        var alpha = NumOps.FromDouble(_options.Alpha);
        var lambda1 = NumOps.FromDouble(_options.Lambda1);
        var lambda2 = NumOps.FromDouble(_options.Lambda2);
        var lambda2Factor = NumOps.FromDouble(_options.Lambda2 * (1 + _options.Beta));

        // Vectorized gradient squared calculation
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);

        // Vectorized n update: n = n + g^2
        var nPlusGradSq = (Vector<T>)Engine.Add(_n!, gradSquared);

        // Vectorized sqrt operations for sigma calculation
        var sqrtNPlusGradSq = (Vector<T>)Engine.Sqrt(nPlusGradSq);
        var sqrtN = (Vector<T>)Engine.Sqrt(_n!);
        var numerator = (Vector<T>)Engine.Subtract(sqrtNPlusGradSq, sqrtN);
        var sigma = (Vector<T>)Engine.Divide(numerator, alpha);

        // Vectorized z update: z = z + g - sigma * params
        var sigmaTimesParams = (Vector<T>)Engine.Multiply(sigma, parameters);
        var gradMinusSigmaParams = (Vector<T>)Engine.Subtract(gradient, sigmaTimesParams);
        _z = (Vector<T>)Engine.Add(_z!, gradMinusSigmaParams);

        // Update n state
        _n = nPlusGradSq;

        // L1 thresholding requires per-element conditional logic
        var newCoefficients = new Vector<T>(parameters.Length);
        var absZ = (Vector<T>)Engine.Abs(_z);
        var signZ = (Vector<T>)Engine.Sign(_z);
        var sqrtNOverAlpha = (Vector<T>)Engine.Divide(sqrtNPlusGradSq, alpha);

        for (int i = 0; i < parameters.Length; i++)
        {
            // L1 proximal operator: sparse solution via thresholding
            if (NumOps.GreaterThan(absZ[i], lambda1))
            {
                // FTRL proximal: numerator = sign(z) * (lambda1 - |z|)
                var lambda1MinusAbsZ = NumOps.Subtract(lambda1, absZ[i]);
                var numeratorValue = NumOps.Multiply(lambda1MinusAbsZ, signZ[i]);
                var denominatorValue = NumOps.Add(lambda2Factor, sqrtNOverAlpha[i]);
                newCoefficients[i] = NumOps.Divide(numeratorValue, denominatorValue);
            }
            else
            {
                newCoefficients[i] = NumOps.Zero;
            }
        }

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Reverses an FTRL gradient update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after FTRL update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// FTRL's reverse update is complex due to its sparse regularization with thresholding.
    /// This method must be called immediately after UpdateParameters while the internal state (_z, _n) is fresh.
    /// Note: FTRL uses L1 regularization which causes sparsity through thresholding, making perfect reversal impossible.
    /// This implementation provides an approximate reverse based on the FTRL update formula.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before an FTRL update.
    /// FTRL is designed for sparse data and uses thresholding (setting small values to zero), which is
    /// irreversible. This reverse method approximates the original parameters using FTRL's internal state.
    /// </para>
    /// </remarks>
    public override Vector<T> ReverseUpdate(Vector<T> updatedParameters, Vector<T> appliedGradients)
    {
        if (updatedParameters == null)
            throw new ArgumentNullException(nameof(updatedParameters));
        if (appliedGradients == null)
            throw new ArgumentNullException(nameof(appliedGradients));

        if (updatedParameters.Length != appliedGradients.Length)
        {
            throw new ArgumentException(
                $"Updated parameters size ({updatedParameters.Length}) must match applied gradients size ({appliedGradients.Length})",
                nameof(appliedGradients));
        }

        if (_previousParameters == null || _previousParameters.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "FTRL optimizer state is not initialized. ReverseUpdate must be called after UpdateSolution.");
        }

        // FTRL's complex proximal gradient descent with L1 thresholding makes exact reversal impossible.
        // Return the pre-update parameters that were saved in UpdateSolution (vectorized copy).
        // This is the best we can do for FTRL due to the irreversible L1 thresholding.
        return new Vector<T>(_previousParameters);
    }

    /// <summary>
    /// Updates the adaptive parameters based on the optimization progress.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how fast the algorithm learns based on its recent performance.
    /// If it's improving, it might speed up. If it's not improving, it might slow down to be more careful.
    /// It's like adjusting your study pace based on how well you're understanding the material.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate,
                NumOps.FromDouble(_options.MinLearningRate),
                NumOps.FromDouble(_options.MaxLearningRate));
        }
    }

    /// <summary>
    /// Updates the options for the FTRL optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer while it's running.
    /// It's like adjusting the controls on a machine while it's operating. This can be useful if you want to
    /// fine-tune the optimizer's behavior based on its performance or other factors.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type FTRLOptimizerOptions.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is FTRLOptimizerOptions<T, TInput, TOutput> ftrlOptions)
        {
            _options = ftrlOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected FTRLOptimizerOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options of the FTRL optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you see what settings the optimizer is currently using.
    /// It's like checking the current settings on a machine to understand how it's configured.
    /// </para>
    /// </remarks>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the FTRL optimizer to a byte array.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts the current state of the optimizer into a format
    /// that can be easily stored or transmitted. It's like taking a snapshot of the optimizer's memory,
    /// including all its settings and learned information, so you can save it or send it somewhere else.
    /// </para>
    /// </remarks>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            writer.Write(_t);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the FTRL optimizer from a byte array.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a previously serialized optimizer state and
    /// reconstructs the optimizer from it. It's like restoring the optimizer's memory from a saved snapshot,
    /// allowing you to continue from where you left off or use a pre-trained optimizer.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<FTRLOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Initializes FTRL optimizer state on the GPU.
    /// </summary>
    /// <param name="parameterCount">Number of parameters.</param>
    /// <param name="backend">GPU backend for memory allocation.</param>
    public override void InitializeGpuState(int parameterCount, IDirectGpuBackend backend)
    {
        if (_gpuStateInitialized && _gpuZ != null && _gpuN != null)
            return;

        // Allocate GPU buffers for z and n state (initialized to zero)
        var zeros = new float[parameterCount];
        _gpuZ = backend.AllocateBuffer(zeros);
        _gpuN = backend.AllocateBuffer(zeros);

        _t = 0;
        _gpuStateInitialized = true;
    }

    /// <summary>
    /// Updates parameters on GPU using FTRL optimization.
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        if (!_gpuStateInitialized || _gpuZ == null || _gpuN == null)
        {
            InitializeGpuState(parameterCount, backend);
        }

        _t++;

        // Call the FTRL GPU kernel
        backend.FtrlUpdate(
            parameters,
            gradients,
            _gpuZ!,
            _gpuN!,
            (float)_options.Alpha,
            (float)_options.Lambda1,
            (float)_options.Lambda2,
            (float)_options.Beta,
            parameterCount
        );
    }

    /// <summary>
    /// Disposes GPU-allocated optimizer state.
    /// </summary>
    public override void DisposeGpuState()
    {
        _gpuZ?.Dispose();
        _gpuZ = null;
        _gpuN?.Dispose();
        _gpuN = null;
        _gpuStateInitialized = false;
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique identifier for storing and retrieving
    /// calculated gradients. It's like creating a label for a file drawer where you store specific calculations.
    /// This helps the optimizer avoid redoing calculations it has already performed, making it more efficient.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A string that uniquely identifies the gradient for the given inputs.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_FTRL_{_options.Alpha}_{_options.Beta}_{_options.Lambda1}_{_options.Lambda2}_{_t}";
    }
}
