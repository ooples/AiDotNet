using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Restart LoRA (ReLoRA) adapter that periodically merges and restarts LoRA training for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// ReLoRA addresses the challenge of continual learning and long-running training by periodically:
/// 1. Merging the LoRA weights into the base layer (accumulating the adaptation)
/// 2. Resetting the LoRA matrices to restart training fresh
/// 3. Continuing training with a clean slate while preserving previous learning
/// </para>
/// <para>
/// This approach:
/// - Prevents catastrophic forgetting by accumulating adaptations into the base layer
/// - Allows continuous adaptation to new data without losing old knowledge
/// - Maintains parameter efficiency by resetting LoRA to small matrices
/// - Enables training on continuously evolving data streams
/// </para>
/// <para><b>For Beginners:</b> ReLoRA is like having multiple rounds of LoRA training.
///
/// Imagine you're fine-tuning a model on data that keeps changing:
/// - Round 1: Train LoRA on dataset A for 1000 steps
/// - Merge: Add the learned changes into the base model
/// - Restart: Reset LoRA matrices and train on dataset B for 1000 steps
/// - Merge: Add these new changes to the (already updated) base model
/// - Repeat...
///
/// Benefits:
/// - Continual learning: Can keep learning from new data indefinitely
/// - No catastrophic forgetting: Old knowledge is preserved in the base layer
/// - Parameter efficient: LoRA matrices stay small even after many restarts
/// - Flexible: Can adapt to distribution shifts and new tasks
///
/// How it works:
/// 1. Train normally with LoRA for N steps (restart interval)
/// 2. At step N: Merge LoRA weights → AccumulatedWeight += LoRA
/// 3. Reset LoRA matrices to zero (fresh start)
/// 4. Continue training for another N steps
/// 5. Repeat indefinitely
///
/// Use cases:
/// - Training on streaming data (news articles, user behavior, etc.)
/// - Adapting to distribution shifts over time
/// - Long-running training sessions that need checkpoints
/// - Multi-task learning with periodic task switches
///
/// Reference: "ReLoRA: High-Rank Training Through Low-Rank Updates" (2023)
/// https://arxiv.org/abs/2307.05695
/// </para>
/// </remarks>
public class ReLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Number of training steps between restart operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The restart interval determines how frequently the LoRA weights are merged and reset.
    /// Typical values:
    /// - Short interval (100-500): Frequent restarts, better for rapidly changing data
    /// - Medium interval (1000-2000): Balance between stability and adaptation
    /// - Long interval (5000+): Fewer restarts, more thorough learning per cycle
    /// </para>
    /// <para><b>For Beginners:</b> This is how many training steps to run before merging and restarting.
    /// Think of it as the length of each training "session" before taking a checkpoint.
    /// </para>
    /// </remarks>
    private readonly int _restartInterval;

    /// <summary>
    /// Current training step counter.
    /// </summary>
    /// <remarks>
    /// This counts up from 0 to restartInterval, then resets to 0 after each restart.
    /// </remarks>
    private int _currentStep;

    /// <summary>
    /// Accumulated weight changes from all previous restart cycles.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix accumulates all LoRA adaptations across restart cycles:
    /// AccumulatedWeight = sum of all (A * B * scaling) across all cycles.
    /// It represents the total learned adaptation that gets added to the base layer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a running total of all the changes made across
    /// all restart cycles. Each time we restart, we add the current LoRA changes to this total.
    /// This is how we prevent forgetting - all previous learning is saved here.
    /// </para>
    /// </remarks>
    private Matrix<T> _accumulatedWeight;

    /// <summary>
    /// Total number of restarts that have occurred.
    /// </summary>
    private int _restartCount;

    /// <summary>
    /// Random number generator for matrix reinitialization.
    /// </summary>
    private static readonly Random _rng = RandomHelper.CreateSecureRandom();

    /// <summary>
    /// Whether to use warmup after each restart.
    /// </summary>
    /// <remarks>
    /// When true, the first few steps after restart use a reduced learning rate to stabilize training.
    /// </remarks>
    private readonly bool _useWarmup;

    /// <summary>
    /// Number of warmup steps to use after each restart.
    /// </summary>
    private readonly int _warmupSteps;

    /// <summary>
    /// Whether to freeze the base layer during training (typical for LoRA).
    /// </summary>
    private readonly bool _freezeBase;

    /// <summary>
    /// Gets the number of training steps between restarts.
    /// </summary>
    public int RestartInterval => _restartInterval;

    /// <summary>
    /// Gets the current step within the current restart cycle.
    /// </summary>
    public int CurrentStep => _currentStep;

    /// <summary>
    /// Gets the total number of restarts that have occurred.
    /// </summary>
    public int RestartCount => _restartCount;

    /// <summary>
    /// Gets a copy of the accumulated weight matrix.
    /// </summary>
    public Matrix<T> GetAccumulatedWeight() => _accumulatedWeight.Clone();

    /// <summary>
    /// Initializes a new ReLoRA adapter with restart-based continual learning.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with ReLoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="restartInterval">Number of steps between restart operations (default: 1000).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training (default: true).</param>
    /// <param name="useWarmup">Whether to use warmup after restarts (default: true).</param>
    /// <param name="warmupSteps">Number of warmup steps after restart (default: 10).</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when restartInterval is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a ReLoRA adapter for continual learning.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt continuously
    /// - rank: Size of the LoRA matrices (lower = more efficient)
    /// - alpha: Strength of the LoRA adaptation
    /// - restartInterval: How often to merge and restart (in training steps)
    /// - freezeBaseLayer: Lock the base layer weights (typical for LoRA)
    /// - useWarmup: Use reduced learning rate after restarts (helps stability)
    /// - warmupSteps: How many steps to warm up for
    ///
    /// The adapter will automatically handle merging and restarting at the specified interval.
    /// You just train normally, and it takes care of the restart logic.
    /// </para>
    /// </remarks>
    public ReLoRAAdapter(
        ILayer<T> baseLayer,
        int rank,
        double alpha = -1,
        int restartInterval = 1000,
        bool freezeBaseLayer = true,
        bool useWarmup = true,
        int warmupSteps = 10)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (restartInterval <= 0)
        {
            throw new ArgumentException("Restart interval must be positive", nameof(restartInterval));
        }

        if (warmupSteps < 0)
        {
            throw new ArgumentException("Warmup steps cannot be negative", nameof(warmupSteps));
        }

        _restartInterval = restartInterval;
        _currentStep = 0;
        _restartCount = 0;
        _freezeBase = freezeBaseLayer;
        _useWarmup = useWarmup;
        _warmupSteps = warmupSteps;

        // Initialize accumulated weight matrix to zero
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        _accumulatedWeight = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                _accumulatedWeight[i, j] = NumOps.Zero;
            }
        }
    }

    /// <summary>
    /// Checks if a restart should be performed based on the current step count.
    /// </summary>
    /// <returns>True if current step has reached the restart interval.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This checks if it's time for a restart.
    /// Returns true when we've completed a full training cycle (reached the interval).
    /// </para>
    /// </remarks>
    public bool ShouldRestart()
    {
        return _currentStep >= _restartInterval;
    }

    /// <summary>
    /// Performs the restart operation: merges current LoRA weights and reinitializes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The restart process:
    /// 1. Merge current LoRA weights: W_accumulated += W_A * W_B * scaling
    /// 2. Reinitialize LoRA matrices: A gets new random values, B reset to zero
    /// 3. Reset step counter to 0
    /// 4. Increment restart count
    /// </para>
    /// <para><b>For Beginners:</b> This performs the "checkpoint and restart" operation.
    ///
    /// Steps:
    /// 1. Save progress: Add current LoRA changes to the accumulated total
    /// 2. Fresh start: Reset LoRA matrices (A gets new random values, B starts at zero)
    /// 3. Reset counter: Start counting steps from 0 again
    ///
    /// After this, training continues normally for another cycle.
    /// The accumulated changes are preserved and will be included in the final output.
    /// </para>
    /// </remarks>
    public void RestartLoRA()
    {
        // Get the current LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Accumulate the LoRA weights
        for (int i = 0; i < _accumulatedWeight.Rows; i++)
        {
            for (int j = 0; j < _accumulatedWeight.Columns; j++)
            {
                _accumulatedWeight[i, j] = NumOps.Add(_accumulatedWeight[i, j], loraWeights[i, j]);
            }
        }

        // Reinitialize LoRA matrices
        Matrix<T> matrixA = _loraLayer.GetMatrixA();
        Matrix<T> matrixB = _loraLayer.GetMatrixB();

        // Reinitialize A with random values (same as initial LoRA initialization)
        T stddev = NumOps.Sqrt(NumOps.Divide(NumOps.One, NumOps.FromDouble(_loraLayer.Rank)));
        for (int i = 0; i < matrixA.Rows; i++)
        {
            for (int j = 0; j < matrixA.Columns; j++)
            {
                matrixA[i, j] = NumOps.Multiply(NumOps.FromDouble(_rng.NextGaussian()), stddev);
            }
        }

        // Reset B to zero
        for (int i = 0; i < matrixB.Rows; i++)
        {
            for (int j = 0; j < matrixB.Columns; j++)
            {
                matrixB[i, j] = NumOps.Zero;
            }
        }

        // Update the LoRA layer's parameters
        int paramIdx = 0;
        Vector<T> loraParams = new Vector<T>(_loraLayer.ParameterCount);

        // Pack matrix A
        for (int i = 0; i < matrixA.Rows; i++)
        {
            for (int j = 0; j < matrixA.Columns; j++)
            {
                loraParams[paramIdx++] = matrixA[i, j];
            }
        }

        // Pack matrix B
        for (int i = 0; i < matrixB.Rows; i++)
        {
            for (int j = 0; j < matrixB.Columns; j++)
            {
                loraParams[paramIdx++] = matrixB[i, j];
            }
        }

        _loraLayer.SetParameters(loraParams);

        // Reset step counter and increment restart count
        _currentStep = 0;
        _restartCount++;
    }

    /// <summary>
    /// Performs the forward pass with accumulated LoRA adaptations.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Base layer output plus accumulated LoRA adaptation plus current LoRA output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes:
    /// output = base_layer(input) + input * AccumulatedWeight + lora_layer(input)
    /// </para>
    /// <para><b>For Beginners:</b> This processes input through three components:
    /// 1. Base layer: The original layer (may have been adapted in previous cycles)
    /// 2. Accumulated weight: All previous LoRA cycles' learning
    /// 3. Current LoRA: The current cycle's adaptation
    ///
    /// All three are added together to produce the final output.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Check if restart is needed
        if (ShouldRestart())
        {
            RestartLoRA();
        }

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Forward through current LoRA layer
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // Apply accumulated weights
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int outputSize = GetOutputShape()[0];

        // Convert input to matrix for accumulated weight multiplication
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = input[i * inputSize + j];
            }
        }

        // Compute accumulated contribution: input * AccumulatedWeight^T
        Matrix<T> accumulatedOutput = inputMatrix.Multiply(_accumulatedWeight.Transpose());

        // Sum all contributions
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                int idx = i * outputSize + j;
                T baseVal = baseOutput[idx];
                T loraVal = loraOutput[idx];
                T accVal = accumulatedOutput[i, j];

                result[idx] = NumOps.Add(NumOps.Add(baseVal, loraVal), accVal);
            }
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through all components.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// Gradients flow through:
    /// 1. Current LoRA layer (always updated)
    /// 2. Base layer (only if not frozen)
    /// 3. Accumulated weights (updated to reflect gradient contribution)
    /// </para>
    /// <para><b>For Beginners:</b> This is where learning happens! Gradients flow backward:
    /// - Update current LoRA matrices
    /// - Update base layer if not frozen
    /// - Note: Accumulated weights are treated as constants during backprop
    ///   (they only change during restart, not during normal training)
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backward through current LoRA layer
        Tensor<T> loraInputGrad = _loraLayer.Backward(outputGradient);

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Backward through accumulated weights
        int batchSize = outputGradient.Shape[0];
        int outputSize = outputGradient.Shape[1];
        int inputSize = GetInputShape()[0];

        // Convert output gradient to matrix
        Matrix<T> gradMatrix = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradMatrix[i, j] = outputGradient[i * outputSize + j];
            }
        }

        // Compute input gradient from accumulated weights: gradient * AccumulatedWeight
        Matrix<T> accInputGrad = gradMatrix.Multiply(_accumulatedWeight);

        // Convert back to tensor
        Vector<T> accInputGradData = new Vector<T>(batchSize * inputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                accInputGradData[idx++] = accInputGrad[i, j];
            }
        }
        Tensor<T> accumulatedInputGrad = new Tensor<T>(new[] { batchSize, inputSize }, accInputGradData);

        // Sum all input gradients
        Tensor<T> inputGrad = new Tensor<T>(loraInputGrad.Shape);
        for (int i = 0; i < loraInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(NumOps.Add(loraInputGrad[i], baseInputGrad[i]), accumulatedInputGrad[i]);
        }

        // Increment step counter
        _currentStep++;

        return inputGrad;
    }

    /// <summary>
    /// Updates parameters with optional warmup after restarts.
    /// </summary>
    /// <param name="learningRate">The base learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// If warmup is enabled, the learning rate is scaled down for the first few steps
    /// after each restart to prevent instability.
    /// Warmup schedule: lr = base_lr * (current_step / warmup_steps)
    /// </para>
    /// <para><b>For Beginners:</b> This updates the model's parameters using gradients.
    ///
    /// After a restart, if warmup is enabled:
    /// - First few steps use a reduced learning rate (gradually increasing)
    /// - This helps the model stabilize after the restart shock
    /// - After warmup, normal learning rate is used
    ///
    /// Think of it like easing back into training after a checkpoint.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        T effectiveLR = learningRate;

        // Apply warmup if enabled and within warmup period
        if (_useWarmup && _currentStep < _warmupSteps)
        {
            T warmupFactor = NumOps.Divide(
                NumOps.FromDouble(_currentStep + 1),
                NumOps.FromDouble(_warmupSteps)
            );
            effectiveLR = NumOps.Multiply(learningRate, warmupFactor);
        }

        // Always update LoRA layer
        _loraLayer.UpdateParameters(effectiveLR);

        // Only update base layer if not frozen
        if (!_freezeBase)
        {
            _baseLayer.UpdateParameters(effectiveLR);
        }

        // Update parameter vector
        UpdateParametersFromLayers();
    }

    /// <summary>
    /// Updates the parameter vector from the current layer states.
    /// </summary>
    protected override void UpdateParametersFromLayers()
    {
        int idx = 0;

        // If base layer is not frozen, pack its parameters first
        if (!_freezeBase)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        // Pack LoRA parameters
        Vector<T> loraParams = _loraLayer.GetParameters();
        for (int i = 0; i < loraParams.Length; i++)
        {
            Parameters[idx++] = loraParams[i];
        }
    }

    /// <summary>
    /// Merges all accumulated adaptations into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with all ReLoRA adaptations (accumulated + current) merged into the base layer's weights.</returns>
    /// <remarks>
    /// <para>
    /// This merges:
    /// 1. All accumulated LoRA weights from previous restart cycles
    /// 2. The current LoRA cycle's weights
    /// into the base layer's weights to create a final standalone layer.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" all the ReLoRA learning into a regular layer.
    ///
    /// This takes:
    /// - All previous cycles' learning (from accumulated weights)
    /// - Current cycle's learning (from current LoRA)
    /// - Base layer weights
    ///
    /// And combines them into a single layer that:
    /// - Works like a normal layer (no special ReLoRA infrastructure needed)
    /// - Contains all the adapted knowledge
    /// - Can be deployed for fast inference
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnected layers
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("ReLoRAAdapter only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get current LoRA weight contribution
        Matrix<T> currentLoRAWeights = _loraLayer.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with all merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge: base_weights + accumulated_weights + current_lora_weights
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            T baseWeight = baseParams[i];
            T accWeight = _accumulatedWeight[row, col];
            T loraWeight = currentLoRAWeights[row, col];

            mergedParams[i] = NumOps.Add(NumOps.Add(baseWeight, accWeight), loraWeight);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Resets the internal state of all layers.
    /// </summary>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _loraLayer.ResetState();
    }

    /// <summary>
    /// Manually triggers a restart (useful for checkpointing or manual control).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This forces an immediate restart, even if the interval hasn't been reached.
    /// Useful when you want to checkpoint at specific points (e.g., after completing a task or dataset).
    /// </para>
    /// </remarks>
    public void ForceRestart()
    {
        RestartLoRA();
    }

    /// <summary>
    /// Resets the step counter without performing a restart (useful for aligning with external training loops).
    /// </summary>
    public void ResetStepCounter()
    {
        _currentStep = 0;
    }
}
