using System.Collections.Generic;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// HRA (Hybrid Rank Adaptation) adapter that combines low-rank and full-rank updates for optimal parameter efficiency.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// HRA addresses a key limitation of standard LoRA: while low-rank updates are efficient, some parameters
/// benefit from full-rank updates. HRA uses a hybrid approach:
/// - Dense low-rank updates for most parameters (efficient, like LoRA)
/// - Sparse full-rank updates for critical parameters (precise, targeted)
/// - Importance-based allocation between the two components
/// </para>
/// <para>
/// The forward computation is: output = base_layer(input) + low_rank(input) + sparse_full_rank(input)
/// where the hybrid allocation provides the best of both worlds.
/// </para>
/// <para><b>For Beginners:</b> HRA is like having two tools instead of one:
///
/// Standard LoRA problem:
/// - Uses only low-rank updates (compressed, efficient)
/// - Some parameters need precise full-rank updates
/// - Full fine-tuning is too expensive
/// - Need something in between
///
/// HRA solution:
/// - Most parameters use low-rank updates (efficient, covers 95% of needs)
/// - Critical parameters get full-rank updates (precise, covers remaining 5%)
/// - Automatically learns which parameters are critical
/// - Best quality with minimal parameter overhead
///
/// Analogy: Think of home renovation:
/// - Low-rank updates: Paint the walls (cheap, covers large area, good enough)
/// - Full-rank updates: Replace key structural beams (expensive, small area, critical)
/// - HRA: Do both where appropriate for best results
///
/// How it works:
/// 1. Start with LoRA-style low-rank matrices (B * A)
/// 2. Add sparse full-rank updates for most important parameters
/// 3. Track importance scores during training
/// 4. Allocate parameter budget optimally between low-rank and sparse full-rank
///
/// Benefits:
/// - Better quality than pure LoRA (full-rank updates where needed)
/// - More efficient than full fine-tuning (most updates are low-rank)
/// - Adaptive: learns which parameters need full-rank updates
/// - Flexible: adjustable sparsity budget for full-rank component
///
/// Use cases:
/// - Tasks where LoRA quality is not quite sufficient
/// - Fine-tuning with specific architectural bottlenecks
/// - When you have slightly more parameter budget than LoRA but much less than full fine-tuning
/// - Domains where certain parameters are known to be critical
///
/// Example parameter comparison for a 1000x1000 layer:
/// - Full fine-tuning: 1,000,000 parameters
/// - Standard LoRA (rank=8): 16,000 parameters (98.4% reduction)
/// - HRA (rank=8, 1% sparsity): 26,000 parameters (97.4% reduction, better quality)
///
/// Reference: Based on "Hybrid Rank Adaptation" research combining low-rank and sparse full-rank approaches
/// </para>
/// </remarks>
public class HRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Sparse full-rank update matrix storing only non-zero entries.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This dictionary maps (row, col) positions to their update values.
    /// Only the most important parameters have non-zero entries here.
    /// This provides targeted full-rank updates while maintaining parameter efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a selective paint touch-up kit.
    /// Instead of repainting the whole wall (full-rank), we only fix the important spots
    /// that need precise attention. The dictionary only stores the spots we're fixing,
    /// saving memory.
    /// </para>
    /// </remarks>
    private Dictionary<(int row, int col), T> _sparseFullRankUpdates;

    /// <summary>
    /// Importance scores for each parameter in the weight matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each score represents how important that parameter is for the adaptation.
    /// Higher scores indicate parameters that should receive full-rank updates.
    /// Lower scores indicate parameters that are fine with low-rank updates.
    /// </para>
    /// <para><b>For Beginners:</b> These scores tell us which parameters are VIPs.
    /// High score = this parameter is critical, give it a full-rank update.
    /// Low score = this parameter is fine with a low-rank approximation.
    /// </para>
    /// </remarks>
    private Matrix<T> _parameterImportance;

    /// <summary>
    /// Gradient accumulator for the sparse full-rank component.
    /// </summary>
    private Dictionary<(int row, int col), T>? _sparseGradients;

    /// <summary>
    /// Cached input from forward pass for computing sparse gradients in backward pass.
    /// </summary>
    private Tensor<T>? _cachedInput;

    /// <summary>
    /// Maximum number of sparse full-rank parameters to allocate.
    /// </summary>
    /// <remarks>
    /// Controls the parameter budget for the sparse full-rank component.
    /// Typical values: 1-5% of total weight parameters.
    /// </remarks>
    private readonly int _maxSparseParams;

    /// <summary>
    /// Sparsity ratio for full-rank updates (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Determines what fraction of parameters can receive full-rank updates.
    /// For example, 0.01 means 1% of parameters can have full-rank updates.
    /// </para>
    /// <para><b>For Beginners:</b> This is your "special attention budget".
    /// If you have 1000 parameters and sparsity=0.01, you can give 10 parameters
    /// the VIP treatment (full-rank updates). Choose wisely!
    /// </para>
    /// </remarks>
    private readonly double _sparsityRatio;

    /// <summary>
    /// Number of training steps between importance updates.
    /// </summary>
    private readonly int _importanceUpdateInterval;

    /// <summary>
    /// Current training step counter.
    /// </summary>
    private int _stepCount;

    /// <summary>
    /// Exponential moving average factor for importance score updates.
    /// </summary>
    /// <remarks>
    /// Controls how quickly importance scores adapt to new gradient information.
    /// Typical values: 0.9 to 0.99 (higher = more smoothing, lower = faster adaptation).
    /// </remarks>
    private readonly double _importanceEMA;

    /// <summary>
    /// Scaling factor for the sparse full-rank component.
    /// </summary>
    private readonly T _sparseScaling;

    /// <summary>
    /// Whether to use dynamic importance-based allocation.
    /// </summary>
    private readonly bool _useDynamicAllocation;

    /// <summary>
    /// Gets the number of active sparse full-rank parameters.
    /// </summary>
    public int ActiveSparseParams => _sparseFullRankUpdates.Count;

    /// <summary>
    /// Gets the maximum allowed sparse parameters.
    /// </summary>
    public int MaxSparseParams => _maxSparseParams;

    /// <summary>
    /// Gets the current sparsity ratio.
    /// </summary>
    public double SparsityRatio => _sparsityRatio;

    /// <summary>
    /// Gets the total number of trainable parameters (low-rank + sparse full-rank).
    /// </summary>
    public override int ParameterCount
    {
        get
        {
            int loraParams = _loraLayer != null ? _loraLayer.ParameterCount : 0;
            int sparseParams = _sparseFullRankUpdates != null ? _sparseFullRankUpdates.Count : 0;
            int baseParams = (_baseLayer != null && !_freezeBaseLayer) ? _baseLayer.ParameterCount : 0;
            return baseParams + loraParams + sparseParams;
        }
    }

    /// <summary>
    /// Initializes a new HRA adapter with hybrid low-rank and sparse full-rank updates.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with HRA.</param>
    /// <param name="rank">The rank of the low-rank decomposition.</param>
    /// <param name="sparsityRatio">Fraction of parameters for sparse full-rank updates (0.0 to 1.0, default: 0.01).</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <param name="importanceUpdateInterval">Steps between importance recalculation (default: 100).</param>
    /// <param name="importanceEMA">EMA factor for importance smoothing (default: 0.95).</param>
    /// <param name="useDynamicAllocation">Whether to dynamically reallocate sparse parameters (default: true).</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an HRA adapter that combines two update strategies.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt
    /// - rank: Size of the low-rank component (typical: 8-16)
    /// - sparsityRatio: Budget for full-rank updates (0.01 = 1% of parameters get special treatment)
    /// - alpha: Strength of the low-rank adaptation
    /// - freezeBaseLayer: Lock original weights (usually true)
    /// - importanceUpdateInterval: How often to reassess which parameters are important
    /// - importanceEMA: How stable importance scores are (higher = more stable)
    /// - useDynamicAllocation: Automatically move sparse budget to most important parameters
    ///
    /// Example:
    /// new HRAAdapter(layer, rank: 8, sparsityRatio: 0.01)
    /// This gives you LoRA-style updates for most parameters, plus precise updates for the top 1%.
    /// </para>
    /// </remarks>
    public HRAAdapter(
        ILayer<T> baseLayer,
        int rank,
        double sparsityRatio = 0.01,
        double alpha = -1,
        bool freezeBaseLayer = true,
        int importanceUpdateInterval = 100,
        double importanceEMA = 0.95,
        bool useDynamicAllocation = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (sparsityRatio < 0.0 || sparsityRatio > 1.0)
        {
            throw new ArgumentException("Sparsity ratio must be between 0 and 1", nameof(sparsityRatio));
        }

        if (importanceEMA <= 0 || importanceEMA >= 1)
        {
            throw new ArgumentException("Importance EMA factor must be between 0 and 1", nameof(importanceEMA));
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int totalWeightParams = inputSize * outputSize;

        _sparsityRatio = sparsityRatio;
        _maxSparseParams = (int)(totalWeightParams * sparsityRatio);
        _importanceUpdateInterval = importanceUpdateInterval;
        _importanceEMA = importanceEMA;
        _useDynamicAllocation = useDynamicAllocation;
        _stepCount = 0;

        // Initialize sparse full-rank updates (empty initially)
        _sparseFullRankUpdates = new Dictionary<(int row, int col), T>();

        // Initialize importance scores (uniform initially)
        _parameterImportance = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                _parameterImportance[i, j] = NumOps.Zero;
            }
        }

        // Sparse scaling factor (typically smaller than LoRA scaling)
        _sparseScaling = NumOps.FromDouble(0.1);

        // Initialize parameters
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromComponents();
    }

    /// <summary>
    /// Performs the forward pass through the HRA adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output, low-rank LoRA output, and sparse full-rank output.</returns>
    /// <remarks>
    /// <para>
    /// The HRA forward pass computes three components:
    /// 1. Base layer output (original behavior)
    /// 2. Low-rank LoRA output: scaling * B * A * input
    /// 3. Sparse full-rank output: sparse_scaling * S * input (where S is sparse)
    /// </para>
    /// <para><b>For Beginners:</b> This processes input through three paths and adds them:
    /// 1. Original layer (base behavior)
    /// 2. LoRA low-rank path (efficient updates for most parameters)
    /// 3. Sparse full-rank path (precise updates for VIP parameters)
    ///
    /// Think of it as a team effort:
    /// - Base layer: The foundation
    /// - Low-rank: The general workforce (handles most of the load efficiently)
    /// - Sparse full-rank: The specialists (handle critical details precisely)
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Cache input for computing sparse gradients in backward pass
        _cachedInput = input;

        // 1. Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // 2. Forward through LoRA layer (low-rank component)
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // 3. Forward through sparse full-rank component
        Tensor<T> sparseOutput = ForwardSparseFullRank(input);

        // Sum all three components
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            T sum = NumOps.Add(baseOutput[i], loraOutput[i]);
            result[i] = NumOps.Add(sum, sparseOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs forward pass through the sparse full-rank component.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sparse full-rank output tensor.</returns>
    /// <remarks>
    /// <para>
    /// Computes output using only the sparse full-rank parameters.
    /// This is a standard matrix multiplication but using a sparse weight matrix.
    /// </para>
    /// <para><b>For Beginners:</b> This applies the "specialist" updates.
    /// Only the VIP parameters (stored in _sparseFullRankUpdates) are used here.
    /// Everything else is treated as zero, maintaining efficiency.
    /// </para>
    /// </remarks>
    private Tensor<T> ForwardSparseFullRank(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int outputSize = GetOutputShape()[0];

        // If no sparse parameters, return zeros
        if (_sparseFullRankUpdates.Count == 0)
        {
            Vector<T> zeroData = new Vector<T>(batchSize * outputSize);
            return new Tensor<T>(new[] { batchSize, outputSize }, zeroData);
        }

        // Convert input to matrix [batchSize, inputSize]
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = input[i * inputSize + j];
            }
        }

        // Compute sparse matrix multiplication
        Matrix<T> output = new Matrix<T>(batchSize, outputSize);
        foreach (var kvp in _sparseFullRankUpdates)
        {
            int row = kvp.Key.row;
            int col = kvp.Key.col;
            T weight = NumOps.Multiply(kvp.Value, _sparseScaling);

            // output[b, row] += weight * input[b, col]
            for (int b = 0; b < batchSize; b++)
            {
                T contribution = NumOps.Multiply(weight, inputMatrix[b, col]);
                output[b, row] = NumOps.Add(output[b, row], contribution);
            }
        }

        // Convert back to tensor
        Vector<T> outputData = new Vector<T>(batchSize * outputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                outputData[idx++] = output[i, j];
            }
        }

        return new Tensor<T>(new[] { batchSize, outputSize }, outputData);
    }

    /// <summary>
    /// Performs the backward pass through the HRA adapter.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients for:
    /// 1. Low-rank LoRA matrices (A and B)
    /// 2. Sparse full-rank parameters
    /// 3. Updates importance scores based on gradient magnitudes
    /// </para>
    /// <para><b>For Beginners:</b> This is where HRA learns which parameters are important!
    /// During backpropagation:
    /// 1. Compute gradients for low-rank component (standard LoRA)
    /// 2. Compute gradients for sparse full-rank parameters
    /// 3. Track which parameters have large gradients (they're important!)
    /// 4. Periodically reassign sparse budget to most important parameters
    ///
    /// This adaptive approach ensures the sparse full-rank budget is always
    /// allocated to the parameters that need it most.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backward through LoRA layer
        Tensor<T> loraInputGrad = _loraLayer.Backward(outputGradient);

        // Backward through sparse full-rank component
        Tensor<T> sparseInputGrad = BackwardSparseFullRank(outputGradient);

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Update importance scores based on gradients
        UpdateImportanceScores(outputGradient);

        // Increment step and check if we should reallocate sparse parameters
        _stepCount++;
        if (_useDynamicAllocation && _stepCount % _importanceUpdateInterval == 0)
        {
            ReallocateSparseParameters();
        }

        // Sum input gradients
        Tensor<T> inputGrad = new Tensor<T>(loraInputGrad.Shape);
        for (int i = 0; i < loraInputGrad.Length; i++)
        {
            T sum = NumOps.Add(loraInputGrad[i], sparseInputGrad[i]);
            inputGrad[i] = NumOps.Add(sum, baseInputGrad[i]);
        }

        return inputGrad;
    }

    /// <summary>
    /// Performs backward pass through the sparse full-rank component.
    /// </summary>
    /// <param name="outputGradient">Output gradient tensor.</param>
    /// <returns>Input gradient tensor.</returns>
    private Tensor<T> BackwardSparseFullRank(Tensor<T> outputGradient)
    {
        int batchSize = outputGradient.Shape[0];
        int outputSize = outputGradient.Shape.Length > 1 ? outputGradient.Shape[1] : outputGradient.Length;
        int inputSize = GetInputShape()[0];

        // Initialize sparse gradients
        _sparseGradients = new Dictionary<(int row, int col), T>();

        // If no sparse parameters, return zeros
        if (_sparseFullRankUpdates.Count == 0)
        {
            Vector<T> zeroData = new Vector<T>(batchSize * inputSize);
            return new Tensor<T>(new[] { batchSize, inputSize }, zeroData);
        }

        // Validate cached input is available
        if (_cachedInput == null)
        {
            throw new InvalidOperationException("Forward must be called before Backward");
        }

        // Convert gradient to matrix
        Matrix<T> gradMatrix = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradMatrix[i, j] = outputGradient[i * outputSize + j];
            }
        }

        // Convert cached input to matrix
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = _cachedInput[i * inputSize + j];
            }
        }

        // Compute input gradients and parameter gradients
        Matrix<T> inputGradMatrix = new Matrix<T>(batchSize, inputSize);

        foreach (var kvp in _sparseFullRankUpdates)
        {
            int row = kvp.Key.row;
            int col = kvp.Key.col;
            T weight = NumOps.Multiply(kvp.Value, _sparseScaling);

            T paramGrad = NumOps.Zero;

            for (int b = 0; b < batchSize; b++)
            {
                // Input gradient: dL/dInput[b, col] += weight * dL/dOutput[b, row]
                T grad = NumOps.Multiply(weight, gradMatrix[b, row]);
                inputGradMatrix[b, col] = NumOps.Add(inputGradMatrix[b, col], grad);

                // Parameter gradient: dL/dWeight[row, col] = Σ_batch (input[b, col] * dL/dOutput[b, row])
                // This is the correct gradient formula for linear layers (outer product of input and output error)
                T inputVal = inputMatrix[b, col];
                T outputGrad = gradMatrix[b, row];
                T gradContribution = NumOps.Multiply(inputVal, outputGrad);
                paramGrad = NumOps.Add(paramGrad, gradContribution);
            }

            _sparseGradients[kvp.Key] = NumOps.Multiply(paramGrad, _sparseScaling);
        }

        // Convert input gradients back to tensor
        Vector<T> inputGradData = new Vector<T>(batchSize * inputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputGradData[idx++] = inputGradMatrix[i, j];
            }
        }

        return new Tensor<T>(new[] { batchSize, inputSize }, inputGradData);
    }

    /// <summary>
    /// Updates importance scores based on current gradient magnitudes.
    /// </summary>
    /// <param name="outputGradient">Output gradient from backward pass.</param>
    /// <remarks>
    /// <para>
    /// Importance is computed using exponential moving average of gradient magnitudes.
    /// Parameters with consistently high gradients are considered important candidates
    /// for sparse full-rank updates.
    /// </para>
    /// <para><b>For Beginners:</b> This identifies which parameters are VIPs.
    ///
    /// We track gradient magnitudes over time using exponential moving average:
    /// - new_importance = 0.95 * old_importance + 0.05 * current_gradient_magnitude
    ///
    /// Parameters with consistently high gradients get high importance scores.
    /// These are the ones that will receive sparse full-rank updates.
    /// </para>
    /// </remarks>
    private void UpdateImportanceScores(Tensor<T> outputGradient)
    {
        int outputSize = GetOutputShape()[0];
        int inputSize = GetInputShape()[0];

        // Get LoRA parameter gradients to estimate per-parameter importance
        Vector<T> loraGradients = _loraLayer.GetParameterGradients();

        // Update importance based on gradient flow through LoRA component
        // This is a proxy for which parameters would benefit from full-rank updates
        Matrix<T> matrixA = _loraLayer.GetMatrixA();
        Matrix<T> matrixB = _loraLayer.GetMatrixB();

        T emaFactor = NumOps.FromDouble(_importanceEMA);
        T oneMinusEma = NumOps.FromDouble(1.0 - _importanceEMA);

        // Estimate per-parameter importance from LoRA gradients
        // Higher LoRA gradients suggest that parameter needs more capacity
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                // Compute approximate gradient magnitude for this weight
                // by looking at contributions through LoRA paths
                T gradMagnitude = NumOps.Zero;

                // Sum contributions from all rank components
                int rank = Rank;
                for (int r = 0; r < rank; r++)
                {
                    // Gradient flows through A[j,r] and B[r,i]
                    int aIndex = j * rank + r;
                    int bIndex = r * outputSize + i;

                    if (aIndex < loraGradients.Length && bIndex < loraGradients.Length)
                    {
                        T contribution = NumOps.Multiply(
                            NumOps.Abs(loraGradients[aIndex]),
                            NumOps.Abs(loraGradients[bIndex]));
                        gradMagnitude = NumOps.Add(gradMagnitude, contribution);
                    }
                }

                // Update importance with EMA
                T oldImportance = _parameterImportance[i, j];
                T newImportance = NumOps.Add(
                    NumOps.Multiply(emaFactor, oldImportance),
                    NumOps.Multiply(oneMinusEma, gradMagnitude));

                _parameterImportance[i, j] = newImportance;
            }
        }
    }

    /// <summary>
    /// Reallocates sparse full-rank parameters to the most important locations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method identifies the top-k most important parameters and assigns
    /// sparse full-rank updates to them. Previously allocated parameters that
    /// are no longer in the top-k are removed.
    /// </para>
    /// <para><b>For Beginners:</b> This is like reassigning specialists to where they're needed most.
    ///
    /// Every few hundred training steps:
    /// 1. Look at all importance scores
    /// 2. Find the top 1% most important parameters
    /// 3. Assign sparse full-rank budget to those parameters
    /// 4. Remove it from parameters that are no longer important
    ///
    /// This ensures the sparse budget is always optimally allocated.
    /// </para>
    /// </remarks>
    private void ReallocateSparseParameters()
    {
        int outputSize = GetOutputShape()[0];
        int inputSize = GetInputShape()[0];

        // Create list of (importance, position) pairs
        var importanceList = new List<(T importance, int row, int col)>();
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                importanceList.Add((_parameterImportance[i, j], i, j));
            }
        }

        // Sort by importance (descending)
        importanceList.Sort((a, b) =>
            Convert.ToDouble(b.importance).CompareTo(Convert.ToDouble(a.importance)));

        // Select top-k positions for sparse full-rank updates
        var newSparseUpdates = new Dictionary<(int row, int col), T>();
        for (int i = 0; i < Math.Min(_maxSparseParams, importanceList.Count); i++)
        {
            var entry = importanceList[i];
            var key = (entry.row, entry.col);

            // Preserve existing values if already allocated, otherwise initialize small random
            if (_sparseFullRankUpdates.ContainsKey(key))
            {
                newSparseUpdates[key] = _sparseFullRankUpdates[key];
            }
            else
            {
                // Initialize new sparse parameter with small random value
                Random rng = RandomHelper.CreateSecureRandom();
                double randVal = (rng.NextDouble() - 0.5) * 0.02; // Small initialization
                newSparseUpdates[key] = NumOps.FromDouble(randVal);
            }
        }

        _sparseFullRankUpdates = newSparseUpdates;
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        // Update LoRA layer
        _loraLayer.UpdateParameters(learningRate);

        // Update sparse full-rank parameters
        if (_sparseGradients != null)
        {
            var updatedSparse = new Dictionary<(int row, int col), T>();
            foreach (var kvp in _sparseFullRankUpdates)
            {
                T currentValue = kvp.Value;
                T gradient = _sparseGradients.ContainsKey(kvp.Key) ? _sparseGradients[kvp.Key] : NumOps.Zero;
                T update = NumOps.Multiply(gradient, learningRate);
                T newValue = NumOps.Subtract(currentValue, update);
                updatedSparse[kvp.Key] = newValue;
            }
            _sparseFullRankUpdates = updatedSparse;
        }

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromComponents();
    }

    /// <summary>
    /// Updates the parameter vector from the current component states.
    /// </summary>
    private void UpdateParametersFromComponents()
    {
        Parameters = new Vector<T>(ParameterCount);
        int idx = 0;

        // Pack base layer parameters if not frozen
        if (!_freezeBaseLayer)
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

        // Pack sparse parameters (just the values, positions are implicit)
        foreach (var kvp in _sparseFullRankUpdates)
        {
            Parameters[idx++] = kvp.Value;
        }
    }

    /// <summary>
    /// Merges the HRA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with both low-rank and sparse full-rank updates merged.</returns>
    /// <remarks>
    /// <para>
    /// This merges both the low-rank LoRA component and the sparse full-rank component
    /// into the base layer's weights, creating a single efficient layer.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" both types of updates for deployment.
    ///
    /// The merged layer includes:
    /// - Original base layer weights
    /// - Low-rank LoRA updates (for general improvements)
    /// - Sparse full-rank updates (for critical parameters)
    ///
    /// Result: A single layer with all adaptations built-in, ready for fast inference.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("HRAAdapter only supports DenseLayer or FullyConnectedLayer base layers");
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;

        // Create merged parameters
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Start with base weights
        for (int i = 0; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Add LoRA contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(mergedParams[i], loraWeights[row, col]);
        }

        // Add sparse full-rank contribution
        foreach (var kvp in _sparseFullRankUpdates)
        {
            int row = kvp.Key.row;
            int col = kvp.Key.col;
            int paramIndex = row * inputSize + col;

            if (paramIndex < weightCount)
            {
                T sparseContribution = NumOps.Multiply(kvp.Value, _sparseScaling);
                mergedParams[paramIndex] = NumOps.Add(mergedParams[paramIndex], sparseContribution);
            }
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Gets a copy of the current parameter importance matrix.
    /// </summary>
    /// <returns>Matrix of importance scores for each parameter.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This lets you see which parameters the model considers important.
    /// High values indicate parameters that are candidates for sparse full-rank updates.
    /// Useful for understanding and debugging the hybrid allocation strategy.
    /// </para>
    /// </remarks>
    public Matrix<T> GetParameterImportance()
    {
        return _parameterImportance.Clone();
    }

    /// <summary>
    /// Gets the positions and values of current sparse full-rank updates.
    /// </summary>
    /// <returns>Dictionary mapping (row, col) positions to update values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This shows you exactly which parameters are receiving
    /// the VIP treatment (full-rank updates). You can inspect this to understand
    /// where the model is allocating its sparse parameter budget.
    /// </para>
    /// </remarks>
    public Dictionary<(int row, int col), T> GetSparseUpdates()
    {
        return new Dictionary<(int row, int col), T>(_sparseFullRankUpdates);
    }

    /// <summary>
    /// Gets all parameters including base, LoRA, and sparse full-rank parameters.
    /// </summary>
    /// <returns>Vector containing all trainable parameters.</returns>
    public override Vector<T> GetParameters()
    {
        Vector<T> allParams = new Vector<T>(ParameterCount);
        int idx = 0;

        // Pack base layer parameters (if not frozen)
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                allParams[idx++] = baseParams[i];
            }
        }

        // Pack LoRA layer parameters
        Vector<T> loraParams = _loraLayer.GetParameters();
        for (int i = 0; i < loraParams.Length; i++)
        {
            allParams[idx++] = loraParams[i];
        }

        // Pack sparse full-rank parameters
        foreach (var kvp in _sparseFullRankUpdates)
        {
            allParams[idx++] = kvp.Value;
        }

        return allParams;
    }

    /// <summary>
    /// Sets all parameters including base, LoRA, and sparse full-rank parameters.
    /// </summary>
    /// <param name="parameters">Vector containing all parameters to set.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        int idx = 0;

        // Unpack base layer parameters (if not frozen)
        if (!_freezeBaseLayer)
        {
            int baseParamCount = _baseLayer.ParameterCount;
            Vector<T> baseParams = new Vector<T>(baseParamCount);
            for (int i = 0; i < baseParamCount; i++)
            {
                baseParams[i] = parameters[idx++];
            }
            _baseLayer.SetParameters(baseParams);
        }

        // Unpack LoRA layer parameters
        int loraParamCount = _loraLayer.ParameterCount;
        Vector<T> loraParams = new Vector<T>(loraParamCount);
        for (int i = 0; i < loraParamCount; i++)
        {
            loraParams[i] = parameters[idx++];
        }
        _loraLayer.SetParameters(loraParams);

        // Unpack sparse full-rank parameters
        // Restore the same keys that exist in the dictionary
        var keys = new List<(int row, int col)>(_sparseFullRankUpdates.Keys);
        foreach (var key in keys)
        {
            _sparseFullRankUpdates[key] = parameters[idx++];
        }

        // Update the unified Parameters vector
        Parameters = parameters.Clone();
    }
}
