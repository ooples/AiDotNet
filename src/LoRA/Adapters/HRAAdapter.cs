using AiDotNet.Attributes;
using AiDotNet.Helpers;
#pragma warning disable CS0649, CS0414, CS0169
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
[AutoParameters]
public partial class HRAAdapter<T> : LoRAAdapterBase<T>
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
    [TrainableParameter]
    private Tensor<T> _sparseValues;

    /// <summary>Row coordinate for each compact sparse value.</summary>
    [Buffer]
    private Tensor<T> _sparseRows;

    /// <summary>Column coordinate for each compact sparse value.</summary>
    [Buffer]
    private Tensor<T> _sparseColumns;

    /// <summary>Number of populated entries in the fixed-capacity sparse tensors.</summary>
    [Buffer]
    private Tensor<T> _activeSparseCount;

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
    [Buffer]
    private Matrix<T> _parameterImportance;

    /// <summary>
    /// Gradient accumulator for the sparse full-rank component.
    /// </summary>
    [Scratch]
    private Tensor<T>? _sparseGradients;

    /// <summary>
    /// Cached input from forward pass for computing sparse gradients in backward pass.
    /// </summary>
    [Scratch]
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
    public int ActiveSparseParams
    {
        get
        {
            int count = (int)Convert.ToDouble(_activeSparseCount[0]);
            if (count < 0) return 0;
            return count > _sparseValues.Length ? _sparseValues.Length : count;
        }
    }

    /// <summary>
    /// Gets the maximum allowed sparse parameters.
    /// </summary>
    public int MaxSparseParams => _maxSparseParams;

    /// <summary>
    /// Gets the current sparsity ratio.
    /// </summary>
    public double SparsityRatio => _sparsityRatio;

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
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        int totalWeightParams = inputSize * outputSize;

        _sparsityRatio = sparsityRatio;
        _maxSparseParams = (int)(totalWeightParams * sparsityRatio);
        _importanceUpdateInterval = importanceUpdateInterval;
        _importanceEMA = importanceEMA;
        _useDynamicAllocation = useDynamicAllocation;
        _stepCount = 0;

        // Keep sparse values and their coordinates in explicit tensors. Values are trainable;
        // coordinates and active length are persistent buffers, so a fresh clone can restore the
        // exact sparse layout without a hand-written flat-vector override.
        _sparseValues = new Tensor<T>([_maxSparseParams]);
        _sparseRows = new Tensor<T>([_maxSparseParams]);
        _sparseColumns = new Tensor<T>([_maxSparseParams]);
        _activeSparseCount = new Tensor<T>([1]);
        _activeSparseCount[0] = NumOps.Zero;

        // Initialize importance scores (uniform initially)
        _parameterImportance = new Matrix<T>(outputSize, inputSize); // zero-filled


        // Sparse scaling factor (typically smaller than LoRA scaling)
        _sparseScaling = NumOps.FromDouble(0.1);

        // The inherited LoRA tensors and the compact sparse tensors now form one generated manifest.
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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Cache input for the sparse-gradient backward only when a manual backward will read it
        SaveForBackward(ref _cachedInput, input);

        // 1. Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // 2. Forward through LoRA layer (low-rank component)
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // 3. Forward through sparse full-rank component
        Tensor<T> sparseOutput = ForwardSparseFullRank(input);

        // Sum all three components
        Tensor<T> result = Engine.TensorAdd(
            Engine.TensorAdd(baseOutput, loraOutput), Engine.Reshape(sparseOutput, baseOutput.Shape.ToArray()));


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
    /// Only the VIP parameters in the compact sparse tensors are used here.
    /// Everything else is treated as zero, maintaining efficiency.
    /// </para>
    /// </remarks>
    private Tensor<T> ForwardSparseFullRank(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        // If no sparse parameters, return zeros
        int activeCount = ActiveSparseParams;
        if (activeCount == 0)
        {
            Vector<T> zeroData = new Vector<T>(batchSize * outputSize);
            return new Tensor<T>(new[] { batchSize, outputSize }, zeroData);
        }

        // Sparse matrix multiplication: output[b, row] += (value * scaling) * input[b, col] for every
        // active entry, i.e. output = input * W^T with the dense [outputSize, inputSize] sparse delta W.
        Tensor<T> sparseDelta = BuildScaledSparseDelta(activeCount, outputSize, inputSize);
        return Engine.TensorMatMul(Engine.Reshape(input, [batchSize, inputSize]), Engine.TensorTranspose(sparseDelta));
    }

    /// <summary>
    /// One-hot encodes the first <paramref name="activeCount"/> stored coordinates against
    /// 0..<paramref name="size"/>-1 as an [activeCount, size] 0/1 tensor.
    /// </summary>
    private Tensor<T> SparseCoordinateOneHot(Tensor<T> coordinates, int activeCount, int size)
    {
        Tensor<T> active = Engine.TensorNarrow(coordinates, 0, 0, activeCount);
        return Engine.TensorEquals(
            Engine.TensorBroadcastTo(Engine.Reshape(active, [activeCount, 1]), [activeCount, size]),
            Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorArange<T>(0, size, 1), [1, size]), [activeCount, size]));
    }

    /// <summary>
    /// Builds the dense [outputSize, inputSize] matrix holding value * scaling at every active
    /// (row, col) sparse coordinate (repeated coordinates accumulate).
    /// </summary>
    private Tensor<T> BuildScaledSparseDelta(int activeCount, int outputSize, int inputSize)
    {
        Tensor<T> weights = Engine.TensorMultiplyScalar(Engine.TensorNarrow(_sparseValues, 0, 0, activeCount), _sparseScaling);
        Tensor<T> weightedRows = Engine.TensorMultiply(
            SparseCoordinateOneHot(_sparseRows, activeCount, outputSize), Engine.Reshape(weights, [activeCount, 1]));
        return Engine.TensorMatMul(
            Engine.TensorTranspose(weightedRows), SparseCoordinateOneHot(_sparseColumns, activeCount, inputSize));
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
        _sparseGradients = new Tensor<T>([_maxSparseParams]);

        // If no sparse parameters, return zeros
        int activeCount = ActiveSparseParams;
        if (activeCount == 0)
        {
            Vector<T> zeroData = new Vector<T>(batchSize * inputSize);
            return new Tensor<T>(new[] { batchSize, inputSize }, zeroData);
        }

        // Validate cached input is available
        if (_cachedInput == null)
        {
            throw new InvalidOperationException("Forward must be called before Backward");
        }

        Tensor<T> gradMatrix = Engine.Reshape(outputGradient, [batchSize, outputSize]);
        Tensor<T> inputMatrix = Engine.Reshape(_cachedInput, [batchSize, inputSize]);

        // Parameter gradient: dL/dWeight[row, col] = Σ_batch (input[b, col] * dL/dOutput[b, row])
        // (outer product of input and output error), scaled like the forward pass. With
        // M = grad^T * input ([outputSize, inputSize]) the k-th entry reads M[row_k, col_k].
        Tensor<T> rowOneHot = SparseCoordinateOneHot(_sparseRows, activeCount, outputSize);
        Tensor<T> colOneHot = SparseCoordinateOneHot(_sparseColumns, activeCount, inputSize);
        Tensor<T> outer = Engine.TensorMatMul(Engine.TensorTranspose(gradMatrix), inputMatrix);
        Tensor<T> paramGrads = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorMatMul(rowOneHot, outer), colOneHot), [1], false),
            _sparseScaling);
        _sparseGradients = activeCount < _maxSparseParams
            ? Engine.TensorConcatenate([paramGrads, new Tensor<T>([_maxSparseParams - activeCount])], 0)
            : paramGrads;

        // Input gradient: dL/dInput[b, col] += (value * scaling) * dL/dOutput[b, row] = grad * W
        return Engine.TensorMatMul(gradMatrix, BuildScaledSparseDelta(activeCount, outputSize, inputSize));
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
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
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
        // Higher LoRA gradients suggest that parameter needs more capacity.
        // Gradient flows through A[j, r] and B[r, i]; the LoRA parameter gradients are laid out as
        // [dA ([inputSize, rank]) row-major, dB ([rank, outputSize]) row-major], so
        // magnitude[i, j] = Σ_r |dA[j, r]| * |dB[r, i]| = (|dB|^T * |dA|^T)[i, j].
        int rank = Rank;
        int aCount = inputSize * rank;
        int bCount = rank * outputSize;
        Tensor<T> gradMagnitude = loraGradients.Length >= aCount + bCount
            ? Engine.TensorMatMul(
                Engine.TensorTranspose(Engine.TensorAbs(Engine.Reshape(
                    Tensor<T>.FromVector(loraGradients.GetSubVector(aCount, bCount)), [rank, outputSize]))),
                Engine.TensorTranspose(Engine.TensorAbs(Engine.Reshape(
                    Tensor<T>.FromVector(loraGradients.GetSubVector(0, aCount)), [inputSize, rank]))))
            : new Tensor<T>([outputSize, inputSize]);

        // Update importance with EMA
        _parameterImportance = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(_parameterImportance), emaFactor),
            Engine.TensorMultiplyScalar(gradMagnitude, oneMinusEma)).ToMatrix();
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
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        int inputSize = GetInputShape()[0];

        // Rank every position by importance (descending; master's List.Sort gave no tie guarantee)
        int total = outputSize * inputSize;
        int[] order = Engine.TensorArgsort(
            Engine.Reshape(Tensor<T>.FromMatrix(_parameterImportance), [total]), 0, true).ToArray();

        // Select top-k positions for sparse full-rank updates
        int selectedCount = Math.Min(_maxSparseParams, total);
        int[] selectedPositions = new int[selectedCount];
        Array.Copy(order, selectedPositions, selectedCount);
        Tensor<int> selected = new Tensor<int>(selectedPositions, [selectedCount]);

        // Existing sparse values laid out densely by (row, col), plus a mask of allocated positions
        int activeCount = ActiveSparseParams;
        Tensor<T> existingValues = new Tensor<T>([total]);
        Tensor<T> existingMask = new Tensor<T>([total]);
        if (activeCount > 0)
        {
            Tensor<T> rowOneHot = SparseCoordinateOneHot(_sparseRows, activeCount, outputSize);
            Tensor<T> colOneHot = SparseCoordinateOneHot(_sparseColumns, activeCount, inputSize);
            Tensor<T> weightedRows = Engine.TensorMultiply(
                rowOneHot, Engine.Reshape(Engine.TensorNarrow(_sparseValues, 0, 0, activeCount), [activeCount, 1]));
            existingValues = Engine.Reshape(
                Engine.TensorMatMul(Engine.TensorTranspose(weightedRows), colOneHot), [total]);
            existingMask = Engine.Reshape(
                Engine.TensorMatMul(Engine.TensorTranspose(rowOneHot), colOneHot), [total]);
        }

        // Preserve existing values if already allocated, otherwise initialize a small random value
        // uniform in [-0.01, 0.01) (master drew (u - 0.5) * 0.02 from an unseeded secure generator).
        Tensor<T> freshValues = Engine.TensorRandomUniformRange<T>(
            [selectedCount], NumOps.FromDouble(-0.01), NumOps.FromDouble(0.01));
        Tensor<T> newValues = Engine.TensorWhere(
            Engine.TensorIndexSelect(existingMask, selected, 0),
            Engine.TensorIndexSelect(existingValues, selected, 0),
            freshValues);

        // Row and column of each selected flat position (row = position / inputSize, col = position % inputSize)
        Tensor<T> rowOfPosition = Engine.Reshape(Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<T>(0, outputSize, 1), [outputSize, 1]), [outputSize, inputSize]), [total]);
        Tensor<T> colOfPosition = Engine.Reshape(Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<T>(0, inputSize, 1), [1, inputSize]), [outputSize, inputSize]), [total]);
        Tensor<T> newRows = Engine.TensorIndexSelect(rowOfPosition, selected, 0);
        Tensor<T> newCols = Engine.TensorIndexSelect(colOfPosition, selected, 0);

        // Entries past the selection are cleared
        if (selectedCount < _maxSparseParams)
        {
            Tensor<T> padding = new Tensor<T>([_maxSparseParams - selectedCount]);
            newValues = Engine.TensorConcatenate([newValues, padding], 0);
            newRows = Engine.TensorConcatenate([newRows, padding], 0);
            newCols = Engine.TensorConcatenate([newCols, padding], 0);
        }

        newValues.AsSpan().CopyTo(_sparseValues.AsWritableSpan());
        newRows.AsSpan().CopyTo(_sparseRows.AsWritableSpan());
        newCols.AsSpan().CopyTo(_sparseColumns.AsWritableSpan());
        _activeSparseCount[0] = NumOps.FromDouble(selectedCount);
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
            // values[0 : activeCount] -= learningRate * grads[0 : activeCount]
            int activeCount = ActiveSparseParams;
            if (activeCount > 0)
            {
                Tensor<T> updated = Engine.TensorSubtract(
                    Engine.TensorNarrow(_sparseValues, 0, 0, activeCount),
                    Engine.TensorMultiplyScalar(Engine.TensorNarrow(_sparseGradients, 0, 0, activeCount), learningRate));
                updated.AsSpan().CopyTo(_sparseValues.AsWritableSpan().Slice(0, activeCount));
            }

        }

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
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
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;

        // Start with base weights and add the LoRA contribution ([inputSize, outputSize]) and the
        // sparse full-rank contribution ([outputSize, inputSize]). DenseLayer stores its weights
        // input-major and FullyConnectedLayer output-major. Biases are kept unchanged.
        bool isOutputMajor = _baseLayer is FullyConnectedLayer<T>;
        Matrix<T> loraWeights = _loraLayer.MergeWeights();
        Vector<T> mergedParams = AddFlatWeightsToParameters(
            baseParams, isOutputMajor ? loraWeights.Transpose() : loraWeights, weightCount);

        int activeCount = ActiveSparseParams;
        if (activeCount > 0)
        {
            Matrix<T> sparseDelta = BuildScaledSparseDelta(activeCount, outputSize, inputSize).ToMatrix();
            mergedParams = AddFlatWeightsToParameters(
                mergedParams, isOutputMajor ? sparseDelta : sparseDelta.Transpose(), weightCount);
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
        var updates = new Dictionary<(int row, int col), T>();
        int activeCount = ActiveSparseParams;
        for (int i = 0; i < activeCount; i++)
        {
            int row = (int)Convert.ToDouble(_sparseRows[i]);
            int col = (int)Convert.ToDouble(_sparseColumns[i]);
            updates[(row, col)] = _sparseValues[i];
        }
        return updates;
    }

}
