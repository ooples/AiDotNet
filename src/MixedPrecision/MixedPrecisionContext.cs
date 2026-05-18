using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MixedPrecision;

/// <summary>
/// Manages master weights (FP32) and working weights (FP16) for mixed-precision training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Mixed-precision training uses two copies of model parameters:
///
/// 1. **Master Weights** (FP32):
///    - High-precision copy of all parameters
///    - Used for parameter updates to maintain accuracy
///    - Stored in memory but not used for forward/backward passes
///
/// 2. **Working Weights** (FP16):
///    - Low-precision copy used for computation
///    - Used in forward and backward passes
///    - Faster and uses less memory
///    - Synced from master weights before each forward pass
///
/// The workflow:
/// 1. Cast master weights (FP32) to working weights (FP16)
/// 2. Forward pass using FP16 weights → faster, less memory
/// 3. Backward pass in FP16 → computes FP16 gradients
/// 4. Cast gradients to FP32 and unscale
/// 5. Update master weights in FP32 → maintains precision
/// 6. Repeat from step 1
///
/// This approach combines the speed of FP16 with the numerical stability of FP32.
/// </para>
/// <para><b>Technical Details:</b> The context maintains:
/// - Dictionary mapping parameter names to FP32 master copies
/// - Dictionary mapping parameter names to FP16 working copies
/// - Synchronization methods to cast between precisions
/// - Integration with LossScaler for gradient management
/// </para>
/// </remarks>
public class MixedPrecisionContext : IDisposable
{
    /// <summary>
    /// Master weights stored in FP32 for precise updates.
    /// </summary>
    private readonly Dictionary<string, Vector<float>> _masterWeights;

    /// <summary>
    /// Working weights stored in FP16 for fast computation.
    /// </summary>
    private readonly Dictionary<string, Vector<Half>> _workingWeights;

    /// <summary>
    /// Loss scaler for gradient scaling and overflow detection.
    /// </summary>
    public LossScaler<float> LossScaler { get; }

    /// <summary>
    /// Per-tensor FP32 master snapshot pool — keyed by tensor reference
    /// identity so the SAME tensor across training steps reuses the same
    /// backing array. Allocated lazily on first
    /// <see cref="GetOrCreateFp32Snapshot"/> call for a given tensor and
    /// resized in place when shape changes. Drives the inline FP32
    /// round-trip in <c>NeuralNetworkBase.TrainWithTape</c>'s mixed-
    /// precision path so each step amortizes the snapshot cost across
    /// the entire training run instead of allocating a new
    /// <c>float[len]</c> per parameter per step (review #1362).
    ///
    /// <para><b>Concurrency:</b> <see cref="System.Collections.Concurrent.ConcurrentDictionary{TKey,TValue}"/>
    /// for thread-safe access from multi-threaded training loops (a feature
    /// gap vs. PyTorch / TensorFlow whose model state is single-thread-per-
    /// instance). The hot path uses <c>TryGetValue</c> + <c>AddOrUpdate</c>,
    /// which are lock-free reads. <b>DO NOT</b> call <c>.Count</c> or
    /// <c>.IsEmpty</c> on this dictionary — those acquire per-bucket
    /// Monitor locks and serialize parallel readers (root-caused 2026-04-22
    /// in <c>DeferredArrayMaterializer</c> where high-fanout parallel
    /// tensor forwards observed 44 s of unmanaged wait per 30 s of work).
    /// If a "pool populated?" indicator is ever needed, add a separate
    /// <c>Interlocked.Increment</c>-driven volatile counter alongside the
    /// dictionary, following the <c>DeferredArrayMaterializer._pendingCount</c>
    /// pattern.</para>
    /// </summary>
    private readonly System.Collections.Concurrent.ConcurrentDictionary<AiDotNet.Tensors.LinearAlgebra.Tensor<float>, float[]> _fp32SnapshotPool
        = new(AiDotNet.Helpers.TensorReferenceComparer<AiDotNet.Tensors.LinearAlgebra.Tensor<float>>.Instance);

    /// <summary>
    /// Get (or grow) the cached FP32 master snapshot for <paramref name="param"/>.
    /// </summary>
    /// <remarks>
    /// The pool's buffer is keyed by tensor REFERENCE identity, not by tensor
    /// content. NeuralNetworkBase's MP path reuses the same Tensor&lt;float&gt;
    /// references across training steps (the optimizer mutates in place), so
    /// the cached buffer is reused step-after-step on the steady state.
    /// Returns a <c>float[]</c> sized to at least <paramref name="param"/>.Length.
    /// Lock-free on the steady-state hot path (cached buffer hits); the
    /// allocate-or-grow path uses ConcurrentDictionary's atomic AddOrUpdate.
    /// </remarks>
    internal float[] GetOrCreateFp32Snapshot(AiDotNet.Tensors.LinearAlgebra.Tensor<float> param)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        int needed = param.Length;
        // Steady-state hot path: lock-free TryGetValue. Lands here every
        // training step on every trainable tensor once warm-up has run.
        if (_fp32SnapshotPool.TryGetValue(param, out var existing) && existing.Length >= needed)
            return existing;
        // Cold / shape-grew path: atomic add-or-update. The valueFactory /
        // updateValueFactory closures may run more than once under contention
        // (ConcurrentDictionary contract), but the result is idempotent — any
        // returned array of length >= needed is acceptable since the next
        // hot-path TryGetValue will return the winning entry.
        return _fp32SnapshotPool.AddOrUpdate(
            param,
            addValueFactory: t => new float[t.Length],
            updateValueFactory: (t, old) => old.Length >= t.Length ? old : new float[t.Length]);
    }

    /// <summary>
    /// Drop the FP32 snapshot pool. Called on Dispose; callers can also
    /// invoke explicitly to free the snapshots between training campaigns.
    /// </summary>
    internal void ClearFp32SnapshotPool() => _fp32SnapshotPool.Clear();

    /// <summary>
    /// Configuration for mixed-precision training.
    /// </summary>
    public MixedPrecisionConfig Config { get; }

    /// <summary>
    /// Whether the context has been initialized with parameters.
    /// </summary>
    public bool IsInitialized { get; private set; }

    /// <summary>
    /// Number of parameters managed by this context.
    /// </summary>
    public long ParameterCount { get; private set; }

    /// <summary>
    /// Gets the names of all parameters being managed.
    /// </summary>
    public IReadOnlyCollection<string> ParameterNames => _masterWeights.Keys;

    /// <summary>
    /// Initializes a new mixed-precision training context.
    /// </summary>
    /// <param name="config">Configuration for mixed-precision training (optional, uses defaults if null).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create one context per neural network model.
    /// The context will manage all the parameter conversions automatically.
    /// </para>
    /// </remarks>
    public MixedPrecisionContext(MixedPrecisionConfig? config = null)
    {
        Config = config ?? new MixedPrecisionConfig();
        _masterWeights = new Dictionary<string, Vector<float>>();
        _workingWeights = new Dictionary<string, Vector<Half>>();

        LossScaler = new LossScaler<float>(
            initialScale: Config.InitialLossScale,
            dynamicScaling: Config.EnableDynamicScaling,
            growthInterval: Config.ScaleGrowthInterval,
            growthFactor: Config.ScaleGrowthFactor,
            backoffFactor: Config.ScaleBackoffFactor,
            minScale: Config.MinLossScale,
            maxScale: Config.MaxLossScale
        );

        IsInitialized = false;
        ParameterCount = 0;
    }

    /// <summary>
    /// Initializes the context with model parameters.
    /// </summary>
    /// <param name="parameters">The model parameters in FP32.</param>
    /// <param name="parameterName">Optional name for the parameters (default: "params").</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this once after creating your model to register
    /// the parameters with the mixed-precision context. The parameters you pass should be in FP32.
    /// </para>
    /// </remarks>
    public void Initialize(Vector<float> parameters, string parameterName = "params")
    {
        if (IsInitialized)
        {
            throw new InvalidOperationException("Context is already initialized. Call Reset() first if you want to re-initialize.");
        }

        _masterWeights[parameterName] = parameters.Clone() as Vector<float>
            ?? throw new InvalidOperationException("Failed to clone parameters.");
        ParameterCount += parameters.Length;
        IsInitialized = true;
    }

    /// <summary>
    /// Initializes the context with multiple named parameter groups.
    /// </summary>
    /// <param name="namedParameters">Dictionary mapping parameter names to parameter vectors.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when you want to manage multiple parameter groups separately,
    /// for example, different layers or different types of parameters (weights vs. biases).
    /// </para>
    /// </remarks>
    public void Initialize(Dictionary<string, Vector<float>> namedParameters)
    {
        if (IsInitialized)
        {
            throw new InvalidOperationException("Context is already initialized. Call Reset() first if you want to re-initialize.");
        }

        foreach (var kvp in namedParameters)
        {
            var name = kvp.Key;
            var parameters = kvp.Value;
            _masterWeights[name] = parameters.Clone() as Vector<float>
                ?? throw new InvalidOperationException($"Failed to clone parameters '{name}'.");
            ParameterCount += parameters.Length;
        }

        IsInitialized = true;
    }

    /// <summary>
    /// Converts master weights (FP32) to working weights (FP16) for forward pass.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this before each forward pass to sync the FP16 working weights
    /// from the FP32 master weights. This ensures the working weights reflect the latest parameter updates.
    /// </para>
    /// </remarks>
    public void CastWeightsToFP16()
    {
        if (!IsInitialized)
        {
            throw new InvalidOperationException("Context not initialized. Call Initialize() first.");
        }

        foreach (var kvp in _masterWeights)
        {
            var name = kvp.Key;
            var masterParams = kvp.Value;
            // Convert FP32 master to FP16 working
            var workingParams = new Vector<Half>(masterParams.Length);

            for (int i = 0; i < masterParams.Length; i++)
            {
                workingParams[i] = (Half)masterParams[i];
            }

            _workingWeights[name] = workingParams;
        }
    }

    /// <summary>
    /// Round-trips the master weights through BF16 (truncate-low-16-bits-of-FP32)
    /// to emulate a forward pass that would have used BF16 working weights.
    /// </summary>
    /// <param name="parameterName">Name of the parameter group (default: "params").</param>
    /// <returns>The BF16-round-tripped weights as an FP32 vector (BF16 has no native CLR type;
    /// the round-tripped vector contains FP32 values whose representation has had
    /// the low 16 mantissa bits cleared via round-to-nearest-even).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> BF16 is "Brain Float 16" — IEEE FP32 with the
    /// low 16 mantissa bits dropped. Same 8-bit exponent as FP32 (so the
    /// dynamic range matches), but only 7 bits of mantissa precision.
    /// </para>
    /// <para>The round-trip is implemented by clearing the low 16 bits of the
    /// IEEE FP32 representation with round-to-nearest-even on the truncated
    /// bits — this is the value any layer would have seen if its forward
    /// weights were materialized as BF16. We return an FP32 vector because
    /// .NET 8 does not ship a primitive BF16 type; the value semantics are
    /// identical to BF16 → FP32.
    /// </para>
    /// </remarks>
    public Vector<float> CastWeightsToBF16(string parameterName = "params")
    {
        if (!IsInitialized)
        {
            throw new InvalidOperationException("Context not initialized. Call Initialize() first.");
        }

        if (!_masterWeights.TryGetValue(parameterName, out var masterParams))
        {
            throw new KeyNotFoundException($"Parameter '{parameterName}' not found. Available parameters: {string.Join(", ", ParameterNames)}");
        }

        var result = new Vector<float>(masterParams.Length);
        for (int i = 0; i < masterParams.Length; i++)
        {
            result[i] = BitConverterHelper.Bf16RoundTrip(masterParams[i]);
        }
        return result;
    }

    /// <summary>
    /// Emulate BF16 → FP32 round-trip by clearing the low 16 bits of the
    /// FP32 mantissa (round-to-nearest-even).
    /// </summary>
    /// <remarks>
    /// BF16 IEEE format = upper 16 bits of FP32 (1 sign + 8 exponent + 7 mantissa).
    /// FP32 → BF16 drops the low 16 bits of the mantissa.
    /// BF16 → FP32 zero-extends those 16 bits.
    /// Net: "zero the low 16 mantissa bits with round-to-nearest-even on the
    /// dropped half".
    /// </remarks>
    // Removed: TruncateFloatToBF16RoundTrip — deduplicated to
    // BitConverterHelper.Bf16RoundTrip (single source of truth, review #1362).

    /// <summary>
    /// Returns true if <paramref name="masterWeight"/> carries low-mantissa
    /// bits that an FP16 / BF16 round-trip would have zeroed out — i.e. the
    /// value still holds full FP32 precision (the master copy was preserved).
    /// </summary>
    /// <remarks>
    /// Test-friendly verification API for the mixed-precision contract
    /// (issue #1354): master parameters live in FP32 and only the working
    /// copy is cast to FP16/BF16 each step. A correctly wired model
    /// retains low-13-mantissa-bit detail in the master state across
    /// optimizer steps; an incorrectly wired one would round-trip the
    /// master through FP16 and lose those bits. Consumers can call this
    /// after Train to assert the master copy is intact without having to
    /// touch BitConverterHelper directly (which is intentionally internal
    /// to keep low-level bit utilities off the public facade — issue
    /// #1354 review feedback). Returns false for exact zero (which has
    /// no mantissa bits to check by definition) so callers can filter
    /// trivial cases without a separate guard.
    /// </remarks>
    public static bool HasFullFP32Precision(float masterWeight)
    {
        if (masterWeight == 0f) return false;
        int bits = BitConverterHelper.SingleToInt32Bits(masterWeight);
        // FP16 cast-and-back zeroes ~13 low mantissa bits; BF16 zeroes 16.
        // Either round-trip clears the low 13 bits, so the low-13-bit mask
        // is the correct discriminator.
        return (bits & 0x00001FFF) != 0;
    }

    /// <summary>
    /// Gets the working weights (FP16) for a parameter group.
    /// </summary>
    /// <param name="parameterName">Name of the parameter group.</param>
    /// <returns>The working weights in FP16.</returns>
    public Vector<Half> GetWorkingWeights(string parameterName = "params")
    {
        if (!_workingWeights.TryGetValue(parameterName, out var workingWeights))
        {
            throw new KeyNotFoundException($"Parameter '{parameterName}' not found. Available parameters: {string.Join(", ", ParameterNames)}");
        }

        return workingWeights;
    }

    /// <summary>
    /// Gets the master weights (FP32) for a parameter group.
    /// </summary>
    /// <param name="parameterName">Name of the parameter group.</param>
    /// <returns>The master weights in FP32.</returns>
    public Vector<float> GetMasterWeights(string parameterName = "params")
    {
        if (!_masterWeights.TryGetValue(parameterName, out var masterWeights))
        {
            throw new KeyNotFoundException($"Parameter '{parameterName}' not found. Available parameters: {string.Join(", ", ParameterNames)}");
        }

        return masterWeights;
    }

    /// <summary>
    /// Updates master weights with FP32 gradients after unscaling.
    /// </summary>
    /// <param name="gradients">The unscaled gradients in FP32.</param>
    /// <param name="learningRate">Learning rate for the update.</param>
    /// <param name="parameterName">Name of the parameter group to update.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This applies the gradient descent update to the FP32 master weights.
    /// The formula is: weights = weights - learningRate * gradients
    /// </para>
    /// </remarks>
    public void UpdateMasterWeights(Vector<float> gradients, float learningRate, string parameterName = "params")
    {
        if (!_masterWeights.TryGetValue(parameterName, out var masterParams))
        {
            throw new KeyNotFoundException($"Parameter '{parameterName}' not found.");
        }

        if (masterParams.Length != gradients.Length)
        {
            throw new ArgumentException($"Gradient length ({gradients.Length}) does not match parameter length ({masterParams.Length}).");
        }

        // Simple SGD update: params -= learningRate * gradients
        for (int i = 0; i < masterParams.Length; i++)
        {
            masterParams[i] -= learningRate * gradients[i];
        }
    }

    /// <summary>
    /// Converts FP16 gradients to FP32, unscales them, and checks for overflow.
    /// </summary>
    /// <param name="gradientsHalf">The scaled gradients in FP16.</param>
    /// <param name="gradientsFloat">Output: unscaled gradients in FP32 (if no overflow).</param>
    /// <returns>True if gradients are valid and update should proceed; false if overflow detected.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a key method in the mixed-precision training loop.
    /// It performs three steps:
    /// 1. Converts FP16 gradients to FP32 (lossless)
    /// 2. Unscales the gradients (divides by loss scale)
    /// 3. Checks for NaN/infinity and adjusts loss scale if needed
    ///
    /// If this returns false, you should skip the parameter update for this iteration.
    /// </para>
    /// </remarks>
    public bool PrepareGradientsForUpdate(Vector<Half> gradientsHalf, out Vector<float> gradientsFloat)
    {
        // Step 1: Cast FP16 gradients to FP32
        gradientsFloat = new Vector<float>(gradientsHalf.Length);
        for (int i = 0; i < gradientsHalf.Length; i++)
        {
            gradientsFloat[i] = (float)gradientsHalf[i];
        }

        // Step 2: Unscale and check for overflow
        bool isValid = LossScaler.UnscaleGradientsAndCheck(gradientsFloat);

        return isValid;
    }

    /// <summary>
    /// Resets the context, clearing all weights and statistics.
    /// </summary>
    public void Reset()
    {
        _masterWeights.Clear();
        _workingWeights.Clear();
        LossScaler.Reset();
        IsInitialized = false;
        ParameterCount = 0;
    }

    /// <summary>
    /// Disposes of the context and releases resources.
    /// </summary>
    public void Dispose()
    {
        _masterWeights.Clear();
        _workingWeights.Clear();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Gets a summary of the context's current state.
    /// </summary>
    /// <returns>A string describing the current state.</returns>
    public override string ToString()
    {
        return $"MixedPrecisionContext: " +
               $"Initialized={IsInitialized}, " +
               $"Parameters={ParameterCount:N0}, " +
               $"Groups={_masterWeights.Count}, " +
               $"LossScale={LossScaler.Scale:F0}, " +
               $"OverflowRate={LossScaler.OverflowRate:P2}";
    }
}
