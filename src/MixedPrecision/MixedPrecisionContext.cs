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
    public int ParameterCount { get; private set; }

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
    /// Gets the working weights (FP16) for a parameter group.
    /// </summary>
    /// <param name="parameterName">Name of the parameter group.</param>
    /// <returns>The working weights in FP16.</returns>
    public Vector<Half> GetWorkingWeights(string parameterName = "params")
    {
        if (!_workingWeights.ContainsKey(parameterName))
        {
            throw new KeyNotFoundException($"Parameter '{parameterName}' not found. Available parameters: {string.Join(", ", ParameterNames)}");
        }

        return _workingWeights[parameterName];
    }

    /// <summary>
    /// Gets the master weights (FP32) for a parameter group.
    /// </summary>
    /// <param name="parameterName">Name of the parameter group.</param>
    /// <returns>The master weights in FP32.</returns>
    public Vector<float> GetMasterWeights(string parameterName = "params")
    {
        if (!_masterWeights.ContainsKey(parameterName))
        {
            throw new KeyNotFoundException($"Parameter '{parameterName}' not found. Available parameters: {string.Join(", ", ParameterNames)}");
        }

        return _masterWeights[parameterName];
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
        if (!_masterWeights.ContainsKey(parameterName))
        {
            throw new KeyNotFoundException($"Parameter '{parameterName}' not found.");
        }

        var masterParams = _masterWeights[parameterName];

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
