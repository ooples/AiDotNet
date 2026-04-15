namespace AiDotNet.Configuration;

/// <summary>
/// Configuration for JIT (Just-In-Time) compilation of model forward/backward passes.
/// </summary>
/// <remarks>
/// <para>
/// JIT compilation traces the model's computation graph on the first call and
/// replays the compiled plan on subsequent calls, eliminating virtual dispatch,
/// per-op allocation, and bounds-checking overhead. Typical speedup is 1.5-3x on
/// CPU and up to 10x on GPU for small batches where dispatch dominates.
/// </para>
/// <para>
/// Under the hood this binds to the <c>TensorCodecOptions</c> in the Tensors
/// package. The builder writes the options into the thread-local codec config
/// before the built model sees any work; <see cref="NeuralNetworks.NeuralNetworkBase{T}.Predict"/>
/// and the tape-based training path then route through their compiled variants
/// via <c>CompiledModelCache</c>.
/// </para>
/// <para><b>For Beginners:</b> JIT compilation is like building a shortcut for your
/// model. The first time you call <c>Predict</c> the library watches which math
/// operations happen in what order, compiles that pattern into a flat fast
/// pipeline, and from then on just replays the pipeline. Your model doesn't
/// change — only the plumbing around it gets faster.</para>
/// <para>
/// When to enable:
/// <list type="bullet">
/// <item>Production inference where Predict is called many times at the same shape.</item>
/// <item>Training loops where the forward+backward is invoked each iteration.</item>
/// <item>Diffusion/autoregressive generation where a sub-network runs tens of times per call.</item>
/// </list>
/// </para>
/// <para>
/// When to disable / when to keep <see cref="ThrowOnFailure"/> off:
/// <list type="bullet">
/// <item>Debugging a model whose forward path has non-Engine tensor accesses
/// (direct span writes, scalar control flow) — those bake at trace time and will
/// replay stale data. The default fallback to eager execution hides this.</item>
/// <item>Tiny models where the compile cost isn't amortized (rare — overhead is
/// one traced forward).</item>
/// </list>
/// </para>
/// <para>
/// Example YAML (binds through <c>YamlConfigApplier</c>):
/// <code>
/// jitCompilation:
///   enabled: true
///   throwOnFailure: false
///   enableDataflowFusion: true
///   enableAttentionFusion: true
/// </code>
/// </para>
/// <para>
/// Example code:
/// <code>
/// var result = await new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(myModel)
///     .ConfigureJitCompilation()            // enable with library defaults
///     .BuildAsync();
/// </code>
/// </para>
/// </remarks>
public sealed class JitCompilationConfig
{
    /// <summary>Default config — compilation enabled, failures fall back silently to eager.</summary>
    public static JitCompilationConfig Default => new()
    {
        Enabled = true,
        ThrowOnFailure = false
    };

    /// <summary>Aggressive config — all fusion and constant-folding passes on. Good for benchmarking.</summary>
    public static JitCompilationConfig Aggressive => new()
    {
        Enabled = true,
        ThrowOnFailure = false,
        EnableDataflowFusion = true,
        EnableAlgebraicBackward = true,
        EnableConvBnFusion = true,
        EnableAttentionFusion = true,
        EnablePointwiseFusion = true,
        EnableConstantFolding = true,
        EnableForwardCSE = true,
        EnableBlasBatch = true
    };

    /// <summary>Explicitly disabled — compiled paths short-circuit to eager. Useful for A/B diffs.</summary>
    public static JitCompilationConfig Disabled => new()
    {
        Enabled = false
    };

    /// <summary>
    /// Master switch. When <c>false</c>, all compilation paths disable themselves and
    /// execution falls through to the eager path (<c>TensorCodecOptions.EnableCompilation = false</c>).
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// When <c>true</c>, a compilation failure in the builder's <c>JitCompiledFunction</c>
    /// wrapper (the one populated by <c>AiModelBuilder.BuildCompiledPredictFunction</c>)
    /// propagates as an exception instead of silently falling back to the eager
    /// Predict path. Use in tests to catch regressions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Scope: this flag is currently consulted only by the builder-level wrapper
    /// — the lower-level <c>NeuralNetworkBase{T}.PredictCompiled</c> and
    /// <c>CompiledTapeTrainingStep{T}.Step</c> have their own catch-all fallbacks
    /// (with Trace warnings logged on the wrapper). Threading strict-mode through
    /// those internal helpers is tracked as a follow-up; for tests, exercising
    /// the wrapper path (built via <c>AiModelBuilder</c>) is sufficient.
    /// </para>
    /// <para>
    /// Leave <c>false</c> in production so a misbehaving op doesn't take down
    /// inference — fallback runs eager + emits a <c>Trace.TraceWarning</c> so
    /// the regression is observable in telemetry.
    /// </para>
    /// </remarks>
    public bool ThrowOnFailure { get; set; }

    /// <summary>
    /// Phase B: Fuse consecutive linear layers into a single multi-layer kernel
    /// that keeps data in registers / L1 across layer boundaries.
    /// </summary>
    public bool EnableDataflowFusion { get; set; } = true;

    /// <summary>
    /// Phase C: Symbolically simplify the backward graph at compile time
    /// (CSE, double-transpose elimination, associative regrouping).
    /// </summary>
    public bool EnableAlgebraicBackward { get; set; } = true;

    /// <summary>
    /// Phase A: SVD-factorize frozen weight matrices for faster inference.
    /// Opt-in because it introduces bounded approximation error.
    /// </summary>
    public bool EnableSpectralDecomposition { get; set; }

    /// <summary>
    /// Maximum approximation error per element for spectral decomposition
    /// (used as <c>energyThreshold = 1.0 - tolerance</c> for SVD rank selection).
    /// </summary>
    public float SpectralErrorTolerance { get; set; } = 1e-5f;

    /// <summary>Maximum hidden dimension for dataflow fusion L1 residency.</summary>
    public int DataflowFusionMaxHidden { get; set; } = 512;

    /// <summary>Phase 4.1: Fold BatchNorm into Conv2D weights at compile time.</summary>
    public bool EnableConvBnFusion { get; set; } = true;

    /// <summary>Phase 4.2: Fuse attention Q@K^T → Softmax → V patterns.</summary>
    public bool EnableAttentionFusion { get; set; } = true;

    /// <summary>Phase 4.3: Merge consecutive pointwise ops into fewer dispatch steps.</summary>
    public bool EnablePointwiseFusion { get; set; } = true;

    /// <summary>Phase 4.5: Precompute static subgraphs at compile time.</summary>
    public bool EnableConstantFolding { get; set; } = true;

    /// <summary>Phase 6.2: Deduplicate identical computations across layers.</summary>
    public bool EnableForwardCSE { get; set; } = true;

    /// <summary>Phase 7.1: Group independent MatMuls into batched calls.</summary>
    public bool EnableBlasBatch { get; set; } = true;

    /// <summary>
    /// Phase 7.3: Mixed precision (fp16 forward, fp32 backward). Opt-in because it
    /// can introduce small numerical drift vs pure fp32.
    /// </summary>
    public bool EnableMixedPrecisionCompilation { get; set; }

    /// <summary>
    /// Validates the config and throws if settings are inconsistent.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when values are out of range.</exception>
    public void Validate()
    {
        // NaN passes both `< 0` and `>= 1` comparisons (NaN is unordered with
        // every value), so the explicit IsNaN check is required to reject it.
        // Without this guard a NaN tolerance would silently become the SVD
        // energy threshold and corrupt every spectral compilation downstream.
        if (float.IsNaN(SpectralErrorTolerance)
            || SpectralErrorTolerance < 0f
            || SpectralErrorTolerance >= 1f)
        {
            throw new InvalidOperationException(
                $"SpectralErrorTolerance must be in [0, 1) and not NaN. Got: {SpectralErrorTolerance}");
        }

        if (DataflowFusionMaxHidden <= 0)
        {
            throw new InvalidOperationException(
                $"DataflowFusionMaxHidden must be positive. Got: {DataflowFusionMaxHidden}");
        }
    }

    /// <summary>
    /// Projects this config onto the Tensors-package <c>TensorCodecOptions</c> and
    /// installs it as the current thread-local config.
    /// </summary>
    /// <remarks>
    /// Called by <see cref="AiModelBuilder{T, TInput, TOutput}"/>.Build() so that
    /// every compiled path the built model touches (<c>CompiledModelCache</c>,
    /// <c>AutoTracer</c>, <c>CompiledTapeTrainingStep</c>) reads the flags
    /// configured here.
    /// </remarks>
    public void ApplyToTensorCodec()
    {
        // Validate at the boundary — we're about to install these values into
        // thread-static codec options where a bad value (negative
        // DataflowFusionMaxHidden, out-of-range SpectralErrorTolerance) would
        // silently corrupt compilation paths instead of failing fast. Running
        // Validate() on every Apply is cheap (simple range checks) and catches
        // deserialized or mutated config before any plan is compiled against it.
        Validate();

        var opts = new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions
        {
            EnableCompilation = Enabled,
            EnableDataflowFusion = EnableDataflowFusion,
            EnableAlgebraicBackward = EnableAlgebraicBackward,
            EnableSpectralDecomposition = EnableSpectralDecomposition,
            SpectralErrorTolerance = SpectralErrorTolerance,
            DataflowFusionMaxHidden = DataflowFusionMaxHidden,
            EnableConvBnFusion = EnableConvBnFusion,
            EnableAttentionFusion = EnableAttentionFusion,
            EnablePointwiseFusion = EnablePointwiseFusion,
            EnableConstantFolding = EnableConstantFolding,
            EnableForwardCSE = EnableForwardCSE,
            EnableBlasBatch = EnableBlasBatch,
            EnableMixedPrecision = EnableMixedPrecisionCompilation
        };
        AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(opts);
    }
}
