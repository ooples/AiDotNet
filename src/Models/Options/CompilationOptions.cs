using System;
using System.IO;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configures compiler graph fusion, multi-bucket CUDA Graph replay, derived-weight inference caching, precision
/// tiers, persistent autotuning, and self-contained Direct-PTX kernel selection for
/// <c>AiModelBuilder.ConfigureCompilation</c>.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Default"/> (<see cref="ExecutionMode.AutoCompiled"/>) is enabled automatically on every
/// <c>AiModelBuilder</c> instance so models benefit from cross-layer graph fusion, multi-bucket CUDA Graph replay on
/// <c>DirectGpu</c> backends, GPU-resident optimizer scalar buffers, device-predicated dynamic routing, and
/// version-invalidated <c>LayerBase&lt;T&gt;</c> derived-weight caching out of the box with zero manual boilerplate.
/// </para>
/// <para>
/// Pass <see cref="Eager"/> to <c>ConfigureCompilation</c> when stepping through layer operations interactively in a
/// C# debugger, or pass <see cref="PureDirectPtx"/> to disable external vendor libraries (<c>cuBLASLt</c>,
/// <c>cuDNN</c>, <c>OpenBLAS</c>, <c>CLBlast</c>) and run exclusively on self-contained native/Direct-PTX kernels.
/// </para>
/// </remarks>
public class CompilationOptions : ModelOptions
{
    private static readonly int[] DefaultSequenceBuckets = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    private static readonly int[] DefaultBatchBuckets = [1, 2, 4, 8, 16, 32, 64, 128, 256];

    /// <summary>Initializes a new instance with default out-of-the-box acceleration settings.</summary>
    public CompilationOptions()
    {
    }

    /// <summary>Initializes a new instance by copying all settings from <paramref name="other"/>.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
    public CompilationOptions(CompilationOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Mode = other.Mode;
        PrecisionTier = other.PrecisionTier;
        EnableHybridFusionTileIr = other.EnableHybridFusionTileIr;
        EnableCudaGraphTrees = other.EnableCudaGraphTrees;
        EnableMultiBucketShapeRouting = other.EnableMultiBucketShapeRouting;
        SequenceLengthBuckets = other.SequenceLengthBuckets is { Length: > 0 } seq
            ? (int[])seq.Clone()
            : (int[])DefaultSequenceBuckets.Clone();
        BatchSizeBuckets = other.BatchSizeBuckets is { Length: > 0 } batch
            ? (int[])batch.Clone()
            : (int[])DefaultBatchBuckets.Clone();
        EnableSegmentedBlockGraphs = other.EnableSegmentedBlockGraphs;
        EnableGpuResidentOptimizerScalars = other.EnableGpuResidentOptimizerScalars;
        EnableDevicePredicatedDynamicRouting = other.EnableDevicePredicatedDynamicRouting;
        EnableDerivedWeightCaching = other.EnableDerivedWeightCaching;
        AllowExternalVendorLibraries = other.AllowExternalVendorLibraries;
        PreferSelfContainedDirectPtx = other.PreferSelfContainedDirectPtx;
        EnableMaxAutotune = other.EnableMaxAutotune;
        AutotuneCacheDirectory = other.AutotuneCacheDirectory;
        CudaGraphWarmupIterations = other.CudaGraphWarmupIterations;
        PersistCompiledProgram = other.PersistCompiledProgram;
    }

    /// <summary>
    /// Default out-of-the-box configuration (<see cref="ExecutionMode.AutoCompiled"/>) enabled automatically on
    /// <c>AiModelBuilder</c>.
    /// </summary>
    public static CompilationOptions Default => new();

    /// <summary>
    /// Eager step-by-step execution preset for interactive debugging (disables graph compilation and CUDA Graph
    /// capture while preserving numerical semantics).
    /// </summary>
    public static CompilationOptions Eager => new()
    {
        Mode = ExecutionMode.Eager,
        EnableHybridFusionTileIr = false,
        EnableCudaGraphTrees = false,
        EnableSegmentedBlockGraphs = false,
        EnableMaxAutotune = false,
    };

    /// <summary>
    /// Maximum-throughput autotuning preset enabling cross-layer tile fusion, multi-bucket CUDA Graph replay,
    /// GPU-resident optimizer scalars, and persistent disk-cached kernel autotuning.
    /// </summary>
    public static CompilationOptions MaxAutotune => new()
    {
        Mode = ExecutionMode.AutoCompiled,
        EnableHybridFusionTileIr = true,
        EnableCudaGraphTrees = true,
        EnableMultiBucketShapeRouting = true,
        EnableSegmentedBlockGraphs = true,
        EnableGpuResidentOptimizerScalars = true,
        EnableDevicePredicatedDynamicRouting = true,
        EnableDerivedWeightCaching = true,
        EnableMaxAutotune = true,
    };

    /// <summary>
    /// Pure self-contained Direct-PTX preset that disables all external vendor libraries (<c>cuBLASLt</c>,
    /// <c>cuDNN</c>, <c>OpenBLAS</c>, <c>CLBlast</c>) and executes exclusively on native Direct-PTX / TileIR kernels.
    /// </summary>
    public static CompilationOptions PureDirectPtx => new()
    {
        Mode = ExecutionMode.AutoCompiled,
        AllowExternalVendorLibraries = false,
        PreferSelfContainedDirectPtx = true,
        EnableHybridFusionTileIr = true,
        EnableCudaGraphTrees = true,
        EnableMultiBucketShapeRouting = true,
        EnableSegmentedBlockGraphs = true,
        EnableGpuResidentOptimizerScalars = true,
        EnableDevicePredicatedDynamicRouting = true,
        EnableDerivedWeightCaching = true,
        EnableMaxAutotune = true,
    };

    /// <summary>Gets or sets the execution mode (<see cref="ExecutionMode.AutoCompiled"/> by default).</summary>
    public ExecutionMode Mode { get; set; } = ExecutionMode.AutoCompiled;

    /// <summary>Gets or sets the numerical precision tier (<see cref="ComputePrecisionTier.FP32"/> by default).</summary>
    public ComputePrecisionTier PrecisionTier { get; set; } = ComputePrecisionTier.FP32;

    /// <summary>Gets or sets whether cross-layer Hybrid Fusion / TileIR passes are enabled.</summary>
    public bool EnableHybridFusionTileIr { get; set; } = true;

    /// <summary>Gets or sets whether CUDA Graph tree capture and replay are enabled when a DirectGpu backend is active.</summary>
    public bool EnableCudaGraphTrees { get; set; } = true;

    /// <summary>
    /// Gets or sets whether variable sequence lengths and batch sizes route to pre-compiled <c>SizeClass</c> shape
    /// buckets sharing one memory pool and optimizer state.
    /// </summary>
    public bool EnableMultiBucketShapeRouting { get; set; } = true;

    /// <summary>Gets or sets the sorted positive sequence-length buckets used for multi-bucket graph routing.</summary>
    public int[] SequenceLengthBuckets { get; set; } = (int[])DefaultSequenceBuckets.Clone();

    /// <summary>Gets or sets the sorted positive batch-size buckets used for multi-bucket graph routing.</summary>
    public int[] BatchSizeBuckets { get; set; } = (int[])DefaultBatchBuckets.Clone();

    /// <summary>
    /// Gets or sets whether gradient-checkpointed segments and <c>CreditRuleBase&lt;T&gt;</c> local-credit blocks
    /// reuse compiled per-segment graphs instead of rebuilding eager tapes every step.
    /// </summary>
    public bool EnableSegmentedBlockGraphs { get; set; } = true;

    /// <summary>
    /// Gets or sets whether <c>GradientBasedOptimizerBase</c> maintains learning-rate, step-count, weight-decay,
    /// gradient-clip, and NaN/Inf anomaly scalars in a fixed-address GPU buffer so learning-rate schedules never
    /// invalidate captured full-step CUDA Graphs.
    /// </summary>
    public bool EnableGpuResidentOptimizerScalars { get; set; } = true;

    /// <summary>
    /// Gets or sets whether sparse Mixture-of-Experts Top-K routing and dynamic halting execute via static-shape
    /// device predication without host <c>.AsSpan()</c> synchronization when running on GPU or inside compiled graphs.
    /// </summary>
    public bool EnableDevicePredicatedDynamicRouting { get; set; } = true;

    /// <summary>
    /// Gets or sets whether <c>LayerBase&lt;T&gt;</c> caches version-invalidated derived parameter tensors (such as
    /// Compact WY Householder operators, weight/spectral-normalized convolution kernels, and fused QKV projections)
    /// during non-differentiable inference passes.
    /// </summary>
    public bool EnableDerivedWeightCaching { get; set; } = true;

    /// <summary>
    /// Gets or sets whether external vendor libraries (<c>cuBLASLt</c>, <c>cuDNN</c>, <c>OpenBLAS</c>,
    /// <c>CLBlast</c>) are permitted as fallback backends. Set to <c>false</c> for pure self-contained Direct-PTX
    /// execution.
    /// </summary>
    public bool AllowExternalVendorLibraries { get; set; } = true;

    /// <summary>
    /// Gets or sets whether self-contained Direct-PTX kernels are preferred ahead of external vendor libraries even
    /// when <see cref="AllowExternalVendorLibraries"/> is <c>true</c>.
    /// </summary>
    public bool PreferSelfContainedDirectPtx { get; set; }

    /// <summary>Gets or sets whether persistent kernel tile autotuning (<c>MaxAutotune</c>) is enabled.</summary>
    public bool EnableMaxAutotune { get; set; } = true;

    /// <summary>
    /// Gets or sets the directory used to persist hardware-specific autotune profiles across runs, or <c>null</c> to
    /// use the default user-local cache directory.
    /// </summary>
    public string? AutotuneCacheDirectory { get; set; }

    /// <summary>Gets or sets the number of warmup executions performed before capturing a static-shape CUDA Graph.</summary>
    public int CudaGraphWarmupIterations { get; set; } = 2;

    /// <summary>Gets or sets whether serialized models embed portable compiled program metadata.</summary>
    public bool PersistCompiledProgram { get; set; } = true;

    /// <summary>
    /// Resolves <paramref name="sequenceLength"/> up to the smallest configured sequence-length bucket that fits it,
    /// or the next power of two when above the largest explicit bucket.
    /// </summary>
    public int ResolveSequenceBucket(int sequenceLength)
    {
        if (sequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), sequenceLength, "Sequence length must be positive.");
        if (!EnableMultiBucketShapeRouting || SequenceLengthBuckets is not { Length: > 0 })
            return sequenceLength;
        return ResolveBucketCore(sequenceLength, SequenceLengthBuckets);
    }

    /// <summary>
    /// Resolves <paramref name="batchSize"/> up to the smallest configured batch-size bucket that fits it, or the
    /// next power of two when above the largest explicit bucket.
    /// </summary>
    public int ResolveBatchBucket(int batchSize)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        if (!EnableMultiBucketShapeRouting || BatchSizeBuckets is not { Length: > 0 })
            return batchSize;
        return ResolveBucketCore(batchSize, BatchSizeBuckets);
    }

    /// <summary>
    /// Resolves the effective persistent autotune cache directory on disk.
    /// </summary>
    public string ResolveEffectiveAutotuneCacheDirectory()
    {
        if (!string.IsNullOrWhiteSpace(AutotuneCacheDirectory))
            return AutotuneCacheDirectory ?? string.Empty;

        string baseDir = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(baseDir))
            baseDir = Path.GetTempPath();
        return Path.Combine(baseDir, "AiDotNet", "AutotuneCache");
    }

    /// <summary>Validates option values and throws if any configuration property is invalid.</summary>
    public void Validate()
    {
        if (!Enum.IsDefined(typeof(ExecutionMode), Mode))
        {
            throw new ArgumentOutOfRangeException(nameof(Mode), Mode, "Unsupported ExecutionMode value.");
        }

        if (!Enum.IsDefined(typeof(ComputePrecisionTier), PrecisionTier))
        {
            throw new ArgumentOutOfRangeException(nameof(PrecisionTier), PrecisionTier, "Unsupported ComputePrecisionTier value.");
        }

        if (CudaGraphWarmupIterations < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(CudaGraphWarmupIterations),
                CudaGraphWarmupIterations,
                "CudaGraphWarmupIterations must be non-negative.");
        }

        if (EnableMultiBucketShapeRouting)
        {
            ValidateBuckets(SequenceLengthBuckets, nameof(SequenceLengthBuckets));
            ValidateBuckets(BatchSizeBuckets, nameof(BatchSizeBuckets));
        }

        if (AutotuneCacheDirectory is not null && string.IsNullOrWhiteSpace(AutotuneCacheDirectory))
        {
            throw new ArgumentException("AutotuneCacheDirectory cannot be empty or whitespace when specified.", nameof(AutotuneCacheDirectory));
        }
    }

    private static void ValidateBuckets(int[]? buckets, string paramName)
    {
        if (buckets is null || buckets.Length == 0)
        {
            throw new ArgumentException($"{paramName} must contain at least one positive bucket size when EnableMultiBucketShapeRouting is true.", paramName);
        }

        int prev = 0;
        for (int i = 0; i < buckets.Length; i++)
        {
            int b = buckets[i];
            if (b <= 0)
            {
                throw new ArgumentOutOfRangeException(paramName, b, $"{paramName}[{i}] must be positive.");
            }
            if (b <= prev)
            {
                throw new ArgumentException($"{paramName} must be strictly increasing (got {b} after {prev} at index {i}).", paramName);
            }
            prev = b;
        }
    }

    private static int ResolveBucketCore(int value, int[] buckets)
    {
        for (int i = 0; i < buckets.Length; i++)
        {
            if (buckets[i] >= value)
                return buckets[i];
        }

        int pow2 = 1;
        while (pow2 < value && pow2 < (1 << 30))
        {
            pow2 <<= 1;
        }
        return Math.Max(value, pow2);
    }
}
