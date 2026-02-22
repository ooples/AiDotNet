namespace AiDotNet.Onnx;

/// <summary>
/// Configuration options for loading and running ONNX models.
/// </summary>
/// <remarks>
/// <para>
/// This class provides comprehensive configuration for ONNX model inference,
/// including execution provider selection, memory optimization, and threading.
/// </para>
/// <para><b>For Beginners:</b> Use defaults for most cases:
/// <code>
/// var options = new OnnxModelOptions();
/// var model = new OnnxModel&lt;float&gt;("model.onnx", options);
/// </code>
/// For GPU acceleration:
/// <code>
/// var options = new OnnxModelOptions { ExecutionProvider = OnnxExecutionProvider.Cuda };
/// </code>
/// </para>
/// </remarks>
public class OnnxModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public OnnxModelOptions() { }

    /// <summary>
    /// Gets or sets the preferred execution provider.
    /// Default is Auto, which selects the best available provider.
    /// </summary>
    public OnnxExecutionProvider ExecutionProvider { get; set; } = OnnxExecutionProvider.Auto;

    /// <summary>
    /// Gets or sets fallback execution providers if the primary fails.
    /// Default fallback order: CUDA → DirectML → CPU
    /// </summary>
    public List<OnnxExecutionProvider> FallbackProviders { get; set; } =
    [
        OnnxExecutionProvider.Cuda,
        OnnxExecutionProvider.DirectML,
        OnnxExecutionProvider.Cpu
    ];

    /// <summary>
    /// Gets or sets the GPU device ID for CUDA/TensorRT/DirectML providers.
    /// Default is 0 (first GPU).
    /// </summary>
    public int GpuDeviceId { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to enable memory pattern optimization.
    /// This can reduce memory allocations during inference.
    /// </summary>
    public bool EnableMemoryPattern { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable memory arena.
    /// This pre-allocates memory to reduce allocation overhead.
    /// </summary>
    public bool EnableMemoryArena { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of threads for CPU execution.
    /// 0 means use the default (typically number of CPU cores).
    /// </summary>
    public int IntraOpNumThreads { get; set; } = 0;

    /// <summary>
    /// Gets or sets the number of threads for parallel operations.
    /// 0 means use the default (typically number of CPU cores).
    /// </summary>
    public int InterOpNumThreads { get; set; } = 0;

    /// <summary>
    /// Gets or sets the graph optimization level.
    /// Higher levels may increase load time but improve inference speed.
    /// </summary>
    public GraphOptimizationLevel OptimizationLevel { get; set; } = GraphOptimizationLevel.All;

    /// <summary>
    /// Gets or sets whether to enable profiling for performance analysis.
    /// </summary>
    public bool EnableProfiling { get; set; } = false;

    /// <summary>
    /// Gets or sets the path for saving profiling output.
    /// Only used if EnableProfiling is true.
    /// </summary>
    public string? ProfileOutputPath { get; set; }

    /// <summary>
    /// Gets or sets custom session options as key-value pairs.
    /// </summary>
    public Dictionary<string, string> CustomOptions { get; set; } = [];

    /// <summary>
    /// Gets or sets the log severity level for ONNX Runtime.
    /// </summary>
    public OnnxLogLevel LogLevel { get; set; } = OnnxLogLevel.Warning;

    /// <summary>
    /// Gets or sets whether to automatically warm up the model after loading.
    /// Warming up runs a single inference to initialize lazy resources.
    /// </summary>
    public bool AutoWarmUp { get; set; } = false;

    /// <summary>
    /// Gets or sets the CUDA memory limit in bytes (0 = no limit).
    /// Only applies to CUDA execution provider.
    /// </summary>
    public long CudaMemoryLimit { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to use CUDA memory arena for better performance.
    /// </summary>
    public bool CudaUseArena { get; set; } = true;

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OnnxModelOptions(OnnxModelOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ExecutionProvider = other.ExecutionProvider;
        FallbackProviders = new List<OnnxExecutionProvider>(other.FallbackProviders);
        GpuDeviceId = other.GpuDeviceId;
        EnableMemoryPattern = other.EnableMemoryPattern;
        EnableMemoryArena = other.EnableMemoryArena;
        IntraOpNumThreads = other.IntraOpNumThreads;
        InterOpNumThreads = other.InterOpNumThreads;
        OptimizationLevel = other.OptimizationLevel;
        EnableProfiling = other.EnableProfiling;
        ProfileOutputPath = other.ProfileOutputPath;
        CustomOptions = new Dictionary<string, string>(other.CustomOptions);
        LogLevel = other.LogLevel;
        AutoWarmUp = other.AutoWarmUp;
        CudaMemoryLimit = other.CudaMemoryLimit;
        CudaUseArena = other.CudaUseArena;
    }

    /// <summary>
    /// Creates default options for CPU execution.
    /// </summary>
    public static OnnxModelOptions ForCpu(int? threads = null) => new()
    {
        ExecutionProvider = OnnxExecutionProvider.Cpu,
        IntraOpNumThreads = threads ?? Environment.ProcessorCount,
        FallbackProviders = []
    };

    /// <summary>
    /// Creates default options for CUDA GPU execution.
    /// </summary>
    public static OnnxModelOptions ForCuda(int deviceId = 0) => new()
    {
        ExecutionProvider = OnnxExecutionProvider.Cuda,
        GpuDeviceId = deviceId,
        FallbackProviders = [OnnxExecutionProvider.Cpu]
    };

    /// <summary>
    /// Creates default options for DirectML GPU execution (Windows).
    /// </summary>
    public static OnnxModelOptions ForDirectML(int deviceId = 0) => new()
    {
        ExecutionProvider = OnnxExecutionProvider.DirectML,
        GpuDeviceId = deviceId,
        FallbackProviders = [OnnxExecutionProvider.Cpu]
    };

    /// <summary>
    /// Creates default options for TensorRT execution (NVIDIA optimized).
    /// </summary>
    public static OnnxModelOptions ForTensorRT(int deviceId = 0) => new()
    {
        ExecutionProvider = OnnxExecutionProvider.TensorRT,
        GpuDeviceId = deviceId,
        FallbackProviders = [OnnxExecutionProvider.Cuda, OnnxExecutionProvider.Cpu]
    };
}

/// <summary>
/// Graph optimization levels for ONNX Runtime.
/// </summary>
public enum GraphOptimizationLevel
{
    /// <summary>
    /// No optimizations.
    /// </summary>
    None = 0,

    /// <summary>
    /// Basic optimizations (constant folding, redundant node elimination).
    /// </summary>
    Basic = 1,

    /// <summary>
    /// Extended optimizations (node fusion, gemm optimization).
    /// </summary>
    Extended = 2,

    /// <summary>
    /// All optimizations enabled.
    /// </summary>
    All = 99
}

/// <summary>
/// Log severity levels for ONNX Runtime.
/// </summary>
public enum OnnxLogLevel
{
    /// <summary>
    /// Log all messages.
    /// </summary>
    Verbose = 0,

    /// <summary>
    /// Log info and above.
    /// </summary>
    Info = 1,

    /// <summary>
    /// Log warnings and above.
    /// </summary>
    Warning = 2,

    /// <summary>
    /// Log errors and above.
    /// </summary>
    Error = 3,

    /// <summary>
    /// Log only fatal errors.
    /// </summary>
    Fatal = 4
}
