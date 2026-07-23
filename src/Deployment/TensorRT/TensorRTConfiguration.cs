namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// Configuration for TensorRT model conversion and execution.
/// </summary>
public class TensorRTConfiguration
{
    /// <summary>
    /// Gets or sets the maximum batch size for inference (default: 1).
    /// </summary>
    public int MaxBatchSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets the maximum workspace size in bytes for TensorRT (default: 1GB).
    /// </summary>
    public long MaxWorkspaceSize { get; set; } = 1L << 30; // 1 GB

    /// <summary>
    /// Gets or sets the inference precision mode (default: FP32).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls the numeric precision for inference:
    /// - FP32: Full precision, most accurate, slowest
    /// - FP16: Half precision, good balance of speed and accuracy
    /// - INT8: Quantized, fastest, requires calibration data
    /// </para>
    /// </remarks>
    public TensorRTPrecision Precision { get; set; } = TensorRTPrecision.FP32;

    /// <summary>
    /// Gets or sets whether to enable Deep Learning Accelerator (DLA) on Jetson devices.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DLA is a dedicated AI accelerator on NVIDIA Jetson devices.
    /// Enables lower power consumption but may not support all operations.
    /// </para>
    /// </remarks>
    public bool EnableDLA { get; set; } = false;

    /// <summary>
    /// Gets or sets the DLA core to use when EnableDLA is true (-1 = disabled, 0+ = specific core).
    /// </summary>
    public int DLACore { get; set; } = -1;

    /// <summary>
    /// Gets or sets whether to enable strict type constraints (default: false).
    /// </summary>
    public bool StrictTypeConstraints { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to enable dynamic shapes (default: false).
    /// </summary>
    public bool EnableDynamicShapes { get; set; } = false;

    /// <summary>
    /// Gets or sets the GPU device ID to use (default: 0).
    /// </summary>
    public int DeviceId { get; set; } = 0;

    /// <summary>
    /// Gets or sets optimization profiles for dynamic shapes.
    /// </summary>
    public List<OptimizationProfileConfig> OptimizationProfiles { get; set; } = new();

    /// <summary>
    /// Gets or sets custom plugin library paths.
    /// </summary>
    public List<string> CustomPluginPaths { get; set; } = new();

    /// <summary>
    /// Gets or sets the path to calibration data for INT8 quantization.
    /// </summary>
    public string? CalibrationDataPath { get; set; }

    /// <summary>
    /// Gets or sets whether to use multi-stream execution (default: false).
    /// </summary>
    public bool EnableMultiStream { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of streams for multi-stream execution (default: 2).
    /// </summary>
    public int NumStreams { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to enable CUDA graph capture (default: false).
    /// </summary>
    public bool EnableCudaGraphs { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to cleanup intermediate files (ONNX, etc.) (default: true).
    /// </summary>
    public bool CleanupIntermediateFiles { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable profiling (default: false).
    /// </summary>
    public bool EnableProfiling { get; set; } = false;

    /// <summary>
    /// Gets or sets the engine cache path for faster reloading.
    /// </summary>
    public string? EngineCachePath { get; set; }

    /// <summary>
    /// Gets or sets the builder optimization level (0-5, higher is more optimization).
    /// </summary>
    public int BuilderOptimizationLevel { get; set; } = 3;

    /// <summary>
    /// Creates a configuration optimized for maximum performance.
    /// </summary>
    public static TensorRTConfiguration ForMaxPerformance()
    {
        return new TensorRTConfiguration
        {
            MaxBatchSize = 32,
            MaxWorkspaceSize = 4L << 30, // 4 GB
            Precision = TensorRTPrecision.FP16,
            EnableMultiStream = true,
            NumStreams = 4,
            EnableCudaGraphs = true,
            BuilderOptimizationLevel = 5
        };
    }

    /// <summary>
    /// Creates a configuration optimized for low latency (batch size 1).
    /// </summary>
    public static TensorRTConfiguration ForLowLatency()
    {
        return new TensorRTConfiguration
        {
            MaxBatchSize = 1,
            MaxWorkspaceSize = 1L << 30, // 1 GB
            Precision = TensorRTPrecision.FP16,
            EnableMultiStream = false,
            EnableCudaGraphs = true,
            BuilderOptimizationLevel = 5
        };
    }

    /// <summary>
    /// Creates a configuration optimized for high throughput.
    /// </summary>
    /// <param name="batchSize">Maximum batch size</param>
    /// <param name="calibrationDataPath">Optional path to calibration data for INT8 quantization. If provided, INT8 will be enabled.</param>
    public static TensorRTConfiguration ForHighThroughput(int batchSize = 64, string? calibrationDataPath = null)
    {
        var precision = string.IsNullOrEmpty(calibrationDataPath) ? TensorRTPrecision.FP16 : TensorRTPrecision.INT8;

        return new TensorRTConfiguration
        {
            MaxBatchSize = batchSize,
            MaxWorkspaceSize = 8L << 30, // 8 GB
            Precision = precision,
            CalibrationDataPath = calibrationDataPath,
            EnableMultiStream = true,
            NumStreams = 8,
            EnableCudaGraphs = false, // CUDA graphs work better with fixed batch sizes
            BuilderOptimizationLevel = 4
        };
    }

    /// <summary>
    /// Creates a configuration with INT8 quantization.
    /// </summary>
    /// <param name="calibrationDataPath">Path to calibration data file (required for INT8 quantization)</param>
    /// <exception cref="ArgumentException">Thrown when calibrationDataPath is null or whitespace</exception>
    public static TensorRTConfiguration ForInt8(string calibrationDataPath)
    {
        if (string.IsNullOrWhiteSpace(calibrationDataPath))
            throw new ArgumentException("Calibration data path cannot be null or whitespace", nameof(calibrationDataPath));

        return new TensorRTConfiguration
        {
            MaxBatchSize = 8,
            MaxWorkspaceSize = 2L << 30, // 2 GB
            Precision = TensorRTPrecision.INT8,
            CalibrationDataPath = calibrationDataPath,
            BuilderOptimizationLevel = 4
        };
    }

    /// <summary>
    /// Validates the configuration and throws exceptions for invalid settings.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when INT8 is enabled without calibration data</exception>
    /// <exception cref="FileNotFoundException">Thrown when calibration data path is specified but file doesn't exist</exception>
    public void Validate()
    {
        // Validate INT8 requires calibration data
        if (Precision == TensorRTPrecision.INT8 && string.IsNullOrWhiteSpace(CalibrationDataPath))
        {
            throw new InvalidOperationException(
                "INT8 precision requires CalibrationDataPath to be provided. " +
                "Either provide calibration data or use a different Precision mode.");
        }

        // Validate calibration data file exists if path is provided
        if (!string.IsNullOrWhiteSpace(CalibrationDataPath) && !File.Exists(CalibrationDataPath))
        {
            throw new FileNotFoundException(
                $"Calibration data file not found: {CalibrationDataPath}. " +
                "Ensure the file exists before building TensorRT engine.",
                CalibrationDataPath);
        }

        // Validate batch size
        if (MaxBatchSize < 1)
        {
            throw new InvalidOperationException(
                $"MaxBatchSize must be at least 1, got: {MaxBatchSize}");
        }

        // Validate workspace size
        if (MaxWorkspaceSize < 0)
        {
            throw new InvalidOperationException(
                $"MaxWorkspaceSize must be non-negative, got: {MaxWorkspaceSize}");
        }

        // Validate optimization level
        if (BuilderOptimizationLevel < 0 || BuilderOptimizationLevel > 5)
        {
            throw new InvalidOperationException(
                $"BuilderOptimizationLevel must be between 0 and 5, got: {BuilderOptimizationLevel}");
        }

        // Validate multi-stream configuration
        if (EnableMultiStream && NumStreams < 1)
        {
            throw new InvalidOperationException(
                $"NumStreams must be at least 1 when EnableMultiStream is true, got: {NumStreams}");
        }
    }
}
