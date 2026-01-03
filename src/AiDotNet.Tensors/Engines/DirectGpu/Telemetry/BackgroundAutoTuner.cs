// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Diagnostics;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Tensors.Engines.DirectGpu.Telemetry;

/// <summary>
/// Background auto-tuner that finds optimal GPU configurations without blocking the main thread.
/// </summary>
/// <remarks>
/// <para>
/// This tuner runs A/B tests in the background to find optimal kernel configurations
/// for the current GPU. Results are cached locally and optionally shared via telemetry.
/// </para>
/// <para><b>Tuning Process:</b></para>
/// <list type="number">
/// <item>Check local cache for existing profile</item>
/// <item>Check telemetry service for community-shared profile</item>
/// <item>If no profile found, run A/B tests on common matrix sizes</item>
/// <item>Cache best configuration locally</item>
/// <item>Optionally submit results to telemetry service</item>
/// </list>
/// </remarks>
public sealed class BackgroundAutoTuner : IDisposable
{
    private readonly IDirectGpuBackend _backend;
    private readonly GpuProfileCache _cache;
    private readonly ITelemetryClient? _telemetryClient;
    private readonly ILogger? _logger;
    private readonly double _minEfficiencyTarget;
    private readonly int _timeoutSeconds;
    private readonly CancellationTokenSource _cts;
    private Task? _tuningTask;
    private bool _disposed;
    private bool _tuningComplete;

    /// <summary>
    /// Gets whether auto-tuning has completed.
    /// </summary>
    public bool IsTuningComplete => _tuningComplete;

    /// <summary>
    /// Gets whether auto-tuning is currently running.
    /// </summary>
    public bool IsRunning => _tuningTask is not null && !_tuningTask.IsCompleted;

    /// <summary>
    /// Creates a new background auto-tuner.
    /// </summary>
    /// <param name="backend">The GPU backend to tune.</param>
    /// <param name="cache">Local profile cache.</param>
    /// <param name="telemetryClient">Optional telemetry client for sharing results.</param>
    /// <param name="minEfficiencyTarget">Minimum efficiency target (0.0 to 1.0).</param>
    /// <param name="timeoutSeconds">Maximum tuning time in seconds.</param>
    /// <param name="logger">Optional logger.</param>
    public BackgroundAutoTuner(
        IDirectGpuBackend backend,
        GpuProfileCache cache,
        ITelemetryClient? telemetryClient = null,
        double minEfficiencyTarget = 0.70,
        int timeoutSeconds = 60,
        ILogger? logger = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _cache = cache ?? throw new ArgumentNullException(nameof(cache));
        _telemetryClient = telemetryClient;
        // Math.Clamp is .NET Core 2.0+ only, use manual implementation for net471 compatibility
        _minEfficiencyTarget = Math.Max(0.1, Math.Min(0.99, minEfficiencyTarget));
        _timeoutSeconds = Math.Max(10, timeoutSeconds);
        _logger = logger;
        _cts = new CancellationTokenSource();
    }

    /// <summary>
    /// Starts background auto-tuning.
    /// </summary>
    /// <remarks>
    /// This method returns immediately. Tuning runs in a background thread.
    /// Check <see cref="IsTuningComplete"/> or await <see cref="WaitForCompletionAsync"/>
    /// to know when tuning has finished.
    /// </remarks>
    public void Start()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(BackgroundAutoTuner));
        }

        if (_tuningTask is not null)
        {
            return; // Already started
        }

        _tuningTask = Task.Run(() => RunTuningAsync(_cts.Token), _cts.Token);
    }

    /// <summary>
    /// Waits for background tuning to complete.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if tuning completed successfully, false if cancelled or failed.</returns>
    public async Task<bool> WaitForCompletionAsync(CancellationToken cancellationToken = default)
    {
        if (_tuningTask is null)
        {
            return _tuningComplete;
        }

        try
        {
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _cts.Token);
            // Task.WaitAsync is .NET 6.0+ only - use Task.Delay with cancellation instead for net471
            var completedTask = await Task.WhenAny(_tuningTask, Task.Delay(Timeout.Infinite, linkedCts.Token)).ConfigureAwait(false);
            if (completedTask != _tuningTask)
            {
                return false; // Cancelled
            }
            await _tuningTask.ConfigureAwait(false); // Propagate any exceptions
            return _tuningComplete;
        }
        catch (OperationCanceledException)
        {
            return false;
        }
    }

    /// <summary>
    /// Cancels background tuning if running.
    /// </summary>
    public void Cancel()
    {
        _cts.Cancel();
    }

    private async Task RunTuningAsync(CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            _logger?.LogInformation("Starting background GPU auto-tuning (timeout: {Timeout}s)...", _timeoutSeconds);

            // Set overall timeout
            using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(_timeoutSeconds));
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token);
            var token = linkedCts.Token;

            var gpuVendor = DirectGpuBackendFactory.DetectedVendor.ToString();
            var gpuModel = GetGpuModel();
            var gpuInfo = CreateGpuInfo(gpuVendor, gpuModel);

            // Define matrix size ranges to tune
            var sizeRanges = new[]
            {
                (MinDim: 64, MaxDim: 256, TestSize: 128),      // Small matrices
                (MinDim: 256, MaxDim: 512, TestSize: 384),     // Medium matrices
                (MinDim: 512, MaxDim: 1024, TestSize: 768),    // Large matrices
                (MinDim: 1024, MaxDim: 4096, TestSize: 2048),  // Very large matrices
            };

            foreach (var range in sizeRanges)
            {
                token.ThrowIfCancellationRequested();

                // Check local cache first
                var cachedProfile = _cache.GetProfile(gpuVendor, gpuModel, range.MinDim, range.MaxDim);
                if (cachedProfile is not null)
                {
                    _logger?.LogDebug("Found cached profile for {Vendor} {Model} ({Min}-{Max}): {Gflops:F1} GFLOPS ({Efficiency:P1})",
                        gpuVendor, gpuModel, range.MinDim, range.MaxDim,
                        cachedProfile.MeasuredGflops, cachedProfile.EfficiencyPercent / 100.0);
                    continue;
                }

                // Check telemetry service for community profile
                if (_telemetryClient is not null && _telemetryClient.IsEnabled)
                {
                    var cloudProfile = await _telemetryClient.GetProfileAsync(
                        gpuInfo, range.MinDim, range.MaxDim, token).ConfigureAwait(false);

                    if (cloudProfile is not null && cloudProfile.EfficiencyPercent >= _minEfficiencyTarget * 100)
                    {
                        // Cache the community profile locally
                        _cache.SetProfile(gpuVendor, gpuModel, range.MinDim, range.MaxDim,
                            cloudProfile.ConfigJson, cloudProfile.MeasuredGflops, cloudProfile.EfficiencyPercent);

                        _logger?.LogInformation("Downloaded community profile for {Vendor} {Model} ({Min}-{Max}): {Gflops:F1} GFLOPS ({Efficiency:P1})",
                            gpuVendor, gpuModel, range.MinDim, range.MaxDim,
                            cloudProfile.MeasuredGflops, cloudProfile.EfficiencyPercent / 100.0);
                        continue;
                    }
                }

                // No profile found - run A/B test
                var result = await RunAbTestAsync(range.TestSize, token).ConfigureAwait(false);

                if (result is not null)
                {
                    // Cache locally
                    _cache.SetProfile(gpuVendor, gpuModel, range.MinDim, range.MaxDim,
                        result.ConfigJson, result.MeasuredGflops, result.EfficiencyPercent);

                    _logger?.LogInformation("Tuned {Vendor} {Model} ({Min}-{Max}): {Gflops:F1} GFLOPS ({Efficiency:P1})",
                        gpuVendor, gpuModel, range.MinDim, range.MaxDim,
                        result.MeasuredGflops, result.EfficiencyPercent / 100.0);

                    // Submit to telemetry if enabled
                    if (_telemetryClient is not null && _telemetryClient.IsEnabled)
                    {
                        await _telemetryClient.SubmitTuningResultAsync(
                            new TuningResultData
                            {
                                MatrixM = range.TestSize,
                                MatrixN = range.TestSize,
                                MatrixK = range.TestSize,
                                ConfigJson = result.ConfigJson,
                                MeasuredGflops = result.MeasuredGflops,
                                EfficiencyPercent = result.EfficiencyPercent
                            },
                            gpuInfo,
                            token).ConfigureAwait(false);
                    }
                }
            }

            _tuningComplete = true;
            _cache.Flush();

            _logger?.LogInformation("GPU auto-tuning completed in {Elapsed:F1}s", stopwatch.Elapsed.TotalSeconds);
        }
        catch (OperationCanceledException)
        {
            _logger?.LogWarning("GPU auto-tuning cancelled after {Elapsed:F1}s", stopwatch.Elapsed.TotalSeconds);
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "GPU auto-tuning failed after {Elapsed:F1}s", stopwatch.Elapsed.TotalSeconds);

            // Report exception to telemetry
            if (_telemetryClient is not null && _telemetryClient.IsEnabled)
            {
                try
                {
                    await _telemetryClient.SubmitExceptionAsync(new ExceptionData
                    {
                        ExceptionType = ex.GetType().Name,
                        ExceptionMessage = ex.Message,
                        StackTrace = ex.StackTrace,
                        InnerExceptionType = ex.InnerException?.GetType().Name,
                        InnerExceptionMessage = ex.InnerException?.Message,
                        Component = "BackgroundAutoTuner",
                        Operation = "RunTuningAsync",
                        AidotnetVersion = GetAidotnetVersion(),
                        DotnetVersion = Environment.Version.ToString(),
                        OsPlatform = Environment.OSVersion.Platform.ToString(),
                        OsVersion = Environment.OSVersion.VersionString,
                        GpuVendor = DirectGpuBackendFactory.DetectedVendor.ToString(),
                        GpuModel = GetGpuModel()
                    }).ConfigureAwait(false);
                }
                catch
                {
                    // Ignore telemetry failures
                }
            }
        }
    }

    private async Task<TuningResult?> RunAbTestAsync(int matrixSize, CancellationToken cancellationToken)
    {
        // Generate test configurations to compare
        var configs = GenerateTestConfigs(matrixSize);
        TuningResult? bestResult = null;

        foreach (var config in configs)
        {
            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                var result = await BenchmarkConfigAsync(matrixSize, config, cancellationToken).ConfigureAwait(false);

                if (bestResult is null || result.MeasuredGflops > bestResult.MeasuredGflops)
                {
                    bestResult = result;
                }

                // Early exit if we've hit our target
                if (result.EfficiencyPercent >= _minEfficiencyTarget * 100)
                {
                    _logger?.LogDebug("Target efficiency {Target:P0} achieved with config: {Config}",
                        _minEfficiencyTarget, config);
                    break;
                }
            }
            catch (Exception ex)
            {
                _logger?.LogDebug(ex, "Config test failed: {Config}", config);
            }
        }

        return bestResult;
    }

    private async Task<TuningResult> BenchmarkConfigAsync(int matrixSize, GemmTestConfig config, CancellationToken cancellationToken)
    {
        // Warm up
        await Task.Run(() => RunGemm(matrixSize, config), cancellationToken).ConfigureAwait(false);

        // Benchmark
        const int iterations = 5;
        var stopwatch = Stopwatch.StartNew();

        for (int i = 0; i < iterations; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Run(() => RunGemm(matrixSize, config), cancellationToken).ConfigureAwait(false);
        }

        stopwatch.Stop();
        var totalSeconds = stopwatch.Elapsed.TotalSeconds;

        // Calculate GFLOPS: 2 * M * N * K * iterations / time
        long flops = 2L * matrixSize * matrixSize * matrixSize * iterations;
        double gflops = flops / totalSeconds / 1e9;

        // Estimate efficiency (assuming ~10 TFLOPS theoretical for mid-range GPU)
        // This is a rough estimate - real implementation would query GPU specs
        double theoreticalGflops = EstimateTheoreticalGflops();
        double efficiency = Math.Min(gflops / theoreticalGflops, 1.0);

        return new TuningResult
        {
            ConfigJson = JsonSerializer.Serialize(config),
            MeasuredGflops = gflops,
            EfficiencyPercent = efficiency * 100
        };
    }

    private void RunGemm(int size, GemmTestConfig config)
    {
        // Allocate test matrices on CPU
        var a = new float[size * size];
        var b = new float[size * size];

        // Initialize with small values to avoid overflow
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = 0.01f * (i % 100);
            b[i] = 0.01f * ((i + 17) % 100);
        }

        // Apply config by setting backend configuration before GEMM
        // The backend reads from this configuration internally
        ApplyGemmConfig(config);

        // Upload to GPU and run GEMM
        using var bufferA = _backend.AllocateBuffer(a);
        using var bufferB = _backend.AllocateBuffer(b);
        using var bufferC = _backend.AllocateBuffer(size * size);

        _backend.Gemm(bufferA, bufferB, bufferC, size, size, size, 1.0f, 0.0f);
        _backend.Synchronize();
    }

    private void ApplyGemmConfig(GemmTestConfig config)
    {
        // Store current config for the backend to read
        // This is passed via a thread-local or cached state that the backend queries
        CurrentTestConfig = config;
    }

    /// <summary>
    /// Gets the current test configuration being benchmarked.
    /// Backends can read this to apply experimental tile sizes.
    /// </summary>
    internal static GemmTestConfig? CurrentTestConfig { get; private set; }

    private GemmTestConfig[] GenerateTestConfigs(int matrixSize)
    {
        // Generate configurations based on matrix size
        if (matrixSize <= 256)
        {
            return new[]
            {
                new GemmTestConfig(16, 16, 16, true, 4),
                new GemmTestConfig(16, 16, 8, true, 4),
                new GemmTestConfig(8, 8, 8, true, 4),
                new GemmTestConfig(32, 32, 8, true, 4),
            };
        }
        else if (matrixSize <= 512)
        {
            return new[]
            {
                new GemmTestConfig(16, 16, 16, true, 4),
                new GemmTestConfig(32, 32, 16, true, 4),
                new GemmTestConfig(16, 16, 8, true, 8),
                new GemmTestConfig(32, 32, 8, true, 4),
            };
        }
        else
        {
            return new[]
            {
                new GemmTestConfig(32, 32, 16, true, 4),
                new GemmTestConfig(32, 32, 32, true, 4),
                new GemmTestConfig(64, 64, 16, true, 4),
                new GemmTestConfig(16, 16, 16, true, 8),
            };
        }
    }

    private double EstimateTheoreticalGflops()
    {
        // Rough estimates based on vendor
        return DirectGpuBackendFactory.DetectedVendor switch
        {
            GpuVendor.NVIDIA => 15000,   // ~15 TFLOPS for mid-range NVIDIA
            GpuVendor.AMD => 10000,      // ~10 TFLOPS for mid-range AMD
            GpuVendor.Intel => 5000,     // ~5 TFLOPS for Intel iGPU
            _ => 8000                    // Conservative default
        };
    }

    private string GetGpuModel()
    {
        var gpus = DirectGpuBackendFactory.GetAvailableGpus();
        return gpus.Length > 0 ? gpus[0].DeviceName : "Unknown";
    }

    private GpuInfoData CreateGpuInfo(string vendor, string model)
    {
        return new GpuInfoData
        {
            Vendor = vendor,
            Model = model,
            OsPlatform = Environment.OSVersion.Platform.ToString()
        };
    }

    private static string GetAidotnetVersion()
    {
        try
        {
            var assembly = typeof(BackgroundAutoTuner).Assembly;
            var version = assembly.GetName().Version;
            return version?.ToString() ?? "unknown";
        }
        catch
        {
            return "unknown";
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _cts.Cancel();
        _cts.Dispose();
        _cache.Flush();
    }

    private sealed class TuningResult
    {
        public string ConfigJson { get; set; } = string.Empty;
        public double MeasuredGflops { get; set; }
        public double EfficiencyPercent { get; set; }
    }

}

/// <summary>
/// Configuration for GEMM kernel parameters during auto-tuning.
/// </summary>
internal readonly struct GemmTestConfig
{
    public int TileM { get; }
    public int TileN { get; }
    public int TileK { get; }
    public bool UseLocalMemory { get; }
    public int VectorWidth { get; }

    public GemmTestConfig(int tileM, int tileN, int tileK, bool useLocalMemory, int vectorWidth)
    {
        TileM = tileM;
        TileN = tileN;
        TileK = tileK;
        UseLocalMemory = useLocalMemory;
        VectorWidth = vectorWidth;
    }

    public override string ToString() =>
        $"Tile({TileM}x{TileN}x{TileK}), Local={UseLocalMemory}, Vec={VectorWidth}";
}
