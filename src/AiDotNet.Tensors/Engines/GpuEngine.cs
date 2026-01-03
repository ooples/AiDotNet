using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// GPU-accelerated tensor engine with automatic backend selection and diagnostic capabilities.
/// Routes work through DirectGpuTensorEngine and provides comprehensive GPU status reporting.
/// </summary>
/// <remarks>
/// <para><b>Backend Selection Order:</b></para>
/// <list type="number">
/// <item>CUDA (NVIDIA GPUs with NVRTC)</item>
/// <item>OpenCL (AMD, Intel, NVIDIA via universal driver)</item>
/// <item>HIP (AMD GPUs with ROCm SDK)</item>
/// </list>
/// <para>
/// Use <see cref="GetAvailableBackends"/> to check which GPU backends are available,
/// and <see cref="GetDiagnosticReport"/> for comprehensive troubleshooting information.
/// </para>
/// </remarks>
public sealed class GpuEngine : DirectGpuTensorEngine
{
    private readonly AdaptiveThresholds _thresholds;

    public GpuEngine()
        : this(AdaptiveThresholds.Default, enableTimingDiagnostics: false)
    {
    }

    public GpuEngine(AdaptiveThresholds thresholds, bool enableTimingDiagnostics = false)
        : base()
    {
        _thresholds = thresholds ?? AdaptiveThresholds.Default;
    }

    public AdaptiveThresholds Thresholds => _thresholds;

    /// <summary>
    /// Gets information about all available GPU backends and their status.
    /// </summary>
    /// <returns>List of backend info including name, availability, and any error messages.</returns>
    public static IReadOnlyList<GpuBackendInfo> GetAvailableBackends()
    {
        var backends = new List<GpuBackendInfo>();

        // Check CUDA
        backends.Add(CheckCudaBackend());

        // Check OpenCL
        backends.Add(CheckOpenClBackend());

        // Check HIP
        backends.Add(CheckHipBackend());

        return backends;
    }

    private static GpuBackendInfo CheckCudaBackend()
    {
        try
        {
            if (!CudaBackend.IsCudaAvailable)
            {
                return new GpuBackendInfo("CUDA", false, "CUDA driver not available or NVRTC not found");
            }

            using var backend = new CudaBackend();
            if (backend.IsAvailable)
            {
                return new GpuBackendInfo("CUDA", true,
                    deviceName: backend.DeviceName,
                    vendor: backend.DeviceVendor,
                    computeUnits: backend.ComputeUnits,
                    globalMemoryBytes: backend.GlobalMemoryBytes);
            }
            return new GpuBackendInfo("CUDA", false, "CUDA backend initialization failed");
        }
        catch (Exception ex)
        {
            return new GpuBackendInfo("CUDA", false, $"Exception: {ex.Message}");
        }
    }

    private static GpuBackendInfo CheckOpenClBackend()
    {
        try
        {
            if (!OpenClBackend.IsOpenClAvailable)
            {
                return new GpuBackendInfo("OpenCL", false, "OpenCL runtime not available");
            }

            using var backend = new OpenClBackend();
            if (backend.IsAvailable)
            {
                return new GpuBackendInfo("OpenCL", true,
                    deviceName: backend.DeviceName,
                    vendor: backend.DeviceVendor,
                    computeUnits: backend.ComputeUnits,
                    globalMemoryBytes: backend.GlobalMemoryBytes);
            }
            return new GpuBackendInfo("OpenCL", false, "OpenCL backend initialization failed - no suitable GPU device found");
        }
        catch (Exception ex)
        {
            return new GpuBackendInfo("OpenCL", false, $"Exception: {ex.Message}");
        }
    }

    private static GpuBackendInfo CheckHipBackend()
    {
        try
        {
            if (!HipBackend.IsHipAvailable)
            {
                return new GpuBackendInfo("HIP", false,
                    "HIP runtime not available. Install AMD ROCm SDK from https://rocm.docs.amd.com/");
            }

            using var backend = new HipBackend();
            if (backend.IsAvailable)
            {
                return new GpuBackendInfo("HIP", true,
                    deviceName: backend.DeviceName,
                    vendor: backend.DeviceVendor,
                    computeUnits: backend.ComputeUnits,
                    globalMemoryBytes: backend.GlobalMemoryBytes,
                    additionalInfo: $"Architecture: {backend.Architecture}");
            }
            return new GpuBackendInfo("HIP", false, "HIP backend initialization failed - kernel compilation may have failed");
        }
        catch (Exception ex)
        {
            return new GpuBackendInfo("HIP", false, $"Exception: {ex.Message}");
        }
    }

    /// <summary>
    /// Generates a comprehensive diagnostic report for GPU availability and configuration.
    /// Use this for troubleshooting GPU acceleration issues.
    /// </summary>
    /// <returns>Multi-line diagnostic report with backend status and recommendations.</returns>
    public static string GetDiagnosticReport()
    {
        var sb = new StringBuilder();
        sb.AppendLine("=== AiDotNet GPU Diagnostic Report ===");
        sb.AppendLine();

        // Check environment variable
        string? envVar = Environment.GetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS");
        sb.AppendLine($"Environment Variable (AIDOTNET_DIRECTGPU_BACKENDS): {envVar ?? "(not set, using defaults: cuda, opencl, hip)"}");
        sb.AppendLine();

        // Get backend status
        var backends = GetAvailableBackends();
        int availableCount = 0;

        sb.AppendLine("--- Backend Status ---");
        foreach (var backend in backends)
        {
            string status = backend.IsAvailable ? "[AVAILABLE]" : "[NOT AVAILABLE]";
            sb.AppendLine($"{backend.Name}: {status}");

            if (backend.IsAvailable)
            {
                availableCount++;
                sb.AppendLine($"  Device: {backend.DeviceName}");
                sb.AppendLine($"  Vendor: {backend.Vendor}");
                sb.AppendLine($"  Compute Units: {backend.ComputeUnits}");
                sb.AppendLine($"  Global Memory: {backend.GlobalMemoryBytes / (1024.0 * 1024 * 1024):F2} GB");
                if (!string.IsNullOrEmpty(backend.AdditionalInfo))
                {
                    sb.AppendLine($"  {backend.AdditionalInfo}");
                }
            }
            else if (!string.IsNullOrEmpty(backend.ErrorMessage))
            {
                sb.AppendLine($"  Reason: {backend.ErrorMessage}");
            }
            sb.AppendLine();
        }

        // Summary and recommendations
        sb.AppendLine("--- Summary ---");
        if (availableCount > 0)
        {
            sb.AppendLine($"GPU acceleration: ENABLED ({availableCount} backend(s) available)");
        }
        else
        {
            sb.AppendLine("GPU acceleration: DISABLED (no backends available)");
            sb.AppendLine();
            sb.AppendLine("--- Recommendations ---");
            sb.AppendLine("1. NVIDIA GPU: Ensure CUDA toolkit is installed with NVRTC");
            sb.AppendLine("   Download: https://developer.nvidia.com/cuda-downloads");
            sb.AppendLine();
            sb.AppendLine("2. AMD GPU: Install ROCm SDK (Linux) or HIP SDK (Windows)");
            sb.AppendLine("   Download: https://rocm.docs.amd.com/");
            sb.AppendLine();
            sb.AppendLine("3. Any GPU: Install OpenCL runtime from your GPU vendor");
            sb.AppendLine("   - NVIDIA: Included with CUDA driver");
            sb.AppendLine("   - AMD: Install Adrenalin drivers");
            sb.AppendLine("   - Intel: Install Intel Graphics driver");
        }

        return sb.ToString();
    }
}

/// <summary>
/// Information about a GPU backend's availability and capabilities.
/// </summary>
public sealed class GpuBackendInfo
{
    public string Name { get; }
    public bool IsAvailable { get; }
    public string? ErrorMessage { get; }
    public string? DeviceName { get; }
    public string? Vendor { get; }
    public int ComputeUnits { get; }
    public long GlobalMemoryBytes { get; }
    public string? AdditionalInfo { get; }

    public GpuBackendInfo(string name, bool isAvailable, string? errorMessage = null,
        string? deviceName = null, string? vendor = null, int computeUnits = 0,
        long globalMemoryBytes = 0, string? additionalInfo = null)
    {
        Name = name;
        IsAvailable = isAvailable;
        ErrorMessage = errorMessage;
        DeviceName = deviceName;
        Vendor = vendor;
        ComputeUnits = computeUnits;
        GlobalMemoryBytes = globalMemoryBytes;
        AdditionalInfo = additionalInfo;
    }

    public override string ToString()
    {
        if (IsAvailable)
        {
            return $"{Name}: {DeviceName} ({Vendor}) - {ComputeUnits} CUs, {GlobalMemoryBytes / (1024.0 * 1024 * 1024):F2} GB";
        }
        return $"{Name}: Not available - {ErrorMessage}";
    }
}
