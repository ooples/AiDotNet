namespace AiDotNet.Models;

/// <summary>
/// Contains statistics about system resource usage during training.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This tracks how much of your computer's resources
/// (CPU, memory, GPU) are being used.
/// </remarks>
public class ResourceUsageStats
{
    /// <summary>
    /// Gets or sets the CPU usage percentage (0-100).
    /// </summary>
    public double CpuUsagePercent { get; set; }

    /// <summary>
    /// Gets or sets the memory usage in megabytes.
    /// </summary>
    public double MemoryUsageMB { get; set; }

    /// <summary>
    /// Gets or sets the total available memory in megabytes.
    /// </summary>
    public double TotalMemoryMB { get; set; }

    /// <summary>
    /// Gets or sets the memory usage percentage (0-100).
    /// </summary>
    public double MemoryUsagePercent { get; set; }

    /// <summary>
    /// Gets or sets the GPU usage percentage (0-100), if available.
    /// </summary>
    public double? GpuUsagePercent { get; set; }

    /// <summary>
    /// Gets or sets the GPU memory usage in megabytes, if available.
    /// </summary>
    public double? GpuMemoryUsageMB { get; set; }

    /// <summary>
    /// Gets or sets the total GPU memory in megabytes, if available.
    /// </summary>
    public double? TotalGpuMemoryMB { get; set; }

    /// <summary>
    /// Gets or sets the GPU memory usage percentage (0-100), if available.
    /// </summary>
    public double? GpuMemoryUsagePercent { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when these stats were recorded.
    /// </summary>
    public DateTime Timestamp { get; set; }
}
