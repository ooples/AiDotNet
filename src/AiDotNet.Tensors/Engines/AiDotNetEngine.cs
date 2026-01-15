using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Global configuration for the AiDotNet execution engine.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNetEngine provides a singleton pattern for managing the active execution engine.
/// By default, operations run on the CPU. Users can switch to GPU or other accelerators
/// by setting the Current property.
/// </para>
/// <para><b>For Beginners:</b> This is like a settings panel for your calculations.
///
/// Example usage:
/// <code>
/// // Default: Use CPU
/// var result = vector1.Add(vector2);  // Runs on CPU
///
/// // Switch to GPU
/// AiDotNetEngine.Current = new GpuEngine();
/// var result2 = vector1.Add(vector2);  // Now runs on GPU!
///
/// // Auto-detect best hardware
/// AiDotNetEngine.AutoDetectAndConfigureGpu();
/// </code>
/// </para>
/// </remarks>
public static class AiDotNetEngine
{
    private static IEngine _current;
    private static readonly object _lock = new object();

    /// <summary>
    /// Static constructor initializes with CPU engine by default.
    /// </summary>
    static AiDotNetEngine()
    {
        _current = new CpuEngine();
    }

    /// <summary>
    /// Gets or sets the current execution engine.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Changing the engine affects all subsequent operations. The change is global
    /// and thread-safe.
    /// </para>
    /// <para><b>For Beginners:</b> This is like choosing between CPU and GPU mode.
    ///
    /// Common patterns:
    /// <code>
    /// // Use CPU (default, works for all types)
    /// AiDotNetEngine.Current = new CpuEngine();
    ///
    /// // Use GPU (faster for float, fallback to CPU for other types)
    /// AiDotNetEngine.Current = new GpuEngine();
    ///
    /// // Auto-detect (recommended)
    /// AiDotNetEngine.AutoDetectAndConfigureGpu();
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when attempting to set to null.</exception>
    public static IEngine Current
    {
        get
        {
            lock (_lock)
            {
                return _current;
            }
        }
        set
        {
            if (value == null)
            {
                throw new ArgumentNullException(nameof(value), "Engine cannot be null");
            }

            lock (_lock)
            {
                _current = value;
            }
        }
    }

    /// <summary>
    /// Automatically detects and configures GPU acceleration if available.     
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method attempts to initialize DirectGpu acceleration. If successful, the Current
    /// engine is switched to DirectGpuTensorEngine. If GPU is not available or initialization fails,
    /// the engine remains on CpuEngine.
    /// </para>
    /// <para><b>For Beginners:</b> Call this once at application startup for automatic optimization.
    ///
    /// <code>
    /// // In your Program.cs or Main():
    /// AiDotNetEngine.AutoDetectAndConfigureGpu();
    ///
    /// // Now all operations will automatically use GPU if available!
    /// </code>
    ///
    /// This is safe to call even if you don't have a GPU - it will just stay on CPU mode.
    /// </para>
    /// </remarks>
    /// <returns>True if GPU was successfully configured, false otherwise.</returns>
    public static bool AutoDetectAndConfigureGpu()
    {
        try
        {
            var gpuEngine = new DirectGpuTensorEngine();

            if (gpuEngine.SupportsGpu)
            {
                Current = gpuEngine;
                Console.WriteLine($"[AiDotNet] GPU acceleration enabled: {gpuEngine.Name}");
                return true;
            }

            gpuEngine.Dispose();
            Console.WriteLine("[AiDotNet] GPU not available, using CPU");
            return false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AiDotNet] Failed to initialize DirectGpu: {ex.Message}");
            Console.WriteLine("[AiDotNet] Falling back to CPU");
            return false;
        }
    }

    /// <summary>
    /// Resets the engine to the default CPU engine.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is useful for testing or when you explicitly want to disable GPU acceleration.
    /// </para>
    /// </remarks>
    public static void ResetToCpu()
    {
        Current = new CpuEngine();
        Console.WriteLine("[AiDotNet] Reset to CPU engine");
    }

    /// <summary>
    /// Begins a GPU execution context for the current engine, if supported.
    /// </summary>
    /// <param name="options">Optional execution options.</param>
    /// <returns>A GPU execution context, or null if GPU is not available.</returns>
    public static GpuExecutionContext? BeginGpuContext(GpuExecutionOptions? options = null)
    {
        if (Current is DirectGpuTensorEngine gpuEngine)
        {
            return gpuEngine.BeginGpuContext(options);
        }

        return null;
    }

    /// <summary>
    /// Executes an action within a GPU execution context for the current engine.
    /// </summary>
    /// <param name="action">The action to execute.</param>
    /// <param name="options">Optional execution options.</param>
    /// <returns>True if executed on GPU, false if GPU is not available.</returns>
    public static bool WithGpuContext(Action<GpuExecutionContext> action, GpuExecutionOptions? options = null)
    {
        if (Current is DirectGpuTensorEngine gpuEngine)
        {
            return gpuEngine.WithGpuContext(action, options);
        }

        return false;
    }

    /// <summary>
    /// Executes a function within a GPU execution context for the current engine.
    /// </summary>
    /// <typeparam name="TResult">The result type.</typeparam>
    /// <param name="func">The function to execute.</param>
    /// <param name="fallback">Fallback function if GPU is not available.</param>
    /// <param name="options">Optional execution options.</param>
    /// <returns>The function result.</returns>
    public static TResult WithGpuContext<TResult>(Func<GpuExecutionContext, TResult> func, Func<TResult> fallback, GpuExecutionOptions? options = null)
    {
        if (func is null)
            throw new ArgumentNullException(nameof(func));
        if (fallback is null)
            throw new ArgumentNullException(nameof(fallback));

        if (Current is DirectGpuTensorEngine gpuEngine)
        {
            return gpuEngine.WithGpuContext(func, fallback, options);
        }

        return fallback();
    }

    /// <summary>
    /// Gets information about the current engine configuration.
    /// </summary>
    /// <returns>A string describing the current engine.</returns>
    public static string GetEngineInfo()
    {
        var engine = Current;
        return $"Engine: {engine.Name}, GPU Support: {engine.SupportsGpu}";
    }
}
