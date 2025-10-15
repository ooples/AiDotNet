using AiDotNet.Compression.Hardware;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that support hardware-specific optimization.
/// </summary>
/// <remarks>
/// <para>
/// This interface should be implemented by models that can apply their own
/// hardware-specific optimizations.
/// </para>
/// <para><b>For Beginners:</b> This marks a model as able to optimize itself for specific hardware.
/// 
/// Models that implement this interface know how to:
/// - Analyze what hardware features are available
/// - Apply optimizations specific to those features
/// - Reconfigure themselves for optimal performance
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TModel">The type of the model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IHardwareOptimizable<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Optimizes the model for the specified hardware capabilities.
    /// </summary>
    /// <param name="capabilities">The hardware capabilities to optimize for.</param>
    /// <returns>The hardware-optimized model.</returns>
    /// <remarks>
    /// <para>
    /// This method applies optimizations to the model based on the specified hardware capabilities.
    /// </para>
    /// <para><b>For Beginners:</b> This lets a model optimize itself for specific hardware.
    /// 
    /// Given information about the hardware:
    /// - The model can apply specific optimizations
    /// - It can reconfigure itself for better performance
    /// - It can choose the best algorithms and data structures
    /// 
    /// This allows for more specialized optimizations than the general InferenceOptimizer provides.
    /// </para>
    /// </remarks>
    TModel OptimizeForHardware(HardwareCapabilities capabilities);
}