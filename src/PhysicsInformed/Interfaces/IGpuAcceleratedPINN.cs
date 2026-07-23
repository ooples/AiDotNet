using AiDotNet.Engines;

namespace AiDotNet.PhysicsInformed.Interfaces;

/// <summary>
/// Interface for Physics-Informed Neural Networks that support GPU acceleration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// For Beginners:
/// This interface marks PINNs that can take advantage of GPU acceleration during training.
/// GPU acceleration can provide significant speedups for:
/// - Batch processing of collocation points
/// - Parallel derivative computations
/// - Matrix operations in the forward and backward passes
///
/// Implementing this interface signals that the PINN can use GPU resources when available.
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("GpuAcceleratedPINN")]
public interface IGpuAcceleratedPINN<T>
{
    /// <summary>
    /// Gets or sets whether GPU acceleration is enabled for training.
    /// </summary>
    bool UseGpuAcceleration { get; set; }

    /// <summary>
    /// Gets or sets the GPU acceleration configuration.
    /// </summary>
    GpuAccelerationConfig? GpuConfig { get; set; }

    /// <summary>
    /// Gets a value indicating whether GPU is currently available and ready for use.
    /// </summary>
    bool IsGpuAvailable { get; }

    /// <summary>
    /// Initializes GPU resources for accelerated training.
    /// </summary>
    /// <param name="config">Optional GPU configuration. If null, uses default settings.</param>
    /// <returns>True if GPU initialization was successful, false otherwise.</returns>
    bool InitializeGpu(GpuAccelerationConfig? config = null);

    /// <summary>
    /// Releases GPU resources.
    /// </summary>
    void ReleaseGpuResources();
}
