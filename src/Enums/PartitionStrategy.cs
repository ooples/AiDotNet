namespace AiDotNet.Enums;

/// <summary>
/// Strategies for partitioning models between cloud and edge devices.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Sometimes you want to split an AI model so that part of it runs on a
/// local device (edge) and part runs in the cloud. This is useful for:
/// - Reducing bandwidth by processing some data locally
/// - Improving privacy by keeping sensitive data on the device
/// - Balancing speed (edge processing) with power (cloud processing)
///
/// Different strategies determine where to split the model:
///
/// - **EarlyLayers**: First few layers run on edge, rest in cloud. Good for preprocessing data
///   locally before sending it to the cloud.
/// - **LateLayers**: Most processing on edge, only final layers in cloud. Good for devices with
///   decent processing power.
/// - **Balanced**: Split in the middle. Good general-purpose strategy.
/// - **Adaptive**: Automatically determines the best split based on network speed, device power,
///   and battery level.
/// - **Manual**: You specify exactly where to split. For advanced users.
/// </remarks>
public enum PartitionStrategy
{
    /// <summary>
    /// Execute early layers on edge, rest on cloud.
    /// Good for local preprocessing before cloud processing.
    /// </summary>
    EarlyLayers,

    /// <summary>
    /// Execute most layers on edge, only final on cloud.
    /// Good for powerful edge devices that can handle most processing.
    /// </summary>
    LateLayers,

    /// <summary>
    /// Balanced partition - splits model in the middle.
    /// Good general-purpose strategy.
    /// </summary>
    Balanced,

    /// <summary>
    /// Adaptively determine partition based on runtime conditions
    /// (network speed, device power, battery level, model size).
    /// Best for dynamic environments.
    /// </summary>
    Adaptive,

    /// <summary>
    /// Manual partition specification - you control exactly where to split.
    /// For advanced users with specific requirements.
    /// </summary>
    Manual
}
