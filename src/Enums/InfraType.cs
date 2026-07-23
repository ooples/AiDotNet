namespace AiDotNet.Enums;

/// <summary>
/// Defines the type of infrastructure component (Tier 3 metadata).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Infrastructure components handle storage, serving, and platform
/// concerns. They don't contain ML logic — they support the ML system.
/// </para>
/// </remarks>
public enum InfraType
{
    /// <summary>
    /// Caching layer for model predictions or intermediate results.
    /// </summary>
    Cache,

    /// <summary>
    /// Model serving endpoint for production inference.
    /// </summary>
    ServingEndpoint,

    /// <summary>
    /// Metrics collection and monitoring infrastructure.
    /// </summary>
    Metrics,

    /// <summary>
    /// Model serialization and persistence infrastructure.
    /// </summary>
    Persistence,

    /// <summary>
    /// Configuration management infrastructure.
    /// </summary>
    Configuration,

    /// <summary>
    /// Logging and telemetry infrastructure.
    /// </summary>
    Telemetry,

    /// <summary>
    /// Security and authentication infrastructure.
    /// </summary>
    Security,

    /// <summary>
    /// Model registry and versioning infrastructure.
    /// </summary>
    Registry
}
