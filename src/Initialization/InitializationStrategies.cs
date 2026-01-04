namespace AiDotNet.Initialization;

/// <summary>
/// Provides factory methods and default instances for initialization strategies.
/// </summary>
/// <remarks>
/// <para>
/// This static class provides convenient access to commonly used initialization strategies
/// as singletons, reducing memory allocations when the same strategy is used across
/// multiple layers or networks.
/// </para>
/// <para><b>Usage:</b></para>
/// <code>
/// // Use lazy initialization for fast network construction
/// var strategy = InitializationStrategies&lt;double&gt;.Lazy;
///
/// // Use eager initialization for traditional behavior
/// var strategy = InitializationStrategies&lt;double&gt;.Eager;
///
/// // Use zero initialization for testing
/// var strategy = InitializationStrategies&lt;double&gt;.Zero;
///
/// // Load from file for transfer learning
/// var strategy = InitializationStrategies&lt;double&gt;.FromFile("weights.json");
/// </code>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class InitializationStrategies<T>
{
    /// <summary>
    /// Gets the default lazy initialization strategy.
    /// </summary>
    /// <remarks>
    /// Lazy initialization defers weight initialization until the first forward pass,
    /// making network construction faster.
    /// </remarks>
    public static IInitializationStrategy<T> Lazy { get; } = new LazyInitializationStrategy<T>();

    /// <summary>
    /// Gets the default eager initialization strategy.
    /// </summary>
    /// <remarks>
    /// Eager initialization initializes weights immediately during layer construction.
    /// This is the traditional approach that ensures all weights are ready before training.
    /// </remarks>
    public static IInitializationStrategy<T> Eager { get; } = new EagerInitializationStrategy<T>();

    /// <summary>
    /// Gets the zero initialization strategy.
    /// </summary>
    /// <remarks>
    /// Zero initialization sets all weights to zero. Use only for testing or specific
    /// architectures that require it. Not recommended for general training.
    /// </remarks>
    public static IInitializationStrategy<T> Zero { get; } = new ZeroInitializationStrategy<T>();

    /// <summary>
    /// Creates a new initialization strategy that loads weights from a file.
    /// </summary>
    /// <param name="filePath">The path to the weights file.</param>
    /// <param name="format">The format of the weights file. Default is Auto-detect.</param>
    /// <returns>An initialization strategy that loads weights from the specified file.</returns>
    /// <remarks>
    /// Unlike the singleton strategies, this creates a new instance each time
    /// because it depends on the specific file path.
    /// </remarks>
    public static IInitializationStrategy<T> FromFile(string filePath, WeightFileFormat format = WeightFileFormat.Auto)
    {
        return new FromFileInitializationStrategy<T>(filePath, format);
    }
}

/// <summary>
/// Provides backward-compatible access to initialization strategies.
/// </summary>
/// <remarks>
/// This class maintains backward compatibility with existing code that uses
/// <c>InitializationStrategy&lt;T&gt;.Lazy</c> etc.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[Obsolete("Use InitializationStrategies<T> instead. This class is maintained for backward compatibility.")]
public static class InitializationStrategy<T>
{
    /// <summary>
    /// Gets the default lazy initialization strategy.
    /// </summary>
    public static IInitializationStrategy<T> Lazy => InitializationStrategies<T>.Lazy;

    /// <summary>
    /// Gets the default eager initialization strategy.
    /// </summary>
    public static IInitializationStrategy<T> Eager => InitializationStrategies<T>.Eager;

    /// <summary>
    /// Gets the zero initialization strategy.
    /// </summary>
    public static IInitializationStrategy<T> Zero => InitializationStrategies<T>.Zero;
}

/// <summary>
/// Specifies the type of initialization strategy to use for layer weights.
/// </summary>
/// <remarks>
/// <para>
/// Use this enum with <see cref="AiDotNet.Training.Memory.TrainingMemoryConfig"/>
/// to configure the default initialization strategy for all layers.
/// </para>
/// </remarks>
public enum InitializationStrategyType
{
    /// <summary>
    /// Eager initialization - initialize weights immediately during layer construction.
    /// This is the traditional approach that ensures all weights are ready before training.
    /// </summary>
    Eager,

    /// <summary>
    /// Lazy initialization - defer weight initialization until the first forward pass.
    /// This makes network construction faster, especially for large models or when
    /// just inspecting network architecture.
    /// </summary>
    Lazy,

    /// <summary>
    /// Zero initialization - set all weights to zero.
    /// Use only for testing or specific architectures. Not recommended for training.
    /// </summary>
    Zero,

    /// <summary>
    /// Load weights from an external file.
    /// Use for transfer learning or resuming training from a checkpoint.
    /// Requires setting the WeightsFilePath property.
    /// </summary>
    FromFile
}
