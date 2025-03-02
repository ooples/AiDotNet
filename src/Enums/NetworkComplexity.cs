namespace AiDotNet.Enums;

/// <summary>
/// Defines the complexity level of a neural network architecture.
/// </summary>
public enum NetworkComplexity
{
    /// <summary>
    /// Simple network with minimal layers, suitable for basic tasks.
    /// </summary>
    Simple,

    /// <summary>
    /// Medium complexity network with a moderate number of layers.
    /// </summary>
    Medium,

    /// <summary>
    /// Deep network with many layers, suitable for complex tasks.
    /// </summary>
    Deep,

    /// <summary>
    /// Very deep network with extensive layers and connections.
    /// </summary>
    VeryDeep,

    /// <summary>
    /// Custom complexity defined by the user.
    /// </summary>
    Custom
}