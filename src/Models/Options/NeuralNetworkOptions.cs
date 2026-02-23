namespace AiDotNet.Models.Options;

/// <summary>
/// Base configuration options for all neural network models.
/// </summary>
/// <remarks>
/// <para>
/// This class provides the foundational options shared by all neural network models,
/// inheriting the Seed property from ModelOptions. Neural network-specific options
/// like learning rate, epochs, and batch size can be added here as the library evolves.
/// </para>
/// <para><b>For Beginners:</b> This contains the basic settings that all neural networks share.
/// More specific neural network types (audio, document, financial, etc.) extend this with
/// domain-specific settings.
/// </para>
/// </remarks>
public class NeuralNetworkOptions : ModelOptions
{
}
