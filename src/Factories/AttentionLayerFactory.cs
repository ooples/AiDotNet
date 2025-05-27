namespace AiDotNet.Factories;

using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.CustomAttentionLayers;

/// <summary>
/// Factory for creating attention layers compatible with different positional encoding techniques.
/// </summary>
/// <remarks>
/// <para>
/// This factory provides methods to create different types of attention layers that are compatible
/// with various positional encoding techniques. Some positional encoding methods modify token
/// representations, while others modify attention mechanisms directly.
/// </para>
/// <para><b>For Beginners:</b> This factory helps choose the right attention layer for each encoding.
/// 
/// Different positional encoding techniques work in different ways:
/// - Some add position information to word embeddings (like Sinusoidal)
/// - Others modify how attention works (like ALiBi and T5RelativeBias)
/// 
/// This factory makes sure the attention layer is compatible with your chosen encoding method.
/// </para>
/// </remarks>
public static class AttentionLayerFactory
{
    /// <summary>
    /// Creates an appropriate attention layer for the specified positional encoding type.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
    /// <param name="positionalEncodingType">The type of positional encoding being used.</param>
    /// <param name="sequenceLength">The maximum sequence length for the attention layer.</param>
    /// <param name="embeddingDimension">The size of each embedding vector.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <param name="activationFunction">The activation function for the attention layer.</param>
    /// <returns>An attention layer compatible with the specified positional encoding type.</returns>
    /// <remarks>
    /// <para>
    /// This method creates and returns an attention layer that is compatible with the specified
    /// positional encoding type. For encoding types that modify token representations (like Sinusoidal
    /// or Learned), a standard MultiHeadAttentionLayer is used. For encoding types that modify
    /// attention mechanisms directly (like ALiBi or T5RelativeBias), specialized attention layers
    /// are created.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the right attention layer for your encoding.
    /// 
    /// You specify:
    /// - The type of positional encoding you're using
    /// - The basic attention parameters (sequence length, dimensions, etc.)
    /// 
    /// The method then creates the appropriate type of attention layer:
    /// - Standard attention for most encoding types
    /// - Specialized attention for ALiBi and T5RelativeBias
    /// 
    /// This ensures that attention and position encoding work together correctly.
    /// </para>
    /// </remarks>
    public static ILayer<T> Create<T>(
        PositionalEncodingType positionalEncodingType,
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        IActivationFunction<T> activationFunction)
    {
        switch (positionalEncodingType)
        {
            case PositionalEncodingType.ALiBi:
                return new ALiBiAttentionLayer<T>(
                    sequenceLength,
                    embeddingDimension,
                    headCount,
                    activationFunction);
                
            case PositionalEncodingType.T5RelativeBias:
                return new T5RelativeBiasAttentionLayer<T>(
                    sequenceLength,
                    embeddingDimension,
                    headCount,
                    activationFunction);
                
            case PositionalEncodingType.None:
            case PositionalEncodingType.Sinusoidal:
            case PositionalEncodingType.Learned:
            case PositionalEncodingType.Relative:
            case PositionalEncodingType.Rotary:
            default:
                // For other encoding types, use the standard multi-head attention
                return new MultiHeadAttentionLayer<T>(
                    sequenceLength,
                    embeddingDimension,
                    headCount,
                    activationFunction);
        }
    }
    
    /// <summary>
    /// Creates an attention layer with additional custom parameters.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
    /// <param name="positionalEncodingType">The type of positional encoding being used.</param>
    /// <param name="sequenceLength">The maximum sequence length for the attention layer.</param>
    /// <param name="embeddingDimension">The size of each embedding vector.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <param name="activationFunction">The activation function for the attention layer.</param>
    /// <param name="parameters">A dictionary of additional parameters for specific attention layers.</param>
    /// <returns>An attention layer with custom parameters.</returns>
    /// <remarks>
    /// <para>
    /// This overload allows specifying additional parameters for certain attention layer types
    /// that support customization beyond the basic parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This version gives you more control over attention settings.
    /// 
    /// In addition to the basic settings, you can provide a dictionary of advanced parameters:
    /// - For ALiBi, you might specify the slope parameter
    /// - For T5RelativeBias, you might set the number of buckets or maximum distance
    /// 
    /// This allows for fine-tuning the attention mechanism for specific use cases.
    /// </para>
    /// </remarks>
    public static ILayer<T> Create<T>(
        PositionalEncodingType positionalEncodingType,
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        IActivationFunction<T> activationFunction,
        Dictionary<string, object> parameters)
    {
        switch (positionalEncodingType)
        {
            case PositionalEncodingType.ALiBi:
                double alibiSlope = parameters.TryGetValue("slope", out object? slopeValue) && slopeValue is double slopeDouble
                    ? slopeDouble
                    : -0.1;
                return new ALiBiAttentionLayer<T>(
                    sequenceLength,
                    embeddingDimension,
                    headCount,
                    activationFunction,
                    alibiSlope);
                
            case PositionalEncodingType.T5RelativeBias:
                int numBuckets = parameters.TryGetValue("numBuckets", out object? bucketsValue) && bucketsValue is int bucketsInt
                    ? bucketsInt
                    : 32;
                    
                int maxDistance = parameters.TryGetValue("maxDistance", out object? distanceValue) && distanceValue is int distanceInt
                    ? distanceInt
                    : 128;
                    
                bool bidirectional = parameters.TryGetValue("bidirectional", out object? biValue) && biValue is bool biBool
                    ? biBool
                    : true;
                    
                return new T5RelativeBiasAttentionLayer<T>(
                    sequenceLength,
                    embeddingDimension,
                    headCount,
                    activationFunction, 
                    numBuckets, 
                    maxDistance, 
                    bidirectional);
                
            default:
                // For other types, fall back to the simpler Create method
                return Create<T>(positionalEncodingType, sequenceLength, embeddingDimension, headCount, activationFunction);
        }
    }
}