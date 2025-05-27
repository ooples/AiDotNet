namespace AiDotNet.Factories;

using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.PositionalEncoding;

/// <summary>
/// Factory for creating positional encoding layers of different types.
/// </summary>
/// <remarks>
/// <para>
/// This factory provides a centralized way to create positional encoding layers
/// based on the PositionalEncodingType enum. It handles the details of creating
/// the specific implementation and provides default parameters.
/// </para>
/// <para><b>For Beginners:</b> This factory makes it easy to create different types of positional encoding.
/// 
/// Think of it like a specialized workshop that knows how to create various position encodings:
/// - You specify which type you want using the PositionalEncodingType enum
/// - The factory handles all the details of creating that specific type
/// - You get back the right encoding ready to use in your transformer model
/// 
/// This simplifies your code and ensures that positional encodings are created consistently
/// throughout your application.
/// </para>
/// </remarks>
public static class PositionalEncodingFactory
{
    /// <summary>
    /// Creates a positional encoding layer of the specified type.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
    /// <param name="encodingType">The type of positional encoding to create.</param>
    /// <param name="maxSequenceLength">The maximum sequence length that the encoding can handle.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    /// <returns>A positional encoding layer of the specified type.</returns>
    /// <exception cref="ArgumentException">Thrown when an invalid encoding type is specified.</exception>
    /// <remarks>
    /// <para>
    /// This method creates and returns a positional encoding layer of the specified type.
    /// Each encoding type has different characteristics and is suitable for different use cases.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the positional encoding you need.
    /// 
    /// You specify:
    /// - The type of encoding (like Sinusoidal, Learned, etc.)
    /// - The maximum sequence length your model will handle
    /// - The size of your embeddings
    /// 
    /// The method then creates the right type of encoding with these specifications.
    /// Different encoding types are better suited for different tasks or model sizes,
    /// so having this flexibility is important.
    /// </para>
    /// </remarks>
    public static ILayer<T> Create<T>(
        PositionalEncodingType encodingType,
        int maxSequenceLength,
        int embeddingSize)
    {
        switch (encodingType)
        {
            case PositionalEncodingType.None:
                // Return a pass-through layer that doesn't modify the input
                return new LambdaLayer<T>(
                    [maxSequenceLength, embeddingSize],
                    [maxSequenceLength, embeddingSize],
                    input => input,
                    (input, grad) => grad,
                    new IdentityActivation<T>() as IActivationFunction<T>);
                
            case PositionalEncodingType.Sinusoidal:
                return new SinusoidalPositionalEncoding<T>(maxSequenceLength, embeddingSize);
                
            case PositionalEncodingType.Learned:
                return new LearnedPositionalEncoding<T>(maxSequenceLength, embeddingSize);
                
            case PositionalEncodingType.Relative:
                return new RelativePositionalEncoding<T>(maxSequenceLength, embeddingSize);
                
            case PositionalEncodingType.Rotary:
                return new RotaryPositionalEncoding<T>(maxSequenceLength, embeddingSize);
                
            case PositionalEncodingType.ALiBi:
                return new ALiBiPositionalEncoding<T>(maxSequenceLength, embeddingSize);
                
            case PositionalEncodingType.T5RelativeBias:
                return new T5RelativeBiasPositionalEncoding<T>(maxSequenceLength, embeddingSize);
                
            default:
                throw new ArgumentException($"Invalid positional encoding type: {encodingType}");
        }
    }
    
    /// <summary>
    /// Creates a positional encoding layer with additional custom parameters.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
    /// <param name="encodingType">The type of positional encoding to create.</param>
    /// <param name="maxSequenceLength">The maximum sequence length that the encoding can handle.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    /// <param name="parameters">A dictionary of additional parameters for specific encoding types.</param>
    /// <returns>A positional encoding layer of the specified type with custom parameters.</returns>
    /// <exception cref="ArgumentException">Thrown when an invalid encoding type is specified.</exception>
    /// <remarks>
    /// <para>
    /// This overload allows specifying additional parameters for certain encoding types
    /// that support customization beyond the basic parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This version gives you more control over encoding settings.
    /// 
    /// In addition to the basic settings, you can provide a dictionary of advanced parameters:
    /// - For Rotary encoding, you might specify the base frequency
    /// - For ALiBi, you might adjust the slope parameter
    /// - For T5RelativeBias, you might set the number of buckets or maximum distance
    /// 
    /// This allows for fine-tuning the encodings for specific use cases or model architectures.
    /// </para>
    /// </remarks>
    public static ILayer<T> Create<T>(
        PositionalEncodingType encodingType,
        int maxSequenceLength,
        int embeddingSize,
        Dictionary<string, object> parameters)
    {
        switch (encodingType)
        {
            case PositionalEncodingType.Rotary:
                double ropeBase = parameters.TryGetValue("base", out object? baseValue) && baseValue is double baseDouble
                    ? baseDouble
                    : 10000.0;
                return new RotaryPositionalEncoding<T>(maxSequenceLength, embeddingSize, ropeBase);
                
            case PositionalEncodingType.ALiBi:
                double alibiSlope = parameters.TryGetValue("slope", out object? slopeValue) && slopeValue is double slopeDouble
                    ? slopeDouble
                    : -0.1;
                return new ALiBiPositionalEncoding<T>(maxSequenceLength, embeddingSize, alibiSlope);
                
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
                    
                return new T5RelativeBiasPositionalEncoding<T>(
                    maxSequenceLength, 
                    embeddingSize, 
                    numBuckets, 
                    maxDistance, 
                    bidirectional);
                
            default:
                // For other types, fall back to the simpler Create method
                return Create<T>(encodingType, maxSequenceLength, embeddingSize);
        }
    }
}