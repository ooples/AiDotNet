namespace AiDotNet.Interfaces;

using System.Threading.Tasks;
using AiDotNet.Compression;

/// <summary>
/// Defines the interface for model compressors.
/// </summary>
/// <remarks>
/// <para>
/// A model compressor provides methods for compressing machine learning models
/// to reduce their size and potentially improve inference speed, while attempting
/// to maintain model accuracy within acceptable thresholds.
/// </para>
/// <para><b>For Beginners:</b> This interface defines what a model compressor can do.
/// 
/// A model compressor is responsible for:
/// - Taking a trained AI model and making it smaller
/// - Applying different compression techniques like quantization or pruning
/// - Providing ways to save the compressed model
/// - Letting you evaluate the compressed model's performance
/// 
/// This interface ensures all compressors provide a consistent set of capabilities.
/// </para>
/// </remarks>
/// <typeparam name="TModel">The type of model to compress.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IModelCompressor<TModel, TInput, TOutput>
    where TModel : class
{
    /// <summary>
    /// Compresses a model using the specified options.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <param name="options">Options for the compression process.</param>
    /// <returns>The compressed model.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the compression technique specified in the options to the
    /// provided model and returns a compressed version of the model.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main method that actually compresses your model.
    /// 
    /// You provide:
    /// - The original model you want to compress
    /// - Options specifying how you want to compress it (technique, settings, etc.)
    /// 
    /// The method returns a compressed version of your model that you can then use
    /// for inference or further processing.
    /// </para>
    /// </remarks>
    TModel Compress(TModel model, object options);

    /// <summary>
    /// Asynchronously compresses a model using the specified options.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <param name="options">Options for the compression process.</param>
    /// <returns>A task representing the asynchronous operation, containing the compressed model.</returns>
    /// <remarks>
    /// <para>
    /// This method asynchronously applies the compression technique specified in the options
    /// to the provided model and returns a compressed version of the model.
    /// </para>
    /// <para><b>For Beginners:</b> This is the async version of the compression method.
    /// 
    /// It works the same as Compress() but doesn't block the current thread while compressing.
    /// This is useful for:
    /// - UI applications where you don't want to freeze the interface
    /// - Server applications where you want to handle other requests while compressing
    /// - Long-running compression tasks on large models
    /// </para>
    /// </remarks>
    Task<TModel> CompressAsync(TModel model, object options);

    /// <summary>
    /// Evaluates the compressed model against the original model.
    /// </summary>
    /// <param name="originalModel">The original uncompressed model.</param>
    /// <param name="compressedModel">The compressed model.</param>
    /// <param name="testInputs">A set of inputs for evaluating the models.</param>
    /// <param name="testOutputs">The expected outputs for the test inputs.</param>
    /// <returns>A result object containing metrics about the compression.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates the performance of the compressed model compared to the
    /// original model using the provided test data.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you how well the compression worked.
    /// 
    /// After compression, you'll want to know:
    /// - How much smaller is the model?
    /// - Is it faster?
    /// - How much accuracy did it lose?
    /// 
    /// This method runs both the original and compressed models on test data and
    /// gives you a detailed comparison of their performance.
    /// </para>
    /// </remarks>
    ModelCompressionResult EvaluateCompression(
        TModel originalModel,
        TModel compressedModel,
        TInput[] testInputs,
        TOutput[] testOutputs);

    /// <summary>
    /// Serializes a compressed model to a specified format.
    /// </summary>
    /// <param name="model">The compressed model to serialize.</param>
    /// <param name="filePath">The path where the serialized model should be saved.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the compressed model to a file in a format that is
    /// optimized for the specific compression technique used.
    /// </para>
    /// <para><b>For Beginners:</b> This saves your compressed model to a file.
    /// 
    /// Different compression techniques might use different formats to get the
    /// maximum benefit from compression. This method handles those details for you,
    /// producing an optimized file that:
    /// - Takes up minimum space
    /// - Can be loaded efficiently
    /// - Preserves the compression benefits
    /// </para>
    /// </remarks>
    void SerializeCompressedModel(TModel model, string filePath);

    /// <summary>
    /// Deserializes a compressed model from a specified format.
    /// </summary>
    /// <param name="filePath">The path where the serialized model is stored.</param>
    /// <returns>The deserialized compressed model.</returns>
    /// <remarks>
    /// <para>
    /// This method loads a compressed model from a file that was previously saved
    /// using the SerializeCompressedModel method.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a compressed model from a file.
    /// 
    /// After you've saved a compressed model, you'll need to load it again later.
    /// This method handles all the details of reading the file and reconstructing
    /// your compressed model so it's ready to use for making predictions.
    /// </para>
    /// </remarks>
    TModel DeserializeCompressedModel(string filePath);
}