namespace AiDotNet.Enums;

/// <summary>
/// Defines the mode of model compression to apply during serialization.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Compression mode determines when and how your model gets compressed.
/// Like choosing between automatically archiving files vs manually selecting what to archive,
/// you can let the system decide the best approach or take control yourself.
/// </para>
/// </remarks>
public enum ModelCompressionMode
{
    /// <summary>
    /// No compression is applied. The model is stored at full size.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you need maximum accuracy and don't care about file size,
    /// or when debugging to ensure compression isn't affecting your results.
    /// </para>
    /// </remarks>
    None,

    /// <summary>
    /// The system automatically selects the best compression strategy based on model characteristics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the recommended default. The system analyzes your model and
    /// chooses the compression approach that provides the best balance of size reduction and
    /// accuracy preservation. Like auto settings on a camera, it works well for most cases.
    /// </para>
    /// </remarks>
    Automatic,

    /// <summary>
    /// Compresses only the model weights, leaving other metadata uncompressed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Weights are the learned parameters that make up most of a model's size.
    /// This mode compresses just those weights while keeping configuration and metadata readable.
    /// Good when you need to inspect model settings but want smaller storage.
    /// </para>
    /// </remarks>
    WeightsOnly,

    /// <summary>
    /// Compresses the entire serialized model including all metadata.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This provides maximum compression by compressing everything.
    /// Best for production deployment where you want the smallest possible file size
    /// and don't need to inspect the model contents directly.
    /// </para>
    /// </remarks>
    Full
}
