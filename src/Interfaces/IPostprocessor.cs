namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a postprocessor that transforms model outputs into final results.
/// </summary>
/// <remarks>
/// <para>
/// This is the core interface for all postprocessing operations in AiDotNet.
/// Unlike preprocessing (which follows sklearn-style Fit/Transform pattern),
/// postprocessing typically doesn't require fitting - it transforms model outputs
/// directly into the desired format.
/// </para>
/// <para><b>For Beginners:</b> A postprocessor is like a translator that:
/// 1. Takes raw model output (like numbers or probabilities)
/// 2. Converts it into something meaningful (like text, labels, or structured data)
///
/// Examples:
/// - Converting softmax outputs to class labels
/// - Decoding text from token IDs
/// - Applying Non-Maximum Suppression to bounding boxes
/// - Cleaning up OCR text output
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (model output).</typeparam>
/// <typeparam name="TOutput">The output data type after postprocessing.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("Postprocessor")]
public interface IPostprocessor<T, TInput, TOutput>
{
    /// <summary>
    /// Gets whether this postprocessor requires configuration before use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Some postprocessors (like label decoders) need configuration (like label mappings).
    /// Returns true after <see cref="Configure"/> has been called for such postprocessors.
    /// Stateless postprocessors always return true.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the postprocessor is ready to use.
    /// Most postprocessors are ready immediately; some need setup first.
    /// </para>
    /// </remarks>
    bool IsConfigured { get; }

    /// <summary>
    /// Gets whether this postprocessor supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Some postprocessors can reverse their transformation (e.g., converting labels
    /// back to indices). Others cannot (e.g., NMS removes information permanently).
    /// </para>
    /// <para><b>For Beginners:</b> If this is true, you can "undo" the transformation.
    /// This is useful for converting human-readable outputs back to model format.
    /// </para>
    /// </remarks>
    bool SupportsInverse { get; }

    /// <summary>
    /// Configures the postprocessor with optional settings.
    /// </summary>
    /// <param name="settings">Optional configuration dictionary.</param>
    /// <remarks>
    /// <para>
    /// Call this method to configure postprocessors that require setup.
    /// For example, a label decoder might need a label-to-index mapping.
    /// Stateless postprocessors can ignore this call.
    /// </para>
    /// <para><b>For Beginners:</b> Use this to set up any options the postprocessor needs.
    /// Many postprocessors work out of the box without configuration.
    /// </para>
    /// </remarks>
    void Configure(Dictionary<string, object>? settings = null);

    /// <summary>
    /// Transforms model output into the final result format.
    /// </summary>
    /// <param name="input">The model output to process.</param>
    /// <returns>The postprocessed result.</returns>
    /// <remarks>
    /// <para>
    /// This is the main method that converts raw model output into useful results.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the magic happens - raw model outputs
    /// become meaningful results like class labels, cleaned text, or structured data.
    /// </para>
    /// </remarks>
    TOutput Process(TInput input);

    /// <summary>
    /// Transforms a batch of model outputs.
    /// </summary>
    /// <param name="inputs">The model outputs to process.</param>
    /// <returns>The postprocessed results.</returns>
    /// <remarks>
    /// <para>
    /// Processes multiple inputs efficiently. The default implementation calls
    /// <see cref="Process"/> for each input, but implementations may override
    /// for batch-optimized processing.
    /// </para>
    /// </remarks>
    IList<TOutput> ProcessBatch(IEnumerable<TInput> inputs);

    /// <summary>
    /// Reverses the postprocessing (if supported).
    /// </summary>
    /// <param name="output">The postprocessed result.</param>
    /// <returns>The original model output format.</returns>
    /// <exception cref="NotSupportedException">Thrown if inverse is not supported.</exception>
    /// <remarks>
    /// <para>
    /// Converts postprocessed results back to model output format.
    /// This is useful for converting user inputs into model-compatible format.
    /// </para>
    /// <para><b>For Beginners:</b> If you have a class label like "cat" and need
    /// to convert it back to model format for comparison, use this method.
    /// </para>
    /// </remarks>
    TInput Inverse(TOutput output);
}
