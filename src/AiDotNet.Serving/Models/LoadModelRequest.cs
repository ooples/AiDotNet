using AiDotNet.Serving.Configuration;

namespace AiDotNet.Serving.Models;

/// <summary>
/// Request to load a new model.
/// </summary>
public class LoadModelRequest
{
    /// <summary>
    /// Gets or sets the unique name for the model.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the file path to the serialized model.
    /// </summary>
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the numeric type used by the model.
    /// Default is Double.
    /// </summary>
    public NumericType NumericType { get; set; } = NumericType.Double;

    /// <summary>
    /// Gets or sets an optional tokenizer source (a local directory/file such as <c>tokenizer.json</c>,
    /// or a HuggingFace model id). When set, a tokenizer is loaded and associated with this model so it
    /// can be served through the OpenAI-compatible API (<c>/v1/chat/completions</c>, <c>/v1/completions</c>).
    /// </summary>
    public string? TokenizerPath { get; set; }
}

