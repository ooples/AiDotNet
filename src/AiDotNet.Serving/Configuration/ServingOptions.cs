namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration options for the model serving framework.
/// This class defines settings for server behavior, request batching, and startup model loading.
/// </summary>
public class ServingOptions
{
    /// <summary>
    /// Gets or sets the port number on which the server will listen.
    /// Default is 5000.
    /// </summary>
    public int Port { get; set; } = 5000;

    /// <summary>
    /// Gets or sets the batching window in milliseconds.
    /// This is the maximum time the batcher will wait before processing accumulated requests.
    /// Default is 10 milliseconds.
    /// </summary>
    public int BatchingWindowMs { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum batch size for inference requests.
    /// If set to 0 or less, there is no limit on batch size.
    /// Default is 100.
    /// </summary>
    public int MaxBatchSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the list of models to load at startup.
    /// Each entry should be a dictionary with "name" and "path" keys.
    /// </summary>
    public List<StartupModel> StartupModels { get; set; } = new();
}

/// <summary>
/// Represents a model to be loaded when the server starts.
/// </summary>
public class StartupModel
{
    /// <summary>
    /// Gets or sets the name of the model.
    /// This will be used as the identifier for API requests.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the file path to the serialized model.
    /// </summary>
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the numeric type used by the model.
    /// Supported values: "double", "float", "decimal"
    /// Default is "double".
    /// </summary>
    public string NumericType { get; set; } = "double";
}
