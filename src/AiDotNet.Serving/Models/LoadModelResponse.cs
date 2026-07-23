namespace AiDotNet.Serving.Models;

/// <summary>
/// Response after loading a model.
/// </summary>
public class LoadModelResponse
{
    /// <summary>
    /// Gets or sets whether the model was loaded successfully.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the error message if loading failed.
    /// </summary>
    public string? Error { get; set; }

    /// <summary>
    /// Gets or sets information about the loaded model.
    /// </summary>
    public ModelInfo? ModelInfo { get; set; }
}

