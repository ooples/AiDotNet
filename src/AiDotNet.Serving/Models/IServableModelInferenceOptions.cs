namespace AiDotNet.Serving.Models;

/// <summary>
/// Internal serving-only inference options derived from the model's facade configuration.
/// </summary>
internal interface IServableModelInferenceOptions
{
    bool EnableBatching { get; }
    bool EnableSpeculativeDecoding { get; }
}

