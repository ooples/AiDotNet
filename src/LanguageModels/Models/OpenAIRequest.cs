using Newtonsoft.Json;

namespace AiDotNet.LanguageModels.Models;

/// <summary>
/// Represents an OpenAI Chat Completions API request.
/// </summary>
internal class OpenAIRequest
{
    [JsonProperty("model")]
    public string Model { get; set; } = "";

    [JsonProperty("messages")]
    public OpenAIMessage[] Messages { get; set; } = Array.Empty<OpenAIMessage>();

    [JsonProperty("temperature")]
    public double Temperature { get; set; }

    [JsonProperty("max_tokens")]
    public int MaxTokens { get; set; }

    [JsonProperty("top_p")]
    public double TopP { get; set; }

    [JsonProperty("frequency_penalty")]
    public double FrequencyPenalty { get; set; }

    [JsonProperty("presence_penalty")]
    public double PresencePenalty { get; set; }
}
