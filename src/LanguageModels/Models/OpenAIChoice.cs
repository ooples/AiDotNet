using Newtonsoft.Json;

namespace AiDotNet.LanguageModels.Models;

/// <summary>
/// Represents a choice in the OpenAI API response.
/// </summary>
internal class OpenAIChoice
{
    [JsonProperty("index")]
    public int Index { get; set; }

    [JsonProperty("message")]
    public OpenAIMessage? Message { get; set; }

    [JsonProperty("finish_reason")]
    public string? FinishReason { get; set; }
}
