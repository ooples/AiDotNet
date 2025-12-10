using Newtonsoft.Json;

namespace AiDotNet.LanguageModels.Models;

/// <summary>
/// Represents a message in the OpenAI Chat Completions API.
/// </summary>
internal class OpenAIMessage
{
    [JsonProperty("role")]
    public string Role { get; set; } = "";

    [JsonProperty("content")]
    public string Content { get; set; } = "";
}
