using Newtonsoft.Json;

namespace AiDotNet.LanguageModels.Models;

/// <summary>
/// Represents an OpenAI Chat Completions API response.
/// </summary>
internal class OpenAIResponse
{
    [JsonProperty("id")]
    public string? Id { get; set; }

    [JsonProperty("choices")]
    public OpenAIChoice[]? Choices { get; set; }

    [JsonProperty("usage")]
    public OpenAIUsage? Usage { get; set; }
}
