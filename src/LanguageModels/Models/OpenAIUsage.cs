using Newtonsoft.Json;

namespace AiDotNet.LanguageModels.Models;

/// <summary>
/// Represents token usage information in the OpenAI API response.
/// </summary>
internal class OpenAIUsage
{
    [JsonProperty("prompt_tokens")]
    public int PromptTokens { get; set; }

    [JsonProperty("completion_tokens")]
    public int CompletionTokens { get; set; }

    [JsonProperty("total_tokens")]
    public int TotalTokens { get; set; }
}
