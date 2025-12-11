using Newtonsoft.Json;

namespace AiDotNet.Tools;

/// <summary>
/// Represents a single SerpAPI search result.
/// </summary>
internal class SerpAPIResult
{
    [JsonProperty("title")]
    public string Title { get; set; } = "";

    [JsonProperty("link")]
    public string Link { get; set; } = "";

    [JsonProperty("snippet")]
    public string Snippet { get; set; } = "";
}
