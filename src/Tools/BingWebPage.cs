using Newtonsoft.Json;

namespace AiDotNet.Tools;

/// <summary>
/// Represents a single Bing web page result.
/// </summary>
internal class BingWebPage
{
    [JsonProperty("name")]
    public string Name { get; set; } = "";

    [JsonProperty("url")]
    public string Url { get; set; } = "";

    [JsonProperty("snippet")]
    public string Snippet { get; set; } = "";
}
