using Newtonsoft.Json;

namespace AiDotNet.Tools;

/// <summary>
/// Response from Bing Search API.
/// </summary>
internal class BingSearchResponse
{
    [JsonProperty("webPages")]
    public BingWebPages? WebPages { get; set; }
}
