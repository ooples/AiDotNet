using Newtonsoft.Json;

namespace AiDotNet.Tools;

/// <summary>
/// Container for Bing web page results.
/// </summary>
internal class BingWebPages
{
    [JsonProperty("value")]
    public BingWebPage[]? Value { get; set; }
}
