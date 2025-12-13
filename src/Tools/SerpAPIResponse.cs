using Newtonsoft.Json;

namespace AiDotNet.Tools;

/// <summary>
/// Response from SerpAPI.
/// </summary>
internal class SerpAPIResponse
{
    [JsonProperty("organic_results")]
    public SerpAPIResult[]? OrganicResults { get; set; }
}
