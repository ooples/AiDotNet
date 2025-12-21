using System.Net.Http;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.Tools;

/// <summary>
/// A production-ready tool that performs web searches using external search APIs.
/// Supports Bing Search API and SerpAPI with configurable options.
/// </summary>
/// <remarks>
/// For Beginners:
/// This tool lets an AI agent search the internet for current information.
///
/// Why this is useful:
/// - **Current events**: Information not in training data or knowledge base
/// - **Real-time data**: Stock prices, weather, news
/// - **Broader knowledge**: Topics beyond your document collection
/// - **Fact-checking**: Verify claims against public information
///
/// Unlike the mock SearchTool, this uses real search APIs:
/// - **Bing Search API**: Microsoft's search, good coverage, affordable
/// - **SerpAPI**: Google search wrapper, comprehensive, slightly more expensive
///
/// Example:
/// <code>
/// var webSearch = new WebSearchTool(
///     apiKey: "your-bing-api-key",
///     provider: SearchProvider.Bing);
///
/// var agent = new Agent&lt;double&gt;(chatModel, new[] { webSearch });
/// var result = await agent.RunAsync("What's the current price of AAPL stock?");
/// // Agent will search the web and get real-time information
/// </code>
///
/// API Setup:
/// - Bing: Get key at portal.azure.com → Bing Search v7
/// - SerpAPI: Get key at serpapi.com
/// </remarks>
public class WebSearchTool : ITool
{
    private readonly HttpClient _httpClient;
    private readonly string _apiKey;
    private readonly SearchProvider _provider;
    private readonly int _defaultResultCount;
    private readonly string _market;

    /// <summary>
    /// Initializes a new instance of the <see cref="WebSearchTool"/> class.
    /// </summary>
    /// <param name="apiKey">The API key for the search provider.</param>
    /// <param name="provider">The search provider to use (default: Bing).</param>
    /// <param name="resultCount">Default number of search results to return (default: 5).</param>
    /// <param name="market">Market/region for search results (default: "en-US").</param>
    /// <param name="httpClient">Optional HTTP client for advanced scenarios.</param>
    /// <exception cref="ArgumentException">Thrown when API key is null or empty.</exception>
    /// <remarks>
    /// For Beginners:
    ///
    /// **Parameters:**
    /// - apiKey: Your search API key (keep this secret!)
    /// - provider: Which search engine to use (Bing or SerpAPI)
    /// - resultCount: How many results to return (3-10 is typical)
    /// - market: Language/region (e.g., "en-US", "en-GB", "fr-FR")
    ///
    /// **Choosing a provider:**
    /// - **Bing**: Easier setup, Azure integration, $3-7 per 1000 searches
    /// - **SerpAPI**: More features, Google results, $50/month for 5000 searches
    ///
    /// **Getting API keys:**
    /// - Bing: Azure Portal → Create Resource → Bing Search v7
    /// - SerpAPI: Sign up at serpapi.com → Dashboard → API Key
    /// </remarks>
    public WebSearchTool(
        string apiKey,
        SearchProvider provider = SearchProvider.Bing,
        int resultCount = 5,
        string market = "en-US",
        HttpClient? httpClient = null)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new ArgumentException("API key cannot be null or empty.", nameof(apiKey));
        }

        if (resultCount < 1 || resultCount > 50)
        {
            throw new ArgumentException("Result count must be between 1 and 50.", nameof(resultCount));
        }

        _apiKey = apiKey;
        _provider = provider;
        _defaultResultCount = resultCount;
        _market = market;
        _httpClient = httpClient ?? new HttpClient();
        _httpClient.Timeout = TimeSpan.FromSeconds(30);
    }

    /// <inheritdoc/>
    public string Name => "WebSearch";

    /// <inheritdoc/>
    public string Description =>
        "Searches the web for current information and returns relevant results. " +
        "Use this when you need up-to-date information, current events, real-time data, " +
        "or information not available in the knowledge base. " +
        "Input should be a search query describing what you want to find. " +
        "Examples: 'latest news about artificial intelligence', 'current weather in London', " +
        "'AAPL stock price today', 'recent breakthroughs in quantum computing'";

    /// <inheritdoc/>
    public string Execute(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "Error: Search query cannot be empty.";
        }

        try
        {
            return _provider switch
            {
                SearchProvider.Bing => SearchBingAsync(input).GetAwaiter().GetResult(),
                SearchProvider.SerpAPI => SearchSerpAPIAsync(input).GetAwaiter().GetResult(),
                _ => "Error: Unsupported search provider."
            };
        }
        catch (Exception ex)
        {
            return $"Error performing web search: {ex.Message}";
        }
    }

    /// <summary>
    /// Performs a search using Bing Search API v7.
    /// </summary>
    private async Task<string> SearchBingAsync(string query)
    {
        var url = $"https://api.bing.microsoft.com/v7.0/search?q={Uri.EscapeDataString(query)}&count={_defaultResultCount}&mkt={_market}";

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.Add("Ocp-Apim-Subscription-Key", _apiKey);

        using var response = await _httpClient.SendAsync(request).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HttpRequestException($"Bing Search API error ({response.StatusCode}): {errorContent}");
        }

        var content = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var searchResponse = JsonConvert.DeserializeObject<BingSearchResponse>(content);

        if (searchResponse?.WebPages?.Value == null || searchResponse.WebPages.Value.Length == 0)
        {
            return $"No web search results found for: '{query}'";
        }

        return FormatBingResults(query, searchResponse.WebPages.Value);
    }

    /// <summary>
    /// Performs a search using SerpAPI.
    /// </summary>
    private async Task<string> SearchSerpAPIAsync(string query)
    {
        var url = $"https://serpapi.com/search.json?q={Uri.EscapeDataString(query)}&num={_defaultResultCount}&api_key={_apiKey}&engine=google";

        using var response = await _httpClient.GetAsync(url).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HttpRequestException($"SerpAPI error ({response.StatusCode}): {errorContent}");
        }

        var content = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var searchResponse = JsonConvert.DeserializeObject<SerpAPIResponse>(content);

        if (searchResponse?.OrganicResults == null || searchResponse.OrganicResults.Length == 0)
        {
            return $"No web search results found for: '{query}'";
        }

        return FormatSerpAPIResults(query, searchResponse.OrganicResults);
    }

    /// <summary>
    /// Formats Bing search results into a readable string.
    /// </summary>
    private static string FormatBingResults(string query, BingWebPage[] results)
    {
        var output = new System.Text.StringBuilder();
        output.AppendLine($"Web search results for '{query}':");
        output.AppendLine();

        for (int i = 0; i < results.Length; i++)
        {
            var result = results[i];
            output.AppendLine($"[{i + 1}] {result.Name}");
            output.AppendLine($"    {result.Snippet}");
            output.AppendLine($"    URL: {result.Url}");
            output.AppendLine();
        }

        return output.ToString().TrimEnd();
    }

    /// <summary>
    /// Formats SerpAPI search results into a readable string.
    /// </summary>
    private static string FormatSerpAPIResults(string query, SerpAPIResult[] results)
    {
        var output = new System.Text.StringBuilder();
        output.AppendLine($"Web search results for '{query}':");
        output.AppendLine();

        for (int i = 0; i < results.Length; i++)
        {
            var result = results[i];
            output.AppendLine($"[{i + 1}] {result.Title}");
            output.AppendLine($"    {result.Snippet}");
            output.AppendLine($"    URL: {result.Link}");
            output.AppendLine();
        }

        return output.ToString().TrimEnd();
    }
}
