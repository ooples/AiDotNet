using System.Net.Http;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// An <see cref="IChatClient{T}"/> for Azure OpenAI. Azure uses the same Chat Completions wire format as
/// OpenAI, so this derives from <see cref="OpenAIChatClient{T}"/> and only changes the endpoint (a
/// per-deployment URL with an <c>api-version</c>) and the authentication header (<c>api-key</c>).
/// </summary>
/// <typeparam name="T">The numeric type used across the AiDotNet ecosystem.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Azure hosts the same OpenAI models but behind your own Azure resource. You
/// address a "deployment" you created, the URL includes an API version, and the key goes in an
/// <c>api-key</c> header instead of <c>Authorization</c>. Everything else is identical to OpenAI, which is
/// why this class reuses the OpenAI request/response logic.
/// </para>
/// </remarks>
public sealed class AzureOpenAIChatClient<T> : OpenAIChatClient<T>
{
    /// <summary>
    /// Initializes a new Azure OpenAI chat client.
    /// </summary>
    /// <param name="apiKey">The Azure OpenAI API key.</param>
    /// <param name="deploymentName">The name of your deployed model.</param>
    /// <param name="resourceEndpoint">The Azure resource base URL, e.g. <c>https://my-resource.openai.azure.com</c>.</param>
    /// <param name="apiVersion">The Azure OpenAI API version (default <c>2024-10-21</c>).</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <exception cref="ArgumentNullException">Thrown when a required argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when a required argument is empty/whitespace.</exception>
    public AzureOpenAIChatClient(
        string apiKey,
        string deploymentName,
        string resourceEndpoint,
        string apiVersion = "2024-10-21",
        HttpClient? httpClient = null)
        : base(apiKey, deploymentName, BuildEndpoint(resourceEndpoint, deploymentName, apiVersion), httpClient)
    {
    }

    /// <inheritdoc/>
    protected override void ApplyAuthentication(HttpRequestMessage request)
    {
        request.Headers.Add("api-key", ApiKey);
    }

    private static string BuildEndpoint(string resourceEndpoint, string deploymentName, string apiVersion)
    {
        Guard.NotNullOrWhiteSpace(resourceEndpoint);
        Guard.NotNullOrWhiteSpace(deploymentName);
        Guard.NotNullOrWhiteSpace(apiVersion);
        var baseUrl = resourceEndpoint.TrimEnd('/');
        return $"{baseUrl}/openai/deployments/{deploymentName}/chat/completions?api-version={apiVersion}";
    }
}
