using System.Net.Http;
using AiDotNet.Configuration;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// An Azure OpenAI connector that applies the same reasoning-model request rules as
/// <see cref="ReasoningOpenAIChatClient{T}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used across the AiDotNet ecosystem.</typeparam>
/// <remarks>
/// <para>
/// Azure hosts the same o-series and GPT-5 weights behind a per-deployment URL and an <c>api-key</c> header, and
/// enforces exactly the same parameter rules as the public endpoint. That is precisely why reasoning-model
/// detection keys off the model id rather than the base URL: a check on the endpoint would classify this client
/// wrongly, while the deployment name resolves through the same profile registry the OpenAI connector uses.
/// </para>
/// <para>
/// One Azure-specific caveat is worth knowing: on Azure the model id this client sees is your <em>deployment
/// name</em>, not the underlying model. A deployment of <c>o3-mini</c> named <c>o3-mini</c> is recognised
/// automatically; one named <c>reasoning-prod</c> is not, because nothing in that name says what it runs. Give the
/// deployment a name that starts with the model family, or supply a registry whose profile claims your naming
/// convention.
/// </para>
/// <para><b>For Beginners:</b> Use this instead of the plain Azure connector when your Azure deployment might be
/// one of the newer reasoning models. It fixes the request settings those models refuse, exactly as the OpenAI
/// version does, and behaves identically to the plain connector for every other model. Name your Azure deployment
/// after the model it runs (for example <c>o3-mini</c>) so the library can tell what it is.</para>
/// <para>
/// The class is left open for derivation, like <see cref="ReasoningOpenAIChatClient{T}"/>, so a gateway that
/// changes only the endpoint shape or the authentication header can subclass it rather than duplicating the
/// reasoning rules.
/// </para>
/// </remarks>
public class ReasoningAzureOpenAIChatClient<T> : ReasoningOpenAIChatClient<T>
{
    /// <summary>Initializes a reasoning-aware Azure OpenAI chat client.</summary>
    /// <param name="apiKey">The Azure OpenAI API key. It is used for the <c>api-key</c> header and nothing else.</param>
    /// <param name="deploymentName">
    /// The name of your deployed model, which also acts as the model id used to recognise a reasoning model.
    /// </param>
    /// <param name="resourceEndpoint">The Azure resource base URL, for example <c>https://my-resource.openai.azure.com</c>.</param>
    /// <param name="apiVersion">The Azure OpenAI API version (default <c>2024-10-21</c>).</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <param name="options">Reasoning-model settings; <c>null</c> uses the defaults and the built-in profiles.</param>
    /// <exception cref="ArgumentNullException">A required argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A required argument is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is outside its permitted range.</exception>
    public ReasoningAzureOpenAIChatClient(
        string apiKey,
        string deploymentName,
        string resourceEndpoint,
        string apiVersion = "2024-10-21",
        HttpClient? httpClient = null,
        ReasoningModelOptions? options = null)
        : base(apiKey, deploymentName, BuildEndpoint(resourceEndpoint, deploymentName, apiVersion), httpClient, options)
    {
    }

    /// <inheritdoc/>
    protected override void ApplyAuthentication(HttpRequestMessage request)
    {
        Guard.NotNull(request);
        request.Headers.Add("api-key", ApiKey);
    }

    private static string BuildEndpoint(string resourceEndpoint, string deploymentName, string apiVersion)
    {
        Guard.NotNullOrWhiteSpace(resourceEndpoint);
        Guard.NotNullOrWhiteSpace(deploymentName);
        Guard.NotNullOrWhiteSpace(apiVersion);
        string baseUrl = resourceEndpoint.TrimEnd('/');
        return baseUrl + "/openai/deployments/" + deploymentName + "/chat/completions?api-version=" + apiVersion;
    }
}
