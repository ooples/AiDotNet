using System.Net.Http.Headers;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Playground.Functions;

/// <summary>Preserves the public URL without running submitted code in the billing worker.</summary>
public sealed class ExecuteCodeProxy(IHttpClientFactory clients, IConfiguration configuration, ILogger<ExecuteCodeProxy> logger)
{
    public const string ClientName = "PlaygroundExecution";
    public const int MaxResponseBytes = 1024 * 1024;
    public const int DefaultTimeoutSeconds = 60;
    public const int MaxTimeoutSeconds = 180;
    private readonly HttpClient _client = clients.CreateClient(ClientName);

    public static void RegisterClient(IServiceCollection services)
    {
        // Functions creates function classes with ActivatorUtilities rather than resolving a
        // registered typed client. Request the named client explicitly in the constructor.
        services.AddHttpClient(ClientName, (provider, client) =>
            ConfigureClient(client, provider.GetRequiredService<IConfiguration>()))
            .ConfigurePrimaryHttpMessageHandler(() => new HttpClientHandler
            {
                AllowAutoRedirect = false,
                UseCookies = false,
                UseDefaultCredentials = false
            });
    }

    public static void ConfigureClient(HttpClient client, IConfiguration configuration)
    {
        var setting = configuration["PLAYGROUND_EXECUTION_TIMEOUT_SECONDS"];
        var seconds = DefaultTimeoutSeconds;
        if (setting is not null && (!int.TryParse(setting, out seconds) || seconds <= 0 || seconds > MaxTimeoutSeconds))
            throw new InvalidOperationException($"Execution timeout must be between 1 and {MaxTimeoutSeconds} seconds.");
        client.Timeout = TimeSpan.FromSeconds(seconds);
        client.MaxResponseContentBufferSize = MaxResponseBytes;
    }

    [Function("execute")]
    public async Task<IActionResult> Run(
        [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = null)] HttpRequest req)
    {
        req.HttpContext.Response.Headers.Append("Access-Control-Allow-Origin", "*");
        req.HttpContext.Response.Headers.Append("Access-Control-Allow-Methods", "POST, OPTIONS");
        req.HttpContext.Response.Headers.Append("Access-Control-Allow-Headers", "Content-Type");

        if (!Uri.TryCreate(configuration["PLAYGROUND_EXECUTION_URL"], UriKind.Absolute, out var endpoint)
            || endpoint.Scheme != Uri.UriSchemeHttps || !string.IsNullOrEmpty(endpoint.UserInfo)
            || !string.IsNullOrEmpty(endpoint.Query) || !string.IsNullOrEmpty(endpoint.Fragment)
            || endpoint.AbsolutePath != "/api/execute")
        {
            logger.LogError("Isolated execution endpoint is not configured correctly");
            return Unavailable(StatusCodes.Status503ServiceUnavailable);
        }

        try
        {
            // Forward only the body. Never propagate cookies, authorization, function keys,
            // webhook signatures, app settings, or ambient credentials to the code runner.
            using var request = new HttpRequestMessage(HttpMethod.Post, endpoint);
            request.Content = new StreamContent(req.Body);
            request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var response = await _client.SendAsync(request, req.HttpContext.RequestAborted);
            if (!string.Equals(response.Content.Headers.ContentType?.MediaType, "application/json", StringComparison.OrdinalIgnoreCase))
                return Unavailable(response.IsSuccessStatusCode ? StatusCodes.Status502BadGateway : (int)response.StatusCode);
            return new ContentResult
            {
                StatusCode = (int)response.StatusCode,
                ContentType = "application/json",
                Content = await response.Content.ReadAsStringAsync(req.HttpContext.RequestAborted)
            };
        }
        catch (Exception ex) when (ex is HttpRequestException or OperationCanceledException)
        {
            logger.LogWarning("Isolated execution service is unavailable");
            return Unavailable(StatusCodes.Status502BadGateway);
        }
    }

    private static ContentResult Unavailable(int status) => new()
    {
        StatusCode = status,
        ContentType = "application/json",
        Content = "{\"success\":false,\"error\":\"Execution service unavailable\"}"
    };
}
