using System.Net.Http.Headers;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Playground.Functions;

/// <summary>Preserves the public URL without running submitted code in the billing worker.</summary>
public sealed class ExecuteCodeProxy(HttpClient client, IConfiguration configuration, ILogger<ExecuteCodeProxy> logger)
{
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
            return new StatusCodeResult(StatusCodes.Status503ServiceUnavailable);
        }

        try
        {
            // Forward only the body. Never propagate cookies, authorization, function keys,
            // webhook signatures, app settings, or ambient credentials to the code runner.
            using var request = new HttpRequestMessage(HttpMethod.Post, endpoint);
            request.Content = new StreamContent(req.Body);
            request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var response = await client.SendAsync(request, req.HttpContext.RequestAborted);
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
            return new StatusCodeResult(StatusCodes.Status502BadGateway);
        }
    }
}
