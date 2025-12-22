using AiDotNet.Serving.Security.ApiKeys;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers.Admin;

/// <summary>
/// Administrative endpoints for managing API keys.
/// </summary>
[ApiController]
[Route("api/admin/api-keys")]
[Authorize(Policy = "AiDotNetAdmin")]
public sealed class ApiKeysController : ControllerBase
{
    private readonly IApiKeyService _apiKeys;

    public ApiKeysController(IApiKeyService apiKeys)
    {
        _apiKeys = apiKeys ?? throw new ArgumentNullException(nameof(apiKeys));
    }

    /// <summary>
    /// Creates a new API key.
    /// </summary>
    [HttpPost]
    [ProducesResponseType(typeof(ApiKeyCreateResponse), StatusCodes.Status200OK)]
    public async Task<IActionResult> Create([FromBody] ApiKeyCreateRequest request, CancellationToken cancellationToken)
    {
        var response = await _apiKeys.CreateAsync(request, cancellationToken).ConfigureAwait(false);
        return Ok(response);
    }

    /// <summary>
    /// Revokes an API key by key id.
    /// </summary>
    [HttpPost("{keyId}/revoke")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> Revoke(string keyId, CancellationToken cancellationToken)
    {
        var revoked = await _apiKeys.RevokeAsync(keyId, cancellationToken).ConfigureAwait(false);
        if (!revoked)
        {
            return NotFound(new { error = "API key not found." });
        }

        return Ok(new { success = true });
    }
}

