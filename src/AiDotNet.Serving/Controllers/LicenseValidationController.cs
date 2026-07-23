using AiDotNet.Serving.Security.Licensing;
using AiDotNet.Validation;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Public endpoint for license key validation. No authentication required — the license key itself
/// is the credential. Marked [AllowAnonymous] so the serving FallbackPolicy
/// (RequireAuthenticatedUser) doesn't block clients whose entire purpose for hitting this endpoint
/// is to validate their license before they have an authenticated session.
/// </summary>
[ApiController]
[Route("api/licenses")]
[AllowAnonymous]
public sealed class LicenseValidationController : ControllerBase
{
    private readonly ILicenseService _licenses;

    public LicenseValidationController(ILicenseService licenses)
    {
        Guard.NotNull(licenses);
        _licenses = licenses;
    }

    /// <summary>
    /// Validates a license key and records an advisory machine activation.
    /// </summary>
    [HttpPost("validate")]
    [ProducesResponseType(typeof(LicenseValidationResponse), StatusCodes.Status200OK)]
    public async Task<IActionResult> Validate([FromBody] LicenseValidateRequest request, CancellationToken cancellationToken)
    {
        var response = await _licenses.ValidateAsync(request, cancellationToken).ConfigureAwait(false);
        return Ok(response);
    }
}
