using AiDotNet.Serving.Security.Licensing;
using AiDotNet.Validation;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Public endpoint for license key validation. No authentication required — the license key itself
/// is the credential.
/// </summary>
[ApiController]
[Route("api/licenses")]
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
