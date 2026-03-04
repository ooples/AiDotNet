using AiDotNet.Serving.Security.Licensing;
using AiDotNet.Validation;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers.Admin;

/// <summary>
/// Administrative endpoints for managing license keys.
/// </summary>
[ApiController]
[Route("api/admin/licenses")]
[Authorize(Policy = "AiDotNetAdmin")]
public sealed class LicensesController : ControllerBase
{
    private readonly ILicenseService _licenses;

    public LicensesController(ILicenseService licenses)
    {
        Guard.NotNull(licenses);
        _licenses = licenses;
    }

    /// <summary>
    /// Creates a new license key. The key string is returned only once.
    /// </summary>
    [HttpPost]
    [ProducesResponseType(typeof(LicenseCreateResponse), StatusCodes.Status200OK)]
    public async Task<IActionResult> Create([FromBody] LicenseCreateRequest request, CancellationToken cancellationToken)
    {
        var response = await _licenses.CreateAsync(request, cancellationToken).ConfigureAwait(false);
        return Ok(response);
    }

    /// <summary>
    /// Lists all license keys with seat usage.
    /// </summary>
    [HttpGet]
    [ProducesResponseType(typeof(List<LicenseInfo>), StatusCodes.Status200OK)]
    public async Task<IActionResult> List(CancellationToken cancellationToken)
    {
        var result = await _licenses.ListAsync(cancellationToken).ConfigureAwait(false);
        return Ok(result);
    }

    /// <summary>
    /// Gets a single license key with its activation history.
    /// </summary>
    [HttpGet("{id:guid}")]
    [ProducesResponseType(typeof(LicenseInfo), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> Get(Guid id, CancellationToken cancellationToken)
    {
        var result = await _licenses.GetAsync(id, cancellationToken).ConfigureAwait(false);
        if (result is null)
        {
            return NotFound(new ProblemDetails
            {
                Status = StatusCodes.Status404NotFound,
                Title = "License not found.",
                Detail = $"No license with ID '{id}' exists."
            });
        }

        return Ok(result);
    }

    /// <summary>
    /// Revokes a license key by ID.
    /// </summary>
    [HttpPost("{id:guid}/revoke")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> Revoke(Guid id, CancellationToken cancellationToken)
    {
        var revoked = await _licenses.RevokeAsync(id, cancellationToken).ConfigureAwait(false);
        if (!revoked)
        {
            return NotFound(new ProblemDetails
            {
                Status = StatusCodes.Status404NotFound,
                Title = "License not found.",
                Detail = $"No license with ID '{id}' exists."
            });
        }

        return Ok(new { success = true });
    }
}
