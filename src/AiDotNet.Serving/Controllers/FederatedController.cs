using AiDotNet.Serving.Models.Federated;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services.Federated;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.Attestation;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Controller for federated learning coordination (HTTP).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This controller provides endpoints for:
/// - Creating a federated run
/// - Allowing clients to join (with attestation gating depending on tier)
/// - Submitting client updates
/// - Aggregating a round
/// </remarks>
[ApiController]
[Route("api/federated")]
[Produces("application/json")]
public class FederatedController : ControllerBase
{
    private readonly IFederatedCoordinatorService _coordinator;
    private readonly ILogger<FederatedController> _logger;
    private readonly ITierResolver _tierResolver;
    private readonly ITierPolicyProvider _tierPolicyProvider;
    private readonly IAttestationVerifier _attestationVerifier;

    public FederatedController(
        IFederatedCoordinatorService coordinator,
        ILogger<FederatedController> logger,
        ITierResolver tierResolver,
        ITierPolicyProvider tierPolicyProvider,
        IAttestationVerifier attestationVerifier)
    {
        _coordinator = coordinator ?? throw new ArgumentNullException(nameof(coordinator));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _tierResolver = tierResolver ?? throw new ArgumentNullException(nameof(tierResolver));
        _tierPolicyProvider = tierPolicyProvider ?? throw new ArgumentNullException(nameof(tierPolicyProvider));
        _attestationVerifier = attestationVerifier ?? throw new ArgumentNullException(nameof(attestationVerifier));
    }

    [HttpPost("runs")]
    [ProducesResponseType(typeof(CreateFederatedRunResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public ActionResult<CreateFederatedRunResponse> CreateRun([FromBody] CreateFederatedRunRequest request)
    {
        try
        {
            return Ok(_coordinator.CreateRun(request));
        }
        catch (FileNotFoundException)
        {
            return NotFound(new { error = "Model not found." });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to create federated run");
            return BadRequest(new { error = "An unexpected error occurred while creating the federated run." });
        }
    }

    [HttpGet("runs/{runId}")]
    [ProducesResponseType(typeof(FederatedRunStatusResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public ActionResult<FederatedRunStatusResponse> GetStatus(string runId)
    {
        try
        {
            return Ok(_coordinator.GetStatus(runId));
        }
        catch (FileNotFoundException)
        {
            return NotFound(new { error = "Federated run not found." });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get status for federated run {RunId}", runId);
            return BadRequest(new { error = "An unexpected error occurred while retrieving the federated run status." });
        }
    }

    [HttpPost("runs/{runId}/join")]
    [ProducesResponseType(typeof(JoinFederatedRunResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status403Forbidden)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<ActionResult<JoinFederatedRunResponse>> JoinRun(string runId, [FromBody] JoinFederatedRunRequest request)
    {
        try
        {
            return Ok(await _coordinator.JoinRunAsync(runId, request, HttpContext.RequestAborted));
        }
        catch (FileNotFoundException)
        {
            return NotFound(new { error = "Federated run not found." });
        }
        catch (UnauthorizedAccessException)
        {
            return StatusCode(StatusCodes.Status403Forbidden, new { error = "Access denied." });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to join federated run {RunId}", runId);
            return BadRequest(new { error = "An unexpected error occurred while joining the federated run." });
        }
    }

    [HttpGet("runs/{runId}/clients/{clientId}/parameters")]
    [ProducesResponseType(typeof(FederatedRunParametersResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status403Forbidden)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public ActionResult<FederatedRunParametersResponse> GetParameters(string runId, int clientId)
    {
        try
        {
            return Ok(_coordinator.GetParameters(runId, clientId));
        }
        catch (FileNotFoundException)
        {
            return NotFound(new { error = "Federated run not found." });
        }
        catch (UnauthorizedAccessException)
        {
            return StatusCode(StatusCodes.Status403Forbidden, new { error = "Access denied." });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get parameters for federated run {RunId}", runId);
            return BadRequest(new { error = "An unexpected error occurred while retrieving federated run parameters." });
        }
    }

    /// <summary>
    /// Downloads the current run artifact, subject to tier enforcement (Options A/B/C).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Free tier (Option A) does not allow artifact download. Pro can download plaintext.
    /// Enterprise receives an encrypted artifact and must request a key with attestation.
    /// </remarks>
    [HttpGet("runs/{runId}/artifact")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status403Forbidden)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public IActionResult DownloadRunArtifact(string runId)
    {
        var tier = _tierResolver.ResolveTier(HttpContext);
        var policy = _tierPolicyProvider.GetPolicy(tier);

        if (!policy.AllowArtifactDownload)
        {
            return StatusCode(StatusCodes.Status403Forbidden, new { error = "Run artifact download is not available for this tier." });
        }

        try
        {
            if (policy.ArtifactIsEncrypted)
            {
                var protectedArtifact = _coordinator.GetOrCreateEncryptedRunArtifact(runId);
                Response.Headers["X-AiDotNet-Artifact-Encrypted"] = "true";
                Response.Headers["X-AiDotNet-Artifact-Algorithm"] = protectedArtifact.Algorithm;
                Response.Headers["X-AiDotNet-Artifact-KeyId"] = protectedArtifact.KeyId;
                return PhysicalFile(protectedArtifact.EncryptedPath, "application/octet-stream", $"{runId}.aidn.enc");
            }

            var path = _coordinator.GetRunArtifactPath(runId);
            Response.Headers["X-AiDotNet-Artifact-Encrypted"] = "false";
            return PhysicalFile(path, "application/octet-stream", Path.GetFileName(path));
        }
        catch (FileNotFoundException ex)
        {
            _logger.LogWarning(ex, "Federated run artifact for {RunId} not found", runId);
            return NotFound(new { error = "Federated run artifact not found." });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to download federated run artifact for {RunId}", runId);
            return StatusCode(StatusCodes.Status500InternalServerError, new { error = "An unexpected error occurred while downloading the run artifact." });
        }
    }

    /// <summary>
    /// Releases the decryption key for an encrypted run artifact (Enterprise / Option C) after attestation verification.
    /// </summary>
    [HttpPost("runs/{runId}/artifact/key")]
    [ProducesResponseType(typeof(ModelArtifactKeyResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status403Forbidden)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> ReleaseRunArtifactKey(string runId, [FromBody] AttestationEvidence? evidence)
    {
        var tier = _tierResolver.ResolveTier(HttpContext);
        var policy = _tierPolicyProvider.GetPolicy(tier);

        if (!policy.AllowKeyRelease)
        {
            return StatusCode(StatusCodes.Status403Forbidden, new { error = "Run artifact key release is not available for this tier." });
        }

        if (policy.RequireAttestationForKeyRelease)
        {
            if (evidence == null)
            {
                return BadRequest(new { error = "Attestation evidence is required." });
            }

            var attestation = await _attestationVerifier.VerifyAsync(evidence, HttpContext.RequestAborted);
            if (!attestation.IsSuccess)
            {
                return StatusCode(StatusCodes.Status403Forbidden, new { error = attestation.FailureReason ?? "Attestation failed." });
            }
        }

        try
        {
            var protectedArtifact = _coordinator.GetOrCreateEncryptedRunArtifact(runId);
            return Ok(new ModelArtifactKeyResponse
            {
                KeyId = protectedArtifact.KeyId,
                Algorithm = protectedArtifact.Algorithm,
                KeyBase64 = Convert.ToBase64String(protectedArtifact.Key),
                NonceBase64 = Convert.ToBase64String(protectedArtifact.Nonce)
            });
        }
        catch (FileNotFoundException ex)
        {
            _logger.LogWarning(ex, "Federated run artifact for {RunId} not found", runId);
            return NotFound(new { error = "Federated run artifact not found." });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to release federated run artifact key for {RunId}", runId);
            return StatusCode(StatusCodes.Status500InternalServerError, new { error = "An unexpected error occurred while releasing the run artifact key." });
        }
    }

    [HttpPost("runs/{runId}/updates")]
    [ProducesResponseType(typeof(SubmitFederatedUpdateResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status403Forbidden)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public ActionResult<SubmitFederatedUpdateResponse> SubmitUpdate(string runId, [FromBody] SubmitFederatedUpdateRequest request)
    {
        try
        {
            return Ok(_coordinator.SubmitUpdate(runId, request));
        }
        catch (FileNotFoundException)
        {
            return NotFound(new { error = "Federated run not found." });
        }
        catch (UnauthorizedAccessException)
        {
            return StatusCode(StatusCodes.Status403Forbidden, new { error = "Access denied." });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to submit update for federated run {RunId}", runId);
            return BadRequest(new { error = "An unexpected error occurred while submitting the update." });
        }
    }

    [HttpPost("runs/{runId}/aggregate")]
    [ProducesResponseType(typeof(AggregateFederatedRoundResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public ActionResult<AggregateFederatedRoundResponse> AggregateRound(string runId)
    {
        try
        {
            return Ok(_coordinator.AggregateRound(runId));
        }
        catch (FileNotFoundException)
        {
            return NotFound(new { error = "Federated run not found." });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to aggregate round for federated run {RunId}", runId);
            return BadRequest(new { error = "An unexpected error occurred while aggregating the round." });
        }
    }
}
