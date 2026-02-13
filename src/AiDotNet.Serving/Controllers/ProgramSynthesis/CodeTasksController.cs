using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Security;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using AiDotNet.Validation;

namespace AiDotNet.Serving.Controllers.ProgramSynthesis;

[ApiController]
[Route("api/program-synthesis/tasks")]
[Produces("application/json")]
public sealed class CodeTasksController : ControllerBase
{
    private readonly IServingCodeTaskExecutor _executor;
    private readonly IServingCodeTaskRequestValidator _validator;
    private readonly IServingCodeTaskResultRedactor _redactor;
    private readonly IServingRequestContextAccessor _requestContextAccessor;
    private readonly IServingProgramSynthesisConcurrencyLimiter _concurrencyLimiter;
    private readonly ServingProgramSynthesisOptions _options;
    private readonly ILogger<CodeTasksController> _logger;

    public CodeTasksController(
        IServingCodeTaskExecutor executor,
        IServingCodeTaskRequestValidator validator,
        IServingCodeTaskResultRedactor redactor,
        IServingRequestContextAccessor requestContextAccessor,
        IServingProgramSynthesisConcurrencyLimiter concurrencyLimiter,
        IOptions<ServingProgramSynthesisOptions> options,
        ILogger<CodeTasksController> logger)
    {
        Guard.NotNull(executor);
        _executor = executor;
        Guard.NotNull(validator);
        _validator = validator;
        Guard.NotNull(redactor);
        _redactor = redactor;
        Guard.NotNull(requestContextAccessor);
        _requestContextAccessor = requestContextAccessor;
        Guard.NotNull(concurrencyLimiter);
        _concurrencyLimiter = concurrencyLimiter;
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
        Guard.NotNull(logger);
        _logger = logger;
    }

    [HttpPost("completion")]
    [ProducesResponseType(typeof(CodeCompletionResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeCompletionResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeCompletionResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Completion([FromBody] CodeCompletionRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeCompletionResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("generation")]
    [ProducesResponseType(typeof(CodeGenerationResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeGenerationResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeGenerationResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Generation([FromBody] CodeGenerationRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeGenerationResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("translation")]
    [ProducesResponseType(typeof(CodeTranslationResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeTranslationResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeTranslationResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Translation([FromBody] CodeTranslationRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeTranslationResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("summarization")]
    [ProducesResponseType(typeof(CodeSummarizationResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeSummarizationResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeSummarizationResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Summarization([FromBody] CodeSummarizationRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeSummarizationResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("bug-detection")]
    [ProducesResponseType(typeof(CodeBugDetectionResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeBugDetectionResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeBugDetectionResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> BugDetection([FromBody] CodeBugDetectionRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeBugDetectionResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("bug-fixing")]
    [ProducesResponseType(typeof(CodeBugFixingResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeBugFixingResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeBugFixingResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> BugFixing([FromBody] CodeBugFixingRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeBugFixingResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("refactoring")]
    [ProducesResponseType(typeof(CodeRefactoringResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeRefactoringResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeRefactoringResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Refactoring([FromBody] CodeRefactoringRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeRefactoringResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("understanding")]
    [ProducesResponseType(typeof(CodeUnderstandingResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeUnderstandingResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeUnderstandingResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Understanding([FromBody] CodeUnderstandingRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeUnderstandingResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("test-generation")]
    [ProducesResponseType(typeof(CodeTestGenerationResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeTestGenerationResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeTestGenerationResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> TestGeneration([FromBody] CodeTestGenerationRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeTestGenerationResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("documentation")]
    [ProducesResponseType(typeof(CodeDocumentationResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeDocumentationResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeDocumentationResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Documentation([FromBody] CodeDocumentationRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeDocumentationResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("search")]
    [ProducesResponseType(typeof(CodeSearchResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeSearchResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeSearchResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Search([FromBody] CodeSearchRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeSearchResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("clone-detection")]
    [ProducesResponseType(typeof(CodeCloneDetectionResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeCloneDetectionResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeCloneDetectionResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> CloneDetection([FromBody] CodeCloneDetectionRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeCloneDetectionResult(), cancellationToken).ConfigureAwait(false);

    [HttpPost("code-review")]
    [ProducesResponseType(typeof(CodeReviewResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(CodeReviewResult), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(CodeReviewResult), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> CodeReview([FromBody] CodeReviewRequest request, CancellationToken cancellationToken) =>
        await ExecuteAsync(request, () => new CodeReviewResult(), cancellationToken).ConfigureAwait(false);

    private async Task<IActionResult> ExecuteAsync(
        CodeTaskRequestBase request,
        Func<CodeTaskResultBase> emptyResultFactory,
        CancellationToken cancellationToken)
    {
        var ctx = _requestContextAccessor.Current ?? new ServingRequestContext
        {
            Tier = ServingTier.Free,
            IsAuthenticated = false
        };

        if (request is null)
        {
            var missingBody = emptyResultFactory();
            missingBody.Language = ProgramLanguage.Generic;
            missingBody.Success = false;
            missingBody.Error = "Request body is required.";
            return BadRequest(_redactor.Redact(missingBody, ctx));
        }

        if (!_validator.TryValidate(request, ctx, out var validationError))
        {
            var invalid = emptyResultFactory();
            invalid.Language = request.Language;
            invalid.RequestId = request.RequestId;
            invalid.Success = false;
            invalid.Error = validationError;
            return BadRequest(_redactor.Redact(invalid, ctx));
        }

        var limits = GetLimits(ctx.Tier);
        var maxTierMs = checked(limits.MaxTaskTimeSeconds * 1000);
        var requestedMs = request.MaxWallClockMilliseconds;
        var effectiveMs = requestedMs is null ? maxTierMs : Math.Min(maxTierMs, requestedMs.Value);

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutCts.CancelAfter(TimeSpan.FromMilliseconds(effectiveMs));

        try
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using var _ = await _concurrencyLimiter.AcquireAsync(ctx.Tier, timeoutCts.Token).ConfigureAwait(false);

            var result = await _executor.ExecuteAsync(request, ctx, timeoutCts.Token).ConfigureAwait(false);
            sw.Stop();

            _logger.LogInformation(
                "CodeTask completed (Tier={Tier}, Task={Task}, Language={Language}, RequestId={RequestId}, Success={Success}, DurationMs={DurationMs})",
                ctx.Tier,
                request.Task,
                request.Language,
                request.RequestId ?? string.Empty,
                result.Success,
                sw.ElapsedMilliseconds);

            return Ok(_redactor.Redact(result, ctx));
        }
        catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
        {
            _logger.LogWarning(
                "CodeTask timed out (Tier={Tier}, Task={Task}, Language={Language}, RequestId={RequestId}, TimeoutMs={TimeoutMs})",
                ctx.Tier,
                request.Task,
                request.Language,
                request.RequestId ?? string.Empty,
                effectiveMs);

            var timedOut = emptyResultFactory();
            timedOut.Language = request.Language;
            timedOut.RequestId = request.RequestId;
            timedOut.Success = false;
            timedOut.Error = $"Request exceeded tier time limit ({effectiveMs} ms).";
            return Ok(_redactor.Redact(timedOut, ctx));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Unhandled program-synthesis task failure (Task={Task}, Language={Language}, Tier={Tier}, RequestId={RequestId}).",
                request.Task, request.Language, ctx.Tier, request.RequestId);

            var failure = emptyResultFactory();
            failure.Language = request.Language;
            failure.RequestId = request.RequestId;
            failure.Success = false;
            failure.Error = "Unhandled task execution error.";

            return StatusCode(StatusCodes.Status500InternalServerError, _redactor.Redact(failure, ctx));
        }
    }

    private ServingProgramSynthesisLimitOptions GetLimits(ServingTier tier) =>
        tier switch
        {
            ServingTier.Premium => _options.Premium,
            ServingTier.Enterprise => _options.Enterprise,
            _ => _options.Free
        };
}

