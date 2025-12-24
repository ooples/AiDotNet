using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Sandboxing.Execution;
using AiDotNet.Serving.Security;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers.ProgramSynthesis;

[ApiController]
[Route("api/program-synthesis/program")]
[Produces("application/json")]
public sealed class ProgramController : ControllerBase
{
    private readonly IProgramSandboxExecutor _executor;
    private readonly IServingProgramEvaluator _evaluator;
    private readonly IServingProgramExecuteResponseRedactor _responseRedactor;
    private readonly IServingProgramEvaluateIoResponseRedactor _evaluateIoResponseRedactor;
    private readonly IServingRequestContextAccessor _requestContextAccessor;
    private readonly ILogger<ProgramController> _logger;

    public ProgramController(
        IProgramSandboxExecutor executor,
        IServingProgramEvaluator evaluator,
        IServingProgramExecuteResponseRedactor responseRedactor,
        IServingProgramEvaluateIoResponseRedactor evaluateIoResponseRedactor,
        IServingRequestContextAccessor requestContextAccessor,
        ILogger<ProgramController> logger)
    {
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _evaluator = evaluator ?? throw new ArgumentNullException(nameof(evaluator));
        _responseRedactor = responseRedactor ?? throw new ArgumentNullException(nameof(responseRedactor));
        _evaluateIoResponseRedactor = evaluateIoResponseRedactor ?? throw new ArgumentNullException(nameof(evaluateIoResponseRedactor));
        _requestContextAccessor = requestContextAccessor ?? throw new ArgumentNullException(nameof(requestContextAccessor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    [HttpPost("execute")]
    [ProducesResponseType(typeof(ProgramExecuteResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ProgramExecuteResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ProgramExecuteResponse), StatusCodes.Status408RequestTimeout)]
    [ProducesResponseType(typeof(ProgramExecuteResponse), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Execute([FromBody] ProgramExecuteRequest request, CancellationToken cancellationToken)
    {
        var ctx = _requestContextAccessor.Current ?? new ServingRequestContext
        {
            Tier = ServingTier.Free,
            IsAuthenticated = false
        };

        if (request is null)
        {
            return BadRequest(_responseRedactor.Redact(new ProgramExecuteResponse
            {
                Success = false,
                Language = ProgramLanguage.Generic,
                ExitCode = -1,
                Error = "Request body is required.",
                ErrorCode = ProgramExecuteErrorCode.InvalidRequest
            }, ctx));
        }

        if (string.IsNullOrWhiteSpace(request.SourceCode))
        {
            return BadRequest(_responseRedactor.Redact(new ProgramExecuteResponse
            {
                Success = false,
                Language = request.Language,
                ExitCode = -1,
                Error = "SourceCode is required.",
                ErrorCode = ProgramExecuteErrorCode.SourceCodeRequired
            }, ctx));
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await _executor.ExecuteAsync(request, ctx, cancellationToken).ConfigureAwait(false);
        sw.Stop();

        _logger.LogInformation(
            "ProgramExecute completed (Tier={Tier}, Language={Language}, SourceChars={SourceChars}, StdInChars={StdInChars}, Success={Success}, ErrorCode={ErrorCode}, DurationMs={DurationMs})",
            ctx.Tier,
            request.Language,
            request.SourceCode?.Length ?? 0,
            request.StdIn?.Length ?? 0,
            result.Success,
            result.ErrorCode?.ToString() ?? "None",
            sw.ElapsedMilliseconds);

        var response = _responseRedactor.Redact(result, ctx);

        if (result.Success)
        {
            return Ok(response);
        }

        if (result.ErrorCode == ProgramExecuteErrorCode.TimeoutOrCanceled)
        {
            return StatusCode(StatusCodes.Status408RequestTimeout, response);
        }

        if (result.ErrorCode is ProgramExecuteErrorCode.InvalidRequest or
            ProgramExecuteErrorCode.SourceCodeRequired or
            ProgramExecuteErrorCode.SourceCodeTooLarge or
            ProgramExecuteErrorCode.StdInTooLarge or
            ProgramExecuteErrorCode.LanguageNotDetected or
            ProgramExecuteErrorCode.SqlNotSupported)
        {
            return BadRequest(response);
        }

        return StatusCode(StatusCodes.Status500InternalServerError, response);
    }

    [HttpPost("evaluate-io")]
    [ProducesResponseType(typeof(ProgramEvaluateIoResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ProgramEvaluateIoResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ProgramEvaluateIoResponse), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> EvaluateIo([FromBody] ProgramEvaluateIoRequest request, CancellationToken cancellationToken)
    {
        var ctx = _requestContextAccessor.Current ?? new ServingRequestContext
        {
            Tier = ServingTier.Free,
            IsAuthenticated = false
        };

        if (request is null)
        {
            var missing = new ProgramEvaluateIoResponse
            {
                Success = false,
                Language = ProgramLanguage.Generic,
                Error = "Request body is required."
            };

            return BadRequest(_evaluateIoResponseRedactor.Redact(missing, ctx));
        }

        try
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = await _evaluator.EvaluateIoAsync(request, ctx, cancellationToken).ConfigureAwait(false);
            sw.Stop();

            _logger.LogInformation(
                "ProgramEvaluateIo completed (Tier={Tier}, Language={Language}, SourceChars={SourceChars}, TestCount={TestCount}, Success={Success}, DurationMs={DurationMs})",
                ctx.Tier,
                request.Language,
                request.SourceCode?.Length ?? 0,
                request.TestCases?.Count ?? 0,
                result.Success,
                sw.ElapsedMilliseconds);

            var redacted = _evaluateIoResponseRedactor.Redact(result, ctx);

            return result.Success
                ? Ok(redacted)
                : BadRequest(redacted);
        }
        catch (OperationCanceledException)
        {
            _logger.LogWarning(
                "ProgramEvaluateIo canceled/timeout (Tier={Tier}, Language={Language}, SourceChars={SourceChars}, TestCount={TestCount})",
                ctx.Tier,
                request.Language,
                request.SourceCode?.Length ?? 0,
                request.TestCases?.Count ?? 0);

            var timeout = new ProgramEvaluateIoResponse
            {
                Success = false,
                Language = request.Language,
                Error = "Evaluation timed out or was canceled."
            };

            return StatusCode(StatusCodes.Status408RequestTimeout, _evaluateIoResponseRedactor.Redact(timeout, ctx));
        }
        catch (Exception)
        {
            _logger.LogError(
                "ProgramEvaluateIo failed (Tier={Tier}, Language={Language}, SourceChars={SourceChars}, TestCount={TestCount})",
                ctx.Tier,
                request.Language,
                request.SourceCode?.Length ?? 0,
                request.TestCases?.Count ?? 0);

            var failed = new ProgramEvaluateIoResponse
            {
                Success = false,
                Language = request.Language,
                Error = "Evaluation failed."
            };

            return StatusCode(StatusCodes.Status500InternalServerError, _evaluateIoResponseRedactor.Redact(failed, ctx));
        }
    }
}
