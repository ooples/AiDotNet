using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Sandboxing.Sql;
using AiDotNet.Serving.Security;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers.ProgramSynthesis;

[ApiController]
[Route("api/program-synthesis/sql")]
[Produces("application/json")]
public class SqlController : ControllerBase
{
    private readonly ISqlSandboxExecutor _executor;
    private readonly IServingSqlExecuteResponseRedactor _responseRedactor;
    private readonly IServingRequestContextAccessor _requestContextAccessor;
    private readonly ILogger<SqlController> _logger;

    public SqlController(
        ISqlSandboxExecutor executor,
        IServingSqlExecuteResponseRedactor responseRedactor,
        IServingRequestContextAccessor requestContextAccessor,
        ILogger<SqlController> logger)
    {
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _responseRedactor = responseRedactor ?? throw new ArgumentNullException(nameof(responseRedactor));
        _requestContextAccessor = requestContextAccessor ?? throw new ArgumentNullException(nameof(requestContextAccessor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    [HttpPost("execute")]
    [ProducesResponseType(typeof(SqlExecuteResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(SqlExecuteResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(SqlExecuteResponse), StatusCodes.Status408RequestTimeout)]
    [ProducesResponseType(typeof(SqlExecuteResponse), StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> Execute([FromBody] SqlExecuteRequest request, CancellationToken cancellationToken)
    {
        var ctx = _requestContextAccessor.Current ?? new ServingRequestContext
        {
            Tier = ServingTier.Free,
            IsAuthenticated = false
        };

        if (request is null)
        {
            return BadRequest(_responseRedactor.Redact(new SqlExecuteResponse
            {
                Success = false,
                Dialect = null,
                Error = "Request body is required."
            }, ctx));
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await _executor.ExecuteAsync(request, ctx, cancellationToken).ConfigureAwait(false);
        sw.Stop();

        _logger.LogInformation(
            "SqlExecute completed (Tier={Tier}, Dialect={Dialect}, QueryChars={QueryChars}, Success={Success}, ErrorCode={ErrorCode}, DurationMs={DurationMs})",
            ctx.Tier,
            request.Dialect,
            request.Query?.Length ?? 0,
            result.Success,
            result.ErrorCode?.ToString() ?? "None",
            sw.ElapsedMilliseconds);

        var response = _responseRedactor.Redact(result, ctx);

        if (result.Success)
        {
            return Ok(response);
        }

        if (result.ErrorCode == SqlExecuteErrorCode.TimeoutOrCanceled)
        {
            return StatusCode(StatusCodes.Status408RequestTimeout, response);
        }

        if (result.ErrorCode is SqlExecuteErrorCode.InvalidRequest or
            SqlExecuteErrorCode.QueryRequired or
            SqlExecuteErrorCode.UnsupportedDialect or
            SqlExecuteErrorCode.DialectMismatch or
            SqlExecuteErrorCode.DialectNotAllowedForTier or
            SqlExecuteErrorCode.DialectNotConfigured or
            SqlExecuteErrorCode.UnknownDbId or
            SqlExecuteErrorCode.UnknownDatasetId or
            SqlExecuteErrorCode.MultiStatementNotAllowed)
        {
            return BadRequest(response);
        }

        return StatusCode(StatusCodes.Status500InternalServerError, response);
    }
}
