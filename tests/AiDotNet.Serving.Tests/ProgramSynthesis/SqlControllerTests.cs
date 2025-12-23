using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Controllers.ProgramSynthesis;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Sandboxing.Sql;
using AiDotNet.Serving.Security;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace AiDotNet.Serving.Tests.ProgramSynthesis;

public sealed class SqlControllerTests
{
    [Fact]
    public async Task Execute_NullRequest_ReturnsBadRequest()
    {
        var controller = CreateController(new FakeSqlSandboxExecutor(_ => Task.FromResult(new SqlExecuteResponse { Success = true })));

        var result = await controller.Execute(null!, CancellationToken.None);

        var badRequest = Assert.IsType<BadRequestObjectResult>(result);
        var payload = Assert.IsType<SqlExecuteResponse>(badRequest.Value);
        Assert.False(payload.Success);
        Assert.Null(payload.Dialect);
    }

    [Fact]
    public async Task Execute_Success_ReturnsOk()
    {
        var controller = CreateController(new FakeSqlSandboxExecutor(_ => Task.FromResult(new SqlExecuteResponse
        {
            Success = true,
            Dialect = SqlDialect.SQLite,
            Columns = new List<string> { "x" },
            Rows = new List<Dictionary<string, SqlValue>>
            {
                new()
                {
                    ["x"] = new SqlValue { Kind = SqlValueKind.Integer, IntegerValue = 1 }
                }
            }
        })));

        var result = await controller.Execute(new SqlExecuteRequest { Dialect = SqlDialect.SQLite, Query = "select 1;" }, CancellationToken.None);

        var ok = Assert.IsType<OkObjectResult>(result);
        var payload = Assert.IsType<SqlExecuteResponse>(ok.Value);
        Assert.True(payload.Success);
        Assert.Equal(SqlDialect.SQLite, payload.Dialect);
    }

    [Fact]
    public async Task Execute_Timeout_Returns408()
    {
        var controller = CreateController(new FakeSqlSandboxExecutor(_ => Task.FromResult(new SqlExecuteResponse
        {
            Success = false,
            Dialect = SqlDialect.SQLite,
            ErrorCode = SqlExecuteErrorCode.TimeoutOrCanceled
        })));

        var result = await controller.Execute(new SqlExecuteRequest { Dialect = SqlDialect.SQLite, Query = "select 1;" }, CancellationToken.None);

        var objectResult = Assert.IsType<ObjectResult>(result);
        Assert.Equal(408, objectResult.StatusCode);
        var payload = Assert.IsType<SqlExecuteResponse>(objectResult.Value);
        Assert.Equal(SqlExecuteErrorCode.TimeoutOrCanceled, payload.ErrorCode);
    }

    private static SqlController CreateController(ISqlSandboxExecutor executor)
    {
        var redactor = new ServingSqlExecuteResponseRedactor();
        var accessor = new ServingRequestContextAccessor { Current = null };
        var logger = NullLogger<SqlController>.Instance;

        return new SqlController(
            executor,
            redactor,
            accessor,
            logger);
    }

    private sealed class FakeSqlSandboxExecutor : ISqlSandboxExecutor
    {
        private readonly Func<SqlExecuteRequest, Task<SqlExecuteResponse>> _handler;

        public FakeSqlSandboxExecutor(Func<SqlExecuteRequest, Task<SqlExecuteResponse>> handler)
        {
            _handler = handler;
        }

        public Task<SqlExecuteResponse> ExecuteAsync(SqlExecuteRequest request, ServingRequestContext requestContext, CancellationToken cancellationToken)
        {
            return _handler(request);
        }
    }
}
