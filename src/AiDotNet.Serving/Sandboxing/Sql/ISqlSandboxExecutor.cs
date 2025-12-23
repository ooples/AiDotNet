using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Sandboxing.Sql;

public interface ISqlSandboxExecutor
{
    Task<SqlExecuteResponse> ExecuteAsync(
        SqlExecuteRequest request,
        ServingRequestContext requestContext,
        CancellationToken cancellationToken);
}
