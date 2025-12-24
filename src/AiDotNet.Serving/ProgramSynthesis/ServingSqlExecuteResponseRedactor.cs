using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

public sealed class ServingSqlExecuteResponseRedactor : IServingSqlExecuteResponseRedactor
{
    public SqlExecuteResponse Redact(SqlExecuteResponse response, ServingRequestContext requestContext)
    {
        if (response is null) throw new ArgumentNullException(nameof(response));
        if (requestContext is null) throw new ArgumentNullException(nameof(requestContext));

        if (requestContext.Tier != ServingTier.Free)
        {
            return response;
        }

        var error = response.Error;
        if (!response.Success && error is not null)
        {
            error = "SQL execution failed.";
        }

        return new SqlExecuteResponse
        {
            Success = response.Success,
            Dialect = response.Dialect,
            Columns = response.Columns,
            Rows = response.Rows,
            Error = error,
            ErrorCode = response.ErrorCode
        };
    }
}
