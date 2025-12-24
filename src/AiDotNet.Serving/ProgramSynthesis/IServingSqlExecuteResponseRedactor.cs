using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

public interface IServingSqlExecuteResponseRedactor
{
    SqlExecuteResponse Redact(SqlExecuteResponse response, ServingRequestContext requestContext);
}
