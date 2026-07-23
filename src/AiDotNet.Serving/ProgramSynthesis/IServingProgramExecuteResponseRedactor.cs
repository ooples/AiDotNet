using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

public interface IServingProgramExecuteResponseRedactor
{
    ProgramExecuteResponse Redact(ProgramExecuteResponse response, ServingRequestContext requestContext);
}
