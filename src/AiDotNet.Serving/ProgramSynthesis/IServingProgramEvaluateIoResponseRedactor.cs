using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

public interface IServingProgramEvaluateIoResponseRedactor
{
    ProgramEvaluateIoResponse Redact(ProgramEvaluateIoResponse response, ServingRequestContext requestContext);
}

