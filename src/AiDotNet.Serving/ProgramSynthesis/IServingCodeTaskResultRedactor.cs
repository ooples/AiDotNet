using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

public interface IServingCodeTaskResultRedactor
{
    CodeTaskResultBase Redact(CodeTaskResultBase result, ServingRequestContext requestContext);
}

