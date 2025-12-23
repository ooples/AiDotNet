using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

public interface IServingCodeTaskRequestValidator
{
    bool TryValidate(
        CodeTaskRequestBase request,
        ServingRequestContext requestContext,
        out string error);
}

