using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

public interface IServingCodeTaskExecutor
{
    Task<CodeTaskResultBase> ExecuteAsync(
        CodeTaskRequestBase request,
        ServingRequestContext requestContext,
        CancellationToken cancellationToken);
}

