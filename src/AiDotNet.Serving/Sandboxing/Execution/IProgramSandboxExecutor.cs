using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Sandboxing.Execution;

public interface IProgramSandboxExecutor
{
    Task<ProgramExecuteResponse> ExecuteAsync(
        ProgramExecuteRequest request,
        ServingRequestContext requestContext,
        CancellationToken cancellationToken);
}
