using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

public interface IServingProgramEvaluator
{
    Task<ProgramEvaluateIoResponse> EvaluateIoAsync(
        ProgramEvaluateIoRequest request,
        ServingRequestContext requestContext,
        CancellationToken cancellationToken);
}
