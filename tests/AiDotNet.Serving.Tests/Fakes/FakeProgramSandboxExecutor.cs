using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Sandboxing.Execution;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Tests.Fakes;

public sealed class FakeProgramSandboxExecutor : IProgramSandboxExecutor
{
    public Task<ProgramExecuteResponse> ExecuteAsync(
        ProgramExecuteRequest request,
        ServingRequestContext requestContext,
        CancellationToken cancellationToken)
    {
        var language = request.Language == ProgramLanguage.Generic ? ProgramLanguage.Python : request.Language;
        var compilationAttempted =
            language is ProgramLanguage.C or ProgramLanguage.CPlusPlus or ProgramLanguage.Rust or ProgramLanguage.Java or ProgramLanguage.CSharp;

        return Task.FromResult(new ProgramExecuteResponse
        {
            Success = true,
            Language = language,
            CompilationAttempted = compilationAttempted,
            CompilationSucceeded = compilationAttempted ? true : null,
            ExitCode = 0,
            StdOut = request.CompileOnly ? string.Empty : "ok",
            StdErr = string.Empty
        });
    }
}
