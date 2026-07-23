using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

public sealed class ServingProgramExecuteResponseRedactor : IServingProgramExecuteResponseRedactor
{
    public ProgramExecuteResponse Redact(ProgramExecuteResponse response, ServingRequestContext requestContext)
    {
        if (response is null) throw new ArgumentNullException(nameof(response));
        if (requestContext is null) throw new ArgumentNullException(nameof(requestContext));

        if (requestContext.Tier != ServingTier.Free)
        {
            return response;
        }

        var error = response.Error;
        if (!response.Success)
        {
            var preserve =
                response.ErrorCode is ProgramExecuteErrorCode.InvalidRequest or
                ProgramExecuteErrorCode.SourceCodeRequired or
                ProgramExecuteErrorCode.SourceCodeTooLarge or
                ProgramExecuteErrorCode.StdInTooLarge or
                ProgramExecuteErrorCode.LanguageNotDetected or
                ProgramExecuteErrorCode.SqlNotSupported or
                ProgramExecuteErrorCode.TimeoutOrCanceled;

            if (!preserve)
            {
                error = response.CompilationAttempted && response.CompilationSucceeded == false
                    ? "Compilation failed."
                    : "Execution failed.";
            }
        }

        return new ProgramExecuteResponse
        {
            Success = response.Success,
            Language = response.Language,
            CompilationAttempted = response.CompilationAttempted,
            CompilationSucceeded = response.CompilationSucceeded,
            CompilationDiagnostics = new List<CompilationDiagnostic>(),
            ExitCode = response.ExitCode,
            StdOut = response.StdOut,
            StdErr = string.Empty,
            StdOutTruncated = response.StdOutTruncated,
            StdErrTruncated = response.StdErrTruncated,
            Error = error,
            ErrorCode = response.ErrorCode
        };
    }
}
