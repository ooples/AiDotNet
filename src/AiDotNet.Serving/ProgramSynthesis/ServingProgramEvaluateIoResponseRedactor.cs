using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Security;
using AiDotNet.Validation;

namespace AiDotNet.Serving.ProgramSynthesis;

public sealed class ServingProgramEvaluateIoResponseRedactor : IServingProgramEvaluateIoResponseRedactor
{
    private readonly IServingProgramExecuteResponseRedactor _executeResponseRedactor;

    public ServingProgramEvaluateIoResponseRedactor(IServingProgramExecuteResponseRedactor executeResponseRedactor)
    {
        Guard.NotNull(executeResponseRedactor);
        _executeResponseRedactor = executeResponseRedactor;
    }

    public ProgramEvaluateIoResponse Redact(ProgramEvaluateIoResponse response, ServingRequestContext requestContext)
    {
        if (response is null) throw new ArgumentNullException(nameof(response));
        if (requestContext is null) throw new ArgumentNullException(nameof(requestContext));

        if (requestContext.Tier != ServingTier.Free)
        {
            return response;
        }

        var error = response.Error;
        if (!response.Success && error is not null)
        {
            var preserve =
                error.Contains("required", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("exceeds tier limit", StringComparison.OrdinalIgnoreCase);

            if (!preserve)
            {
                error = "Evaluation failed.";
            }
        }

        var redactedResults = response.TestResults
            .Select(x => new ProgramEvaluateIoTestResult
            {
                TestCase = x.TestCase,
                Passed = x.Passed,
                FailureReason = x.Passed ? null : x.FailureReason,
                Execution = _executeResponseRedactor.Redact(x.Execution, requestContext)
            })
            .ToList();

        return new ProgramEvaluateIoResponse
        {
            Success = response.Success,
            Language = response.Language,
            TotalTests = response.TotalTests,
            PassedTests = response.PassedTests,
            PassRate = response.PassRate,
            TestResults = redactedResults,
            Error = error
        };
    }
}

