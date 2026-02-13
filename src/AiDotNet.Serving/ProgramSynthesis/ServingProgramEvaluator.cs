using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Sandboxing.Execution;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Options;
using AiDotNet.Validation;

namespace AiDotNet.Serving.ProgramSynthesis;

public sealed class ServingProgramEvaluator : IServingProgramEvaluator
{
    private readonly IProgramSandboxExecutor _executor;
    private readonly IServingProgramExecuteResponseRedactor _executeResponseRedactor;
    private readonly ServingProgramSynthesisOptions _options;
    private readonly ILogger<ServingProgramEvaluator> _logger;

    public ServingProgramEvaluator(
        IProgramSandboxExecutor executor,
        IServingProgramExecuteResponseRedactor executeResponseRedactor,
        IOptions<ServingProgramSynthesisOptions> options,
        ILogger<ServingProgramEvaluator> logger)
    {
        Guard.NotNull(executor);
        _executor = executor;
        Guard.NotNull(executeResponseRedactor);
        _executeResponseRedactor = executeResponseRedactor;
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
        Guard.NotNull(logger);
        _logger = logger;
    }

    public async Task<ProgramEvaluateIoResponse> EvaluateIoAsync(
        ProgramEvaluateIoRequest request,
        ServingRequestContext requestContext,
        CancellationToken cancellationToken)
    {
        if (request is null) throw new ArgumentNullException(nameof(request));
        if (requestContext is null) throw new ArgumentNullException(nameof(requestContext));

        var limits = GetLimits(requestContext.Tier);

        if (string.IsNullOrWhiteSpace(request.SourceCode))
        {
            return new ProgramEvaluateIoResponse
            {
                Success = false,
                Language = request.Language,
                Error = "SourceCode is required."
            };
        }

        if (request.SourceCode.Length > limits.MaxRequestChars)
        {
            return new ProgramEvaluateIoResponse
            {
                Success = false,
                Language = request.Language,
                Error = $"SourceCode exceeds tier limit ({limits.MaxRequestChars} chars)."
            };
        }

        var testCases = request.TestCases ?? new List<ProgramInputOutputExample>();
        if (testCases.Count == 0)
        {
            return new ProgramEvaluateIoResponse
            {
                Success = false,
                Language = request.Language,
                Error = "TestCases are required."
            };
        }

        if (testCases.Count > limits.MaxListItems)
        {
            return new ProgramEvaluateIoResponse
            {
                Success = false,
                Language = request.Language,
                Error = $"TestCases exceeds tier limit ({limits.MaxListItems})."
            };
        }

        for (var i = 0; i < testCases.Count; i++)
        {
            var example = testCases[i] ?? new ProgramInputOutputExample();

            if (example.Input.Length > limits.MaxRequestChars)
            {
                return new ProgramEvaluateIoResponse
                {
                    Success = false,
                    Language = request.Language,
                    Error = $"TestCases[{i}].Input exceeds tier limit ({limits.MaxRequestChars} chars)."
                };
            }

            if (example.ExpectedOutput.Length > limits.MaxRequestChars)
            {
                return new ProgramEvaluateIoResponse
                {
                    Success = false,
                    Language = request.Language,
                    Error = $"TestCases[{i}].ExpectedOutput exceeds tier limit ({limits.MaxRequestChars} chars)."
                };
            }
        }

        var results = new List<ProgramEvaluateIoTestResult>(testCases.Count);
        var passedCount = 0;
        var resolvedLanguage = request.Language;

        foreach (var testCase in testCases)
        {
            cancellationToken.ThrowIfCancellationRequested();

            ProgramExecuteResponse execution;
            try
            {
                execution = await _executor.ExecuteAsync(
                        new ProgramExecuteRequest
                        {
                            Language = request.Language,
                            AllowedLanguages = request.AllowedLanguages ?? new List<ProgramLanguage>(),
                            PreferredLanguage = request.PreferredLanguage,
                            AllowUndetectedLanguageFallback = request.AllowUndetectedLanguageFallback,
                            SourceCode = request.SourceCode,
                            StdIn = testCase.Input
                        },
                        requestContext,
                        cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Program evaluation execution failed.");
                execution = new ProgramExecuteResponse
                {
                    Success = false,
                    Language = ProgramLanguage.Generic,
                    ExitCode = -1,
                    Error = "Execution failed."
                };
            }

            resolvedLanguage = execution.Language;

            var redactedExecution = _executeResponseRedactor.Redact(execution, requestContext);
            var passed = execution.Success &&
                         string.Equals(
                             NormalizeForComparison(execution.StdOut),
                             NormalizeForComparison(testCase.ExpectedOutput),
                             StringComparison.Ordinal);

            if (passed)
            {
                passedCount++;
            }

            var failureReason = passed
                ? null
                : !execution.Success
                    ? redactedExecution.Error ?? "Execution failed."
                    : "Output mismatch.";

            results.Add(new ProgramEvaluateIoTestResult
            {
                TestCase = testCase,
                Passed = passed,
                FailureReason = failureReason,
                Execution = redactedExecution
            });
        }

        var total = testCases.Count;
        var passRate = total <= 0 ? 0.0 : (double)passedCount / total;

        return new ProgramEvaluateIoResponse
        {
            Success = true,
            Language = resolvedLanguage,
            TotalTests = total,
            PassedTests = passedCount,
            PassRate = passRate,
            TestResults = results
        };
    }

    private ServingProgramSynthesisLimitOptions GetLimits(ServingTier tier) =>
        tier switch
        {
            ServingTier.Premium => _options.Premium,
            ServingTier.Enterprise => _options.Enterprise,
            _ => _options.Free
        };

    private static string NormalizeForComparison(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }

        return value.Replace("\r\n", "\n").Trim();
    }
}
