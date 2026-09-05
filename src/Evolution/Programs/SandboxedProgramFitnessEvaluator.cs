using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>
/// Scores a candidate program by running it on input/output examples through the asynchronous sandbox contract, and
/// records why each failing case failed.
/// </summary>
/// <remarks>
/// <para>
/// This evaluator is the sibling of <see cref="InputOutputProgramFitnessEvaluator"/> and computes the same quality
/// — the fraction of examples whose captured standard output matches the expected output — but it drives
/// <see cref="IProgramExecutionEngine.ExecuteAsync"/> instead of the synchronous overload, and that difference
/// shows up in the diagnostics. A candidate that hung, one that failed to compile, and one that ran and printed the
/// wrong answer are three genuinely different situations, and the synchronous contract collapses all of them into
/// "false plus a message". Here each failing example is labelled with its own code, and an example whose output was
/// cut off at the sandbox's output cap says so rather than silently counting as a mismatch of unknown origin.
/// </para>
/// <para>
/// Running asynchronously also matters for throughput: a population being scored against a process sandbox spends
/// nearly all of its time waiting for child processes, so keeping those waits off the thread pool lets the sandbox's
/// own concurrency limit be the thing that bounds parallelism.
/// </para>
/// <para>
/// Everything reaching a diagnostic is sanitized, truncated, and marked redacted, and at most eight example-level
/// diagnostics are retained, so a program that prints a megabyte to standard error cannot bloat a checkpoint or leak
/// raw payloads into a log. <see cref="VersionHash"/> folds in the examples and the comparison mode, so changing the
/// test set correctly invalidates older checkpoints.
/// </para>
/// <para><b>For Beginners:</b> This runs a generated program once per example, compares what it printed with what
/// it should have printed, and reports the fraction it got right. Because you supply the runner, you decide how
/// dangerous code is contained — a separate process or a container, never this one. When a program fails, the
/// result tells you which kind of failure it was: too slow, would not compile, crashed, or simply wrong. That
/// distinction is what lets you tell "the model is writing slow code" apart from "the model is writing broken
/// code".</para>
/// </remarks>
public sealed class SandboxedProgramFitnessEvaluator : IProgramFitnessEvaluator
{
    private const int MaxRetainedExampleDiagnostics = 8;
    private const int MaxErrorMessageLength = 200;

    private readonly IProgramExecutionEngine _engine;
    private readonly ReadOnlyCollection<ProgramInputOutputExample> _examples;
    private readonly ProgramOutputComparison _comparison;

    /// <summary>Initializes a sandboxed input/output evaluator.</summary>
    /// <param name="engine">The sandbox or runner that executes candidate programs.</param>
    /// <param name="examples">The input/output cases the candidate must satisfy; at least one is required.</param>
    /// <param name="comparison">How captured output is compared with expected output.</param>
    /// <param name="id">A stable evaluator identifier.</param>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> or <paramref name="examples"/> is <c>null</c>, or an example is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="examples"/> is empty, or <paramref name="id"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="comparison"/> is not a defined value.</exception>
    public SandboxedProgramFitnessEvaluator(
        IProgramExecutionEngine engine,
        IEnumerable<ProgramInputOutputExample> examples,
        ProgramOutputComparison comparison = ProgramOutputComparison.TrimmedOrdinal,
        string id = "program-sandbox-evaluator")
    {
        Guard.NotNull(engine);
        Guard.NotNull(examples);
        Guard.NotNullOrWhiteSpace(id);
        if (!Enum.IsDefined(typeof(ProgramOutputComparison), comparison))
            throw new ArgumentOutOfRangeException(nameof(comparison));

        var copy = new List<ProgramInputOutputExample>();
        foreach (ProgramInputOutputExample example in examples)
        {
            if (example is null) throw new ArgumentNullException(nameof(examples), "Examples cannot be null.");
            copy.Add(new ProgramInputOutputExample { Input = example.Input, ExpectedOutput = example.ExpectedOutput });
        }

        if (copy.Count == 0) throw new ArgumentException("At least one example is required.", nameof(examples));

        _engine = engine;
        _examples = new ReadOnlyCollection<ProgramInputOutputExample>(copy);
        _comparison = comparison;
        Id = id.Trim();
        VersionHash = BuildVersionHash(copy, comparison);
    }

    /// <inheritdoc/>
    public string Id { get; }

    /// <inheritdoc/>
    public string VersionHash { get; }

    /// <summary>Gets the input/output cases this evaluator scores against.</summary>
    public IReadOnlyList<ProgramInputOutputExample> Examples => _examples;

    /// <summary>Gets how captured output is compared with expected output.</summary>
    public ProgramOutputComparison Comparison => _comparison;

    /// <inheritdoc/>
    public async ValueTask<EvolutionTaskResult> EvaluateAsync(
        ProgramGenome candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(candidate);
        Guard.NotNull(context);

        int passed = 0;
        var diagnostics = new List<EvolutionDiagnostic>();

        for (int index = 0; index < _examples.Count; index++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                return Canceled(index);
            }

            ProgramInputOutputExample example = _examples[index];
            var request = new ProgramExecuteRequest
            {
                Language = candidate.Language,
                SourceCode = candidate.Source,
                StdIn = example.Input ?? string.Empty
            };

            ProgramExecuteResponse response;
            try
            {
                response = await _engine.ExecuteAsync(request, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                return Canceled(index);
            }
#pragma warning disable CA1031
            catch (Exception exception)
#pragma warning restore CA1031
            {
                AddDiagnostic(diagnostics, index, "engine_threw", exception.GetType().Name);
                continue;
            }

            if (response is null)
            {
                AddDiagnostic(diagnostics, index, "engine_threw", "The engine returned no response.");
                continue;
            }

            if (!response.Success)
            {
                AddDiagnostic(diagnostics, index, DescribeFailure(response), DescribeFailureDetail(response));
                continue;
            }

            if (Matches(response.StdOut, example.ExpectedOutput))
            {
                passed++;
            }
            else
            {
                AddDiagnostic(
                    diagnostics,
                    index,
                    "output_mismatch",
                    response.StdOutTruncated
                        ? "The captured output did not match the expected output and was truncated at the sandbox output cap."
                        : "The captured output did not match the expected output.");
            }
        }

        double quality = (double)passed / _examples.Count;
        var descriptors = new Dictionary<string, double>(StringComparer.Ordinal)
        {
            ["passRate"] = quality
        };

        return new EvolutionTaskResult(
            EvolutionEvaluationStatus.Completed,
            quality,
            EvolutionOptimizationDirection.Maximize,
            descriptors,
            costUnits: _examples.Count,
            diagnostics: diagnostics);
    }

    private static EvolutionTaskResult Canceled(int completedExamples) => new(
        EvolutionEvaluationStatus.Canceled,
        costUnits: completedExamples,
        diagnostics: new[] { new EvolutionDiagnostic("program_sandbox_canceled", "Evaluation was canceled.") });

    private static string DescribeFailure(ProgramExecuteResponse response) => response.ErrorCode switch
    {
        ProgramExecuteErrorCode.TimeoutOrCanceled => "timeout",
        ProgramExecuteErrorCode.CompilationFailed => "compile_failed",
        ProgramExecuteErrorCode.ExecutionFailed => "execution_failed",
        ProgramExecuteErrorCode.LanguageNotDetected => "language_not_detected",
        ProgramExecuteErrorCode.SqlNotSupported => "language_not_supported",
        ProgramExecuteErrorCode.SourceCodeTooLarge => "source_too_large",
        ProgramExecuteErrorCode.StdInTooLarge => "input_too_large",
        ProgramExecuteErrorCode.SourceCodeRequired => "source_required",
        ProgramExecuteErrorCode.InvalidRequest => "invalid_request",
        _ => "execution_failed"
    };

    private static string DescribeFailureDetail(ProgramExecuteResponse response)
    {
        string? error = response.Error;
        string reason;
        if (error is null || error.Trim().Length == 0)
        {
            reason = "The engine reported a failure without a message.";
        }
        else
        {
            reason = error;
        }
        return string.Concat(
            reason,
            " (exit ",
            response.ExitCode.ToString(CultureInfo.InvariantCulture),
            ")");
    }

    private bool Matches(string? actual, string? expected)
    {
        string left = ProgramText.Normalize(actual ?? string.Empty);
        string right = ProgramText.Normalize(expected ?? string.Empty);
        switch (_comparison)
        {
            case ProgramOutputComparison.Ordinal:
                return string.Equals(left, right, StringComparison.Ordinal);
            case ProgramOutputComparison.TrimmedOrdinal:
                return string.Equals(left.Trim(), right.Trim(), StringComparison.Ordinal);
            case ProgramOutputComparison.TrimmedOrdinalIgnoreCase:
                return string.Equals(left.Trim(), right.Trim(), StringComparison.OrdinalIgnoreCase);
            default:
                return string.Equals(
                    ProgramText.CollapseWhitespace(left),
                    ProgramText.CollapseWhitespace(right),
                    StringComparison.Ordinal);
        }
    }

    private static void AddDiagnostic(List<EvolutionDiagnostic> diagnostics, int exampleIndex, string code, string detail)
    {
        if (diagnostics.Count >= MaxRetainedExampleDiagnostics) return;
        string message = string.Concat(
            "Example ",
            (exampleIndex + 1).ToString(CultureInfo.InvariantCulture),
            ": ",
            ProgramText.Bound(ProgramText.Sanitize(detail), MaxErrorMessageLength));
        diagnostics.Add(new EvolutionDiagnostic("program_sandbox_" + code, message, isRedacted: true));
    }

    private static string BuildVersionHash(
        List<ProgramInputOutputExample> examples,
        ProgramOutputComparison comparison)
    {
        var components = new List<string>
        {
            "program-sandbox-evaluator-v1",
            ((int)comparison).ToString(CultureInfo.InvariantCulture)
        };

        foreach (ProgramInputOutputExample example in examples)
        {
            components.Add(example.Input ?? string.Empty);
            components.Add(example.ExpectedOutput ?? string.Empty);
        }

        return "program-sandbox-" + EvolutionHash.Combine(components);
    }
}
