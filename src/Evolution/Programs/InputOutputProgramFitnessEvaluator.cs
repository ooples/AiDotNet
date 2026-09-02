using System.Collections.ObjectModel;
using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>Scores a candidate program by running it on input/output examples through a caller-supplied engine.</summary>
/// <remarks>
/// <para>
/// Quality is the fraction of <see cref="ProgramInputOutputExample"/> cases whose captured output matches the
/// expected output, so it is always between zero and one and always comparable between candidates. Execution goes
/// through the <see cref="IProgramExecutionEngine"/> the caller supplies, which is the whole point: generated
/// program text is untrusted, the core library never runs it in the AiDotNet process, and the sandbox, container,
/// or remote runner that does run it stays entirely under the caller's control. A candidate that crashes, times
/// out, or prints the wrong answer simply scores lower; it never stops the run.
/// </para>
/// <para>
/// Failure text from the engine is truncated and stripped of control characters before it reaches a diagnostic, so
/// a program that prints a megabyte to standard error cannot bloat a checkpoint or leak raw payloads into a log.
/// At most eight example-level diagnostics are retained. <see cref="VersionHash"/> incorporates a hash of the
/// examples and the comparison mode, so changing the test set correctly invalidates older checkpoints.
/// </para>
/// <para><b>For Beginners:</b> This evaluator checks a generated program the way a teacher marks homework: it runs
/// the program on each example input and compares what it printed with the expected answer, then reports the
/// fraction it got right. You supply the "runner" that actually executes the code, because running code a language
/// model wrote is a security decision only you can make — a container or an isolated process is the usual choice.
/// Pick a <see cref="ProgramOutputComparison"/> that matches how precise the expected output has to be.</para>
/// </remarks>
public sealed class InputOutputProgramFitnessEvaluator : IProgramFitnessEvaluator
{
    private const int MaxRetainedExampleDiagnostics = 8;
    private const int MaxErrorMessageLength = 200;

    private readonly IProgramExecutionEngine _engine;
    private readonly ReadOnlyCollection<ProgramInputOutputExample> _examples;
    private readonly ProgramOutputComparison _comparison;

    /// <summary>Initializes an input/output evaluator.</summary>
    /// <param name="engine">The sandbox or runner that executes candidate programs.</param>
    /// <param name="examples">The input/output cases the candidate must satisfy; at least one is required.</param>
    /// <param name="comparison">How captured output is compared with expected output.</param>
    /// <param name="id">A stable evaluator identifier.</param>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> or <paramref name="examples"/> is <c>null</c>, or an example is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="examples"/> is empty, or <paramref name="id"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="comparison"/> is not a defined value.</exception>
    public InputOutputProgramFitnessEvaluator(
        IProgramExecutionEngine engine,
        IEnumerable<ProgramInputOutputExample> examples,
        ProgramOutputComparison comparison = ProgramOutputComparison.TrimmedOrdinal,
        string id = "program-io-evaluator")
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
    public ValueTask<EvolutionTaskResult> EvaluateAsync(
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
                return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
                    EvolutionEvaluationStatus.Canceled,
                    costUnits: index,
                    diagnostics: new[] { new EvolutionDiagnostic("program_io_canceled", "Evaluation was canceled.") }));
            }

            ProgramInputOutputExample example = _examples[index];
            bool executed;
            string output;
            string? errorMessage;
            try
            {
                executed = _engine.TryExecute(
                    candidate.Language, candidate.Source, example.Input, out output, out errorMessage, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
                    EvolutionEvaluationStatus.Canceled,
                    costUnits: index,
                    diagnostics: new[] { new EvolutionDiagnostic("program_io_canceled", "Execution was canceled.") }));
            }
#pragma warning disable CA1031
            catch (Exception exception)
#pragma warning restore CA1031
            {
                AddDiagnostic(diagnostics, index, "engine_threw", exception.GetType().Name);
                continue;
            }

            if (!executed)
            {
                AddDiagnostic(diagnostics, index, "execution_failed", errorMessage ?? "The engine reported no output.");
                continue;
            }

            if (Matches(output, example.ExpectedOutput)) passed++;
            else AddDiagnostic(diagnostics, index, "output_mismatch", "The captured output did not match the expected output.");
        }

        double quality = (double)passed / _examples.Count;
        var descriptors = new Dictionary<string, double>(StringComparer.Ordinal)
        {
            ["passRate"] = quality
        };

        return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
            EvolutionEvaluationStatus.Completed,
            quality,
            EvolutionOptimizationDirection.Maximize,
            descriptors,
            costUnits: _examples.Count,
            diagnostics: diagnostics));
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
                    ProgramText.CollapseWhitespace(left), ProgramText.CollapseWhitespace(right), StringComparison.Ordinal);
        }
    }

    private static void AddDiagnostic(List<EvolutionDiagnostic> diagnostics, int exampleIndex, string code, string detail)
    {
        if (diagnostics.Count >= MaxRetainedExampleDiagnostics) return;
        string message = string.Concat(
            "Example ", (exampleIndex + 1).ToString(CultureInfo.InvariantCulture), ": ",
            ProgramText.Bound(ProgramText.Sanitize(detail), MaxErrorMessageLength));
        diagnostics.Add(new EvolutionDiagnostic("program_io_" + code, message, isRedacted: true));
    }

    private static string BuildVersionHash(List<ProgramInputOutputExample> examples, ProgramOutputComparison comparison)
    {
        var components = new List<string>
        {
            "program-io-evaluator-v1",
            ((int)comparison).ToString(CultureInfo.InvariantCulture)
        };

        foreach (ProgramInputOutputExample example in examples)
        {
            components.Add(example.Input ?? string.Empty);
            components.Add(example.ExpectedOutput ?? string.Empty);
        }

        return "program-io-" + EvolutionHash.Combine(components);
    }
}
