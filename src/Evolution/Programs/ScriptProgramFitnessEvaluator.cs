using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution.Programs.Metrics;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.Validation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Evolution.Programs;

/// <summary>
/// Scores a candidate by running a caller-supplied evaluator script inside the same sandbox, under the same limits,
/// and reading the JSON metrics it prints.
/// </summary>
/// <remarks>
/// <para>
/// The contract is deliberately small. The evaluator script runs as the sandboxed program; the candidate's source
/// arrives on its standard input; and it prints exactly one JSON object to standard output:
/// <c>{"quality": 0.83, "descriptors": {"length": 0.4}, "objectives": [0.83, 0.1], "artifacts": {"note": "…"}}</c>.
/// Only <c>quality</c> is required and it must be a finite number. Descriptors and objectives must be finite too;
/// artifacts are free text that becomes bounded, redacted diagnostics so a chatty evaluator cannot bloat a
/// checkpoint. Anything else in the object is ignored, so an evaluator may print extra fields for its own use.
/// </para>
/// <para>
/// Two things here are stricter than the reference implementation. First, the evaluator script is sandboxed exactly
/// like the candidates: it goes through the same <see cref="IProgramExecutionEngine"/>, so an evaluator with an
/// infinite loop is killed by the same wall-clock limit instead of hanging a worker, and one that prints without
/// stopping is truncated by the same output cap. OpenEvolve writes its evaluator to a file and executes it
/// unsandboxed in its worker processes, which means a bad evaluator is a worse problem than a bad candidate.
/// Second, the metrics are schema-checked: a non-finite quality, a descriptor that is not a number, or output that
/// is not a JSON object produces a <see cref="EvolutionEvaluationStatus.Failed"/> result with a diagnostic naming
/// the problem, never an exception and never a silently-zero score that would look like a legitimately bad program.
/// </para>
/// <para>
/// Because the sandbox deliberately gives the script a pinned path and a scrubbed environment, an evaluator script
/// should judge the candidate's <em>text</em> — its structure, its style, a domain rule. Scoring a candidate by
/// running it belongs in <see cref="SandboxedProgramFitnessEvaluator"/>, which executes the candidate itself.
/// </para>
/// <para><b>For Beginners:</b> Sometimes the question "is this program good?" needs judgement that a table of
/// expected outputs cannot express. This class lets you write that judgement as a small script: it receives the
/// candidate's code on standard input and prints a score as JSON. Your script runs in the same locked-down sandbox
/// as the programs being evolved, so a mistake in your scoring script cannot hang or harm the run either — it just
/// gets reported as a failed evaluation.</para>
/// </remarks>
public sealed class ScriptProgramFitnessEvaluator : IProgramFitnessEvaluator
{
    private const string QualityProperty = "quality";
    private const string DescriptorsProperty = "descriptors";
    private const string ObjectivesProperty = "objectives";
    private const string ArtifactsProperty = "artifacts";
    private const int MaxDiagnosticDetailLength = 200;

    private const string MetricsProperty = "metrics";

    private readonly IProgramExecutionEngine _engine;
    private readonly string _script;
    private readonly ScriptProgramEvaluationOptions _options;
    private readonly ProgramMetricAggregator? _metricAggregator;

    /// <summary>Initializes an evaluator that runs <paramref name="evaluatorScript"/> against every candidate.</summary>
    /// <param name="engine">The sandbox that runs the evaluator script; the same limits apply as to candidates.</param>
    /// <param name="evaluatorScript">The evaluator source, which must contain the configured entry-point marker.</param>
    /// <param name="options">How the script is validated and how its output is bounded, or <c>null</c> for defaults.</param>
    /// <param name="id">A stable evaluator identifier.</param>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> or <paramref name="evaluatorScript"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="evaluatorScript"/> is empty or white space, does not contain
    /// <see cref="ScriptProgramEvaluationOptions.EntryPointMarker"/> while it is required, or <paramref name="id"/>
    /// is empty or white space.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">A value in <paramref name="options"/> is out of range.</exception>
    /// <param name="metricAggregator">
    /// Optional rule for reducing a metric dictionary to a quality when the script reports no <c>quality</c>
    /// property; <c>null</c> keeps the stricter behaviour of failing such a script.
    /// </param>
    public ScriptProgramFitnessEvaluator(
        IProgramExecutionEngine engine,
        string evaluatorScript,
        ScriptProgramEvaluationOptions? options = null,
        string id = "program-script-evaluator",
        ProgramMetricAggregator? metricAggregator = null)
    {
        Guard.NotNull(engine);
        Guard.NotNull(evaluatorScript);
        Guard.NotNullOrWhiteSpace(id);

        ScriptProgramEvaluationOptions resolved = (options ?? new ScriptProgramEvaluationOptions()).Clone();
        resolved.Validate();

        string script = string.IsNullOrWhiteSpace(evaluatorScript)
            ? resolved.EvaluatorScript ?? string.Empty
            : evaluatorScript;
        if (string.IsNullOrWhiteSpace(script))
            throw new ArgumentException("The evaluator script cannot be empty.", nameof(evaluatorScript));

        if (resolved.RequireEntryPoint &&
            script.IndexOf(resolved.EntryPointMarker, StringComparison.Ordinal) < 0)
        {
            throw new ArgumentException(
                $"The evaluator script does not contain the required entry point '{resolved.EntryPointMarker}'.",
                nameof(evaluatorScript));
        }

        _engine = engine;
        _script = script;
        _options = resolved;
        _metricAggregator = metricAggregator;
        Id = id.Trim();
        VersionHash = "program-script-" + EvolutionHash.Combine(new[]
        {
            // The aggregation rule changes what a given script scores, so two evaluators that differ only in it
            // must not share a version hash and pass a checkpoint-compatibility check.
            metricAggregator is null
                ? "no-metric-aggregation"
                : "metric-aggregation:" + metricAggregator.GetOptions().ToString(),
            "program-script-evaluator-v1",
            ProgramText.Normalize(script),
            ((int)resolved.EvaluatorScriptLanguage).ToString(CultureInfo.InvariantCulture),
            ((int)resolved.Direction).ToString(CultureInfo.InvariantCulture),
            resolved.EntryPointMarker
        });
    }

    /// <inheritdoc/>
    public string Id { get; }

    /// <inheritdoc/>
    public string VersionHash { get; }

    /// <summary>Gets the language the evaluator script is executed as.</summary>
    public ProgramLanguage EvaluatorScriptLanguage => _options.EvaluatorScriptLanguage;

    /// <summary>Gets an independent copy of the options this evaluator was constructed with.</summary>
    /// <returns>A copy, so inspecting the configuration cannot change a running evaluator.</returns>
    public ScriptProgramEvaluationOptions GetOptions() => _options.Clone();

    /// <inheritdoc/>
    public async ValueTask<EvolutionTaskResult> EvaluateAsync(
        ProgramGenome candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(candidate);
        Guard.NotNull(context);

        if (cancellationToken.IsCancellationRequested)
        {
            return Canceled();
        }

        var request = new ProgramExecuteRequest
        {
            Language = _options.EvaluatorScriptLanguage,
            SourceCode = _script,
            StdIn = candidate.Source
        };

        ProgramExecuteResponse response;
        try
        {
            response = await _engine.ExecuteAsync(request, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            return Canceled();
        }
#pragma warning disable CA1031
        catch (Exception exception)
#pragma warning restore CA1031
        {
            return Failed("engine_threw", exception.GetType().Name);
        }

        if (response is null)
        {
            return Failed("engine_threw", "The engine returned no response.");
        }

        if (!response.Success)
        {
            string code = response.ErrorCode == ProgramExecuteErrorCode.TimeoutOrCanceled
                ? "timeout"
                : "script_failed";
            return Failed(code, response.Error ?? "The evaluator script did not complete.");
        }

        if (response.StdOutTruncated)
        {
            return Failed(
                "metrics_truncated",
                "The evaluator script printed more than the sandbox output cap, so its metrics are incomplete.");
        }

        return Parse(response.StdOut);
    }

    /// <summary>Reduces the script's reported metrics to one quality using the configured rule.</summary>
    /// <param name="payload">The parsed evaluator output.</param>
    /// <param name="issues">Receives one diagnostic per metric that could not be combined.</param>
    /// <param name="quality">The combined quality when the method returns <see langword="true"/>.</param>
    /// <returns><see langword="true"/> when a finite quality was produced.</returns>
    /// <remarks>
    /// Metrics are read from a <c>metrics</c> object when the script supplies one, otherwise from the top-level
    /// properties, which is the shape the reference implementation's evaluators print. Structural properties are
    /// excluded so a descriptor block is never mistaken for a metric. Every value that cannot be combined is
    /// reported as a diagnostic rather than dropped, which is the difference between a low score you can explain
    /// and one you cannot.
    /// </remarks>
    private bool TryAggregateQuality(JObject payload, List<EvolutionDiagnostic> issues, out double quality)
    {
        quality = 0d;
        if (_metricAggregator is null) return false;

        JObject? source = payload[MetricsProperty] as JObject;
        var metrics = new Dictionary<string, ProgramMetricValue>(StringComparer.Ordinal);
        foreach (JProperty property in (source ?? payload).Properties())
        {
            if (source is null && IsStructuralProperty(property.Name)) continue;
            if (string.IsNullOrWhiteSpace(property.Name)) continue;

            switch (property.Value.Type)
            {
                case JTokenType.Integer:
                case JTokenType.Float:
                    if (TryReadFinite(property.Value, out double number))
                    {
                        metrics[property.Name.Trim()] = ProgramMetricValue.Number(number);
                    }
                    else
                    {
                        issues.Add(new EvolutionDiagnostic(
                            "metric_not_finite",
                            $"Metric '{ProgramText.Bound(property.Name, 48)}' is not a finite number."));
                    }

                    break;
                case JTokenType.Boolean:
                    metrics[property.Name.Trim()] = ProgramMetricValue.Flag(property.Value.Value<bool>());
                    break;
                case JTokenType.String:
                    metrics[property.Name.Trim()] = ProgramMetricValue.Text(
                        ProgramText.Bound(property.Value.Value<string>() ?? string.Empty, 200));
                    break;
                default:
                    break;
            }
        }

        if (metrics.Count == 0) return false;

        ProgramMetricAggregationResult result = _metricAggregator.Aggregate(metrics);
        foreach (ProgramMetricIssue issue in result.Issues)
        {
            issues.Add(new EvolutionDiagnostic(
                "metric_not_combined",
                $"Metric '{ProgramText.Bound(issue.MetricName, 48)}' was not combined ({issue.Reason}): " +
                ProgramText.Bound(issue.Description, 120)));
        }

        if (!result.HasFiniteValue) return false;
        quality = result.Value;
        return true;
    }

    /// <summary>Reports whether a top-level property describes the result's shape rather than a metric.</summary>
    private static bool IsStructuralProperty(string name) =>
        string.Equals(name, QualityProperty, StringComparison.Ordinal) ||
        string.Equals(name, DescriptorsProperty, StringComparison.Ordinal) ||
        string.Equals(name, ObjectivesProperty, StringComparison.Ordinal) ||
        string.Equals(name, ArtifactsProperty, StringComparison.Ordinal) ||
        string.Equals(name, MetricsProperty, StringComparison.Ordinal);

    private EvolutionTaskResult Parse(string stdOut)
    {
        JObject payload;
        try
        {
            string trimmed = ExtractJsonObject(stdOut);
            if (trimmed.Length == 0)
            {
                return Failed("invalid_metrics", "The evaluator script printed no JSON object.");
            }

            payload = JObject.Parse(trimmed);
        }
        catch (JsonReaderException exception)
        {
            return Failed("invalid_metrics", "The evaluator script printed malformed JSON: " + exception.Message);
        }
        catch (JsonSerializationException exception)
        {
            return Failed("invalid_metrics", "The evaluator script printed malformed JSON: " + exception.Message);
        }
        catch (FormatException exception)
        {
            // A numeric literal the reader accepted but the framework cannot represent.
            return Failed("invalid_metrics", "The evaluator script printed an unreadable number: " + exception.Message);
        }
        catch (OverflowException exception)
        {
            // .NET Framework throws rather than saturating on an out-of-range exponent.
            return Failed("invalid_metrics", "The evaluator script printed an out-of-range number: " + exception.Message);
        }

        var metricIssues = new List<EvolutionDiagnostic>();
        if (!TryReadFinite(payload[QualityProperty], out double quality))
        {
            // An evaluator written for the reference implementation reports a flat metric dictionary and no
            // "quality" at all, so without a reduction rule it would fail outright here. With one configured, the
            // same script scores by combined_score or by the mean of its numeric metrics, exactly as upstream.
            if (_metricAggregator is null || !TryAggregateQuality(payload, metricIssues, out quality))
            {
                return Failed(
                    "invalid_metrics",
                    _metricAggregator is null
                        ? $"The '{QualityProperty}' property is missing or not a finite number."
                        : $"The '{QualityProperty}' property is missing and no finite metric could be combined " +
                          "in its place.");
            }
        }

        var descriptors = new Dictionary<string, double>(StringComparer.Ordinal);
        if (payload[DescriptorsProperty] is JObject descriptorObject)
        {
            foreach (JProperty property in descriptorObject.Properties())
            {
                if (string.IsNullOrWhiteSpace(property.Name))
                {
                    return Failed("invalid_metrics", "A descriptor name was empty.");
                }

                if (!TryReadFinite(property.Value, out double value))
                {
                    return Failed(
                        "invalid_metrics",
                        $"Descriptor '{ProgramText.Bound(property.Name, 48)}' is not a finite number.");
                }

                descriptors[property.Name.Trim()] = value;
            }
        }
        else if (payload[DescriptorsProperty] is JToken descriptorToken && descriptorToken.Type != JTokenType.Null)
        {
            return Failed("invalid_metrics", $"The '{DescriptorsProperty}' property must be a JSON object.");
        }

        var objectives = new List<double>();
        if (payload[ObjectivesProperty] is JArray objectiveArray)
        {
            foreach (JToken item in objectiveArray)
            {
                if (!TryReadFinite(item, out double value))
                {
                    return Failed("invalid_metrics", $"The '{ObjectivesProperty}' array contains a non-finite value.");
                }

                objectives.Add(value);
            }
        }
        else if (payload[ObjectivesProperty] is JToken objectiveToken && objectiveToken.Type != JTokenType.Null)
        {
            return Failed("invalid_metrics", $"The '{ObjectivesProperty}' property must be a JSON array.");
        }

        // Metrics that could not be combined lead the diagnostics: a score derived from a partial metric set is
        // explainable only if the caller can see which values were left out and why.
        var diagnostics = new List<EvolutionDiagnostic>(metricIssues);
        if (payload[ArtifactsProperty] is JObject artifactObject && _options.MaxArtifactCount > 0)
        {
            foreach (JProperty property in artifactObject.Properties())
            {
                if (diagnostics.Count >= _options.MaxArtifactCount) break;
                if (string.IsNullOrWhiteSpace(property.Name)) continue;

                JToken value = property.Value;
                string text = value.Type == JTokenType.Null
                    ? string.Empty
                    : value.Type == JTokenType.Object || value.Type == JTokenType.Array
                        ? value.ToString(Formatting.None)
                        : value.ToString();

                diagnostics.Add(new EvolutionDiagnostic(
                    "program_script_artifact",
                    string.Concat(
                        ProgramText.Bound(ProgramText.Sanitize(property.Name.Trim()), 48),
                        ": ",
                        ProgramText.Bound(ProgramText.Sanitize(text), _options.MaxArtifactLength)),
                    isRedacted: true));
            }
        }

        return new EvolutionTaskResult(
            EvolutionEvaluationStatus.Completed,
            quality,
            _options.Direction,
            descriptors,
            objectives,
            costUnits: 1,
            diagnostics: diagnostics);
    }

    /// <summary>Finds the outermost JSON object in a script's standard output.</summary>
    /// <remarks>
    /// Scripts frequently print progress lines before their result. Rather than demanding pristine output, the
    /// span between the first opening brace and the last closing brace is parsed, and anything that is not a valid
    /// object is then reported as malformed rather than guessed at.
    /// </remarks>
    private static string ExtractJsonObject(string stdOut)
    {
        if (string.IsNullOrEmpty(stdOut)) return string.Empty;
        int start = stdOut.IndexOf('{');
        int end = stdOut.LastIndexOf('}');
        return start >= 0 && end > start ? stdOut.Substring(start, end - start + 1) : string.Empty;
    }

    private static bool TryReadFinite(JToken? token, out double value)
    {
        value = 0.0;
        if (token is null) return false;

        switch (token.Type)
        {
            case JTokenType.Integer:
            case JTokenType.Float:
                value = token.Value<double>();
                break;
            case JTokenType.Boolean:
                value = token.Value<bool>() ? 1.0 : 0.0;
                break;
            default:
                return false;
        }

        return !double.IsNaN(value) && !double.IsInfinity(value);
    }

    private static EvolutionTaskResult Canceled() => new(
        EvolutionEvaluationStatus.Canceled,
        diagnostics: new[] { new EvolutionDiagnostic("program_script_canceled", "Evaluation was canceled.") });

    private static EvolutionTaskResult Failed(string code, string detail) => new(
        EvolutionEvaluationStatus.Failed,
        costUnits: 1,
        diagnostics: new[]
        {
            new EvolutionDiagnostic(
                "program_script_" + code,
                ProgramText.Bound(ProgramText.Sanitize(detail), MaxDiagnosticDetailLength),
                isRedacted: true)
        });
}
