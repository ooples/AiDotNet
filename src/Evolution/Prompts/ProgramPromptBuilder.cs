using System.Globalization;
using System.Text;
using AiDotNet.Agentic.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution.Programs;
using AiDotNet.Validation;

// AiDotNet.PromptEngineering.Templates is imported project-wide and also declares a ChatMessage type.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Evolution.Prompts;

/// <summary>Turns a parent program and its surroundings into the bounded, redacted messages sent to a model.</summary>
/// <remarks>
/// <para>
/// The builder is the only place prompt text is assembled, and it is a pure function of three things: the options
/// it was constructed with, the <see cref="ProgramPromptContext"/> it is given, and the
/// <see cref="StableRandom"/> stream it draws from. No clock, no culture, no process-global generator, no file
/// system at build time. That is what makes a run reproducible: replay the same seed and every request body is
/// byte-identical, which is the property a benchmark needs and the reference OpenEvolve sampler cannot offer,
/// because it samples its diverse programs and its template variations from Python's unseeded module-level
/// generator.
/// </para>
/// <para>
/// Validation happens when the builder is constructed, not when a prompt is rendered. Every template it will ever
/// use is checked so that each of its placeholders is resolvable — either supplied by the builder, declared as a
/// custom variable, or declared as a template variation — and the offending names are listed if not. Upstream
/// discovers the same class of mistake as a <c>KeyError</c> from inside a formatting call, part-way through a run
/// that has already spent money.
/// </para>
/// <para>
/// Everything that reaches a prompt is bounded and scrubbed. Program text is truncated to a per-program ceiling,
/// execution output is redacted by <see cref="PromptTextRedactor"/> and capped in bytes, diagnostics are limited
/// in number, and the finished message is held under a total character ceiling by dropping optional sections in a
/// fixed order — inspirations first, then history, then artifacts — before any hard truncation is considered.
/// Whether anything was dropped is reported through <see cref="ProgramPromptResult.WasTruncated"/>.
/// </para>
/// <para><b>For Beginners:</b> This assembles the actual message your search sends to the AI: the program to
/// improve, how it scored, what was tried before, a few example programs, whatever the program printed when it
/// ran, and the instruction to propose a change. You create one builder with your settings and reuse it for every
/// proposal. Because it is completely deterministic, running the same search twice with the same seed sends the
/// AI exactly the same words both times — which is what lets you compare two experiments honestly.</para>
/// </remarks>
public sealed class ProgramPromptBuilder
{
    private const string TruncationNotice = "... (prompt truncated to fit the configured limit)";
    private const string DiagnosticsHeading = "## Evaluation Diagnostics";
    private const string TaskHeading = "# Task Description";
    private const int MaxDiagnosticMessageChars = 400;

    private readonly ProgramEvolutionPromptOptions _promptOptions;
    private readonly ProgramEvolutionOptions _programOptions;
    private readonly ProgramPromptTemplateSet _templates;
    private readonly List<string> _variationNames;
    private readonly string _fenceLabel;

    /// <summary>Initializes a prompt builder and validates every template it will render.</summary>
    /// <param name="promptOptions">Prompt content and size settings; <c>null</c> uses the defaults.</param>
    /// <param name="programOptions">Language, evolve-block, and diff-marker settings; <c>null</c> uses the defaults.</param>
    /// <param name="templateSet">
    /// An explicit template set, or <c>null</c> to build one from
    /// <see cref="ProgramEvolutionPromptOptions.BuildTemplateSet"/>.
    /// </param>
    /// <exception cref="ArgumentException">
    /// An option is invalid, a template placeholder cannot be resolved, or a custom variable or template variation
    /// shadows a name the builder already supplies.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is outside its permitted range.</exception>
    /// <exception cref="DirectoryNotFoundException">A configured template directory does not exist.</exception>
    /// <exception cref="InvalidDataException">A configured fragments file is not a JSON object of string values.</exception>
    public ProgramPromptBuilder(
        ProgramEvolutionPromptOptions? promptOptions = null,
        ProgramEvolutionOptions? programOptions = null,
        ProgramPromptTemplateSet? templateSet = null)
    {
        ProgramEvolutionPromptOptions promptCopy = (promptOptions ?? new ProgramEvolutionPromptOptions()).Clone();
        promptCopy.Validate();
        ProgramEvolutionOptions programCopy = (programOptions ?? new ProgramEvolutionOptions()).Clone();
        programCopy.Validate();

        _promptOptions = promptCopy;
        _programOptions = programCopy;
        _templates = templateSet ?? promptCopy.BuildTemplateSet();
        _fenceLabel = ProgramLanguageDetector.GetFenceLabel(programCopy.Language);

        var variationNames = new List<string>(promptCopy.TemplateVariations.Keys);
        variationNames.Sort(StringComparer.Ordinal);
        _variationNames = variationNames;

        ValidateResolvable();
        VersionHash = "program-prompt-builder-" + EvolutionHash.Combine(BuildVersionComponents());
    }

    /// <summary>Gets a hash over the templates and the settings that shape every prompt this builder renders.</summary>
    /// <remarks>
    /// Fold it into an operator's version identity so that editing a template or a size limit is visible to
    /// checkpoint resume rather than silently changing what the model is shown mid-run.
    /// </remarks>
    public string VersionHash { get; }

    /// <summary>Gets the validated template set this builder renders from.</summary>
    public ProgramPromptTemplateSet Templates => _templates;

    /// <summary>Gets a copy of the prompt settings this builder was constructed with.</summary>
    /// <returns>An independent copy; mutating it does not affect the builder.</returns>
    public ProgramEvolutionPromptOptions GetPromptOptions() => _promptOptions.Clone();

    /// <summary>Gets a copy of the program settings this builder was constructed with.</summary>
    /// <returns>An independent copy; mutating it does not affect the builder.</returns>
    public ProgramEvolutionOptions GetProgramOptions() => _programOptions.Clone();

    /// <summary>Builds the messages that ask a model to improve the parent program.</summary>
    /// <param name="context">The parent program and everything shown alongside it.</param>
    /// <param name="random">
    /// The stream template variations and diverse-example sampling are drawn from, or <c>null</c> to derive a
    /// stream from the parent's identity so the result stays deterministic.
    /// </param>
    /// <returns>The rendered prompt and the record of how it was assembled.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="context"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">The context is internally inconsistent.</exception>
    public ProgramPromptResult Build(ProgramPromptContext context, StableRandom? random = null)
    {
        Guard.NotNull(context);
        context.Validate();

        StableRandom stream = random ?? StableRandom.CreateStream(StreamSeedFor(context.Parent), 0UL);
        ProgramPromptEvolutionMode mode = _promptOptions.ResolveEvolutionMode(context.Parent.Source.Length);
        ProgramPromptTemplateKey userKey = mode == ProgramPromptEvolutionMode.FullRewrite
            ? ProgramPromptTemplateKey.FullRewriteUser
            : ProgramPromptTemplateKey.DiffUser;

        // Variations and diverse sampling both draw from the stream, so they are
        // resolved once, before any budget retry, and reused across retries. A
        // retry that redrew would make the text depend on how many retries ran.
        Dictionary<string, string> variations = DrawVariations(stream);
        IReadOnlyList<ProgramPromptExample> diverse = SelectDiverse(context, stream);

        string userText = string.Empty;
        bool truncated = false;
        for (int stage = 0; stage <= 3; stage++)
        {
            userText = RenderUserMessage(context, userKey, variations, diverse, stage);
            if (userText.Length <= _promptOptions.MaxPromptChars) break;

            truncated = true;
            if (stage == 3) userText = Truncate(userText, _promptOptions.MaxPromptChars);
        }

        string systemText = RenderSystemMessage(context);
        return new ProgramPromptResult(systemText, userText, mode, userKey, variations, truncated);
    }

    /// <summary>Builds the messages that ask a model to score a program against named criteria.</summary>
    /// <param name="program">The program under review.</param>
    /// <param name="criteria">The criteria to score, one per line in the prompt.</param>
    /// <param name="responseSchema">
    /// The JSON shape the answer must take, or <c>null</c> for a schema derived from
    /// <paramref name="criteria"/>. Pair it with <see cref="ChatOptions.ResponseJsonSchema"/> so providers that
    /// support constrained decoding cannot return unparseable text at all.
    /// </param>
    /// <returns>A system message followed by the scoring request.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="program"/> or <paramref name="criteria"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="criteria"/> is empty or holds a null or blank entry.</exception>
    public IReadOnlyList<ChatMessage> BuildEvaluationMessages(
        ProgramGenome program,
        IReadOnlyList<string> criteria,
        string? responseSchema = null)
    {
        Guard.NotNull(program);
        Guard.NotNull(criteria);
        if (criteria.Count == 0)
        {
            throw new ArgumentException("At least one evaluation criterion is required.", nameof(criteria));
        }

        var criteriaBuilder = new StringBuilder();
        var schemaBuilder = new StringBuilder("{\n");
        for (int index = 0; index < criteria.Count; index++)
        {
            string criterion = criteria[index];
            if (string.IsNullOrWhiteSpace(criterion))
            {
                throw new ArgumentException("An evaluation criterion cannot be empty or white space.", nameof(criteria));
            }

            string trimmed = criterion.Trim();
            if (index > 0) criteriaBuilder.Append('\n');
            criteriaBuilder.Append((index + 1).ToString(CultureInfo.InvariantCulture)).Append(". ").Append(trimmed);
            schemaBuilder.Append("  \"").Append(ToJsonName(trimmed)).Append("\": <number between 0.0 and 1.0>,\n");
        }

        schemaBuilder.Append("  \"reasoning\": <one short sentence>\n}");

        var values = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["current_program"] = BoundProgram(program.Source),
            ["language"] = _fenceLabel,
            ["criteria"] = criteriaBuilder.ToString(),
            ["response_schema"] = responseSchema ?? schemaBuilder.ToString()
        };

        AddCustomVariables(values);
        string system = _promptOptions.EvaluatorSystemMessage is { } configured && configured.Trim().Length > 0
            ? configured
            : RenderWithDefaults(
                _templates.GetTemplate(ProgramPromptTemplateKey.EvaluatorSystemMessage),
                BaseTemplateValues(includeFeatureDimensions: false));

        return new List<ChatMessage>
        {
            ChatMessage.System(system),
            ChatMessage.User(RenderWithDefaults(_templates.GetTemplate(ProgramPromptTemplateKey.Evaluation), values))
        };
    }

    private string RenderSystemMessage(ProgramPromptContext context)
    {
        Dictionary<string, string> values = BaseTemplateValues(includeFeatureDimensions: true);
        values["feature_dimensions"] = DescribeFeatureDimensions(context);

        string system;
        if (_promptOptions.SystemMessage is { } configured && configured.Trim().Length > 0)
        {
            if (_promptOptions.SystemMessageMode == ProgramPromptSystemMessageMode.Literal)
            {
                system = configured;
            }
            else
            {
                // Validate() already proved the name resolves, so a miss here would
                // be a library bug rather than a user mistake.
                ProgramEvolutionPromptOptions.TryParseTemplateKey(configured, out ProgramPromptTemplateKey key);
                system = RenderWithDefaults(_templates.GetTemplate(key), values);
            }
        }
        else
        {
            system = RenderWithDefaults(_templates.GetTemplate(ProgramPromptTemplateKey.SystemMessage), values);
        }

        if (_promptOptions.ProgramsAsChangesDescription)
        {
            string changesInstructions = RenderWithDefaults(
                _templates.GetTemplate(ProgramPromptTemplateKey.SystemMessageChangesDescription),
                BaseTemplateValues(includeFeatureDimensions: false));
            var wrapperValues = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["system_message"] = system,
                ["system_message_changes_description"] = changesInstructions
            };
            AddCustomVariables(wrapperValues);
            system = RenderWithDefaults(
                _templates.GetTemplate(ProgramPromptTemplateKey.SystemMessageWithChangesDescription), wrapperValues);
        }

        if (_promptOptions.ExtraSystemText is { } extra && extra.Trim().Length > 0)
        {
            system = system.Length > 0 ? system + "\n\n" + extra.Trim() : extra.Trim();
        }

        return system;
    }

    private string RenderUserMessage(
        ProgramPromptContext context,
        ProgramPromptTemplateKey userKey,
        Dictionary<string, string> variations,
        IReadOnlyList<ProgramPromptExample> diverse,
        int stage)
    {
        bool includeInspirations = stage < 1;
        bool includeHistory = stage < 2;
        bool includeArtifacts = stage < 3;

        var values = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["metrics"] = DescribeMetricsInline(context.ParentMetrics),
            ["fitness_score"] = FormatScore(context.ParentQuality),
            ["feature_coords"] = DescribeFeatureCoordinates(context),
            ["feature_dimensions"] = DescribeFeatureDimensions(context),
            ["improvement_areas"] = DescribeImprovementAreas(context),
            ["evolution_history"] = includeHistory
                ? DescribeHistory(context, diverse, includeInspirations)
                : string.Empty,
            // The program is always the program, even in changes-description mode. Substituting the description for
            // it would leave the model editing source it was never shown, while the diff is still applied to that
            // source; the description belongs in the wrapper section below, which is where the reference
            // implementation puts it too.
            ["current_program"] = BoundProgram(context.Parent.Source),
            ["language"] = _fenceLabel,
            ["artifacts"] = includeArtifacts && _promptOptions.IncludeArtifacts
                ? DescribeArtifacts(context)
                : string.Empty,
            ["task_description"] = DescribeTask(),
            ["diagnostics"] = _promptOptions.IncludeDiagnostics ? DescribeDiagnostics(context) : string.Empty,
            ["search_marker"] = _programOptions.Diff.SearchMarker,
            ["divider_marker"] = _programOptions.Diff.DividerMarker,
            ["replace_marker"] = _programOptions.Diff.ReplaceMarker,
            ["evolve_block_instructions"] = DescribeEvolveBlocks()
        };

        foreach (KeyValuePair<string, string> variation in variations) values[variation.Key] = variation.Value;
        AddCustomVariables(values);

        string user = RenderWithDefaults(_templates.GetTemplate(userKey), values);
        if (!_promptOptions.ProgramsAsChangesDescription) return user;

        var wrapperValues = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["user_message"] = user,
            ["changes_description"] = (ResolveChangesDescription(context) ?? string.Empty).TrimEnd()
        };
        AddCustomVariables(wrapperValues);
        return RenderWithDefaults(
            _templates.GetTemplate(ProgramPromptTemplateKey.UserMessageWithChangesDescription), wrapperValues);
    }

    /// <summary>Returns the parent's changes description, or the configured starting text when it has none.</summary>
    /// <remarks>
    /// A seed program has no history, so its description is missing on the first generation. Showing the model an
    /// empty summary and asking it to edit that gives it nothing to write a SEARCH block against; the configured
    /// starting text gives it something real to replace, and is used only where a description is genuinely absent.
    /// </remarks>
    private string? ResolveChangesDescription(ProgramPromptContext context)
    {
        string? description = context.ChangesDescription;
        if (!string.IsNullOrWhiteSpace(description)) return description;
        return string.IsNullOrWhiteSpace(_promptOptions.InitialChangesDescription)
            ? description
            : _promptOptions.InitialChangesDescription;
    }

    private Dictionary<string, string> DrawVariations(StableRandom random)
    {
        var chosen = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!_promptOptions.UseTemplateStochasticity) return chosen;

        // Ordinal-sorted names, so the sequence of draws never depends on the
        // iteration order of a dictionary.
        foreach (string name in _variationNames)
        {
            IReadOnlyList<string> wordings = _promptOptions.TemplateVariations[name];
            chosen[name] = wordings.Count == 1 ? wordings[0] : wordings[random.NextInt(wordings.Count)];
        }

        return chosen;
    }

    private IReadOnlyList<ProgramPromptExample> SelectDiverse(ProgramPromptContext context, StableRandom random)
    {
        var selected = new List<ProgramPromptExample>();
        int top = Math.Min(_promptOptions.NumTopPrograms, context.TopPrograms.Count);
        int remaining = context.TopPrograms.Count - top;
        if (_promptOptions.NumDiversePrograms <= 0 || remaining <= 0) return selected;

        var pool = new List<ProgramPromptExample>(remaining);
        for (int index = top; index < context.TopPrograms.Count; index++) pool.Add(context.TopPrograms[index]);

        int wanted = Math.Min(_promptOptions.NumDiversePrograms, pool.Count);
        for (int drawn = 0; drawn < wanted; drawn++)
        {
            int pick = random.NextInt(pool.Count);
            selected.Add(pool[pick]);
            pool.RemoveAt(pick);
        }

        return selected;
    }

    private string DescribeHistory(
        ProgramPromptContext context,
        IReadOnlyList<ProgramPromptExample> diverse,
        bool includeInspirations)
    {
        var shown = new HashSet<string>(StringComparer.Ordinal);
        var values = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["previous_attempts"] = DescribeAttempts(context),
            ["top_programs"] = DescribeTopPrograms(context, diverse, shown),
            ["inspirations_section"] = includeInspirations && _promptOptions.IncludeInspirations
                ? DescribeInspirations(context, shown)
                : string.Empty
        };

        AddCustomVariables(values);
        return RenderWithDefaults(_templates.GetTemplate(ProgramPromptTemplateKey.EvolutionHistory), values).Trim();
    }

    private string DescribeAttempts(ProgramPromptContext context)
    {
        if (!_promptOptions.IncludePreviousAttempts || _promptOptions.NumPreviousAttempts == 0) return string.Empty;
        int count = Math.Min(_promptOptions.NumPreviousAttempts, context.PreviousAttempts.Count);
        if (count == 0) return string.Empty;

        ProgramPromptTemplate template = _templates.GetTemplate(ProgramPromptTemplateKey.PreviousAttempt);
        var builder = new StringBuilder();
        int first = context.PreviousAttempts.Count - count;
        for (int index = context.PreviousAttempts.Count - 1; index >= first; index--)
        {
            ProgramPromptAttempt attempt = context.PreviousAttempts[index];
            var values = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["attempt_number"] = attempt.AttemptNumber.ToString(CultureInfo.InvariantCulture),
                ["changes"] = attempt.ChangesDescription is { } changes && changes.Trim().Length > 0
                    ? PromptTextRedactor.Redact(changes.Trim())
                    : _templates.RenderFragment(ProgramPromptFragmentKey.AttemptUnknownChanges),
                ["performance"] = DescribeMetricsInline(attempt.Metrics),
                ["outcome"] = DescribeOutcome(context.Direction, attempt)
            };

            AddCustomVariables(values);
            if (builder.Length > 0) builder.Append("\n\n");
            builder.Append(RenderWithDefaults(template, values).TrimEnd());
        }

        return builder.ToString();
    }

    private string DescribeOutcome(EvolutionOptimizationDirection direction, ProgramPromptAttempt attempt)
    {
        int improved = 0;
        int regressed = 0;
        int compared = 0;
        foreach (KeyValuePair<string, double> pair in Ordered(attempt.Metrics))
        {
            if (!attempt.ParentMetrics.TryGetValue(pair.Key, out double parentValue)) continue;
            compared++;
            int sign = pair.Value.CompareTo(parentValue);
            if (sign == 0) continue;
            bool better = direction == EvolutionOptimizationDirection.Maximize ? sign > 0 : sign < 0;
            if (better) improved++;
            else regressed++;
        }

        if (compared == 0 || (improved == 0 && regressed == 0))
        {
            return _templates.RenderFragment(ProgramPromptFragmentKey.AttemptMixedMetrics);
        }

        if (regressed == 0) return _templates.RenderFragment(ProgramPromptFragmentKey.AttemptAllMetricsImproved);
        if (improved == 0) return _templates.RenderFragment(ProgramPromptFragmentKey.AttemptAllMetricsRegressed);
        return _templates.RenderFragment(ProgramPromptFragmentKey.AttemptMixedMetrics);
    }

    private string DescribeTopPrograms(
        ProgramPromptContext context,
        IReadOnlyList<ProgramPromptExample> diverse,
        HashSet<string> shown)
    {
        ProgramPromptTemplate template = _templates.GetTemplate(ProgramPromptTemplateKey.TopProgram);
        var builder = new StringBuilder();
        int top = Math.Min(_promptOptions.NumTopPrograms, context.TopPrograms.Count);
        for (int index = 0; index < top; index++)
        {
            ProgramPromptExample example = context.TopPrograms[index];
            shown.Add(example.Genome.Id);
            if (builder.Length > 0) builder.Append("\n\n");
            builder.Append(RenderExample(
                template,
                example,
                (index + 1).ToString(CultureInfo.InvariantCulture),
                DescribeKeyFeatures(example, ProgramPromptFragmentKey.TopProgramMetricsPrefix, withValues: true)));
        }

        if (diverse.Count == 0) return builder.ToString();

        if (builder.Length > 0) builder.Append("\n\n");
        builder.Append("## ")
            .Append(_templates.RenderFragment(ProgramPromptFragmentKey.DiverseProgramsTitle))
            .Append("\n\n");

        for (int index = 0; index < diverse.Count; index++)
        {
            ProgramPromptExample example = diverse[index];
            shown.Add(example.Genome.Id);
            if (index > 0) builder.Append("\n\n");
            builder.Append(RenderExample(
                template,
                example,
                "D" + (index + 1).ToString(CultureInfo.InvariantCulture),
                DescribeKeyFeatures(example, ProgramPromptFragmentKey.DiverseProgramMetricsPrefix, withValues: false)));
        }

        return builder.ToString();
    }

    private string DescribeInspirations(ProgramPromptContext context, HashSet<string> shown)
    {
        if (context.Inspirations.Count == 0) return string.Empty;

        ProgramPromptTemplate template = _templates.GetTemplate(ProgramPromptTemplateKey.InspirationProgram);
        var builder = new StringBuilder();
        int rendered = 0;
        foreach (ProgramPromptExample example in context.Inspirations)
        {
            // Dedup by canonical content identity, not by archive id: two entries
            // holding byte-identical programs must not both be quoted, which is
            // exactly what upstream's id-only check lets through.
            if (!shown.Add(example.Genome.Id)) continue;
            rendered++;
            if (builder.Length > 0) builder.Append("\n\n");

            var values = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["program_number"] = rendered.ToString(CultureInfo.InvariantCulture),
                ["score"] = FormatScore(example.Quality),
                ["program_type"] = DescribeExampleKind(example),
                ["language"] = _promptOptions.ProgramsAsChangesDescription ? "text" : _fenceLabel,
                ["program_snippet"] = DescribeExampleSource(example),
                ["unique_features"] = DescribeUniqueFeatures(example)
            };

            AddCustomVariables(values);
            builder.Append(RenderWithDefaults(template, values).TrimEnd());
        }

        if (rendered == 0) return string.Empty;

        var sectionValues = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["inspiration_programs"] = builder.ToString()
        };
        AddCustomVariables(sectionValues);
        return RenderWithDefaults(_templates.GetTemplate(ProgramPromptTemplateKey.InspirationsSection), sectionValues);
    }

    private string RenderExample(
        ProgramPromptTemplate template,
        ProgramPromptExample example,
        string number,
        string keyFeatures)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["program_number"] = number,
            ["score"] = FormatScore(example.Quality),
            ["language"] = _promptOptions.ProgramsAsChangesDescription ? "text" : _fenceLabel,
            ["program_snippet"] = DescribeExampleSource(example),
            ["key_features"] = keyFeatures
        };

        AddCustomVariables(values);
        return RenderWithDefaults(template, values).TrimEnd();
    }

    private string DescribeExampleSource(ProgramPromptExample example)
    {
        if (!_promptOptions.ProgramsAsChangesDescription) return BoundProgram(example.Genome.Source);
        string changes = example.ChangesDescription ?? string.Empty;
        return changes.Trim().Length == 0
            ? _templates.RenderFragment(ProgramPromptFragmentKey.AttemptUnknownChanges)
            : BoundProgram(PromptTextRedactor.Redact(changes));
    }

    private string DescribeKeyFeatures(ProgramPromptExample example, ProgramPromptFragmentKey prefixKey, bool withValues)
    {
        string prefix = _templates.RenderFragment(prefixKey);
        var parts = new List<string>();
        int limit = Math.Max(1, _promptOptions.NumTopPrograms);
        foreach (KeyValuePair<string, double> pair in Ordered(example.Descriptors))
        {
            if (parts.Count >= limit) break;
            parts.Add(withValues
                ? prefix + " " + pair.Key + " (" + FormatCoordinate(pair.Value) + ")"
                : prefix + " " + pair.Key);
        }

        return parts.Count == 0
            ? _templates.RenderFragment(ProgramPromptFragmentKey.NoFeatureCoordinates)
            : string.Join(", ", parts);
    }

    private string DescribeExampleKind(ProgramPromptExample example)
    {
        switch (example.Kind)
        {
            case ProgramPromptExampleKind.Diverse:
                return _templates.RenderFragment(ProgramPromptFragmentKey.InspirationTypeDiverse);
            case ProgramPromptExampleKind.Migrant:
                return _templates.RenderFragment(ProgramPromptFragmentKey.InspirationTypeMigrant);
            case ProgramPromptExampleKind.Random:
                return _templates.RenderFragment(ProgramPromptFragmentKey.InspirationTypeRandom);
            default:
                break;
        }

        double score = example.Quality ?? 0.0;
        if (score >= 0.8) return _templates.RenderFragment(ProgramPromptFragmentKey.InspirationTypeHighPerformer);
        if (score >= 0.6) return _templates.RenderFragment(ProgramPromptFragmentKey.InspirationTypeAlternative);
        if (score >= 0.4) return _templates.RenderFragment(ProgramPromptFragmentKey.InspirationTypeExperimental);
        return _templates.RenderFragment(ProgramPromptFragmentKey.InspirationTypeExploratory);
    }

    private string DescribeUniqueFeatures(ProgramPromptExample example)
    {
        var features = new List<string>();
        int limit = Math.Max(1, _promptOptions.NumTopPrograms);

        if (example.ChangesDescription is { } changes
            && changes.Trim().Length > 0
            && _promptOptions.IncludeChangesUnderChars is { } changesLimit
            && changes.Length < changesLimit)
        {
            features.Add(_templates.RenderFragment(
                ProgramPromptFragmentKey.InspirationChangesPrefix,
                new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["changes"] = PromptTextRedactor.Redact(changes.Trim())
                }));
        }

        foreach (KeyValuePair<string, double> pair in Ordered(example.Descriptors))
        {
            if (features.Count >= limit) break;
            if (pair.Value >= 0.9)
            {
                features.Add(_templates.RenderFragment(
                    ProgramPromptFragmentKey.InspirationMetricsExcellent,
                    new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["name"] = pair.Key,
                        ["value"] = FormatCoordinate(pair.Value)
                    }));
            }
            else if (pair.Value <= 0.3)
            {
                features.Add(_templates.RenderFragment(
                    ProgramPromptFragmentKey.InspirationMetricsAlternative,
                    new Dictionary<string, string>(StringComparer.Ordinal) { ["name"] = pair.Key }));
            }
        }

        if (features.Count < limit)
        {
            int lines = example.Genome.LineCount;
            if (_promptOptions.ConciseImplementationMaxLines is { } concise && lines <= concise)
            {
                features.Add(_templates.RenderFragment(ProgramPromptFragmentKey.InspirationConciseImplementation));
            }
            else if (_promptOptions.ComprehensiveImplementationMinLines is { } comprehensive && lines >= comprehensive)
            {
                features.Add(_templates.RenderFragment(ProgramPromptFragmentKey.InspirationComprehensiveImplementation));
            }
        }

        if (features.Count == 0)
        {
            features.Add(_templates.RenderFragment(
                ProgramPromptFragmentKey.InspirationNoFeatures,
                new Dictionary<string, string>(StringComparer.Ordinal) { ["type"] = DescribeExampleKind(example) }));
        }

        if (features.Count > limit) features.RemoveRange(limit, features.Count - limit);
        return string.Join(", ", features);
    }

    private string DescribeImprovementAreas(ProgramPromptContext context)
    {
        var areas = new List<string>();

        if (context.PreviousQuality.HasValue && context.ParentQuality.HasValue)
        {
            double previous = context.PreviousQuality.Value;
            double current = context.ParentQuality.Value;
            var values = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["previous"] = FormatScore(previous),
                ["current"] = FormatScore(current)
            };

            if (Math.Abs(current - previous) < _promptOptions.FitnessStableBand)
            {
                areas.Add(_templates.RenderFragment(ProgramPromptFragmentKey.FitnessStable, values));
            }
            else
            {
                bool better = context.Direction == EvolutionOptimizationDirection.Maximize
                    ? current > previous
                    : current < previous;
                areas.Add(_templates.RenderFragment(
                    better ? ProgramPromptFragmentKey.FitnessImproved : ProgramPromptFragmentKey.FitnessDeclined,
                    values));
            }
        }

        if (context.FeatureDimensions.Count > 0)
        {
            string coordinates = DescribeFeatureCoordinates(context);
            areas.Add(coordinates.Length == 0 || IsNoCoordinates(coordinates)
                ? _templates.RenderFragment(ProgramPromptFragmentKey.NoFeatureCoordinates)
                : _templates.RenderFragment(
                    ProgramPromptFragmentKey.ExploringRegion,
                    new Dictionary<string, string>(StringComparer.Ordinal) { ["features"] = coordinates }));
        }

        if (_promptOptions.SuggestSimplificationAfterChars is { } threshold
            && context.Parent.Source.Length > threshold)
        {
            areas.Add(_templates.RenderFragment(
                ProgramPromptFragmentKey.CodeTooLong,
                new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["threshold"] = threshold.ToString(CultureInfo.InvariantCulture)
                }));
        }

        if (_promptOptions.IncludeCoverageHints && context.EmptyNeighborCells.Count > 0)
        {
            areas.Add(_templates.RenderFragment(
                ProgramPromptFragmentKey.CoverageHint,
                new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["cells"] = string.Join("; ", context.EmptyNeighborCells)
                }));
        }

        if (areas.Count == 0) areas.Add(_templates.RenderFragment(ProgramPromptFragmentKey.NoSpecificGuidance));

        var builder = new StringBuilder();
        for (int index = 0; index < areas.Count; index++)
        {
            if (index > 0) builder.Append('\n');
            builder.Append("- ").Append(areas[index]);
        }

        return builder.ToString();
    }

    private string DescribeFeatureCoordinates(ProgramPromptContext context)
    {
        if (!_promptOptions.IncludeFeatureCoordinates) return string.Empty;

        var parts = new List<string>();
        for (int index = 0; index < context.FeatureDimensions.Count; index++)
        {
            string dimension = context.FeatureDimensions[index];
            if (!context.ParentDescriptors.TryGetValue(dimension, out double value)) continue;

            var part = new StringBuilder(dimension).Append('=').Append(FormatCoordinate(value));
            // Showing the bin alongside the raw value is what tells the model where
            // the grid still has room; upstream renders the value alone, which is a
            // number without a scale.
            if (index < context.FeatureBins.Count && index < context.FeatureBinCounts.Count)
            {
                part.Append(" [bin ")
                    .Append((context.FeatureBins[index] + 1).ToString(CultureInfo.InvariantCulture))
                    .Append('/')
                    .Append(context.FeatureBinCounts[index].ToString(CultureInfo.InvariantCulture))
                    .Append(']');
            }

            parts.Add(part.ToString());
        }

        return parts.Count == 0
            ? _templates.RenderFragment(ProgramPromptFragmentKey.NoFeatureCoordinates)
            : string.Join(", ", parts);
    }

    private static string DescribeFeatureDimensions(ProgramPromptContext context) =>
        context.FeatureDimensions.Count == 0 ? "none" : string.Join(", ", context.FeatureDimensions);

    private string DescribeMetricsInline(IReadOnlyDictionary<string, double> metrics)
    {
        if (metrics.Count == 0) return "none recorded";
        var parts = new List<string>();
        foreach (KeyValuePair<string, double> pair in Ordered(metrics))
        {
            parts.Add(pair.Key + ": " + FormatScore(pair.Value));
        }

        return string.Join(", ", parts);
    }

    private string DescribeArtifacts(ProgramPromptContext context)
    {
        if (context.Artifacts.Count == 0 || _promptOptions.MaxArtifactCount == 0) return string.Empty;

        var builder = new StringBuilder();
        int shown = 0;
        foreach (ProgramPromptArtifact artifact in context.Artifacts)
        {
            if (shown >= _promptOptions.MaxArtifactCount) break;
            string content = _promptOptions.ArtifactSecurityFilter
                ? PromptTextRedactor.RedactAndBound(
                    artifact.Content, _promptOptions.MaxArtifactBytes, TruncationMarker(), out _)
                : BoundArtifactWithoutRedaction(artifact.Content);

            if (content.Length == 0) continue;
            shown++;
            if (builder.Length > 0) builder.Append("\n\n");
            builder.Append("### ").Append(artifact.Name).Append("\n```\n").Append(content).Append("\n```");
        }

        if (shown == 0) return string.Empty;
        return "## " + _templates.RenderFragment(ProgramPromptFragmentKey.ArtifactTitle) + "\n\n" + builder;
    }

    private string TruncationMarker() => _templates.RenderFragment(
        ProgramPromptFragmentKey.ArtifactTruncated,
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["bytes"] = PromptTextRedactor.FormatBytes(_promptOptions.MaxArtifactBytes)
        });

    private string BoundArtifactWithoutRedaction(string content)
    {
        string bounded = PromptTextRedactor.BoundToUtf8Bytes(content, _promptOptions.MaxArtifactBytes, out bool truncated);
        if (!truncated) return bounded;
        string marker = TruncationMarker();
        return bounded.Length > 0 ? bounded + "\n" + marker : marker;
    }

    private string DescribeDiagnostics(ProgramPromptContext context)
    {
        if (context.Diagnostics.Count == 0 || _promptOptions.MaxDiagnostics == 0) return string.Empty;

        var builder = new StringBuilder(DiagnosticsHeading);
        int shown = 0;
        foreach (EvolutionDiagnostic diagnostic in context.Diagnostics)
        {
            if (shown >= _promptOptions.MaxDiagnostics) break;
            shown++;
            builder.Append("\n- ").Append(diagnostic.Code);
            if (diagnostic.Message.Length > 0)
            {
                builder.Append(": ")
                    .Append(Truncate(PromptTextRedactor.Redact(diagnostic.Message), MaxDiagnosticMessageChars));
            }
        }

        if (context.Diagnostics.Count > shown)
        {
            builder.Append("\n- and ")
                .Append((context.Diagnostics.Count - shown).ToString(CultureInfo.InvariantCulture))
                .Append(" more.");
        }

        return builder.ToString();
    }

    private string DescribeTask()
    {
        if (_promptOptions.TaskDescription is not { } description || description.Trim().Length == 0)
        {
            return string.Empty;
        }

        return TaskHeading + "\n" + description.Trim();
    }

    private string DescribeEvolveBlocks()
    {
        if (!_programOptions.EnforceEvolveBlocks) return string.Empty;
        EvolveBlockMarkers markers = _programOptions.ResolveEvolveBlockMarkers();
        return "Change only the lines between " + markers.Start + " and " + markers.End +
            "; edits anywhere else are rejected.";
    }

    private string BoundProgram(string source)
    {
        string text = source ?? string.Empty;
        return text.Length <= _promptOptions.MaxProgramSnippetChars
            ? text
            : Truncate(text, _promptOptions.MaxProgramSnippetChars);
    }

    private string FormatScore(double? value) =>
        value.HasValue
            ? value.Value.ToString(
                "F" + _promptOptions.ScoreDecimals.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture)
            : "unknown";

    private static string FormatCoordinate(double value) => value.ToString("0.####", CultureInfo.InvariantCulture);

    private Dictionary<string, string> BaseTemplateValues(bool includeFeatureDimensions)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["language"] = _fenceLabel,
            ["task_description"] = _promptOptions.TaskDescription?.Trim() ?? string.Empty
        };

        if (includeFeatureDimensions) values["feature_dimensions"] = "none";
        AddCustomVariables(values);
        return values;
    }

    private void AddCustomVariables(Dictionary<string, string> values)
    {
        foreach (KeyValuePair<string, string> pair in _promptOptions.CustomVariables)
        {
            if (!values.ContainsKey(pair.Key)) values[pair.Key] = pair.Value;
        }
    }

    private static string RenderWithDefaults(ProgramPromptTemplate template, Dictionary<string, string> values)
    {
        foreach (string name in template.Placeholders)
        {
            if (!values.ContainsKey(name)) values[name] = string.Empty;
        }

        return template.Render(values);
    }

    private bool IsNoCoordinates(string coordinates) => string.Equals(
        coordinates,
        _templates.RenderFragment(ProgramPromptFragmentKey.NoFeatureCoordinates),
        StringComparison.Ordinal);

    private static IEnumerable<KeyValuePair<string, double>> Ordered(IReadOnlyDictionary<string, double> values)
    {
        var names = new List<string>(values.Keys);
        names.Sort(StringComparer.Ordinal);
        foreach (string name in names) yield return new KeyValuePair<string, double>(name, values[name]);
    }

    private static string Truncate(string text, int maximumLength)
    {
        if (maximumLength <= 0) return string.Empty;
        if (text.Length <= maximumLength) return text;
        int keep = maximumLength - TruncationNotice.Length - 1;
        return keep <= 0
            ? text.Substring(0, maximumLength)
            : text.Substring(0, keep) + "\n" + TruncationNotice;
    }

    private static ulong StreamSeedFor(ProgramGenome genome)
    {
        ulong seed = 1469598103934665603UL;
        foreach (char character in genome.Id)
        {
            seed = unchecked((seed ^ character) * 1099511628211UL);
        }

        return seed;
    }

    /// <summary>Converts a human-readable criterion into the JSON field name the evaluation prompt asks for.</summary>
    /// <param name="criterion">The criterion as it is written in the prompt.</param>
    /// <returns>
    /// A lower-case, underscore-separated field name, or <c>"score"</c> when the criterion holds no letters or
    /// digits at all.
    /// </returns>
    /// <remarks>
    /// A judge that reads the model's answer must look for exactly the field names
    /// <see cref="BuildEvaluationMessages"/> asked for, so both sides derive them here rather than each guessing.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="criterion"/> is <c>null</c>.</exception>
    public static string ToCriterionFieldName(string criterion)
    {
        Guard.NotNull(criterion);
        return ToJsonName(criterion.Trim());
    }

    private static string ToJsonName(string criterion)
    {
        var builder = new StringBuilder(criterion.Length);
        foreach (char character in criterion)
        {
            if (char.IsLetterOrDigit(character)) builder.Append(char.ToLowerInvariant(character));
            else if (builder.Length > 0 && builder[builder.Length - 1] != '_') builder.Append('_');
        }

        string name = builder.ToString().Trim('_');
        return name.Length == 0 ? "score" : name;
    }

    private void ValidateResolvable()
    {
        var supplied = new HashSet<string>(StringComparer.Ordinal);
        foreach (ProgramPromptTemplateKey key in ProgramPromptTemplateSet.TemplateKeys)
        {
            foreach (string name in ProgramPromptTemplateSet.SuppliedPlaceholders(key)) supplied.Add(name);
        }

        foreach (string name in _promptOptions.CustomVariables.Keys)
        {
            if (!supplied.Contains(name)) continue;
            throw new ArgumentException(
                $"The custom variable '{name}' shadows a value the prompt builder already supplies. Choose another name.",
                "promptOptions");
        }

        foreach (string name in _variationNames)
        {
            if (supplied.Contains(name))
            {
                throw new ArgumentException(
                    $"The template variation '{name}' shadows a value the prompt builder already supplies. Choose another name.",
                    "promptOptions");
            }

            if (_promptOptions.CustomVariables.ContainsKey(name))
            {
                throw new ArgumentException(
                    $"'{name}' is declared both as a custom variable and as a template variation.",
                    "promptOptions");
            }
        }

        var unresolved = new List<string>();
        foreach (ProgramPromptTemplateKey key in ProgramPromptTemplateSet.TemplateKeys)
        {
            var allowed = new HashSet<string>(ProgramPromptTemplateSet.SuppliedPlaceholders(key), StringComparer.Ordinal);
            foreach (string name in _promptOptions.CustomVariables.Keys) allowed.Add(name);
            foreach (string name in _variationNames) allowed.Add(name);

            foreach (string placeholder in _templates.GetTemplate(key).Placeholders)
            {
                if (allowed.Contains(placeholder)) continue;
                unresolved.Add(ProgramPromptTemplateSet.TemplateFileStem(key) + ".{" + placeholder + "}");
            }
        }

        if (unresolved.Count > 0)
        {
            // Configure time, not run time: upstream surfaces the same mistake as a
            // KeyError from inside a format call, part-way through a paid run.
            throw new ArgumentException(
                "These prompt placeholders cannot be resolved: " + string.Join(", ", unresolved) +
                ". Add them to CustomVariables or TemplateVariations, or remove them from the template.",
                "promptOptions");
        }
    }

    private IEnumerable<string> BuildVersionComponents()
    {
        yield return "program-prompt-builder-v1";
        yield return _templates.VersionHash;
        yield return _programOptions.Language.ToString();
        yield return _programOptions.Diff.SearchMarker;
        yield return _programOptions.Diff.DividerMarker;
        yield return _programOptions.Diff.ReplaceMarker;
        yield return _programOptions.EnforceEvolveBlocks ? "enforce" : "free";
        yield return ((int)_promptOptions.EvolutionMode).ToString(CultureInfo.InvariantCulture);
        yield return ((int)_promptOptions.SystemMessageMode).ToString(CultureInfo.InvariantCulture);
        yield return _promptOptions.SystemMessage ?? string.Empty;
        yield return _promptOptions.EvaluatorSystemMessage ?? string.Empty;
        yield return _promptOptions.ExtraSystemText ?? string.Empty;
        yield return _promptOptions.TaskDescription ?? string.Empty;
        yield return _promptOptions.NumTopPrograms.ToString(CultureInfo.InvariantCulture);
        yield return _promptOptions.NumDiversePrograms.ToString(CultureInfo.InvariantCulture);
        yield return _promptOptions.NumPreviousAttempts.ToString(CultureInfo.InvariantCulture);
        yield return _promptOptions.MaxArtifactBytes.ToString(CultureInfo.InvariantCulture);
        yield return _promptOptions.MaxPromptChars.ToString(CultureInfo.InvariantCulture);
        yield return _promptOptions.MaxProgramSnippetChars.ToString(CultureInfo.InvariantCulture);
        yield return _promptOptions.ScoreDecimals.ToString(CultureInfo.InvariantCulture);
        yield return _promptOptions.ProgramsAsChangesDescription ? "changes" : "source";
        yield return _promptOptions.UseTemplateStochasticity ? "varying" : "fixed";

        foreach (string name in _variationNames)
        {
            yield return name;
            foreach (string wording in _promptOptions.TemplateVariations[name]) yield return wording;
        }

        var customNames = new List<string>(_promptOptions.CustomVariables.Keys);
        customNames.Sort(StringComparer.Ordinal);
        foreach (string name in customNames)
        {
            yield return name;
            yield return _promptOptions.CustomVariables[name];
        }
    }
}
