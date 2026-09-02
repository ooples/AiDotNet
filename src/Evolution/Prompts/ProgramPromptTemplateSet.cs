using System.Text;
using AiDotNet.Enums;
using AiDotNet.Validation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Evolution.Prompts;

/// <summary>The complete, validated collection of prompt texts and phrases a program-evolution run renders.</summary>
/// <remarks>
/// <para>
/// A set holds one <see cref="ProgramPromptTemplate"/> for every <see cref="ProgramPromptTemplateKey"/> and one
/// short phrase for every <see cref="ProgramPromptFragmentKey"/>. It is immutable: <see cref="With"/> and
/// <see cref="WithFragment"/> return a new set, and <see cref="LoadFromDirectory(string)"/> layers files from a
/// directory over the shipped defaults. Every construction path validates, so an invalid set cannot exist — a
/// template that drops a structurally required slot, or a phrase that asks for an argument it will never be
/// given, is refused at the point it is configured rather than at the point a model is first called.
/// </para>
/// <para>
/// Defaults are compiled into the assembly rather than read from files beside it, so they can never be missing,
/// truncated, or shadowed by a stale copy on disk. Overrides are read as UTF-8 by file stem, which is the same
/// layout the reference OpenEvolve implementation uses, so an existing template directory transfers unchanged —
/// but read correctly: upstream opens template files in the platform's default encoding, which turns every
/// non-ASCII character in a UTF-8 template into mojibake on a Windows machine. Upstream also carries a set of
/// in-module default templates that nothing ever reads and that have drifted out of step with the files that are
/// actually loaded; there is one source of truth here.
/// </para>
/// <para>
/// <see cref="VersionHash"/> covers every template and phrase, so folding it into an operator's version identity
/// makes an edited prompt visible to checkpoint resume: a run cannot silently continue with different wording
/// than it started with.
/// </para>
/// <para><b>For Beginners:</b> This is the wording your AI-driven search uses, all in one place. You get a
/// complete, working set for free by calling <see cref="CreateDefault"/>. To change any part of it, either call
/// <see cref="With"/> with your own text or drop a <c>.txt</c> file into a folder and call
/// <see cref="LoadFromDirectory(string)"/> — file <c>diff_user.txt</c> replaces the diff request,
/// <c>system_message.txt</c> replaces the standing instructions, and so on. Whatever you supply is checked
/// straight away, so a typo in a fill-in-the-blank name is reported while you are setting up rather than after
/// the run has been going for an hour.</para>
/// </remarks>
public sealed class ProgramPromptTemplateSet
{
    /// <summary>The file name, inside a template directory, that carries phrase overrides.</summary>
    public const string FragmentsFileName = "fragments.json";

    /// <summary>The largest phrase text accepted, in characters.</summary>
    public const int MaxFragmentLength = 4_096;

    private static readonly IReadOnlyList<ProgramPromptTemplateKey> AllTemplateKeys = new[]
    {
        ProgramPromptTemplateKey.SystemMessage,
        ProgramPromptTemplateKey.EvaluatorSystemMessage,
        ProgramPromptTemplateKey.DiffUser,
        ProgramPromptTemplateKey.FullRewriteUser,
        ProgramPromptTemplateKey.EvolutionHistory,
        ProgramPromptTemplateKey.PreviousAttempt,
        ProgramPromptTemplateKey.TopProgram,
        ProgramPromptTemplateKey.InspirationsSection,
        ProgramPromptTemplateKey.InspirationProgram,
        ProgramPromptTemplateKey.Evaluation,
        ProgramPromptTemplateKey.SystemMessageChangesDescription,
        ProgramPromptTemplateKey.SystemMessageWithChangesDescription,
        ProgramPromptTemplateKey.UserMessageWithChangesDescription
    };

    private static readonly IReadOnlyList<ProgramPromptFragmentKey> AllFragmentKeys = new[]
    {
        ProgramPromptFragmentKey.FitnessImproved,
        ProgramPromptFragmentKey.FitnessDeclined,
        ProgramPromptFragmentKey.FitnessStable,
        ProgramPromptFragmentKey.ExploringRegion,
        ProgramPromptFragmentKey.NoFeatureCoordinates,
        ProgramPromptFragmentKey.CodeTooLong,
        ProgramPromptFragmentKey.NoSpecificGuidance,
        ProgramPromptFragmentKey.CoverageHint,
        ProgramPromptFragmentKey.AttemptUnknownChanges,
        ProgramPromptFragmentKey.AttemptAllMetricsImproved,
        ProgramPromptFragmentKey.AttemptAllMetricsRegressed,
        ProgramPromptFragmentKey.AttemptMixedMetrics,
        ProgramPromptFragmentKey.TopProgramMetricsPrefix,
        ProgramPromptFragmentKey.DiverseProgramsTitle,
        ProgramPromptFragmentKey.DiverseProgramMetricsPrefix,
        ProgramPromptFragmentKey.InspirationTypeDiverse,
        ProgramPromptFragmentKey.InspirationTypeMigrant,
        ProgramPromptFragmentKey.InspirationTypeRandom,
        ProgramPromptFragmentKey.InspirationTypeHighPerformer,
        ProgramPromptFragmentKey.InspirationTypeAlternative,
        ProgramPromptFragmentKey.InspirationTypeExperimental,
        ProgramPromptFragmentKey.InspirationTypeExploratory,
        ProgramPromptFragmentKey.InspirationChangesPrefix,
        ProgramPromptFragmentKey.InspirationMetricsExcellent,
        ProgramPromptFragmentKey.InspirationMetricsAlternative,
        ProgramPromptFragmentKey.InspirationNoFeatures,
        ProgramPromptFragmentKey.ArtifactTitle,
        ProgramPromptFragmentKey.ArtifactTruncated,
        ProgramPromptFragmentKey.InspirationConciseImplementation,
        ProgramPromptFragmentKey.InspirationComprehensiveImplementation
    };

    private readonly Dictionary<ProgramPromptTemplateKey, ProgramPromptTemplate> _templates;
    private readonly Dictionary<ProgramPromptFragmentKey, ProgramPromptTemplate> _fragments;

    private ProgramPromptTemplateSet(
        Dictionary<ProgramPromptTemplateKey, ProgramPromptTemplate> templates,
        Dictionary<ProgramPromptFragmentKey, ProgramPromptTemplate> fragments)
    {
        _templates = templates;
        _fragments = fragments;
        VersionHash = ComputeVersionHash(templates, fragments);
    }

    /// <summary>Gets a hash over every template and phrase in this set.</summary>
    /// <remarks>Fold it into an operator's version identity so edited wording is visible to checkpoint resume.</remarks>
    public string VersionHash { get; }

    /// <summary>Gets every template key this set resolves, in a stable order.</summary>
    public static IReadOnlyList<ProgramPromptTemplateKey> TemplateKeys => AllTemplateKeys;

    /// <summary>Gets every phrase key this set resolves, in a stable order.</summary>
    public static IReadOnlyList<ProgramPromptFragmentKey> FragmentKeys => AllFragmentKeys;

    /// <summary>Creates the set of texts and phrases that ship with the library.</summary>
    /// <returns>A validated set covering every key.</returns>
    public static ProgramPromptTemplateSet CreateDefault()
    {
        var templates = new Dictionary<ProgramPromptTemplateKey, ProgramPromptTemplate>();
        foreach (ProgramPromptTemplateKey key in AllTemplateKeys)
        {
            templates[key] = BuildTemplate(key, ProgramPromptDefaults.TemplateText(key));
        }

        var fragments = new Dictionary<ProgramPromptFragmentKey, ProgramPromptTemplate>();
        foreach (ProgramPromptFragmentKey key in AllFragmentKeys)
        {
            fragments[key] = BuildFragment(key, ProgramPromptDefaults.FragmentText(key));
        }

        return new ProgramPromptTemplateSet(templates, fragments);
    }

    /// <summary>Loads a directory of overrides over the shipped defaults.</summary>
    /// <param name="directory">A directory holding <c>&lt;stem&gt;.txt</c> overrides and an optional fragments file.</param>
    /// <returns>A validated set where present files replace the corresponding defaults.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="directory"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="directory"/> is empty or white space, or an override is invalid.</exception>
    /// <exception cref="DirectoryNotFoundException"><paramref name="directory"/> does not exist.</exception>
    /// <exception cref="InvalidDataException">The fragments file is not a JSON object of string values.</exception>
    public static ProgramPromptTemplateSet LoadFromDirectory(string directory) =>
        LoadFromDirectory(directory, CreateDefault());

    /// <summary>Loads a directory of overrides over an existing set.</summary>
    /// <param name="directory">A directory holding <c>&lt;stem&gt;.txt</c> overrides and an optional fragments file.</param>
    /// <param name="baseSet">The set the files are layered over.</param>
    /// <returns>A validated set where present files replace the corresponding entries of <paramref name="baseSet"/>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="directory"/> or <paramref name="baseSet"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="directory"/> is empty or white space, or an override is invalid.</exception>
    /// <exception cref="DirectoryNotFoundException"><paramref name="directory"/> does not exist.</exception>
    /// <exception cref="InvalidDataException">The fragments file is not a JSON object of string values.</exception>
    public static ProgramPromptTemplateSet LoadFromDirectory(string directory, ProgramPromptTemplateSet baseSet)
    {
        Guard.NotNullOrWhiteSpace(directory);
        Guard.NotNull(baseSet);

        string root = Path.GetFullPath(directory);
        if (!Directory.Exists(root))
        {
            // A silently ignored missing directory is how a run ends up quietly
            // using defaults the operator believed they had replaced.
            throw new DirectoryNotFoundException($"The prompt template directory '{root}' does not exist.");
        }

        var templates = new Dictionary<ProgramPromptTemplateKey, ProgramPromptTemplate>(baseSet._templates);
        foreach (ProgramPromptTemplateKey key in AllTemplateKeys)
        {
            string path = Path.Combine(root, ProgramPromptDefaults.TemplateFileStem(key) + ".txt");
            if (!File.Exists(path)) continue;
            templates[key] = BuildTemplate(key, ReadUtf8(path));
        }

        var fragments = new Dictionary<ProgramPromptFragmentKey, ProgramPromptTemplate>(baseSet._fragments);
        string fragmentsPath = Path.Combine(root, FragmentsFileName);
        if (File.Exists(fragmentsPath)) ApplyFragmentsFile(ReadUtf8(fragmentsPath), fragmentsPath, fragments);

        return new ProgramPromptTemplateSet(templates, fragments);
    }

    /// <summary>Returns a copy of this set with one template replaced.</summary>
    /// <param name="key">The template to replace.</param>
    /// <param name="text">The replacement text.</param>
    /// <returns>A new validated set; this instance is unchanged.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="key"/> is undefined, or <paramref name="text"/> is malformed or omits a structurally
    /// required placeholder.
    /// </exception>
    public ProgramPromptTemplateSet With(ProgramPromptTemplateKey key, string text)
    {
        var templates = new Dictionary<ProgramPromptTemplateKey, ProgramPromptTemplate>(_templates)
        {
            [key] = BuildTemplate(key, text)
        };
        return new ProgramPromptTemplateSet(templates, new Dictionary<ProgramPromptFragmentKey, ProgramPromptTemplate>(_fragments));
    }

    /// <summary>Returns a copy of this set with one phrase replaced.</summary>
    /// <param name="key">The phrase to replace.</param>
    /// <param name="text">The replacement text.</param>
    /// <returns>A new validated set; this instance is unchanged.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="key"/> is undefined, or <paramref name="text"/> is malformed, too long, omits a required
    /// argument, or asks for an argument the phrase is never given.
    /// </exception>
    public ProgramPromptTemplateSet WithFragment(ProgramPromptFragmentKey key, string text)
    {
        var fragments = new Dictionary<ProgramPromptFragmentKey, ProgramPromptTemplate>(_fragments)
        {
            [key] = BuildFragment(key, text)
        };
        return new ProgramPromptTemplateSet(new Dictionary<ProgramPromptTemplateKey, ProgramPromptTemplate>(_templates), fragments);
    }

    /// <summary>Gets the template registered for a key.</summary>
    /// <param name="key">The template key.</param>
    /// <returns>The parsed template.</returns>
    /// <exception cref="ArgumentException"><paramref name="key"/> is not a defined key.</exception>
    public ProgramPromptTemplate GetTemplate(ProgramPromptTemplateKey key)
    {
        if (!_templates.TryGetValue(key, out ProgramPromptTemplate? template))
        {
            throw new ArgumentException($"'{key}' is not a defined prompt template key.", nameof(key));
        }

        return template;
    }

    /// <summary>Gets the phrase registered for a key.</summary>
    /// <param name="key">The phrase key.</param>
    /// <returns>The parsed phrase.</returns>
    /// <exception cref="ArgumentException"><paramref name="key"/> is not a defined key.</exception>
    public ProgramPromptTemplate GetFragment(ProgramPromptFragmentKey key)
    {
        if (!_fragments.TryGetValue(key, out ProgramPromptTemplate? fragment))
        {
            throw new ArgumentException($"'{key}' is not a defined prompt fragment key.", nameof(key));
        }

        return fragment;
    }

    /// <summary>Renders a phrase that takes no arguments.</summary>
    /// <param name="key">The phrase key.</param>
    /// <returns>The phrase text.</returns>
    /// <exception cref="ArgumentException"><paramref name="key"/> is not a defined key or the phrase takes arguments.</exception>
    public string RenderFragment(ProgramPromptFragmentKey key)
    {
        ProgramPromptTemplate fragment = GetFragment(key);
        if (!fragment.IsConstant)
        {
            throw new ArgumentException(
                $"The prompt fragment '{key}' takes arguments; call the overload that supplies them.", nameof(key));
        }

        return fragment.Text;
    }

    /// <summary>Renders a phrase with the arguments it declares.</summary>
    /// <param name="key">The phrase key.</param>
    /// <param name="values">A value for each argument the phrase names.</param>
    /// <returns>The rendered phrase.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="key"/> is not a defined key.</exception>
    /// <exception cref="KeyNotFoundException">An argument the phrase names has no entry in <paramref name="values"/>.</exception>
    public string RenderFragment(ProgramPromptFragmentKey key, IReadOnlyDictionary<string, string> values) =>
        GetFragment(key).Render(values);

    /// <summary>Gets the argument names a phrase is allowed to use.</summary>
    /// <param name="key">The phrase key.</param>
    /// <returns>The declared argument names, which may be empty.</returns>
    /// <exception cref="ArgumentException"><paramref name="key"/> is not a defined key.</exception>
    public static IReadOnlyList<string> DeclaredFragmentArguments(ProgramPromptFragmentKey key) =>
        ProgramPromptDefaults.FragmentArguments(key);

    /// <summary>Gets the placeholder names the prompt builder supplies to a template.</summary>
    /// <param name="key">The template key.</param>
    /// <returns>
    /// The names always available to that template. A template may also use names declared as custom variables or
    /// as template variations; the prompt builder checks those when it is constructed.
    /// </returns>
    /// <exception cref="ArgumentException"><paramref name="key"/> is not a defined key.</exception>
    public static IReadOnlyList<string> SuppliedPlaceholders(ProgramPromptTemplateKey key) =>
        ProgramPromptDefaults.SuppliedPlaceholders(key);

    /// <summary>Gets the placeholders a template must keep to remain structurally usable.</summary>
    /// <param name="key">The template key.</param>
    /// <returns>The required placeholder names, which may be empty.</returns>
    /// <exception cref="ArgumentException"><paramref name="key"/> is not a defined key.</exception>
    public static IReadOnlyList<string> RequiredPlaceholders(ProgramPromptTemplateKey key) =>
        ProgramPromptDefaults.RequiredPlaceholders(key);

    /// <summary>Gets the file stem an override for a template key is read from.</summary>
    /// <param name="key">The template key.</param>
    /// <returns>The file stem, without the <c>.txt</c> extension.</returns>
    /// <exception cref="ArgumentException"><paramref name="key"/> is not a defined key.</exception>
    public static string TemplateFileStem(ProgramPromptTemplateKey key) => ProgramPromptDefaults.TemplateFileStem(key);

    /// <summary>Gets the name a phrase override is read under inside the fragments file.</summary>
    /// <param name="key">The phrase key.</param>
    /// <returns>The fragment name.</returns>
    /// <exception cref="ArgumentException"><paramref name="key"/> is not a defined key.</exception>
    public static string FragmentName(ProgramPromptFragmentKey key) => ProgramPromptDefaults.FragmentName(key);

    /// <summary>Returns a short description that never echoes template text.</summary>
    /// <returns>The template and phrase counts and the version hash.</returns>
    public override string ToString() =>
        $"ProgramPromptTemplateSet(templates={_templates.Count}, fragments={_fragments.Count}, hash={VersionHash.Substring(0, 12)})";

    private static ProgramPromptTemplate BuildTemplate(ProgramPromptTemplateKey key, string text)
    {
        Guard.NotNull(text);
        IReadOnlyList<string> required = ProgramPromptDefaults.RequiredPlaceholders(key);
        var template = new ProgramPromptTemplate(text);
        foreach (string name in required)
        {
            if (template.ContainsPlaceholder(name)) continue;
            throw new ArgumentException(
                $"The prompt template '{key}' must keep the '{{{name}}}' placeholder; without it the section it " +
                "wraps would never reach the model.",
                nameof(text));
        }

        return template;
    }

    private static ProgramPromptTemplate BuildFragment(ProgramPromptFragmentKey key, string text)
    {
        Guard.NotNull(text);
        if (text.Length > MaxFragmentLength)
        {
            throw new ArgumentException(
                $"The prompt fragment '{key}' cannot exceed {MaxFragmentLength} characters.", nameof(text));
        }

        IReadOnlyList<string> declared = ProgramPromptDefaults.FragmentArguments(key);
        var fragment = new ProgramPromptTemplate(text);

        foreach (string placeholder in fragment.Placeholders)
        {
            if (Contains(declared, placeholder)) continue;
            throw new ArgumentException(
                $"The prompt fragment '{key}' asks for '{{{placeholder}}}', which it is never given. " +
                $"Allowed arguments: {Describe(declared)}.",
                nameof(text));
        }

        foreach (string name in declared)
        {
            if (fragment.ContainsPlaceholder(name)) continue;
            // Upstream ships a fragment formatted with a {changes} argument whose
            // text has no such slot, so the description is silently discarded from
            // every prompt. Requiring declared arguments makes that loud.
            throw new ArgumentException(
                $"The prompt fragment '{key}' must use its '{{{name}}}' argument; a fragment that drops an " +
                "argument silently discards the value it was given.",
                nameof(text));
        }

        return fragment;
    }

    private static void ApplyFragmentsFile(
        string json,
        string path,
        Dictionary<ProgramPromptFragmentKey, ProgramPromptTemplate> fragments)
    {
        JObject document;
        try
        {
            JToken token = JToken.Parse(json);
            if (token is not JObject parsed)
            {
                throw new InvalidDataException($"The prompt fragments file '{path}' must contain a JSON object.");
            }

            document = parsed;
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException($"The prompt fragments file '{path}' is not valid JSON.", exception);
        }

        foreach (ProgramPromptFragmentKey key in AllFragmentKeys)
        {
            JToken? value = document[ProgramPromptDefaults.FragmentName(key)];
            if (value is null) continue;
            if (value.Type != JTokenType.String)
            {
                throw new InvalidDataException(
                    $"The prompt fragment '{ProgramPromptDefaults.FragmentName(key)}' in '{path}' must be a string.");
            }

            fragments[key] = BuildFragment(key, value.Value<string>() ?? string.Empty);
        }
    }

    private static string ReadUtf8(string path)
    {
        // Explicit UTF-8 rather than the platform default: upstream reads template
        // files in whatever the machine's code page happens to be, which turns every
        // non-ASCII character of a UTF-8 template into mojibake on Windows.
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var reader = new StreamReader(stream, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false), detectEncodingFromByteOrderMarks: true);
        return reader.ReadToEnd();
    }

    private static string ComputeVersionHash(
        Dictionary<ProgramPromptTemplateKey, ProgramPromptTemplate> templates,
        Dictionary<ProgramPromptFragmentKey, ProgramPromptTemplate> fragments)
    {
        var components = new List<string> { "program-prompt-template-set-v1" };
        foreach (ProgramPromptTemplateKey key in AllTemplateKeys)
        {
            components.Add(ProgramPromptDefaults.TemplateFileStem(key));
            components.Add(templates[key].Text);
        }

        foreach (ProgramPromptFragmentKey key in AllFragmentKeys)
        {
            components.Add(ProgramPromptDefaults.FragmentName(key));
            components.Add(fragments[key].Text);
        }

        return EvolutionHash.Combine(components);
    }

    private static bool Contains(IReadOnlyList<string> names, string value)
    {
        for (int index = 0; index < names.Count; index++)
        {
            if (string.Equals(names[index], value, StringComparison.Ordinal)) return true;
        }

        return false;
    }

    private static string Describe(IReadOnlyList<string> names) =>
        names.Count == 0 ? "(none)" : string.Join(", ", names);
}
