using System.IO;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Evolution.Programs;

/// <summary>Serializes <see cref="ProgramGenome"/> instances for portable evolution checkpoints.</summary>
/// <remarks>
/// <para>
/// The payload is a compact JSON object with a fixed property order — <c>v</c>, <c>language</c>, <c>source</c>,
/// then <c>description</c> when one is present — written with the invariant culture, so serializing the same
/// genome twice produces byte-identical text. That determinism is a contract requirement:
/// <c>EvolutionEngine&lt;TGenome&gt;</c> compares checkpointed seed payloads with ordinal string equality when
/// resuming, and a codec that reordered properties would make every resume look incompatible.
/// </para>
/// <para>
/// <see cref="VersionHash"/> is folded into the engine's compatibility hash. Change it whenever the payload shape
/// changes so older checkpoints are refused outright rather than being misread, and keep <see cref="Id"/> stable
/// for the lifetime of the format. Malformed or truncated payloads raise <see cref="InvalidDataException"/> as the
/// interface requires, never a JSON-library exception.
/// </para>
/// <para><b>For Beginners:</b> A long evolution run needs to be able to stop and start again, which means writing
/// every candidate program to disk and reading it back. This class is the save-file format for one program: it
/// turns a genome into a small line of JSON and turns that JSON back into an identical genome. You only need it
/// when you give the engine a checkpoint store; runs that never pause do not use it at all.</para>
/// </remarks>
public sealed class ProgramGenomeCodec : IEvolutionGenomeCodec<ProgramGenome>
{
    /// <summary>The payload schema version written into every serialized genome.</summary>
    public const int PayloadVersion = 1;

    /// <inheritdoc/>
    public string Id => "program-genome";

    /// <inheritdoc/>
    public string VersionHash => "program-genome-v1";

    /// <inheritdoc/>
    public string Serialize(ProgramGenome genome)
    {
        if (genome is null) throw new ArgumentNullException(nameof(genome));

        var payload = new JObject
        {
            ["v"] = PayloadVersion,
            ["language"] = genome.Language.ToString(),
            ["source"] = genome.Source
        };

        if (genome.Description is not null) payload["description"] = genome.Description;
        return payload.ToString(Formatting.None);
    }

    /// <inheritdoc/>
    public ProgramGenome Deserialize(string payload)
    {
        if (payload is null) throw new ArgumentNullException(nameof(payload));

        JObject parsed;
        try
        {
            parsed = JObject.Parse(payload);
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The program genome payload is not valid JSON.", exception);
        }

        JToken? versionToken = parsed["v"];
        if (versionToken is null || versionToken.Type != JTokenType.Integer)
            throw new InvalidDataException("The program genome payload is missing its integer version field.");
        int version = versionToken.Value<int>();
        if (version != PayloadVersion)
            throw new InvalidDataException($"Unsupported program genome payload version {version}.");

        JToken? languageToken = parsed["language"];
        string? languageName = languageToken?.Value<string>();
        if (string.IsNullOrWhiteSpace(languageName))
            throw new InvalidDataException("The program genome payload is missing its language field.");

        ProgramLanguage language;
        try
        {
            language = (ProgramLanguage)Enum.Parse(typeof(ProgramLanguage), languageName, ignoreCase: false);
        }
        catch (ArgumentException exception)
        {
            throw new InvalidDataException($"Unknown program language '{languageName}'.", exception);
        }

        JToken? sourceToken = parsed["source"];
        string? source = sourceToken?.Value<string>();
        if (source is null) throw new InvalidDataException("The program genome payload is missing its source field.");

        string? description = parsed["description"]?.Value<string>();

        try
        {
            return new ProgramGenome(source, language, description);
        }
        catch (ArgumentException exception)
        {
            throw new InvalidDataException("The program genome payload does not describe a valid genome.", exception);
        }
    }
}
