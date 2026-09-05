namespace AiDotNet.Configuration;

/// <summary>The starting points a program-evolution run begins from, expressed as plain source text.</summary>
/// <remarks>
/// <para>
/// A search has to start somewhere. For a program-evolution run that somewhere is one or more complete programs,
/// and this is the configuration-file-friendly way to supply them: every entry becomes one seed genome, in order.
/// The typed counterpart, <c>ConfigureEvolutionSeeds&lt;TGenome&gt;</c>, takes genome objects instead and is what a
/// caller with a custom genome type uses; this type exists so the same thing can be said in YAML, where a genome
/// object cannot be written down but a program can.
/// </para>
/// <para>
/// Seeds configured here are added ahead of any already listed on <c>ProgramEvolutionOptions.SeedPrograms</c>, so a
/// configuration file can extend a program's built-in seeds rather than having to restate them. Duplicates are
/// harmless: the engine recognises a repeated candidate and does not spend budget scoring it twice.
/// </para>
/// <para><b>For Beginners:</b> Put your existing, working program here — even a slow or naive one. Evolution
/// improves what it is given, so a reasonable starting program reaches a good answer far sooner than starting from
/// nothing. Several entries are useful when you already have genuinely different approaches worth keeping apart,
/// because the search will develop each of them instead of merging them into one.</para>
/// </remarks>
public sealed class EvolutionSeedOptions
{
    private IList<string>? _programSources;

    /// <summary>Gets or sets the seed program sources, one per starting candidate.</summary>
    public IList<string> ProgramSources
    {
        get => _programSources ??= new List<string>();
        set => _programSources = value;
    }

    /// <summary>Validates every entry and returns an independent copy.</summary>
    /// <returns>A validated copy whose list is not shared with this instance.</returns>
    /// <exception cref="ArgumentException">An entry is <c>null</c>, empty, or white space.</exception>
    public EvolutionSeedOptions SnapshotAndValidate()
    {
        var copy = new List<string>();
        foreach (string source in ProgramSources)
        {
            if (source is null || source.Trim().Length == 0)
            {
                throw new ArgumentException(
                    "A seed program cannot be null, empty, or white space.", nameof(ProgramSources));
            }

            copy.Add(source);
        }

        return new EvolutionSeedOptions { ProgramSources = copy };
    }
}
