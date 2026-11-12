using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// The CMU Pronouncing Dictionary (Carnegie Mellon University; cmusphinx/cmudict, BSD-2-Clause): ARPAbet pronunciations
/// of 126,052 English words with lexical stress, first pronunciation of each.
/// </summary>
/// <remarks>The dictionary is an embedded resource built by <c>tools/reference-data/g2p_resources.py</c> and loaded
/// once, on first use.</remarks>
internal static class CmuPronouncingDictionary
{
    private const string ResourceName = "AiDotNet.TextToSpeech.FrontEnd.cmudict.tsv.gz";
    private static readonly System.Lazy<Dictionary<string, string[]>> Entries = new(Load);

    /// <summary>The ARPAbet phones of <paramref name="word"/> (lower case), or null when the word is not listed.</summary>
    public static string[]? Lookup(string word) => Entries.Value.TryGetValue(word, out var phones) ? phones : null;

    private static Dictionary<string, string[]> Load()
    {
        var assembly = typeof(CmuPronouncingDictionary).Assembly;
        using var stream = assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidDataException($"The embedded resource {ResourceName} is missing.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, Encoding.UTF8);
        var entries = new Dictionary<string, string[]>(130_000, System.StringComparer.Ordinal);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            int tab = line.IndexOf('\t');
            if (tab <= 0) continue;
            entries[line.Substring(0, tab)] = line.Substring(tab + 1).Split(' ');
        }
        return entries;
    }
}
