using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the enwik8 character-level LM data loader.
/// </summary>
/// <remarks>
/// <para>
/// enwik8 is the standard character-level Wikipedia language modeling benchmark
/// (Hutter Prize): the first 100M bytes of an English Wikipedia XML dump.
/// Models are evaluated in bits-per-character (BPC); SOTA values typically
/// sit in the 0.95–1.10 range. Canonical split: first 90M chars train,
/// next 5M val, last 5M test.
/// </para>
/// </remarks>
public sealed class Enwik8DataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Sequence length (characters per sample). Default is 512.</summary>
    public int SequenceLength { get; set; } = 512;
    /// <summary>
    /// Vocabulary size cap. enwik8 contains ~205 unique bytes; the default
    /// 256 covers the full byte alphabet. Set lower to clip rare bytes.
    /// </summary>
    public int VocabularySize { get; set; } = 256;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
