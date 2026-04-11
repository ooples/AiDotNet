using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the <see cref="TinyShakespeareDataLoader{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Tiny Shakespeare is a standard character-level language modeling benchmark
/// consisting of ~1MB of Shakespeare's works, byte-tokenized as 256-entry
/// vocabulary. Used for small-scale language model validation — cheap enough
/// to train in minutes on CPU, rich enough to produce coherent Shakespeare-
/// like text when the model is working correctly.
/// </para>
/// </remarks>
public sealed class TinyShakespeareDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>
    /// Optional path to a tinyshakespeare.txt file. When null, uses a small
    /// public-domain Shakespeare excerpt bundled directly in the loader, which
    /// allows tests to run with no network or filesystem dependencies.
    /// </summary>
    public string? DataPath { get; set; }

    /// <summary>Sequence length for language modeling (context window). Default is 64.</summary>
    public int SequenceLength { get; set; } = 64;

    /// <summary>
    /// Fraction of the corpus used for training. The remainder is held out for
    /// validation. Default 0.9.
    /// </summary>
    public double TrainFraction { get; set; } = 0.9;

    /// <summary>Optional maximum number of sequences to materialize.</summary>
    public int? MaxSamples { get; set; }
}
