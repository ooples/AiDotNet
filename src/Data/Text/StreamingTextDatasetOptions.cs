namespace AiDotNet.Data.Text;

/// <summary>
/// Configuration options for the streaming text dataset.
/// </summary>
/// <remarks>
/// <para>
/// Streaming text datasets load text data lazily from disk, enabling training on
/// datasets larger than memory (e.g., The Pile, RedPajama, C4).
/// </para>
/// </remarks>
public sealed class StreamingTextDatasetOptions
{
    /// <summary>Root data path containing text files. Required.</summary>
    public string? DataPath { get; set; }
    /// <summary>Sequence length for each sample. Default is 2048.</summary>
    public int SequenceLength { get; set; } = 2048;
    /// <summary>Vocabulary size for token ID bounds checking. Default is 50257 (GPT-2).</summary>
    public int VocabularySize { get; set; } = 50257;
    /// <summary>File pattern to match. Default is "*.txt".</summary>
    public string FilePattern { get; set; } = "*.txt";
    /// <summary>Shuffle files before reading. Default is true.</summary>
    public bool ShuffleFiles { get; set; } = true;
    /// <summary>Random seed for shuffling. When null, uses non-deterministic seed.</summary>
    public int? Seed { get; set; }
    /// <summary>Optional maximum number of samples to produce.</summary>
    public int? MaxSamples { get; set; }
}
