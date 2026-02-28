using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the GLUE benchmark data loader.
/// </summary>
/// <remarks>
/// <para>
/// GLUE (General Language Understanding Evaluation) contains 9 NLU tasks for evaluating
/// language models: CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI.
/// </para>
/// </remarks>
public sealed class GlueDataLoaderOptions
{
    /// <summary>GLUE sub-task to load. Default is SST2.</summary>
    public GlueTask Task { get; set; } = GlueTask.SST2;
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Maximum sequence length in tokens. Default is 128.</summary>
    public int MaxSequenceLength { get; set; } = 128;
    /// <summary>Maximum vocabulary size. Default is 30000.</summary>
    public int VocabularySize { get; set; } = 30000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
