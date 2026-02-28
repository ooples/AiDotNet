using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Configuration options for the SuperGLUE benchmark data loader.
/// </summary>
/// <remarks>
/// <para>
/// SuperGLUE is a more challenging successor to GLUE with 8 tasks: BoolQ, CB, COPA,
/// MultiRC, ReCoRD, RTE, WiC, WSC.
/// </para>
/// </remarks>
public sealed class SuperGlueDataLoaderOptions
{
    /// <summary>SuperGLUE sub-task to load. Default is BoolQ.</summary>
    public SuperGlueTask Task { get; set; } = SuperGlueTask.BoolQ;
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Maximum sequence length in tokens. Default is 256.</summary>
    public int MaxSequenceLength { get; set; } = 256;
    /// <summary>Maximum vocabulary size. Default is 30000.</summary>
    public int VocabularySize { get; set; } = 30000;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
}
