using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the DocVQA (Document Visual Question Answering) data loader.
/// </summary>
/// <remarks>
/// <para>
/// DocVQA contains document images with questions and answers. Standard benchmark for
/// document understanding models combining OCR and visual reasoning.
/// </para>
/// </remarks>
public sealed class DocVqaDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Image width after resizing. Default is 224.</summary>
    public int ImageWidth { get; set; } = 224;
    /// <summary>Image height after resizing. Default is 224.</summary>
    public int ImageHeight { get; set; } = 224;
    /// <summary>Maximum question token length. Default is 64.</summary>
    public int MaxQuestionLength { get; set; } = 64;
    /// <summary>Maximum answer character length for text encoding. Default is 128.</summary>
    public int MaxAnswerLength { get; set; } = 128;
}
