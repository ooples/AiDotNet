using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for the Music Structure Analyzer model.
/// </summary>
/// <remarks>
/// <para>
/// The Music Structure Analyzer segments songs into structural sections (intro, verse, chorus,
/// bridge, outro) using a neural network trained on annotated music datasets. It combines
/// self-similarity matrix features with a segmentation network.
/// </para>
/// <para>
/// <b>For Beginners:</b> This model listens to a song and identifies its sections—where the
/// verse begins, where the chorus kicks in, and where the bridge or outro happens. It's like
/// creating an automatic table of contents for a song.
/// </para>
/// </remarks>
public class MusicStructureAnalyzerOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 512;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the hidden dimension.</summary>
    public int HiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumLayers { get; set; } = 4;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Gets or sets the number of section labels.</summary>
    public int NumSections { get; set; } = 8;

    /// <summary>Gets or sets the section label names.</summary>
    public string[] SectionLabels { get; set; } = ["intro", "verse", "chorus", "bridge", "outro", "instrumental", "pre-chorus", "other"];

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    #endregion
}
