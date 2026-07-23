using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// Configuration options for the data2vec 2.0 self-supervised audio foundation model.
/// </summary>
/// <remarks>
/// <para>
/// data2vec 2.0 (Baevski et al., 2023, Meta) is a self-supervised learning framework that
/// predicts contextualized latent representations rather than modality-specific targets.
/// Version 2.0 is 16x faster than v1 through efficient data encoding and a novel self-distillation
/// objective. It achieves strong results on speech, vision, and language tasks with the same method.
/// </para>
/// <para>
/// <b>For Beginners:</b> data2vec 2.0 is a foundation model that learns by predicting its own
/// hidden representations - like studying by explaining things to yourself. Unlike HuBERT which
/// predicts discrete labels, data2vec predicts rich continuous features. Version 2.0 is much
/// faster to train while maintaining quality. It works for audio, images, and text.
/// </para>
/// </remarks>
public class Data2Vec2Options : ModelOptions
{
    #region Audio Settings

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model hidden dimension.</summary>
    public int HiddenDim { get; set; } = 768;

    /// <summary>Gets or sets the number of transformer layers.</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>Gets or sets the feed-forward inner dimension.</summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>Gets or sets the model variant ("base" or "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the convolutional feature extractor channels.</summary>
    public int[] ConvChannels { get; set; } = [512, 512, 512, 512, 512, 512, 512];

    /// <summary>Gets or sets the convolutional feature extractor kernel sizes.</summary>
    public int[] ConvKernels { get; set; } = [10, 3, 3, 3, 3, 2, 2];

    #endregion

    #region Self-Distillation

    /// <summary>Gets or sets the teacher EMA decay rate.</summary>
    public double EMADecay { get; set; } = 0.999;

    /// <summary>Gets or sets the masking probability for training.</summary>
    public double MaskProbability { get; set; } = 0.65;

    /// <summary>Gets or sets the mask span length.</summary>
    public int MaskSpanLength { get; set; } = 10;

    /// <summary>Gets or sets the number of top-k layers averaged for teacher targets.</summary>
    public int TopKLayers { get; set; } = 8;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
