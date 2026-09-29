using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the TrOCR document model.
/// </summary>
public class TrOCROptions : DocumentNeuralNetworkOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    /// <remarks>
    /// <para>
    /// ImageHeight, ImageWidth, MaxSequenceLength, NumEncoderLayers, NumDecoderLayers, PatchSize
    /// and VocabSize are declared by DocumentNeuralNetworkOptions and were previously left unset,
    /// with the values living in constructor parameters instead. They carry TrOCR-Base's
    /// published values here (Li et al. 2021: 384x384 input, 16x16 patches, a 12-layer encoder
    /// over a 6-layer decoder, and the 50265-token RoBERTa vocabulary).
    /// </para>
    /// </remarks>
    public TrOCROptions()
    {
        ImageHeight = 384;
        ImageWidth = 384;
        MaxSequenceLength = 128;
        NumEncoderLayers = 12;
        NumDecoderLayers = 6;
        PatchSize = 16;
        VocabSize = 50265;
        EncoderHiddenDim = 768;
        DecoderHiddenDim = 768;
        NumEncoderHeads = 12;
        NumDecoderHeads = 12;
    }

    /// <summary>
    /// Gets or sets the width of the vision encoder. Default: 768.
    /// </summary>
    /// <remarks>
    /// <para>
    /// TrOCR sizes its encoder and decoder independently, so the single inherited HiddenDim
    /// cannot describe it and both halves are declared here.
    /// </para>
    /// </remarks>
    public int EncoderHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the width of the text decoder. Default: 768.
    /// </summary>
    public int DecoderHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads in the encoder. Default: 12.
    /// </summary>
    public int NumEncoderHeads { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads in the decoder. Default: 12.
    /// </summary>
    public int NumDecoderHeads { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for gradient descent parameter updates. Default: 0.0001.
    /// </summary>
    public double LearningRate { get; set; } = 0.0001;
}
