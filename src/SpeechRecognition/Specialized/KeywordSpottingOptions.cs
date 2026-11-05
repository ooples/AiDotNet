using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>Options for Keyword Spotting: lightweight wake-word and command detection.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the KeywordSpotting model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class KeywordSpottingOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public KeywordSpottingOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KeywordSpottingOptions(KeywordSpottingOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        LearningRate = other.LearningRate;
        LearningRateDecay = other.LearningRateDecay;
        Language = other.Language;
        Vocabulary = other.Vocabulary.ToArray();
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the number of ReLU units in each fully connected hidden layer.</summary>
    /// <value>Defaults to 128, matching the paper's small 3x128 Deep KWS network.</value>
    public int EncoderDim { get; set; } = 128;

    /// <summary>Gets or sets the number of fully connected ReLU hidden layers.</summary>
    /// <value>Defaults to 3, matching the paper's small 3x128 Deep KWS network.</value>
    public int NumEncoderLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets an attention-head count kept for source compatibility with older Conformer-based
    /// configurations. The paper's feed-forward Deep KWS network has no attention, so it is not read.
    /// </summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Gets or sets the number of log-filterbank energies per acoustic frame.</summary>
    /// <value>Defaults to 40, as specified by Chen, Parada, and Heigold (2014).</value>
    public int NumMels { get; set; } = 40;
    /// <summary>Gets or sets the number of output tokens the model predicts over.</summary>
    public int VocabSize { get; set; } = 100;
    /// <summary>Gets or sets the most tokens decoded for one utterance.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets a dropout probability kept for source compatibility; the Deep KWS layers have no dropout, so it is not read.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the initial stochastic-gradient-descent learning rate.</summary>
    /// <remarks>The paper specifies exponentially decayed asynchronous SGD but does not publish its numeric initial rate.</remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>Gets or sets the per-update exponential learning-rate multiplier.</summary>
    /// <remarks>The paper specifies exponential decay but does not publish the numeric decay coefficient.</remarks>
    public double LearningRateDecay { get; set; } = 0.99;

    /// <summary>Gets or sets the language code (for example "en") reported for a transcript when none is passed or detected.</summary>
    public string Language { get; set; } = "en";

    /// <summary>Gets or sets output label names, with index zero reserved for the non-keyword/filler label.</summary>
    public string[] Vocabulary { get; set; } = Array.Empty<string>();
}
