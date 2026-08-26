using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Streaming;

/// <summary>Options for Kyutai Moshi: full-duplex spoken dialogue model.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the KyutaiMoshi model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class KyutaiMoshiOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public KyutaiMoshiOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KyutaiMoshiOptions(KyutaiMoshiOptions other)
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
        WeightDecay = other.WeightDecay;
        WarmupFraction = other.WarmupFraction;
        TotalTrainingSteps = other.TotalTrainingSteps;
        MaxGradientNorm = other.MaxGradientNorm;
        Language = other.Language;
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 24;
    public int NumAttentionHeads { get; set; } = 8;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 5000;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the maximum fine-tuning learning rate.</summary>
    /// <value>Defaults to 2e-6.</value>
    /// <remarks>This is the recommended starting value in Kyutai's official Moshi fine-tuning configuration.</remarks>
    public double LearningRate { get; set; } = 2e-6;

    /// <summary>Gets or sets AdamW's decoupled weight decay.</summary>
    /// <value>Defaults to 0.1.</value>
    /// <remarks>This matches the official Moshi fine-tuning configuration.</remarks>
    public double WeightDecay { get; set; } = 0.1;

    /// <summary>Gets or sets the fraction of the one-cycle schedule used for warmup.</summary>
    /// <value>Defaults to 0.05.</value>
    /// <remarks>Kyutai's official recipe reaches peak learning rate after five percent of the total steps.</remarks>
    public double WarmupFraction { get; set; } = 0.05;

    /// <summary>Gets or sets the total number of one-cycle scheduler steps.</summary>
    /// <value>Defaults to 2,000.</value>
    /// <remarks>This matches the official quick-training Moshi configuration.</remarks>
    public int TotalTrainingSteps { get; set; } = 2000;

    /// <summary>Gets or sets the global gradient-norm limit.</summary>
    /// <value>Defaults to 1.0.</value>
    /// <remarks>The official trainer clips the trainable parameter norm before every AdamW step.</remarks>
    public double MaxGradientNorm { get; set; } = 1.0;

    public string Language { get; set; } = "en";
}
