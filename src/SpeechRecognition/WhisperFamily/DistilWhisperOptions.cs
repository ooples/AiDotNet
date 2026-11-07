using AiDotNet.Audio.Whisper;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.WhisperFamily;

/// <summary>Options for Distil-Whisper, built on openai-whisper's network.</summary>
/// <remarks>
/// <para>The defaults are Distil-Whisper: 32 encoder and 2 decoder layers, width 1280, 20 heads, 128 mel bins and 51866 tokens.</para>
/// <para><b>For Beginners:</b> These options set the size of the Whisper encoder and decoder, the language and how
/// transcription decodes.</para>
/// </remarks>
public class DistilWhisperOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public DistilWhisperOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DistilWhisperOptions(DistilWhisperOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumMels = other.NumMels;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        DecoderDim = other.DecoderDim;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        Language = other.Language;
        WordTimestamps = other.WordTimestamps;
        Temperatures = (double[])other.Temperatures.Clone();
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
    }

    /// <summary>Log-mel bins of the input window (80 up to large-v2, 128 from large-v3 on).</summary>
    public int NumMels { get; set; } = 128;

    /// <summary>Encoder width (<c>n_audio_state</c>).</summary>
    public int EncoderDim { get; set; } = 1280;

    /// <summary>Encoder blocks (<c>n_audio_layer</c>).</summary>
    public int NumEncoderLayers { get; set; } = 32;

    /// <summary>Decoder width (<c>n_text_state</c>).</summary>
    public int DecoderDim { get; set; } = 1280;

    /// <summary>Decoder blocks (<c>n_text_layer</c>).</summary>
    public int NumDecoderLayers { get; set; } = 2;

    /// <summary>Attention heads of every encoder and decoder block (<c>n_audio_head</c> = <c>n_text_head</c>).</summary>
    public int NumAttentionHeads { get; set; } = 20;

    /// <summary>Token vocabulary (<c>n_vocab</c>): 51,865 for the multilingual models up to large-v2, 51,866 from large-v3 on.</summary>
    public int VocabSize { get; set; } = 51866;

    /// <summary>Decoder positions (<c>n_text_ctx</c>: 448).</summary>
    public int MaxTextLength { get; set; } = 448;

    /// <summary>Language code of the audio; null detects it from the first 30 seconds.</summary>
    public string? Language { get; set; } = "en";

    /// <summary>Whether transcription times every word (openai-whisper's <c>word_timestamps</c>).</summary>
    public bool WordTimestamps { get; set; } = false;

    /// <summary>Decoding temperatures tried in turn when a window's decoding fails the compression or log-probability
    /// checks (openai-whisper's default: 0 to 1 in steps of 0.2).</summary>
    public double[] Temperatures { get; set; } = { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };

    /// <summary>Path to an exported ONNX model, when one is used.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    // openai-whisper's ModelDimensions for these options: 30-second windows of 1500 encoder positions.
    internal WhisperDimensions ToDimensions()
    {
        if (EncoderDim != DecoderDim) throw new NotSupportedException("openai-whisper's network ties the encoder and decoder widths.");
        return new WhisperDimensions(NumMels, 1500, EncoderDim, NumAttentionHeads, NumEncoderLayers, VocabSize, MaxTextLength, DecoderDim,
            NumAttentionHeads, NumDecoderLayers);
    }
}
