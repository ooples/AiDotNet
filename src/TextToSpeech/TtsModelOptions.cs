using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// Base configuration options for text-to-speech models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the TtsModel model. Default values follow the original paper settings.</para>
/// </remarks>
public class TtsModelOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public TtsModelOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TtsModelOptions(TtsModelOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MelChannels = other.MelChannels;
        HopSize = other.HopSize;
        FftSize = other.FftSize;
        HiddenDim = other.HiddenDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        MaxMelLength = other.MaxMelLength;
        DropoutRate = other.DropoutRate;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro's released ISTFTNet renderer is fixed at 24000 Hz; the native constructor rejects other rates.", ModelType = typeof(EndToEnd.Kokoro<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Bark's released EnCodec renderer operates at 24000 Hz; BarkOptions.Validate rejects every other sample rate.", ModelType = typeof(CodecBased.BarkModel<>))]
    public int SampleRate { get; set; } = 22050;

    /// <summary>Gets or sets the number of mel-spectrogram frequency channels.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro predicts waveform spectra directly with ISTFTNet and has no mel-spectrogram representation.", ModelType = typeof(EndToEnd.Kokoro<>))]
    public virtual int MelChannels { get; set; } = 80;

    /// <summary>Gets or sets the hop size for mel-spectrogram computation.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro has no mel frontend; its released ISTFTNet upsampling and inverse-STFT hop geometry are fixed by the decoder.", ModelType = typeof(EndToEnd.Kokoro<>))]
    public int HopSize { get; set; } = 256;

    /// <summary>Gets or sets the FFT window size.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro has no configurable analysis FFT; its released ISTFTNet synthesis transform has fixed decoder geometry.", ModelType = typeof(EndToEnd.Kokoro<>))]
    public int FftSize { get; set; } = 1024;

    /// <summary>Gets or sets the model hidden dimension.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro's released style and decoder interfaces require hidden width 512; the native constructor rejects other widths.", ModelType = typeof(EndToEnd.Kokoro<>))]
    public int HiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("CSM configures its native Llama layer counts through BackboneLayers and DecoderLayers.", ModelType = typeof(CodecBased.CSM<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("MinMo configures SenseVoice depth through LargeEncoderLayers, SmallEncoderBlocks and SmallEncoderTpBlocks.", ModelType = typeof(MultiModal.MinMo<>))]
    public virtual int NumEncoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("CSM configures native attention heads through BackboneHeads, DecoderHeads and MimiHeads.", ModelType = typeof(CodecBased.CSM<>))]
    public virtual int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the phoneme/text vocabulary size.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro uses the released 178-symbol phoneme vocabulary; the native constructor rejects other vocabulary sizes.", ModelType = typeof(EndToEnd.Kokoro<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("ChatTTS derives its text vocabulary from its fixed tokenizer; audio vocabulary is configured separately by CodebookSize.", ModelType = typeof(CodecBased.ChatTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("SPEAR-TTS derives its combined vocabulary from the phoneme alphabet, special tokens and SemanticCodes; this inherited value records that total.", ModelType = typeof(CodecBased.SPEARTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Parler-TTS derives this metadata from PromptVocabulary, the option that configures its transcript embedding table.", ModelType = typeof(DescriptionBased.ParlerTTS<>))]
    public int VocabSize { get; set; } = 256;

    /// <summary>Gets or sets the maximum input text length.</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Gets or sets the maximum output audio length in mel frames.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro emits waveforms rather than mel frames; duration logits and Speed determine its output length.", ModelType = typeof(EndToEnd.Kokoro<>))]
    public int MaxMelLength { get; set; } = 2048;

    /// <summary>Gets or sets the dropout rate.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("DiTTo uses this inherited setting only in its frozen SpeechT5 encoder, which always runs in inference mode; its trainable diffusion transformer has no dropout.", ModelType = typeof(FlowDiffusion.DiTToTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("BigVGAN uses deterministic convolutional generator and discriminator blocks, without dropout.", ModelType = typeof(Vocoders.BigVGAN<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Voicebox configures dropout through AudioAttentionDropout, AudioActivationDropout, DurationAttentionDropout and DurationActivationDropout.", ModelType = typeof(CodecBased.Voicebox<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("MetaVoice configures native GPT dropout through TrainingDropout; this inherited generic TTS setting does not configure its actors.", ModelType = typeof(VoiceCloning.MetaVoice1B<>))]

    [AiDotNet.Attributes.OptionBehaviourExempt("CSM's native Llama and Mimi blocks use deterministic attention without dropout.", ModelType = typeof(CodecBased.CSM<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("MinMo configures dropout on its native Qwen adapters through LoraDropout; its other native blocks do not use this generic setting.", ModelType = typeof(MultiModal.MinMo<>))]
    public virtual double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the ONNX model path.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the weight decay.</summary>
    public double WeightDecay { get; set; } = 0.01;
}
