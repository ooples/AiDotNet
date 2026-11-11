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
        // Derived options restore their own native stride arrays after this base constructor.
        // Copy the shared slot without dispatching a derived setter against not-yet-restored geometry.
        _hopSize = other.HopSize;
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
    [AiDotNet.Attributes.OptionBehaviourExempt("The released OpenVoice converter config fixes audio at 22050 Hz; the native constructor rejects other rates (myshell-ai/OpenVoice checkpoints/converter/config.json).", ModelType = typeof(VoiceCloning.OpenVoice<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("The released OpenVoiceV2 converter config fixes audio at 22050 Hz; the native constructor rejects other rates (myshell-ai/OpenVoiceV2 converter/config.json).", ModelType = typeof(VoiceCloning.OpenVoiceV2<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Dia's audio is its Descript Audio Codec's; the native constructor rejects any other rate (nari-labs/dia config.json).", ModelType = typeof(CodecBased.Dia<>))]
    public virtual int SampleRate { get; set; } = 22050;

    /// <summary>Gets or sets the number of mel-spectrogram frequency channels.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro predicts waveform spectra directly with ISTFTNet and has no mel-spectrogram representation.", ModelType = typeof(EndToEnd.Kokoro<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("E3 TTS sections 3.2-3.3 predict the raw waveform with a BERT-conditioned U-Net and have no mel representation (arXiv:2311.00945).", ModelType = typeof(FlowDiffusion.E3TTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Moshi uses Mimi's waveform SEANet/RVQ, not mel channels (Defossez et al. 2024 section 3.3); native construction requires MelChannels=0.", ModelType = typeof(MultiModal.Moshi<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Moshi ASR uses Mimi waveform codes, not mel channels; native construction requires MelChannels=0 (Defossez et al. 2024 section 3.3).", ModelType = typeof(AiDotNet.SpeechRecognition.Streaming.Moshi<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("NaturalSpeech 2 consumes waveform codec latents (Shen et al. 2023, §3.1). Its constructors reject mel widths other than zero and the inherited legacy default of 80.", ModelType = typeof(CodecBased.NaturalSpeech2<>))]
    public virtual int MelChannels { get; set; } = 80;

    /// <summary>Gets or sets the hop size for mel-spectrogram computation.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro has no mel frontend; its released ISTFTNet upsampling and inverse-STFT hop geometry are fixed by the decoder.", ModelType = typeof(EndToEnd.Kokoro<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("OpenVoice's released HiFi-GAN strides [8,8,2,2] fix its hop at 256; the native constructor rejects other hops (myshell-ai/OpenVoice checkpoints/converter/config.json).", ModelType = typeof(VoiceCloning.OpenVoice<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("OpenVoiceV2's released HiFi-GAN strides [8,8,2,2] fix its hop at 256; the native constructor rejects other hops (myshell-ai/OpenVoiceV2 converter/config.json).", ModelType = typeof(VoiceCloning.OpenVoiceV2<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Moshi has no mel-analysis hop; Mimi's actual codec strides are MimiRatios (Defossez et al. 2024 section 3.3); native construction requires HopSize=0.", ModelType = typeof(MultiModal.Moshi<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Moshi ASR has no mel-analysis hop; Mimi's codec strides are MimiRatios; native construction requires HopSize=0 (Defossez et al. 2024 section 3.3).", ModelType = typeof(AiDotNet.SpeechRecognition.Streaming.Moshi<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("IndexTTS2's hop is the product of its BigVGAN upsampling rates (VocoderUpsampleRates); the native constructor rejects a hop that differs from it.", ModelType = typeof(Latest.IndexTTS2<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("EmotiVoice's hop is the product of its HiFi-GAN upsampling rates (UpsampleRates); the native constructor rejects a hop that differs from it.", ModelType = typeof(StyleEmotion.EmotiVoice<>))]
    public virtual int HopSize { get => _hopSize; set => _hopSize = value; }
    private int _hopSize = 256;

    /// <summary>Gets or sets the FFT window size.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro has no configurable analysis FFT; its released ISTFTNet synthesis transform has fixed decoder geometry.", ModelType = typeof(EndToEnd.Kokoro<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Moshi has no analysis FFT; its optional Mimi STFT critics use MimiDiscriminatorWindows (Defossez et al. 2024 section 3.3); native construction requires FftSize=0.", ModelType = typeof(MultiModal.Moshi<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Moshi ASR has no analysis FFT; its native constructor requires FftSize=0 (Defossez et al. 2024 section 3.3).", ModelType = typeof(AiDotNet.SpeechRecognition.Streaming.Moshi<>))]
    public int FftSize { get; set; } = 1024;

    /// <summary>Gets or sets the model hidden dimension.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro's released style and decoder interfaces require hidden width 512; the native constructor rejects other widths.", ModelType = typeof(EndToEnd.Kokoro<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Tacotron (Wang et al. 2017) sizes its parts with TacotronOptions' own fields (EmbeddingDim, CbhgChannels, AttentionDim, DecoderRnnDim); the inherited width has no part to size.", ModelType = typeof(Classic.Tacotron<>))]
    public virtual int HiddenDim { get; set; } = 256;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Orpheus is a decoder-only Llama causal language model with no encoder; its native constructor requires NumEncoderLayers=0 (canopyai/Orpheus-TTS finetune/train.py).", ModelType = typeof(CodecBased.OrpheusTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("OuteTTS uses LlamaForCausalLM or Qwen3ForCausalLM without a separate text encoder; its native constructor requires NumEncoderLayers=0 (OuteAI/Llama-OuteTTS-1.0-1B config.json).", ModelType = typeof(Latest.OuteTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("CSM configures its native Llama layer counts through BackboneLayers and DecoderLayers.", ModelType = typeof(CodecBased.CSM<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("MinMo configures SenseVoice depth through LargeEncoderLayers, SmallEncoderBlocks and SmallEncoderTpBlocks.", ModelType = typeof(MultiModal.MinMo<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Tacotron (Wang et al. 2017) sizes its parts with TacotronOptions' own fields (PrenetSizes, EncoderBankSize, CbhgChannels); the inherited layer count has no part to size.", ModelType = typeof(Classic.Tacotron<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Parler-TTS's description encoder depth is TextEncoderLayers (Flan-T5's); its decoder depth is NumDecoderLayers. The inherited encoder count has no part to size.", ModelType = typeof(DescriptionBased.ParlerTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("SpeedySpeech's encoder is a stack of dilated residual convolutions whose depth is EncoderDilations (Vainer and Dusek 2020, section 3.2); it has no Transformer layers.", ModelType = typeof(Classic.SpeedySpeech<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Dia sizes its text encoder with DiaOptions.EncoderLayers and its decoder with DecoderLayers (nari-labs/dia config.json).", ModelType = typeof(CodecBased.Dia<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("EmotiVoice sizes its encoder with EmotiVoiceOptions.EncoderLayers (netease-youdao/EmotiVoice config).", ModelType = typeof(StyleEmotion.EmotiVoice<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("ChatTTS's GPT is a decoder-only Llama with no text encoder (2noise/ChatTTS ChatTTS/config/config.py); its native constructor requires NumEncoderLayers=0.", ModelType = typeof(CodecBased.ChatTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("SpeechGPT is a decoder-only LLaMA over text and HuBERT units (Zhang et al. 2023, section 3.2) with no text encoder; its depth is SpeechGPTOptions.Layers and the frozen HuBERT's is HubertLayer.", ModelType = typeof(MultiModal.SpeechGPT<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("LLaMA-Omni's speech encoder depth is LlamaOmniOptions.EncoderLayers, its LLM's Layers and its speech decoder's SpeechDecoderLayers (Fang et al. 2024, section 2); it has no separate text encoder.", ModelType = typeof(MultiModal.LlamaOmni<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Fish Speech is a decoder-only dual autoregressive transformer (Liao et al. 2024, section 3.1) with no text encoder; its slow transformer's depth is NumDecoderLayers and the fast one's FastLayers.", ModelType = typeof(CodecBased.FishSpeech<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Fish Speech 1.5 is a decoder-only dual autoregressive transformer (Liao et al. 2024, section 3.1) with no text encoder; its slow transformer's depth is NumDecoderLayers and the fast one's FastLayers.", ModelType = typeof(CodecBased.FishSpeechV15<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("IndexTTS's text-to-code model is a decoder-only GPT (Deng et al. 2025, section 2.2) with no text encoder; its depth is IndexTTSOptions.Layers and the conditioner's ConditionBlocks.", ModelType = typeof(CodecBased.IndexTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("IndexTTS2's text-to-semantic model is a decoder-only GPT (Zhou et al. 2025, section 3.1) with no text encoder; its depth is IndexTTS2Options.Layers.", ModelType = typeof(Latest.IndexTTS2<>))]
    public virtual int NumEncoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of decoder layers.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("OpenVoice's released HiFi-GAN decoder has four upsampling stages [8,8,2,2]; other depths are rejected by the native constructor (myshell-ai/OpenVoice checkpoints/converter/config.json).", ModelType = typeof(VoiceCloning.OpenVoice<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("OpenVoiceV2's released HiFi-GAN decoder has four upsampling stages [8,8,2,2]; other depths are rejected by the native constructor (myshell-ai/OpenVoiceV2 converter/config.json).", ModelType = typeof(VoiceCloning.OpenVoiceV2<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Tacotron's decoder is the paper's fixed attention GRU plus two residual GRUs (§3.3); the inherited layer count has no part to size.", ModelType = typeof(Classic.Tacotron<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Non-Attentive Tacotron uses two autoregressive decoder LSTMs (Shen et al. 2020, §3). Native construction accepts two and the inherited legacy value of six.", ModelType = typeof(Classic.NonAttentiveTacotron<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("SpeedySpeech's decoder is a stack of dilated residual convolutions whose depth is DecoderDilations (Vainer and Dusek 2020, section 3.2); it has no Transformer layers.", ModelType = typeof(Classic.SpeedySpeech<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("EmotiVoice sizes its decoder with EmotiVoiceOptions.DecoderLayers (netease-youdao/EmotiVoice config).", ModelType = typeof(StyleEmotion.EmotiVoice<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("ProDiff's decoder is its diffusion denoiser, a WaveNet sized by DenoiserLayers and DenoiserChannels (Huang et al. 2022, section 3.3); the FastSpeech 2 mel decoder depth does not exist in it.", ModelType = typeof(Classic.ProDiff<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("MeloTTS's waveform decoder is VITS's HiFi-GAN generator, staged by MeloTTSOptions.UpsampleRates (myshell-ai/MeloTTS configs); it has no Transformer decoder layers.", ModelType = typeof(EndToEnd.MeloTTS<>))]
    public virtual int NumDecoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("CSM configures native attention heads through BackboneHeads, DecoderHeads and MimiHeads.", ModelType = typeof(CodecBased.CSM<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Deep Voice 3 section 3.6 uses a single dot-product alignment per convolutional attention hop, not configurable multi-head attention; the native constructor requires NumHeads=1 (arXiv:1710.07654).", ModelType = typeof(Classic.DeepVoice3<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Tacotron's attention is single-head additive (Bahdanau) attention, §3.3; there are no heads.", ModelType = typeof(Classic.Tacotron<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Non-Attentive Tacotron uses Gaussian duration upsampling (Shen et al. 2020, §3.1). Native construction accepts zero and the inherited legacy value of eight.", ModelType = typeof(Classic.NonAttentiveTacotron<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("SpeedySpeech is fully convolutional and has no attention (Vainer and Dusek 2020, section 3.2).", ModelType = typeof(Classic.SpeedySpeech<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Dia sets its encoder, decoder and cross-attention heads separately (DiaOptions.EncoderHeads, DecoderHeads, CrossHeads; nari-labs/dia config.json).", ModelType = typeof(CodecBased.Dia<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("EmotiVoice's attention heads are EmotiVoiceOptions.Heads (netease-youdao/EmotiVoice config).", ModelType = typeof(StyleEmotion.EmotiVoice<>))]
    public virtual int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the phoneme/text vocabulary size.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro uses the released 178-symbol phoneme vocabulary; the native constructor rejects other vocabulary sizes.", ModelType = typeof(EndToEnd.Kokoro<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("ChatTTS derives its text vocabulary from its fixed tokenizer; audio vocabulary is configured separately by CodebookSize.", ModelType = typeof(CodecBased.ChatTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("SPEAR-TTS derives its combined vocabulary from the phoneme alphabet, special tokens and SemanticCodes; this inherited value records that total.", ModelType = typeof(CodecBased.SPEARTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Parler-TTS derives this metadata from PromptVocabulary, the option that configures its transcript embedding table.", ModelType = typeof(DescriptionBased.ParlerTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("The native embedding is sized from the NATSpeech phone symbols and reserves. Native construction accepts their count and the inherited legacy value of 256.", ModelType = typeof(Latest.MegaTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("The native embedding is sized from the NATSpeech phone symbols and reserves. Native construction accepts their count and the inherited legacy value of 256.", ModelType = typeof(Latest.MegaTTS2<>))]
    public int VocabSize { get; set; } = 256;

    /// <summary>Gets or sets the maximum input text length.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("EmotiVoice's ESPnet ScaledPositionalEncoding extends to any length, so its text has no length bound.", ModelType = typeof(StyleEmotion.EmotiVoice<>))]
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Gets or sets the maximum output audio length in mel frames.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Vevo's released AR MaxLength budget bounds codec/mel generation (Amphion 26f6883 ar_model.py); native construction rejects a separate nonzero MaxMelLength.", ModelType = typeof(CodecBased.Vevo<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Kokoro emits waveforms rather than mel frames; duration logits and Speed determine its output length.", ModelType = typeof(EndToEnd.Kokoro<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Moshi generates codec audio rather than mel frames (Defossez et al. 2024 sections 3.3-3.4); native construction requires MaxMelLength=0.", ModelType = typeof(MultiModal.Moshi<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Moshi ASR produces text from waveform codes, not mel frames; native construction requires MaxMelLength=0 (Defossez et al. 2024 section 3.3 and Appendix C).", ModelType = typeof(AiDotNet.SpeechRecognition.Streaming.Moshi<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("EmotiVoice's mel length is the duration predictor's, with positions for any length (ESPnet ScaledPositionalEncoding).", ModelType = typeof(StyleEmotion.EmotiVoice<>))]
    public int MaxMelLength { get; set; } = 2048;

    /// <summary>Gets or sets the dropout rate.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("SpeechGPT's LLaMA-7B fine-tunes without dropout (0nutation/SpeechGPT's training scripts leave LlamaConfig's attention dropout at 0); the native constructor requires DropoutRate=0; LoRA dropout is SpeechGPTOptions.LoraDropout.", ModelType = typeof(MultiModal.SpeechGPT<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("LLaMA-Omni's Llama-3.1 LLM and speech decoder train without dropout (ictnlp/LLaMA-Omni leaves the configs' attention dropout at 0); the native constructor requires DropoutRate=0.", ModelType = typeof(MultiModal.LlamaOmni<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("ChatTTS's released GPT is a Hugging Face LlamaModel whose config leaves attention_dropout at 0 (2noise/ChatTTS ChatTTS/config/config.py); its native constructor requires DropoutRate=0.", ModelType = typeof(CodecBased.ChatTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Llasa's released Llama config fixes attention_dropout=0 (HKUSTAudio/Llasa-1B cef8ce6); native construction requires DropoutRate=0, and X-codec2's encoders are frozen.", ModelType = typeof(CodecBased.Llasa<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Vevo's released AR and flow Llama configurations use zero dropout, and frozen HuBERT runs in inference mode (Amphion 26f6883); native construction requires DropoutRate=0.", ModelType = typeof(CodecBased.Vevo<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Moshi's released streaming transformer fixes attention dropout to zero (kyutai-labs/moshi e6a55d2 modules/transformer.py); construction requires DropoutRate=0. MimiQuantizerDropout controls codec quantizer dropout separately.", ModelType = typeof(MultiModal.Moshi<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Moshi ASR uses the same zero-dropout streaming transformer (kyutai-labs/moshi e6a55d2 modules/transformer.py); construction requires DropoutRate=0. MimiQuantizerDropout controls codec quantizer dropout separately.", ModelType = typeof(AiDotNet.SpeechRecognition.Streaming.Moshi<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Vocos's published ConvNeXt backbone has no dropout (Siuzdak 2024, Figure 2 and Section 3.2; gemelo-ai/vocos ConvNeXtBlock); the native constructor accepts DropoutRate=0 only.", ModelType = typeof(Vocoders.Vocos<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("WaveNet's selected gated residual architecture and the r9y9 mulaw256 reference preset use zero dropout (van den Oord et al. 2016, Figure 4 and Equation 2); the native constructor accepts DropoutRate=0 only.", ModelType = typeof(Vocoders.WaveNet<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("WaveRNN's published recurrent equations and dual-softmax architecture have no dropout (Kalchbrenner et al. 2018, Equation 2 and Figure 1); the native constructor accepts DropoutRate=0 only.", ModelType = typeof(Vocoders.WaveRNN<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("FreGrad's published frequency-aware residual network has no dropout (Nguyen et al. 2024, Figure 2 and Section 3.2; kaistmm/fregrad model.py); the native constructor accepts DropoutRate=0 only.", ModelType = typeof(Vocoders.FreGrad<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("PriorGrad's published DiffWave vocoder has no dropout (Lee et al. 2022, Section 4.2; Microsoft NeuralSpeech PriorGrad-vocoder/model.py); the native constructor accepts DropoutRate=0 only.", ModelType = typeof(Vocoders.PriorGrad<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("DiffWave's published gated residual network has no dropout (Kong et al. 2021, Figure 2 and Appendix C; lmnt-com/diffwave model.py); the native constructor accepts DropoutRate=0 only.", ModelType = typeof(Vocoders.DiffWave<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("WaveGrad's UBlock, DBlock and FiLM modules have no dropout (Chen et al. 2021, Appendix A, Figures 4-6); the native constructor accepts DropoutRate=0 only.", ModelType = typeof(Vocoders.WaveGrad<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("DiTTo uses this inherited setting only in its frozen SpeechT5 encoder, which always runs in inference mode; its trainable diffusion transformer has no dropout.", ModelType = typeof(FlowDiffusion.DiTToTTS<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("BigVGAN uses deterministic convolutional generator and discriminator blocks, without dropout.", ModelType = typeof(Vocoders.BigVGAN<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Voicebox configures dropout through AudioAttentionDropout, AudioActivationDropout, DurationAttentionDropout and DurationActivationDropout.", ModelType = typeof(CodecBased.Voicebox<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("MetaVoice configures native GPT dropout through TrainingDropout; this inherited generic TTS setting does not configure its actors.", ModelType = typeof(VoiceCloning.MetaVoice1B<>))]

    [AiDotNet.Attributes.OptionBehaviourExempt("CSM's native Llama and Mimi blocks use deterministic attention without dropout.", ModelType = typeof(CodecBased.CSM<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("UniAudio's author MegaByteModel and ResidualAttentionBlock contain no dropout (yangdongchao/UniAudio, UniAudio/model.py); native options require zero dropout.", ModelType = typeof(CodecBased.UniAudio<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("MinMo configures dropout on its native Qwen adapters through LoraDropout; its other native blocks do not use this generic setting.", ModelType = typeof(MultiModal.MinMo<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Tacotron applies dropout only in its pre-nets (PrenetDropout, 0.5, Table 1); the inherited rate has no part to act on.", ModelType = typeof(Classic.Tacotron<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("The released Step-1 decoder does not apply attention dropout (stepfun-ai/Step-Audio-TTS-3B modeling_step1.py). Native construction requires zero.", ModelType = typeof(MultiModal.StepAudio<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("SpeedySpeech's residual blocks are convolution, ReLU and batch normalization without dropout (Vainer and Dusek 2020, section 3.2).", ModelType = typeof(Classic.SpeedySpeech<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("Dia's released encoder and decoder apply no dropout (nari-labs/dia layers.py).", ModelType = typeof(CodecBased.Dia<>))]
    public virtual double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the ONNX model path.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the weight decay.</summary>
    [AiDotNet.Attributes.OptionBehaviourExempt("Tacotron trains with plain Adam and no weight decay (§4); the inherited decay has no part to act on.", ModelType = typeof(Classic.Tacotron<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("SpeedySpeech trains with Adam at its default parameters and no weight decay (Vainer and Dusek 2020, section 4.3).", ModelType = typeof(Classic.SpeedySpeech<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("EmotiVoice's released recipe trains with Adam (0.5, 0.9, 1e-9) and no weight decay.", ModelType = typeof(StyleEmotion.EmotiVoice<>))]
    [AiDotNet.Attributes.OptionBehaviourExempt("ProDiff trains with Adam (0.9, 0.98, 1e-9) on the rsqrt schedule and no weight decay (Huang et al. 2022, section 4.1).", ModelType = typeof(Classic.ProDiff<>))]
    public double WeightDecay { get; set; } = 0.01;
}
