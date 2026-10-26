namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Options for <see cref="MegaTTS3{T}"/>: the release's five networks (the diffusion transformer, the duration
/// language model, the Whisper aligner, the Qwen2 grapheme-to-phoneme model and the WaveVAE), its inference settings and the
/// paper's training recipe.</summary>
/// <remarks>
/// <para>The defaults are the release's (bytedance/MegaTTS3: <c>dit.py</c>, <c>duration_lm/config.yaml</c>,
/// <c>whisper_small.py</c>, <c>g2p/config.json</c>, <c>wavvae_v3.py</c>, <c>infer_cli.py</c>) and, for training, the paper's
/// (Jiang et al. 2025, §3 and Appendix A.1).</para>
/// <para><b>For Beginners:</b> MegaTTS 3 copies a voice from a few seconds of it. These options size its parts and set how
/// it samples; the defaults load the released checkpoints.</para>
/// </remarks>
public class MegaTTS3Options : TtsModelOptions
{
    public MegaTTS3Options()
    {
        SampleRate = 24000;
        HopSize = 240;
        FftSize = 1200;
        HiddenDim = 1024;
        LearningRate = 5e-5;
        WeightDecay = 0;
        DropoutRate = 0;
    }

    // ---------------------------------------------------------------- the diffusion transformer (dit.py)

    /// <summary>Transformer layers (24).</summary>
    public int DitLayers { get; set; } = 24;

    /// <summary>Attention heads (16).</summary>
    public int DitHeads { get; set; } = 16;

    /// <summary>The WaveVAE latent's channels (32).</summary>
    public int LatentChannels { get; set; } = 32;

    /// <summary>The local condition's width (512).</summary>
    public int LocalConditionDim { get; set; } = 512;

    /// <summary>The context mask's projection width (16).</summary>
    public int ContextMaskDim { get; set; } = 16;

    /// <summary>The transformer's phone vocabulary (302; the last is the CFG mask token).</summary>
    public int DitPhoneVocabulary { get; set; } = 302;

    /// <summary>The transformer's tone vocabulary (32; the last is the CFG mask token).</summary>
    public int DitToneVocabulary { get; set; } = 32;

    /// <summary>The phone encoder's layers (6).</summary>
    public int PhoneEncoderLayers { get; set; } = 6;

    /// <summary>The phone encoder's heads (4).</summary>
    public int PhoneEncoderHeads { get; set; } = 4;

    // ---------------------------------------------------------------- the duration language model (duration_lm/config.yaml)

    /// <summary>The duration model's phone vocabulary (174: the release's 171 phones and three reserved tokens).</summary>
    public int DurationPhoneVocabulary { get; set; } = 174;

    /// <summary>The duration model's text encoder width (512).</summary>
    public int DurationEncoderWidth { get; set; } = 512;

    /// <summary>The text encoder's feed-forward width (1024).</summary>
    public int DurationEncoderFilter { get; set; } = 1024;

    /// <summary>The text encoder's heads (2).</summary>
    public int DurationEncoderHeads { get; set; } = 2;

    /// <summary>The text encoder's layers (4).</summary>
    public int DurationEncoderLayers { get; set; } = 4;

    /// <summary>The duration model's tone vocabulary (23: 20 and three reserved).</summary>
    public int DurationToneVocabulary { get; set; } = 23;

    /// <summary>The duration decoder's width (512).</summary>
    public int DurationWidth { get; set; } = 512;

    /// <summary>The duration decoder's layers (8).</summary>
    public int DurationLayers { get; set; } = 8;

    /// <summary>The duration decoder's heads (8).</summary>
    public int DurationHeads { get; set; } = 8;

    /// <summary>Duration codes (128: a phone lasts at most 127 frames).</summary>
    public int DurationCodeSize { get; set; } = 128;

    // ---------------------------------------------------------------- the aligner (whisper_small.py)

    /// <summary>The aligner's width (512).</summary>
    public int AlignerWidth { get; set; } = 512;

    /// <summary>The aligner's heads (8).</summary>
    public int AlignerHeads { get; set; } = 8;

    /// <summary>The aligner's encoder and decoder layers (6 each).</summary>
    public int AlignerLayers { get; set; } = 6;

    /// <summary>The aligner's vocabulary (6800).</summary>
    public int AlignerVocabulary { get; set; } = 6800;

    // ---------------------------------------------------------------- the grapheme-to-phoneme model (g2p/config.json, Qwen2-0.5B)

    /// <summary>The g2p model's vocabulary (168,896).</summary>
    public int G2pVocabSize { get; set; } = 168896;

    /// <summary>The g2p model's width (896).</summary>
    public int G2pHidden { get; set; } = 896;

    /// <summary>The g2p model's feed-forward width (4864).</summary>
    public int G2pIntermediate { get; set; } = 4864;

    /// <summary>The g2p model's layers (24).</summary>
    public int G2pLayers { get; set; } = 24;

    /// <summary>The g2p model's attention heads (14).</summary>
    public int G2pHeads { get; set; } = 14;

    /// <summary>The g2p model's key-value heads (2).</summary>
    public int G2pKeyValueHeads { get; set; } = 2;

    /// <summary>The g2p model's RoPE base (1,000,000).</summary>
    public double G2pRopeTheta { get; set; } = 1000000;

    /// <summary>The g2p tokenizer's <c>tokenizer.json</c> text; <see cref="MegaTTS3{T}.LoadReferenceWeights"/> reads it from
    /// the checkpoint.</summary>
    public string? G2pTokenizerJson { get; set; }

    // ---------------------------------------------------------------- the WaveVAE (wavvae_v3.py)

    /// <summary>The SEANet encoder's downsampling ratios (6, 5, 4, 4, 2: 960 samples a latent frame).</summary>
    public int[] VaeEncoderRatios { get; set; } = { 6, 5, 4, 4, 2 };

    /// <summary>The encoder's base filters (32).</summary>
    public int VaeEncoderFilters { get; set; } = 32;

    /// <summary>The encoder's output width (512).</summary>
    public int VaeEncoderDimension { get; set; } = 512;

    /// <summary>The latent upsampler's channels (320).</summary>
    public int VaeUpsamplerChannels { get; set; } = 320;

    /// <summary>The latent upsampler's factor (4).</summary>
    public int VaeLatentUpsampling { get; set; } = 4;

    /// <summary>The generator's base filters (128).</summary>
    public int VaeGeneratorFilters { get; set; } = 128;

    /// <summary>The generator's upsampling ratios (5, 4, 4, 3).</summary>
    public int[] VaeGeneratorRatios { get; set; } = { 5, 4, 4, 3 };

    // ---------------------------------------------------------------- inference (infer_cli.py)

    /// <summary>Diffusion steps (<c>--time_step</c>, 32).</summary>
    public int Steps { get; set; } = 32;

    /// <summary>The text guidance weight (<c>--p_w</c>, "intelligibility", 1.6).</summary>
    public double TextGuidance { get; set; } = 1.6;

    /// <summary>The speaker guidance weight (<c>--t_w</c>, "similarity", 2.5).</summary>
    public double SpeakerGuidance { get; set; } = 2.5;

    /// <summary>The random disturbance of predicted durations (<c>dur_disturb</c>, 0.1).</summary>
    public double DurationDisturbance { get; set; } = 0.1;

    /// <summary>The scale of predicted durations (<c>dur_alpha</c>, 1.0).</summary>
    public double DurationScale { get; set; } = 1.0;

    /// <summary>The seed of the duration disturbance, the diffusion noise, the posterior draw and language detection.</summary>
    public int SamplingSeed { get; set; } = 0;

    /// <summary>English chunks of at most this many UTF-8 bytes (130).</summary>
    public int EnglishChunkBytes { get; set; } = 130;

    /// <summary>Chinese chunks of about this many ideographs (60).</summary>
    public int ChineseChunkLimit { get; set; } = 60;

    /// <summary>The g2p model's most generated tokens (256).</summary>
    public int G2pMaxNewTokens { get; set; } = 256;

    /// <summary>The aligner's most decoded tokens (768).</summary>
    public int AlignerMaxTokens { get; set; } = 768;

    // ---------------------------------------------------------------- training (Jiang et al. 2025, §3, Appendix A.1)

    /// <summary>Adam's β1 (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Adam's β2 (0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>Linear warm-up steps from a rate of 0 (10,000), then the constant rate; 0 trains at the rate from the
    /// first step.</summary>
    public int WarmupSteps { get; set; } = 10000;

    /// <summary>The prompt's share of a training latent, drawn from U(min, max) (0.1, 0.9).</summary>
    public double PromptFractionMin { get; set; } = 0.1;

    /// <summary>See <see cref="PromptFractionMin"/>.</summary>
    public double PromptFractionMax { get; set; } = 0.9;

    /// <summary>The probability of dropping the speaker prompt (p_spk, 0.10).</summary>
    public double SpeakerDropProbability { get; set; } = 0.10;

    /// <summary>The probability of also dropping the text when the prompt is dropped (50 %).</summary>
    public double TextDropProbability { get; set; } = 0.5;

    /// <summary>Whether the transformer reads one random anchor frame per phone in training (the paper's sparse alignment);
    /// false reads every frame, as inference does.</summary>
    public bool SparseAlignment { get; set; } = true;

    /// <summary>The seed of training's draws.</summary>
    public int TrainingSeed { get; set; } = 0;
}
