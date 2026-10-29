namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for <see cref="IndexTTS{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are IndexTTS 1.0 (Deng et al. 2025; <c>IndexTeam/Index-TTS</c> <c>config.yaml</c>): a 1024-wide,
/// 20-layer GPT-2 with 16 heads over 12000 BPE text tokens and 8194 mel codes (8192 DVAE codes, start 8192, stop 8193),
/// conditioned by 32 perceiver latents pooled from a 6-block, 512-wide conformer over the prompt's 100-band log mel
/// (24 kHz, n_fft 1024, hop 256); a DVAE of 8192 codes, one per 1024 samples; and BigVGAN turning the GPT's latents
/// into 24 kHz audio, upsampling ×1024 from 1536 channels with SnakeBeta, conditioned on a 512-wide ECAPA-TDNN speaker
/// embedding of the prompt. <see cref="Version1_5"/> gives IndexTTS 1.5 (1280 wide, 24 layers, 20 heads).</para>
/// <para>Generation follows the release's <c>infer</c>: beam search over 3 beams that samples its candidates (top-k 30,
/// top-p 0.8, temperature 1, repetition penalty 10), up to 600 codes a segment of at most 120 text tokens; runs of more
/// than 30 silent codes (code 52) shrink to 10. The paper and the release publish no training recipe; training follows
/// XTTS's GPT recipe, whose UnifiedVoice IndexTTS's is: 0.01 × text cross-entropy + mel cross-entropy, AdamW at 5e-6 with
/// betas (0.9, 0.96), weight decay 0.01 on weights only, the rate halved at steps 900k, 2.7M and 5.4M.</para>
/// <para><b>For Beginners:</b> These settings size IndexTTS's networks, control how it picks each piece of sound, and set
/// how it trains; the defaults reproduce the released 1.0 model.</para>
/// </remarks>
public class IndexTTSOptions : TtsModelOptions
{
    /// <summary>Creates the IndexTTS 1.0 configuration.</summary>
    public IndexTTSOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 256;
        FftSize = 1024;
        HiddenDim = 1024;
        NumHeads = 16;
        VocabSize = 12000;
        MaxTextLength = 402;
        DropoutRate = 0.1;
        LearningRate = 5e-6;
        WeightDecay = 0.01;
    }

    /// <summary>The IndexTTS 1.5 configuration (<c>IndexTeam/IndexTTS-1.5</c>): a 1280-wide, 24-layer GPT with 20 heads,
    /// up to 600 text tokens and 800 mel codes, and a BigVGAN reading 1280-wide latents.</summary>
    public static IndexTTSOptions Version1_5() => new()
    {
        HiddenDim = 1280,
        NumHeads = 20,
        Layers = 24,
        MaxTextLength = 600,
        MaxMelTokens = 800,
    };

    // ---------------------------------------------------------------- text

    /// <summary>The release's SentencePiece model (<c>bpe.model</c>), which turning text into tokens needs.</summary>
    public string? BpeModelPath { get; set; }

    /// <summary>The most text tokens a segment holds (<c>max_text_tokens_per_segment</c>, 120).</summary>
    public int SegmentTokens { get; set; } = 120;

    // ---------------------------------------------------------------- the GPT

    /// <summary>GPT-2 layers (20).</summary>
    public int Layers { get; set; } = 20;

    /// <summary>The most mel codes a sequence holds (<c>max_mel_tokens</c>, 605).</summary>
    public int MaxMelTokens { get; set; } = 605;

    /// <summary>The mel-code vocabulary: the DVAE's codes plus the start and stop codes (8194).</summary>
    public int MelCodes { get; set; } = 8194;

    /// <summary>Samples per mel code (<c>mel_length_compression</c>, 1024): BigVGAN's upsampling.</summary>
    public int SamplesPerCode { get; set; } = 1024;

    /// <summary>Perceiver latents the prompt is pooled into (32).</summary>
    public int ConditionLatents { get; set; } = 32;

    /// <summary>The conditioning conformer's width (512).</summary>
    public int ConditionDim { get; set; } = 512;

    /// <summary>The conditioning conformer's feed-forward width (2048).</summary>
    public int ConditionLinearUnits { get; set; } = 2048;

    /// <summary>The conditioning conformer's (and perceiver's) heads (8).</summary>
    public int ConditionHeads { get; set; } = 8;

    /// <summary>The conditioning conformer's blocks (6).</summary>
    public int ConditionBlocks { get; set; } = 6;

    /// <summary>The perceiver's feed-forward multiple (2).</summary>
    public int PerceiverMultiple { get; set; } = 2;

    // ---------------------------------------------------------------- generation

    /// <summary>Sample the beam candidates (<c>do_sample</c>, true); otherwise take the best.</summary>
    public bool Sample { get; set; } = true;

    /// <summary>Beams (3).</summary>
    public int Beams { get; set; } = 3;

    /// <summary>Top-k (30).</summary>
    public int TopK { get; set; } = 30;

    /// <summary>Top-p (0.8).</summary>
    public double TopP { get; set; } = 0.8;

    /// <summary>Temperature (1.0).</summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>Repetition penalty (10).</summary>
    public double RepetitionPenalty { get; set; } = 10.0;

    /// <summary>The most mel codes generated a segment (<c>max_mel_tokens</c>, 600).</summary>
    public int MaxGeneratedTokens { get; set; } = 600;

    /// <summary>The DVAE code of silence, whose long runs are shrunk (52).</summary>
    public int SilentCode { get; set; } = 52;

    /// <summary>More silent codes than this in a segment trigger the shrinking (30).</summary>
    public int MaxSilentCodes { get; set; } = 30;

    /// <summary>The silent codes kept in a row when shrinking (10).</summary>
    public int KeptSilentCodes { get; set; } = 10;

    /// <summary>The seed of generation's sampling.</summary>
    public int SamplingSeed { get; set; } = 1234;

    // ---------------------------------------------------------------- the DVAE

    /// <summary>The DVAE's first downsampling width (512; doubled per layer).</summary>
    public int DvaeHidden { get; set; } = 512;

    /// <summary>The DVAE's stride-2 layers (2).</summary>
    public int DvaeLayers { get; set; } = 2;

    /// <summary>The DVAE's residual blocks (3).</summary>
    public int DvaeResBlocks { get; set; } = 3;

    /// <summary>The DVAE's codebook width (512).</summary>
    public int DvaeCodebookDim { get; set; } = 512;

    /// <summary>The DVAE's downsampling kernel (3).</summary>
    public int DvaeKernel { get; set; } = 3;

    // ---------------------------------------------------------------- BigVGAN

    /// <summary>BigVGAN's first width (1536).</summary>
    public int VocoderInitialChannels { get; set; } = 1536;

    /// <summary>BigVGAN's upsampling rates (<c>[4, 4, 4, 4, 2, 2]</c>, ×1024).</summary>
    public int[] VocoderUpsampleRates { get; set; } = { 4, 4, 4, 4, 2, 2 };

    /// <summary>BigVGAN's upsampling kernels (<c>[8, 8, 4, 4, 4, 4]</c>).</summary>
    public int[] VocoderUpsampleKernels { get; set; } = { 8, 8, 4, 4, 4, 4 };

    /// <summary>BigVGAN's AMP block kernels (<c>[3, 7, 11]</c>).</summary>
    public int[] VocoderResBlockKernels { get; set; } = { 3, 7, 11 };

    /// <summary>BigVGAN's AMP block dilations (<c>[1, 3, 5]</c> each; the release's AMPBlock1 takes three).</summary>
    public int[][] VocoderResBlockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    /// <summary>The speaker embedding's width (<c>speaker_embedding_dim</c>, 512).</summary>
    public int SpeakerEmbeddingDim { get; set; } = 512;

    /// <summary>ECAPA-TDNN's widths per stage (<c>[512, 512, 512, 512, 1536]</c>).</summary>
    public int[] SpeakerEncoderChannels { get; set; } = { 512, 512, 512, 512, 1536 };

    /// <summary>ECAPA-TDNN's kernels per stage (<c>[5, 3, 3, 3, 1]</c>).</summary>
    public int[] SpeakerEncoderKernels { get; set; } = { 5, 3, 3, 3, 1 };

    /// <summary>ECAPA-TDNN's dilations per stage (<c>[1, 2, 3, 4, 1]</c>).</summary>
    public int[] SpeakerEncoderDilations { get; set; } = { 1, 2, 3, 4, 1 };

    /// <summary>ECAPA-TDNN's attentive pooling width (128).</summary>
    public int SpeakerEncoderAttentionChannels { get; set; } = 128;

    /// <summary>ECAPA-TDNN's squeeze-excitation width (128).</summary>
    public int SpeakerEncoderSeChannels { get; set; } = 128;

    /// <summary>ECAPA-TDNN's Res2Net scale (8).</summary>
    public int SpeakerEncoderScale { get; set; } = 8;

    // ---------------------------------------------------------------- training (XTTS's GPT recipe)

    /// <summary>The text cross-entropy's weight (XTTS's <c>gpt_loss_text_ce_weight</c>, 0.01).</summary>
    public double TextLossWeight { get; set; } = 0.01;

    /// <summary>The mel cross-entropy's weight (XTTS's <c>gpt_loss_mel_ce_weight</c>, 1.0).</summary>
    public double MelLossWeight { get; set; } = 1.0;

    /// <summary>AdamW's β1 (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>AdamW's β2 (0.96).</summary>
    public double Beta2 { get; set; } = 0.96;

    /// <summary>AdamW's ε (1e-8).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>The steps at which the rate is multiplied by <see cref="LearningRateDecay"/> (MultiStepLR,
    /// <c>[900000, 2700000, 5400000]</c>).</summary>
    public int[] LearningRateMilestones { get; set; } = { 900000, 2700000, 5400000 };

    /// <summary>The factor at each milestone (0.5).</summary>
    public double LearningRateDecay { get; set; } = 0.5;

    /// <summary>The seed of training's dropout.</summary>
    public int TrainingSeed { get; set; } = 1234;
}
