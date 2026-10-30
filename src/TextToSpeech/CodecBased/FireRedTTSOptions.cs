namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which of FireRedTTS's networks a training step updates.</summary>
public enum FireRedTtsStage
{
    /// <summary>The text-to-speech language model (§3.2), on a recording's semantic tokens and speaker embedding.</summary>
    TextToSpeech,

    /// <summary>The flow-matching Mel decoder (§3.3.1), on a recording's tokens and 16 kHz mel, with the timbre from another
    /// recording of the speaker.</summary>
    FlowDecoder,

    /// <summary>The semantic-aware speech tokenizer (§3.1, eq. 1) with the acoustic encoder: the encoder, product quantizer
    /// and the SSL and mel decoders; HuBERT stays frozen.</summary>
    SpeechTokenizer,

    /// <summary>The streamable decoder's Mel codec (§3.3.2), against its mel discriminator in alternate turns.</summary>
    MelCodec,

    /// <summary>The streamable decoder's multi-stream acoustic language model (§3.3.2), on a recording's semantic tokens
    /// and its Mel codec streams.</summary>
    AcousticLanguageModel,
}

/// <summary>Which of the paper's Mel decoders turns semantic tokens into the mel (§3.3).</summary>
public enum FireRedTtsMelDecoder
{
    /// <summary>The flow-matching decoder (§3.3.1), the release's: higher quality, not streamable.</summary>
    FlowMatching,

    /// <summary>The streamable decoder (§3.3.2): a multi-stream acoustic LM generates a causal Mel codec's four streams.
    /// No weights were released; it trains from scratch.</summary>
    Streamable,
}

/// <summary>Which BigVGAN-v2 vocoder turns the 16 kHz, 10 ms mel into audio.</summary>
public enum FireRedTtsVocoder
{
    /// <summary>The release's: 24 kHz, the mel upsampled ×240 (rates 5·3·2·2·2·2).</summary>
    Release24k,

    /// <summary>The paper's super-resolution vocoder (§3.3.3): 48 kHz, ×480. The paper gives no upsampling rates; these are
    /// 5·4·3·2·2·2 with kernels twice the rate (11 for 5, as the release's first). No weights exist; it trains from scratch.</summary>
    Paper48k,
}

/// <summary>An emotion of FireRedTTS's instruction tuning (§4.2).</summary>
public enum FireRedTtsEmotion
{
    Neutral,
    Happy,
    Sad,
    Angry,
}

/// <summary>Options for <see cref="FireRedTTS{T}"/>: the release's sizes (FireRedTeam/FireRedTTS <c>config_24k.json</c>),
/// decoding as its <c>synthesize</c> runs it, and the training recipes.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> the defaults build the released model; load its weights with
/// <see cref="FireRedTTS{T}.LoadReferenceWeights"/>. Smaller values make a model to train from scratch.</para>
/// </remarks>
public class FireRedTTSOptions : TtsModelOptions
{
    public FireRedTTSOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 960;
        FftSize = 1024;
        HiddenDim = 1024;
        NumHeads = 16;
        NumDecoderLayers = 30;
        VocabSize = 57341;
        MaxTextLength = 400;
        DropoutRate = 0.1;
        LearningRate = 5e-6;
        WeightDecay = 0.01;
    }

    // ---------------------------------------------------------------- the language model

    /// <summary>The most text tokens the GPT's positions hold (<c>gpt_max_text_tokens</c>).</summary>
    public int MaxTextTokens { get; set; } = 402;

    /// <summary>The most semantic tokens the GPT's positions hold (<c>gpt_max_audio_tokens</c>).</summary>
    public int MaxAudioTokens { get; set; } = 630;

    /// <summary>The audio vocabulary: the tokenizer's 16,384 tokens plus the start and stop tokens.</summary>
    public int AudioTokens { get; set; } = 16386;

    /// <summary>The text start token (<c>[START]</c>).</summary>
    public int StartTextToken { get; set; } = 57187;

    /// <summary>The text stop token (<c>[STOP]</c>).</summary>
    public int StopTextToken { get; set; } = 57184;

    /// <summary>The speaker embedding's projection's hidden width (<c>reference_embedding</c>).</summary>
    public int ReferenceHidden { get; set; } = 256;

    // ---------------------------------------------------------------- the speaker (acoustic) encoder

    /// <summary>The ECAPA-TDNN's channels per stage (the last aggregates).</summary>
    public int[] SpeakerChannels { get; set; } = { 512, 512, 512, 512, 1536 };
    public int[] SpeakerKernels { get; set; } = { 5, 3, 3, 3, 1 };
    public int[] SpeakerDilations { get; set; } = { 1, 2, 3, 4, 1 };
    public int SpeakerAttentionChannels { get; set; } = 128;
    public int SpeakerSeChannels { get; set; } = 128;
    public int SpeakerRes2NetScale { get; set; } = 4;

    /// <summary>The speaker embedding's width (also the tokenizer's model width, whose decoder adds it).</summary>
    public int SpeakerDim { get; set; } = 512;

    // ---------------------------------------------------------------- the flow-matching Mel decoder

    /// <summary>The speech tokenizer's codewords per sub-codebook (two sub-codebooks make 16,384 tokens).</summary>
    public int CodebookSize { get; set; } = 128;
    public int CodebookDim { get; set; } = 128;
    public int FlowDim { get; set; } = 512;
    public int FlowHeads { get; set; } = 8;
    public int FlowLinearUnits { get; set; } = 2048;
    public int FlowBlocks { get; set; } = 6;

    /// <summary>The encoder layers with cross-attention to the prompt (<c>srcattention_end_index</c> + 1).</summary>
    public int FlowCrossAttentionBlocks { get; set; } = 3;
    public double FlowDropout { get; set; } = 0.01;
    public int FlowRegulatorBlocks { get; set; } = 4;
    public int PromptEncoderHidden { get; set; } = 384;
    public int PromptEncoderReduction { get; set; } = 4;
    public int PromptEncoderLayers { get; set; } = 2;
    public int PromptEncoderBlocks { get; set; } = 5;
    public int PromptEncoderKernel { get; set; } = 3;
    public int[] FlowDecoderChannels { get; set; } = { 256, 256 };
    public int FlowDecoderHeads { get; set; } = 8;
    public int FlowDecoderHeadDim { get; set; } = 64;
    public int FlowDecoderBlocksPerLevel { get; set; } = 4;
    public int FlowDecoderMidBlocks { get; set; } = 12;

    /// <summary>Classifier-free guidance at inference (eq. 7: α = 0.7).</summary>
    public double InferenceCfgRate { get; set; } = 0.7;

    /// <summary>The probability training drops μ (§3.3.1: 20 %).</summary>
    public double TrainingCfgRate { get; set; } = 0.2;

    /// <summary>The Euler steps of the flow (the release's <c>n_timesteps=10</c>).</summary>
    public int FlowSteps { get; set; } = 10;

    // ---------------------------------------------------------------- the vocoder

    /// <summary>The vocoder: the release's 24 kHz or the paper's 48 kHz. Setting it also sets <see cref="TtsModelOptions.SampleRate"/>,
    /// <see cref="TtsModelOptions.HopSize"/> and the upsampling.</summary>
    public FireRedTtsVocoder Vocoder
    {
        get => _vocoder;
        set
        {
            _vocoder = value;
            if (value == FireRedTtsVocoder.Paper48k)
            {
                SampleRate = 48000;
                HopSize = 1920;
                VocoderUpsampleRates = new[] { 5, 4, 3, 2, 2, 2 };
                VocoderUpsampleKernels = new[] { 11, 8, 6, 4, 4, 4 };
            }
            else
            {
                SampleRate = 24000;
                HopSize = 960;
                VocoderUpsampleRates = new[] { 5, 3, 2, 2, 2, 2 };
                VocoderUpsampleKernels = new[] { 11, 7, 4, 4, 4, 4 };
            }
        }
    }

    private FireRedTtsVocoder _vocoder = FireRedTtsVocoder.Release24k;

    public int VocoderInitialChannels { get; set; } = 1536;
    public int[] VocoderUpsampleRates { get; set; } = { 5, 3, 2, 2, 2, 2 };
    public int[] VocoderUpsampleKernels { get; set; } = { 11, 7, 4, 4, 4, 4 };
    public int[] VocoderResBlockKernels { get; set; } = { 3, 7, 11 };
    public int[][] VocoderResBlockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    // ---------------------------------------------------------------- the speech tokenizer

    /// <summary>The HuBERT features' width (chinese-hubert-large: 1024).</summary>
    public int SslDim { get; set; } = 1024;

    /// <summary>The tokenizer encoder's width (<c>n_model_size</c>).</summary>
    public int TokenizerDim { get; set; } = 512;
    public int[] TokenizerDownsampleStrides { get; set; } = { 1, 1, 1, 2 };

    /// <summary>The SSL decoder's and the mel decoder's upsampling strides (the release config's <c>upsample_scales</c>).</summary>
    public int[][] TokenizerUpsampleStrides { get; set; } = { new[] { 2, 1 }, new[] { 2, 1, 1, 1 } };
    public int HubertLayers { get; set; } = 24;
    public int HubertDim { get; set; } = 1024;
    public int HubertHeads { get; set; } = 16;
    public int HubertIntermediate { get; set; } = 4096;
    public int[] HubertConvDims { get; set; } = { 512, 512, 512, 512, 512, 512, 512 };
    public int HubertPositionalKernel { get; set; } = 128;
    public int HubertPositionalGroups { get; set; } = 16;

    /// <summary>Whether HuBERT's input is layer-normalized (the checkpoint's <c>task.normalize</c>; chinese-hubert-large: true).</summary>
    public bool HubertNormalizeWave { get; set; } = true;

    /// <summary>The codebooks' EMA decay (DPQ-VAE §4.1.2: 0.9 for PQ-VAE).</summary>
    public double CodebookEmaDecay { get; set; } = 0.9;

    /// <summary>The cluster size below which a codeword is replaced (the 1S code's <c>threshold_ema_dead_code</c>).</summary>
    public double DeadCodeThreshold { get; set; } = 1.0;

    /// <summary>Eq. 1's weights: λ_vq, λ_s, λ_a.</summary>
    public double VqLossWeight { get; set; } = 1.0;
    public double SslLossWeight { get; set; } = 1000.0;
    public double MelLossWeight { get; set; } = 1.0;

    /// <summary>Dual decoding's weight γ on the pre-quantized reconstruction (DPQ-VAE eq. 4): 1, falling linearly to
    /// <see cref="DualDecodingFinalWeight"/> between <see cref="DualDecodingDecayStart"/> and
    /// <see cref="DualDecodingDecayEnd"/>.</summary>
    public double DualDecodingFinalWeight { get; set; } = 0.1;
    public int DualDecodingDecayStart { get; set; } = 20000;
    public int DualDecodingDecayEnd { get; set; } = 80000;

    // ---------------------------------------------------------------- the streamable decoder (§3.3.2)

    /// <summary>The Mel decoder synthesis uses.</summary>
    public FireRedTtsMelDecoder MelDecoder { get; set; } = FireRedTtsMelDecoder.FlowMatching;

    /// <summary>The Mel codec's width (SoCodec's <c>n_model_size</c> at the speaker embedding's width, which its decoder
    /// adds).</summary>
    public int CodecDim { get; set; } = 512;

    /// <summary>The codec's strides: the 10 ms mel to 20 ms frames.</summary>
    public int[] CodecStrides { get; set; } = { 2 };

    /// <summary>The streams (§3.3.2: four of 16,384 codewords, each two codebooks of <see cref="CodebookSize"/>).</summary>
    public int CodecStreams { get; set; } = 4;
    public int CodecCodebookDim { get; set; } = 128;

    /// <summary>The residual units' convolution groups (SoCodec: 32).</summary>
    public int CodecConvGroups { get; set; } = 32;

    /// <summary>The codec's EMA decay and commitment weight (SoCodec's config: 0.99, 1.0).</summary>
    public double CodecEmaDecay { get; set; } = 0.99;
    public double CodecCommitmentWeight { get; set; } = 1.0;

    /// <summary>The codec's losses: L1 mel reconstruction and the LSGAN generator loss (no paper or code gives them).</summary>
    public double CodecReconstructionWeight { get; set; } = 1.0;
    public double CodecAdversarialWeight { get; set; } = 1.0;

    /// <summary>The mel discriminator's window lengths (frames) and width (MegaTTS's).</summary>
    public int[] DiscriminatorWindows { get; set; } = { 32, 64, 128 };
    public int DiscriminatorHidden { get; set; } = 192;

    /// <summary>The acoustic LM's sizes (MusicGen-small: 24 layers, 1024 wide, 16 heads, a 4096 feed-forward).</summary>
    public int AcousticLmDim { get; set; } = 1024;
    public int AcousticLmHeads { get; set; } = 16;
    public int AcousticLmLayers { get; set; } = 24;
    public int AcousticLmFeedForward { get; set; } = 4096;
    public double AcousticLmDropout { get; set; }

    /// <summary>d in sᵢ₋d + aᵢ (unstated in the paper; 3 spans MusicGen's delay pattern over four streams).</summary>
    public int Lookahead { get; set; } = 3;

    /// <summary>The acoustic LM's sampling (MusicGen's defaults: top-k 250, temperature 1).</summary>
    public int AcousticTopK { get; set; } = 250;
    public double AcousticTemperature { get; set; } = 1.0;


    // ---------------------------------------------------------------- decoding

    /// <summary>Sample the semantic tokens (the release: true); false takes the most likely token.</summary>
    public bool Sample { get; set; } = true;
    public int TopK { get; set; } = 30;
    public double TopP { get; set; } = 0.85;
    public double Temperature { get; set; } = 0.75;
    public double RepetitionPenalty { get; set; } = 2.0;

    /// <summary>Sequences sampled per segment (<c>num_return_sequences</c>: 7).</summary>
    public int Candidates { get; set; } = 7;

    /// <summary>The kept sequence's rank by length, shortest first (the release keeps the second shortest, 1).</summary>
    public int CandidateRank { get; set; } = 1;

    /// <summary>The most semantic tokens per segment (the release: the GPT's limit, 627).</summary>
    public int MaxNewTokens { get; set; } = 627;

    /// <summary>Keeps decoding to semantic tokens and the stop token (off: the release samples from every audio token, and a
    /// start token would fail in the flow). An untrained model needs it.</summary>
    public bool ConstrainToSemanticTokens { get; set; }

    /// <summary>The release's <c>text_split</c> drops a last piece of 15 characters or more; false (the default) keeps
    /// it.</summary>
    public bool ReleaseTextSplit { get; set; }

    /// <summary>The seed of the sampled tokens and the flow's noise.</summary>
    public int SamplingSeed { get; set; } = 1234;

    // ---------------------------------------------------------------- instruction tuning (§4.2)

    /// <summary>Adds the emotion and paralinguistic embeddings and the paper's tags to the tokenizer.</summary>
    public bool InstructionTuning { get; set; }

    // ---------------------------------------------------------------- training

    /// <summary>The network a training step updates.</summary>
    public FireRedTtsStage Stage { get; set; } = FireRedTtsStage.TextToSpeech;
    public int TrainingSeed { get; set; } = 7;

    /// <summary>The flow's and the acoustic LM's warm-up steps (their recipes: CosyVoice's 2,500 and MusicGen's 4,000; the other
    /// recipe values are the stages' declared optimizers).</summary>
    public int FlowWarmupSteps { get; set; } = 2500;
    public int AcousticLmWarmupSteps { get; set; } = 4000;

    /// <summary>XTTS's loss weights: text 0.01, audio 1.</summary>
    public double TextLossWeight { get; set; } = 0.01;
    public double AudioLossWeight { get; set; } = 1.0;

    /// <summary>The GPT's AdamW (XTTS: 0.9, 0.96, 1e-8; rate <see cref="TtsModelOptions.LearningRate"/>, decay on weights).</summary>
    public double Beta1 { get; set; } = 0.9;
    public double Beta2 { get; set; } = 0.96;
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>XTTS's MultiStepLR milestones (×0.5 each).</summary>
    public int[] LearningRateMilestones { get; set; } = { 900000, 2700000, 5400000 };
    public double LearningRateDecay { get; set; } = 0.5;


}
