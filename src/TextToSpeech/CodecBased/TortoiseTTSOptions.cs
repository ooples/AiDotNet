namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The independently trained TorToiSe component selected by <see cref="TortoiseTTSOptions.TrainingComponent"/>.</summary>
public enum TortoiseTrainingComponent
{
    /// <summary>The GPT-2-style text-to-speech-code prior.</summary>
    Autoregressive,
    /// <summary>The contrastive language/voice pretraining model used to rank autoregressive samples.</summary>
    Clvp,
    /// <summary>The code-conditioned diffusion decoder.</summary>
    Diffusion,
    /// <summary>The four-times-compressed 8,192-entry VQ-VAE speech tokenizer.</summary>
    SpeechTokenizer,
    /// <summary>The UnivNet waveform generator.</summary>
    UnivNet
}

/// <summary>Options for TorToiSe: Better Speech Synthesis Through Scaling (Betker, 2023).</summary>
/// <remarks>
/// Defaults reproduce the released standard preset. Smaller values may be supplied for experiments and tests, but
/// the topology is unchanged: VQ-VAE codes, a GPT-2 prior, CLVP candidate ranking, a latent-conditioned diffusion
/// decoder, and the 24 kHz UnivNet-c32 vocoder.
/// </remarks>
public class TortoiseTTSOptions : CodecTtsOptions
{
    /// <summary>UnivNet's original auxiliary STFT, discriminator and pretraining settings, used only by the vocoder stage.</summary>
    public AiDotNet.TextToSpeech.Vocoders.UnivNetOptions VocoderTraining { get; set; } = new();
    /// <summary>Creates the paper/release configuration.</summary>
    public TortoiseTTSOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 256;
        FftSize = 1024;
        VocabSize = 256;
        MaxTextLength = 402;
        LLMDim = 1024;
        NumLLMLayers = 30;
        NumHeads = 16;
        NumEncoderLayers = 6;
        NumCodebooks = 1;
        CodebookSize = 8192;
        CodecFrameRate = 22;
        MaxCodecFrames = 604;
        LanguageModelName = "GPT-2";
        DropoutRate = 0.1;
    }

    /// <summary>The component optimized by the TTS training entry points.</summary>
    public TortoiseTrainingComponent TrainingComponent { get; set; } = TortoiseTrainingComponent.Autoregressive;
    /// <summary>Optional original tokenizer.json path; null uses the released 255-token BPE resource.</summary>
    public string? TextTokenizerPath { get; set; }
    /// <summary>The VQ-VAE and autoregressive conditioner's rate (22.05 kHz).</summary>
    public int TokenizerSampleRate { get; set; } = 22050;
    /// <summary>Mel bands read by the speech tokenizer and autoregressive conditioning encoder (80).</summary>
    public int AutoregressiveMelChannels { get; set; } = 80;
    /// <summary>Optional checkpoint mel_norms values; null uses the released 80-band values.</summary>
    public double[]? AutoregressiveMelNormalization { get; set; }
    /// <summary>FFT size used for 22.05 kHz conditioning features (1024).</summary>
    public int AutoregressiveFftSize { get; set; } = 1024;
    /// <summary>Hop used for 22.05 kHz conditioning features (256).</summary>
    public int AutoregressiveHopSize { get; set; } = 256;
    /// <summary>Maximum samples in one autoregressive conditioning clip (132,300, or six seconds).</summary>
    public int ConditioningSamples { get; set; } = 132300;
    /// <summary>Maximum conditioning clips averaged by the released model (2).</summary>
    public int MaxConditioningInputs { get; set; } = 2;
    /// <summary>GPT-2 feed-forward width (4,096).</summary>
    public int AutoregressiveFeedForwardDim { get; set; } = 4096;
    /// <summary>Text start token (255).</summary>
    public int TextStartToken { get; set; } = 255;
    /// <summary>Text stop/padding token (0).</summary>
    public int TextStopToken { get; set; } = 0;
    /// <summary>Speech start token (8,192).</summary>
    public int SpeechStartToken { get; set; } = 8192;
    /// <summary>Speech stop token (8,193).</summary>
    public int SpeechStopToken { get; set; } = 8193;
    /// <summary>Weight of the text cross-entropy beside the speech-code loss (0.01).</summary>
    public double TextLossWeight { get; set; } = 0.01;
    /// <summary>Number of autoregressive candidates passed to CLVP (256 in the standard preset).</summary>
    public int NumAutoregressiveSamples { get; set; } = 256;
    /// <summary>Nucleus probability used by the paper's autoregressive sampler (0.8).</summary>
    public double TopP { get; set; } = 0.8;
    /// <summary>Autoregressive sampling temperature (0.8).</summary>
    public double Temperature { get; set; } = 0.8;
    /// <summary>Penalty applied to a code already present in the generated sequence (2).</summary>
    public double RepetitionPenalty { get; set; } = 2.0;
    /// <summary>Minimum generated speech codes before a stop token may terminate a candidate.</summary>
    public int MinimumAutoregressiveTokens { get; set; } = 0;
    /// <summary>Generation/padding limit in the released synthesis API (500), separate from the checkpoint's 604-position capacity.</summary>
    public int MaximumAutoregressiveTokens { get; set; } = 500;
    /// <summary>CLVP transformer/latent width (768).</summary>
    public int ClvpDim { get; set; } = 768;
    /// <summary>CLVP transformer depth for each of its two towers (20).</summary>
    public int ClvpLayers { get; set; } = 20;
    /// <summary>CLVP attention heads (12).</summary>
    public int ClvpHeads { get; set; } = 12;
    /// <summary>CLVP maximum text positions (350).</summary>
    public int ClvpTextLength { get; set; } = 350;
    /// <summary>CLVP maximum speech positions in the released checkpoint (430).</summary>
    public int ClvpSpeechLength { get; set; } = 430;
    /// <summary>Token dropout used while training CLVP (15%).</summary>
    public double ClvpTokenDropout { get; set; } = 0.15;
    /// <summary>Diffusion model width (1,024).</summary>
    public int DiffusionDim { get; set; } = 1024;
    /// <summary>Alternating attention/convolution diffusion layers (10).</summary>
    public int DiffusionLayers { get; set; } = 10;
    /// <summary>Diffusion attention heads (16).</summary>
    public int DiffusionHeads { get; set; } = 16;
    /// <summary>Training diffusion steps used by the release (4,000).</summary>
    public int DiffusionTrainingSteps { get; set; } = 4000;
    /// <summary>Spaced DDPM steps in the released standard preset (200).</summary>
    public int DiffusionInferenceSteps { get; set; } = 200;
    /// <summary>Released classifier-free guidance k (2), ramped over the selected DDPM schedule.</summary>
    public double ClassifierFreeGuidanceScale { get; set; } = 2.0;
    /// <summary>Probability of dropping all diffusion conditioning during training (15%).</summary>
    public double DiffusionConditioningDropout { get; set; } = 0.15;
    /// <summary>Maximum 24 kHz reference samples read by the diffusion speaker encoder (102,400).</summary>
    public int DiffusionConditioningSamples { get; set; } = 102400;
    /// <summary>VQ-VAE latent channels (512).</summary>
    public int TokenizerLatentDim { get; set; } = 512;
    /// <summary>Released dVAE code-vector width (512).</summary>
    public int TokenizerCodeDim { get; set; } = 512;
    /// <summary>Residual blocks in each dVAE encoder/decoder (3).</summary>
    public int TokenizerResidualBlocks { get; set; } = 3;
    /// <summary>Temporal compression of the tokenizer (4).</summary>
    public int TokenizerCompression { get; set; } = 4;
    /// <summary>VQ-VAE commitment-loss multiplier.</summary>
    public double TokenizerCommitmentWeight { get; set; } = 0.25;
    /// <summary>UnivNet input noise channels (64).</summary>
    public int UnivNetNoiseDim { get; set; } = 64;
    /// <summary>UnivNet-c32 generator channels (32).</summary>
    public int UnivNetChannels { get; set; } = 32;
    /// <summary>UnivNet LVC upsampling rates (8, 8, 4; product 256).</summary>
    public int[] UnivNetUpsampleRates { get; set; } = [8, 8, 4];
    /// <summary>UnivNet LVC dilations (1, 3, 9, 27).</summary>
    public int[] UnivNetDilations { get; set; } = [1, 3, 9, 27];
    /// <summary>UnivNet kernel-predictor hidden channels (64).</summary>
    public int UnivNetKernelPredictorHidden { get; set; } = 64;
    /// <summary>UnivNet kernel-predictor convolution width (3).</summary>
    public int UnivNetKernelPredictorConvSize { get; set; } = 3;
    /// <summary>Log-mel frames appended by the released vocoder and cropped from its output (10).</summary>
    public int UnivNetPaddingFrames { get; set; } = 10;
    /// <summary>Seed for candidate sampling, diffusion noise, and UnivNet noise.</summary>
    public int SamplingSeed { get; set; } = 1234;
}
