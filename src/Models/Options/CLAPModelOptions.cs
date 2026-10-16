using AiDotNet.Audio.Fingerprinting;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for CLAP (Contrastive Language-Audio Pretraining) models.
/// </summary>
/// <remarks>
/// <para>
/// Defaults are LAION-CLAP's published "HTSAT-tiny + RoBERTa" model with feature fusion (Wu et al. 2023,
/// "Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation";
/// reference <c>LAION-AI/CLAP</c>, <c>model_configs/HTSAT-tiny.json</c> and <c>htsat.create_htsat_model</c>):
/// a 48 kHz, 64-band log-mel front end from 50 Hz to 14 kHz over 10-second clips; the HTS-AT audio encoder
/// (Chen et al. 2022), a four-stage hierarchical Swin Transformer; the RoBERTa-base text encoder (Liu et al.
/// 2019); and <c>Linear → ReLU → Linear</c> projections into a 512-dimensional joint space.
/// </para>
/// <para>
/// <b>For Beginners:</b> These knobs control how big and deep the audio and text encoders are. The defaults match
/// the published CLAP checkpoint and produce 512-dim contrastive embeddings. Override individual fields to build
/// a smaller / faster variant.
/// </para>
/// </remarks>
public class CLAPModelOptions : AudioNeuralNetworkOptions
{
    /// <summary>Initializes a new instance with the LAION-CLAP HTSAT-tiny + RoBERTa-base defaults.</summary>
    public CLAPModelOptions()
    {
        // Front end (HTSAT-tiny.json audio_cfg).
        SampleRate = 48_000;
        StftWindowSize = 1024;
        HopLength = 480;
        NumMelBands = 64;
        MelMinFrequency = 50.0;
        MelMaxFrequency = 14_000.0;
        ClipSamples = 480_000;
        EnableFeatureFusion = true;

        // HTS-AT tiny (htsat.create_htsat_model, model_name == "tiny").
        SpecSize = 256;
        AudioPatchSize = 4;
        AudioEmbedDim = 96;
        AudioStageDepths = new[] { 2, 2, 6, 2 };
        AudioStageHeads = new[] { 4, 8, 16, 32 };
        SwinWindowSize = 8;
        AudioMlpRatio = 4;
        AudioDropPathRate = 0.1;
        // HTSAT: SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2).
        SpecAugmentTimeDropWidth = 64;
        SpecAugmentTimeStripes = 2;
        SpecAugmentFreqDropWidth = 8;
        SpecAugmentFreqStripes = 2;

        // RoBERTa-base (roberta-base config.json).
        VocabSize = 50_265;
        MaxTextLength = 77;
        TextMaxPositions = 514;
        TextHiddenDim = 768;
        TextEncoderLayers = 12;
        TextEncoderHeads = 12;
        TextIntermediateDim = 3072;
        TextPaddingTokenId = 1;
        TextDropoutRate = 0.1;
        TextLayerNormEpsilon = 1e-5;

        // Joint space (model.py: joint_embed_shape = 512, logit_scale = log(1 / 0.07)).
        ProjectionDim = 512;
        InitialTemperature = 0.07;
        TextEncoderPath = null;
    }

    /// <summary>
    /// Initializes a new instance by copying every property from <paramref name="other"/>. Throws when
    /// <paramref name="other"/> is null.
    /// </summary>
    /// <param name="other">Source options to copy. Must not be null.</param>
    public CLAPModelOptions(CLAPModelOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        // Inherited properties — copy explicitly because none of the base classes expose a copy constructor.
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ReturnTimestamps = other.ReturnTimestamps;
        SampleRate = other.SampleRate;
        StftWindowSize = other.StftWindowSize;
        HopLength = other.HopLength;
        NumMelBands = other.NumMelBands;
        MelMinFrequency = other.MelMinFrequency;
        MelMaxFrequency = other.MelMaxFrequency;
        ClipSamples = other.ClipSamples;
        EnableFeatureFusion = other.EnableFeatureFusion;
        SpecSize = other.SpecSize;
        AudioPatchSize = other.AudioPatchSize;
        AudioEmbedDim = other.AudioEmbedDim;
        // Arrays are copied, not aliased: the clone must not change when the source's stages are edited.
        AudioStageDepths = (int[])other.AudioStageDepths.Clone();
        AudioStageHeads = (int[])other.AudioStageHeads.Clone();
        SwinWindowSize = other.SwinWindowSize;
        AudioMlpRatio = other.AudioMlpRatio;
        AudioDropPathRate = other.AudioDropPathRate;
        SpecAugmentTimeDropWidth = other.SpecAugmentTimeDropWidth;
        SpecAugmentTimeStripes = other.SpecAugmentTimeStripes;
        SpecAugmentFreqDropWidth = other.SpecAugmentFreqDropWidth;
        SpecAugmentFreqStripes = other.SpecAugmentFreqStripes;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        TextMaxPositions = other.TextMaxPositions;
        TextHiddenDim = other.TextHiddenDim;
        TextEncoderLayers = other.TextEncoderLayers;
        TextEncoderHeads = other.TextEncoderHeads;
        TextIntermediateDim = other.TextIntermediateDim;
        TextPaddingTokenId = other.TextPaddingTokenId;
        TextDropoutRate = other.TextDropoutRate;
        TextLayerNormEpsilon = other.TextLayerNormEpsilon;
        ProjectionDim = other.ProjectionDim;
        InitialTemperature = other.InitialTemperature;
        TextEncoderPath = other.TextEncoderPath;
    }

    // ── Log-mel front end ──────────────────────────────────────────────────────────────────────────

    /// <summary>Audio sample rate in Hz the encoder is configured for.</summary>
    /// <value>Default 48000 (CLAP training rate).</value>
    /// <remarks><para><b>For Beginners:</b> CLAP was trained at 48 kHz — resample audio to this rate first.</para></remarks>
    public int SampleRate { get; set; }

    /// <summary>STFT window (and FFT) size in samples.</summary>
    /// <value>Default 1024.</value>
    /// <remarks><para><b>For Beginners:</b> Audio is cut into overlapping windows this many samples long before measuring which frequencies each contains; longer windows resolve pitch more finely but blur timing.</para></remarks>
    public int StftWindowSize { get; set; }

    /// <summary>STFT hop length in samples.</summary>
    /// <value>Default 480 (10 ms at 48 kHz).</value>
    /// <remarks><para><b>For Beginners:</b> How far each STFT window moves forward; it sets how many spectrogram frames a second of audio becomes.</para></remarks>
    public int HopLength { get; set; }

    /// <summary>Number of mel-frequency bands.</summary>
    /// <value>Default 64.</value>
    /// <remarks><para><b>For Beginners:</b> How many frequency bands, spaced the way human hearing spaces pitch, the spectrogram keeps; it is the height of the spectrogram image.</para></remarks>
    public int NumMelBands { get; set; }

    /// <summary>Lowest frequency of the mel filterbank, in Hz.</summary>
    /// <value>Default 50 (HTSAT-tiny.json <c>fmin</c>).</value>
    /// <remarks><para><b>For Beginners:</b> Sound below this frequency is ignored by the mel filterbank.</para></remarks>
    public double MelMinFrequency { get; set; }

    /// <summary>Highest frequency of the mel filterbank, in Hz.</summary>
    /// <value>Default 14000 (HTSAT-tiny.json <c>fmax</c>).</value>
    /// <remarks>
    /// <para>Clamped to the Nyquist frequency when the sample rate is lower.</para>
    /// <para><b>For Beginners:</b> Sound above this frequency is ignored by the mel filterbank.</para>
    /// </remarks>
    public double MelMaxFrequency { get; set; }

    /// <summary>Samples per clip: shorter audio is repeated and zero-padded to this length, longer audio is
    /// cropped (no fusion) or fused (feature fusion).</summary>
    /// <value>Default 480000 (10 s at 48 kHz).</value>
    /// <remarks><para><b>For Beginners:</b> The model always looks at one fixed-length clip; this is that length in samples, and shorter or longer audio is fitted to it.</para></remarks>
    public int ClipSamples { get; set; }

    /// <summary>
    /// Whether long clips use feature fusion (Wu et al. 2023 §3.3): a shrunk global view plus three random local
    /// crops, merged by attentional feature fusion in the patch embedding.
    /// </summary>
    /// <value>Default true (the 630k-audioset-fusion checkpoint).</value>
    /// <remarks>
    /// The two published variants also differ in their front end, and this follows them: the fusion models read
    /// mels computed by torchaudio (HTK mel scale, unnormalized filters) in the data pipeline, the non-fusion
    /// models compute their own with torchlibrosa (librosa's Slaney scale and normalization).
    /// <para><b>For Beginners:</b> With fusion on, a long recording is summarized by a shrunk view of the whole thing
    /// plus three close-up crops, so the model sees both the overall shape and some detail; with it off, a long
    /// recording is simply cut to one clip.</para>
    /// </remarks>
    public bool EnableFeatureFusion { get; set; }

    // ── HTS-AT audio encoder ───────────────────────────────────────────────────────────────────────

    /// <summary>Side of the square spectrogram image HTS-AT reads.</summary>
    /// <value>Default 256. With 64 mel bands the clip is folded into 256 / 64 = 4 stacked time chunks.</value>
    /// <remarks><para><b>For Beginners:</b> The spectrogram is stretched and folded into a square image of this side length before the vision-style encoder reads it.</para></remarks>
    public int SpecSize { get; set; }

    /// <summary>Patch size (and stride) of the patch embedding.</summary>
    /// <value>Default 4.</value>
    /// <remarks><para><b>For Beginners:</b> The spectrogram image is cut into square patches of this size; each patch becomes one token.</para></remarks>
    public int AudioPatchSize { get; set; }

    /// <summary>Embedding width of the first stage; it doubles at every patch merging.</summary>
    /// <value>Default 96 (HTS-AT tiny), so the final width is 96 · 2³ = 768.</value>
    /// <remarks><para><b>For Beginners:</b> How many numbers describe each patch token in the first stage; deeper stages use more.</para></remarks>
    public int AudioEmbedDim { get; set; }

    /// <summary>Swin blocks per stage; one stage per entry, with a patch merging between consecutive stages.</summary>
    /// <value>Default [2, 2, 6, 2].</value>
    /// <remarks><para><b>For Beginners:</b> How many transformer blocks each stage stacks; more blocks give more capacity at more cost.</para></remarks>
    public int[] AudioStageDepths { get; set; }

    /// <summary>Attention heads per stage; one entry per stage.</summary>
    /// <value>Default [4, 8, 16, 32].</value>
    /// <remarks><para><b>For Beginners:</b> How many attention heads each stage uses; more heads let a block attend to several patterns at once.</para></remarks>
    public int[] AudioStageHeads { get; set; }

    /// <summary>Swin window size (W-MSA / SW-MSA).</summary>
    /// <value>Default 8 (HTS-AT).</value>
    /// <remarks><para><b>For Beginners:</b> Attention is computed inside small square windows of this many patches, which keeps it affordable on large images.</para></remarks>
    public int SwinWindowSize { get; set; }

    /// <summary>Ratio of the Swin MLP's hidden width to the block width.</summary>
    /// <value>Default 4.</value>
    /// <remarks><para><b>For Beginners:</b> How much wider the feed-forward part of each block is than the block itself.</para></remarks>
    public int AudioMlpRatio { get; set; }

    /// <summary>Stochastic-depth rate of the last Swin block; the rate grows linearly from 0 across blocks.</summary>
    /// <value>Default 0.1 (HTS-AT <c>drop_path_rate</c>).</value>
    /// <remarks><para><b>For Beginners:</b> During training, whole blocks are randomly skipped with up to this probability, which regularizes a deep network.</para></remarks>
    public double AudioDropPathRate { get; set; }

    /// <summary>Exclusive upper bound of the width, in frames, of a time stripe zeroed during training.</summary>
    /// <value>Default 64 (HTS-AT's SpecAugmentation).</value>
    /// <remarks><para><b>For Beginners:</b> SpecAugment hides short spans of time during training so the model does not rely on any single moment; this bounds how long each hidden span can be.</para></remarks>
    public int SpecAugmentTimeDropWidth { get; set; }

    /// <summary>Time stripes zeroed per training sample; 0 turns time masking off.</summary>
    /// <value>Default 2.</value>
    /// <remarks><para><b>For Beginners:</b> How many spans of time SpecAugment hides in each training example.</para></remarks>
    public int SpecAugmentTimeStripes { get; set; }

    /// <summary>Exclusive upper bound of the width, in mel bins, of a frequency stripe zeroed during training.</summary>
    /// <value>Default 8.</value>
    /// <remarks><para><b>For Beginners:</b> SpecAugment also hides narrow frequency bands during training; this bounds how wide each band can be.</para></remarks>
    public int SpecAugmentFreqDropWidth { get; set; }

    /// <summary>Frequency stripes zeroed per training sample; 0 turns frequency masking off.</summary>
    /// <value>Default 2.</value>
    /// <remarks><para><b>For Beginners:</b> How many frequency bands SpecAugment hides in each training example.</para></remarks>
    public int SpecAugmentFreqStripes { get; set; }

    // ── RoBERTa text encoder ───────────────────────────────────────────────────────────────────────

    /// <summary>Text vocabulary size — RoBERTa BPE vocab.</summary>
    /// <value>Default 50265.</value>
    /// <remarks><para><b>For Beginners:</b> How many distinct word pieces the text tokenizer knows; it must match the tokenizer you use.</para></remarks>
    public int VocabSize { get; set; }

    /// <summary>Token length captions are truncated / padded to.</summary>
    /// <value>Default 77 (LAION-CLAP's tokenizer call: <c>max_length=77</c>).</value>
    /// <remarks><para><b>For Beginners:</b> Captions are cut or padded to this many tokens before encoding.</para></remarks>
    public int MaxTextLength { get; set; }

    /// <summary>Rows of RoBERTa's position table (positions start after the padding id).</summary>
    /// <value>Default 514.</value>
    /// <remarks><para><b>For Beginners:</b> How many token positions the text encoder has learned embeddings for; longer inputs cannot be encoded.</para></remarks>
    public int TextMaxPositions { get; set; }

    /// <summary>Hidden width of the text encoder.</summary>
    /// <value>Default 768 (RoBERTa-base).</value>
    /// <remarks><para><b>For Beginners:</b> How many numbers describe each token inside the text encoder.</para></remarks>
    public int TextHiddenDim { get; set; }

    /// <summary>Number of text encoder layers.</summary>
    /// <value>Default 12 (RoBERTa-base).</value>
    /// <remarks><para><b>For Beginners:</b> How many transformer layers the text encoder stacks.</para></remarks>
    public int TextEncoderLayers { get; set; }

    /// <summary>Attention heads per text encoder layer.</summary>
    /// <value>Default 12 (RoBERTa-base).</value>
    /// <remarks><para><b>For Beginners:</b> How many attention heads each text encoder layer uses.</para></remarks>
    public int TextEncoderHeads { get; set; }

    /// <summary>Width of the text encoder's feed-forward intermediate projection.</summary>
    /// <value>Default 3072 (RoBERTa-base).</value>
    /// <remarks><para><b>For Beginners:</b> How wide the feed-forward part of each text encoder layer is.</para></remarks>
    public int TextIntermediateDim { get; set; }

    /// <summary>RoBERTa's padding token id (<c>&lt;pad&gt;</c>).</summary>
    /// <value>Default 1.</value>
    /// <remarks><para><b>For Beginners:</b> The token id used to fill short captions up to the fixed length; positions holding it are ignored by attention.</para></remarks>
    public int TextPaddingTokenId { get; set; }

    /// <summary>Hidden and attention-probability dropout of the text encoder.</summary>
    /// <value>Default 0.1 (RoBERTa-base).</value>
    /// <remarks><para><b>For Beginners:</b> During training, this fraction of activations is randomly zeroed to reduce overfitting; it has no effect at inference.</para></remarks>
    public double TextDropoutRate { get; set; }

    /// <summary>LayerNorm epsilon of the text encoder.</summary>
    /// <value>Default 1e-5 (RoBERTa-base).</value>
    /// <remarks><para><b>For Beginners:</b> A tiny constant added inside layer normalization so it never divides by zero.</para></remarks>
    public double TextLayerNormEpsilon { get; set; }

    // ── Shared projection + contrastive learning ───────────────────────────────────────────────────

    /// <summary>Dimension of the joint audio-text embedding space.</summary>
    /// <value>Default 512.</value>
    /// <remarks><para><b>For Beginners:</b> Both encoders end with a small MLP into this shared dimension, where
    /// audio and text vectors are compared by cosine similarity.</para></remarks>
    public int ProjectionDim { get; set; }

    /// <summary>Initial temperature τ for the contrastive softmax; the model learns <c>log(1/τ)</c>.</summary>
    /// <value>Default 0.07.</value>
    /// <remarks><para><b>For Beginners:</b> Controls how sharply the model separates matching from non-matching audio-caption pairs; lower values sharpen it, and the model keeps learning it during training.</para></remarks>
    public double InitialTemperature { get; set; }

    /// <summary>
    /// Gets or sets the text encoder path.
    /// </summary>
    /// <value>Default null (the built-in RoBERTa tower).</value>
    /// <remarks><para><b>For Beginners:</b> Path to an exported ONNX text encoder. Leave it unset to use the built-in RoBERTa tower.</para></remarks>
    public string? TextEncoderPath { get; set; }

    /// <summary>Number of HTS-AT stages.</summary>
    internal int AudioStageCount => AudioStageDepths.Length;

    /// <summary>Width of the audio encoder's output: <c>AudioEmbedDim · 2^(stages − 1)</c>.</summary>
    internal int AudioFeatureDim => AudioEmbedDim << (AudioStageDepths.Length - 1);

    /// <summary>
    /// Throws if a value this model requires has been left unset or is inconsistent.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is invalid.</exception>
    public void Validate()
    {
        ValidateCore();
        static void Positive(int value, string name)
        {
            if (value <= 0) throw new ArgumentException($"{name} must be positive; got {value}.", name);
        }
        Positive(SampleRate, nameof(SampleRate));
        Positive(StftWindowSize, nameof(StftWindowSize));
        Positive(HopLength, nameof(HopLength));
        Positive(NumMelBands, nameof(NumMelBands));
        Positive(ClipSamples, nameof(ClipSamples));
        Positive(SpecSize, nameof(SpecSize));
        Positive(AudioPatchSize, nameof(AudioPatchSize));
        Positive(AudioEmbedDim, nameof(AudioEmbedDim));
        Positive(SwinWindowSize, nameof(SwinWindowSize));
        Positive(AudioMlpRatio, nameof(AudioMlpRatio));
        Positive(VocabSize, nameof(VocabSize));
        Positive(MaxTextLength, nameof(MaxTextLength));
        Positive(TextHiddenDim, nameof(TextHiddenDim));
        Positive(TextEncoderLayers, nameof(TextEncoderLayers));
        Positive(TextEncoderHeads, nameof(TextEncoderHeads));
        Positive(TextIntermediateDim, nameof(TextIntermediateDim));
        Positive(ProjectionDim, nameof(ProjectionDim));
        if (MelMinFrequency < 0 || MelMaxFrequency <= MelMinFrequency)
            throw new ArgumentException($"The mel band [{MelMinFrequency}, {MelMaxFrequency}] Hz is empty.", nameof(MelMaxFrequency));
        if (SpecSize % NumMelBands != 0)
            throw new ArgumentException(
                $"SpecSize ({SpecSize}) must be a multiple of NumMelBands ({NumMelBands}): HTS-AT folds time into SpecSize / NumMelBands chunks.",
                nameof(SpecSize));
        int frames = ClipSamples / HopLength + 1;
        if (frames > SpecSize * (SpecSize / NumMelBands))
            throw new ArgumentException(
                $"A clip of {ClipSamples} samples gives {frames} frames, more than the {SpecSize * (SpecSize / NumMelBands)} the image holds.",
                nameof(ClipSamples));
        if (AudioStageDepths is null || AudioStageHeads is null || AudioStageDepths.Length == 0
            || AudioStageDepths.Length != AudioStageHeads.Length)
            throw new ArgumentException("AudioStageDepths and AudioStageHeads need one entry per stage.", nameof(AudioStageDepths));
        int grid = SpecSize / AudioPatchSize;
        if (SpecSize % AudioPatchSize != 0 || grid % (1 << (AudioStageDepths.Length - 1)) != 0)
            throw new ArgumentException(
                $"The {grid} x {grid} patch grid must halve evenly at each of the {AudioStageDepths.Length - 1} patch mergings.",
                nameof(SpecSize));
        for (int s = 0; s < AudioStageDepths.Length; s++)
        {
            Positive(AudioStageDepths[s], nameof(AudioStageDepths));
            int width = AudioEmbedDim << s;
            if (AudioStageHeads[s] <= 0 || width % AudioStageHeads[s] != 0)
                throw new ArgumentException($"Stage {s} width {width} is not divisible by its {AudioStageHeads[s]} heads.", nameof(AudioStageHeads));
        }
        Positive(SpecAugmentTimeDropWidth, nameof(SpecAugmentTimeDropWidth));
        Positive(SpecAugmentFreqDropWidth, nameof(SpecAugmentFreqDropWidth));
        if (SpecAugmentTimeStripes < 0 || SpecAugmentFreqStripes < 0)
            throw new ArgumentException("SpecAugment stripe counts cannot be negative.", nameof(SpecAugmentTimeStripes));
        if (AudioDropPathRate < 0 || AudioDropPathRate >= 1)
            throw new ArgumentException("AudioDropPathRate must be in [0, 1).", nameof(AudioDropPathRate));
        if (TextHiddenDim % TextEncoderHeads != 0)
            throw new ArgumentException($"TextHiddenDim ({TextHiddenDim}) must be divisible by TextEncoderHeads ({TextEncoderHeads}).", nameof(TextEncoderHeads));
        if (TextPaddingTokenId < 0 || TextPaddingTokenId >= VocabSize)
            throw new ArgumentException("TextPaddingTokenId must be a token of the vocabulary.", nameof(TextPaddingTokenId));
        if (MaxTextLength + TextPaddingTokenId + 1 > TextMaxPositions)
            throw new ArgumentException(
                $"MaxTextLength ({MaxTextLength}) needs {MaxTextLength + TextPaddingTokenId + 1} RoBERTa positions; TextMaxPositions is {TextMaxPositions}.",
                nameof(TextMaxPositions));
        if (TextDropoutRate < 0 || TextDropoutRate >= 1)
            throw new ArgumentException("TextDropoutRate must be in [0, 1).", nameof(TextDropoutRate));
        if (TextLayerNormEpsilon <= 0)
            throw new ArgumentException("TextLayerNormEpsilon must be positive.", nameof(TextLayerNormEpsilon));
        if (InitialTemperature <= 0)
            throw new ArgumentException("InitialTemperature must be positive.", nameof(InitialTemperature));
    }
}
