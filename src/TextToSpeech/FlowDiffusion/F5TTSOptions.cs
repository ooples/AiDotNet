namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Options for F5-TTS (Chen et al. 2024): a DiT with ConvNeXt V2 text refinement trained by text-guided
/// speech-infilling flow matching.</summary>
/// <remarks>
/// <para>
/// Defaults are F5-TTS Base (§4, App. B; reference SWivid/F5-TTS <c>configs/F5TTS_Base.yaml</c>, <c>model/cfm.py</c>):
/// 22 DiT layers, 16 heads of 64, width 1024, feed-forward 2048 (×2), dropout 0.1, rotary embedding on the first head;
/// text embedding 512 with 4 ConvNeXt V2 blocks (FFN 1024); a vocabulary of 2545 characters plus the filler;
/// 24 kHz, 1024-point STFT, hop 256, 100 mel bins; infilling spans of 70–100 %, audio-condition drop 0.3, joint drop 0.2;
/// AdamW at 7.5e-5 warmed up linearly over 20K updates then decayed linearly, gradient clip 1; inference with 32 NFE,
/// sway coefficient −1 and CFG strength 2.
/// </para>
/// <para><b>For Beginners:</b> These options configure the F5TTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class F5TTSOptions : TtsModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public F5TTSOptions(F5TTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NumLayers = other.NumLayers;
        HeadDim = other.HeadDim;
        FeedForwardMultiplier = other.FeedForwardMultiplier;
        TextDim = other.TextDim;
        TextConvLayers = other.TextConvLayers;
        RotaryHeads = other.RotaryHeads;
        MaskFractionMin = other.MaskFractionMin;
        MaskFractionMax = other.MaskFractionMax;
        AudioDropProbability = other.AudioDropProbability;
        ConditionDropProbability = other.ConditionDropProbability;
        NumFunctionEvaluations = other.NumFunctionEvaluations;
        CfgStrength = other.CfgStrength;
        SwayCoefficient = other.SwayCoefficient;
        FramesPerCharacter = other.FramesPerCharacter;
        Speed = other.Speed;
        WarmupSteps = other.WarmupSteps;
        TotalSteps = other.TotalSteps;
        MaxGradientNorm = other.MaxGradientNorm;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates F5-TTS Base.</summary>
    public F5TTSOptions()
    {
        SampleRate = 24000;
        MelChannels = 100;
        HopSize = 256;
        FftSize = 1024;
        HiddenDim = 1024;
        NumHeads = 16;
        VocabSize = 2545;
        DropoutRate = 0.1;
        LearningRate = 7.5e-5;
        WeightDecay = 0.01;
        MaxMelLength = 4096;
    }

    /// <summary>Gets or sets the DiT blocks (22).</summary>
    public int NumLayers { get; set; } = 22;

    /// <summary>Gets or sets the attention head width (64).</summary>
    public int HeadDim { get; set; } = 64;

    /// <summary>Gets or sets the feed-forward width multiplier (2: 1024/2048).</summary>
    public int FeedForwardMultiplier { get; set; } = 2;

    /// <summary>Gets or sets the text embedding width (512).</summary>
    public int TextDim { get; set; } = 512;

    /// <summary>Gets or sets the ConvNeXt V2 blocks refining the text (4).</summary>
    public int TextConvLayers { get; set; } = 4;

    /// <summary>Gets or sets how many leading heads carry the rotary embedding (1, the reference configuration; null for all).</summary>
    public int? RotaryHeads { get; set; } = 1;

    /// <summary>Gets or sets the smallest masked fraction of a training utterance (0.7).</summary>
    public double MaskFractionMin { get; set; } = 0.7;

    /// <summary>Gets or sets the largest masked fraction of a training utterance (1.0).</summary>
    public double MaskFractionMax { get; set; } = 1.0;

    /// <summary>Gets or sets the probability of dropping the audio condition (0.3).</summary>
    public double AudioDropProbability { get; set; } = 0.3;

    /// <summary>Gets or sets the probability of dropping audio and text together (0.2).</summary>
    public double ConditionDropProbability { get; set; } = 0.2;

    /// <summary>Gets or sets the number of Euler steps at synthesis (32 NFE).</summary>
    public int NumFunctionEvaluations { get; set; } = 32;

    /// <summary>Gets or sets the classifier-free guidance strength α (2).</summary>
    public double CfgStrength { get; set; } = 2.0;

    /// <summary>Gets or sets the sway sampling coefficient s (−1).</summary>
    public double SwayCoefficient { get; set; } = -1.0;

    /// <summary>Gets or sets the frames per character when synthesizing without a prompt, whose speaking rate the paper
    /// otherwise uses (6, about 15 characters per second at 93.75 frames per second).</summary>
    public double FramesPerCharacter { get; set; } = 6.0;

    /// <summary>Gets or sets the speaking-rate multiplier at synthesis (1).</summary>
    public double Speed { get; set; } = 1.0;

    /// <summary>Gets or sets the learning-rate warmup updates (20 000).</summary>
    public int WarmupSteps { get; set; } = 20000;

    /// <summary>Gets or sets the total updates over which the rate decays linearly (1.2M).</summary>
    public int TotalSteps { get; set; } = 1200000;

    /// <summary>Gets or sets the gradient-norm clip (1).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>Gets or sets the seed of the training draws and the sampling noise, for repeatable runs.</summary>
    public int SamplingSeed { get; set; }
}
