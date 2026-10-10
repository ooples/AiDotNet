namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options for YourTTS (Casanova et al. 2022): VITS conditioned on external H/ASP speaker embeddings, with
/// optional language embeddings and an optional speaker consistency loss, for zero-shot multi-speaker and multilingual
/// synthesis.</summary>
/// <remarks>
/// <para>
/// The paper (§2, §3.3) and its released recipe (Coqui <c>recipes/vctk/yourtts/train_yourtts.py</c>, <c>VitsArgs</c>):
/// raw characters with blanks between them; a 10-layer text encoder of 192 channels, widened to 196 by the 4-dimensional
/// language embedding concatenated to every character when there are several languages; VITS's 16-layer posterior
/// encoder, 4-coupling flow and stochastic duration predictor; a HiFi-GAN decoder with V1's sizes and type-2 residual
/// blocks (the recipe notes the paper's models were trained with type-2 blocks); 512-dimensional d-vectors from the
/// H/ASP speaker encoder conditioning the posterior, the flow, the duration predictor and the decoder; 16 kHz audio,
/// 1024-point FFT, hop 256; AdamW (β = 0.8, 0.99, weight decay 0.01) at 2e-4 decayed 0.999875 per epoch, batch 64; the
/// speaker consistency loss with α = 9 when it is used (the paper's "+SCL" fine-tuning).
/// </para>
/// <para><b>For Beginners:</b> These options configure the YourTTS model. Default values follow the original paper settings.</para>
/// </remarks>
public class YourTTSOptions : VITSOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public YourTTSOptions(YourTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        SpeakerEncoderDim = other.SpeakerEncoderDim;
        SpeakerEncoderFilters = (int[])other.SpeakerEncoderFilters.Clone();
        UseSpeakerConsistencyLoss = other.UseSpeakerConsistencyLoss;
        SpeakerConsistencyLossWeight = other.SpeakerConsistencyLossWeight;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public YourTTSOptions()
    {
        SampleRate = 16000;
        NumEncoderLayers = 10;
        AddBlank = true;
        ResblockType = 2;
        UpdatesPerEpoch = 679;
    }

    /// <summary>Gets or sets the width of the H/ASP speaker embedding (512).</summary>
    public int SpeakerEncoderDim { get; set; } = 512;

    /// <summary>Gets or sets the H/ASP speaker encoder's four stage widths (32, 64, 128, 256: the released model's;
    /// a checkpoint loads only into the widths it was trained with).</summary>
    public int[] SpeakerEncoderFilters { get; set; } = [32, 64, 128, 256];

    /// <summary>Gets or sets whether training adds the speaker consistency loss (false: the base training; the paper
    /// fine-tunes with it for 50k steps).</summary>
    public bool UseSpeakerConsistencyLoss { get; set; }

    /// <summary>Gets or sets α, the speaker consistency loss's weight (9).</summary>
    public double SpeakerConsistencyLossWeight { get; set; } = 9.0;
}
