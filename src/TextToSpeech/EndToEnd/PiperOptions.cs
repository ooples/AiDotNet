namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options for Piper (Rhasspy): VITS trained with Piper's recipe, at one of its three model sizes.</summary>
/// <remarks>
/// <para>
/// Piper has no paper; its definition is its training code (rhasspy/piper <c>piper_train</c>). Its network is VITS
/// unchanged (<c>vits/models.py</c>) with the stochastic duration predictor; what Piper sets differently is the model
/// size (<c>__main__.py</c>, <c>vits/config.py</c>) and the phoneme-id framing of piper-phonemize (pad 0 after every id,
/// beginning-of-sentence 1 and end-of-sentence 2):
/// </para>
/// <list type="bullet">
/// <item><b>medium</b> (the default): VITS's 192-wide text encoder, latent and flow, with the HiFi-GAN V2-style decoder of
/// <c>ModelAudioConfig.low_quality</c> (type-2 residual blocks with kernels 3, 5, 7 and dilations (1, 2), (2, 6), (3, 12);
/// rates 8, 8, 4 from 256 channels with kernels 16, 16, 8);</item>
/// <item><b>x-low</b> (<see cref="XLow"/>): the same decoder with 96-wide encoder and latent and a 384-wide filter;</item>
/// <item><b>high</b> (<see cref="High"/>): VITS's HiFi-GAN V1 decoder.</item>
/// </list>
/// <para>A multi-speaker Piper voice has a 512-wide speaker table. Training uses AdamW (β = 0.8, 0.99, ε = 1e-9, PyTorch's
/// default weight decay 0.01) at 2e-4 decayed 0.999875 per epoch, batch 32 (<c>TRAINING.md</c>).</para>
/// <para><b>For Beginners:</b> These options configure the Piper model. Default values follow Piper's medium voices.</para>
/// </remarks>
public class PiperOptions : VITSOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PiperOptions(PiperOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        PadId = other.PadId;
        BosId = other.BosId;
        EosId = other.EosId;
        InterspersePad = other.InterspersePad;
    }

    /// <summary>Creates Piper's medium configuration.</summary>
    public PiperOptions()
    {
        AddBlank = false;
        SpeakerEmbeddingDim = 512;
        ResblockType = 2;
        ResblockKernelSizes = [3, 5, 7];
        ResblockDilationSizes = [[1, 2], [2, 6], [3, 12]];
        UpsampleRates = [8, 8, 4];
        UpsampleInitialChannels = 256;
        UpsampleKernelSizes = [16, 16, 8];
    }

    /// <summary>Piper's x-low configuration: 96-wide encoder and latent, 384-wide filter.</summary>
    public static PiperOptions XLow() => new() { HiddenDim = 96, InterChannels = 96, FilterChannels = 384 };

    /// <summary>Piper's medium configuration (the default).</summary>
    public static PiperOptions Medium() => new();

    /// <summary>Piper's high configuration: VITS's HiFi-GAN V1 decoder.</summary>
    public static PiperOptions High() => new()
    {
        ResblockType = 1,
        ResblockKernelSizes = [3, 7, 11],
        ResblockDilationSizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        UpsampleRates = [8, 8, 2, 2],
        UpsampleInitialChannels = 512,
        UpsampleKernelSizes = [16, 16, 4, 4],
    };

    /// <summary>Gets or sets the padding id placed after every id (0, <c>_</c>).</summary>
    public int PadId { get; set; }

    /// <summary>Gets or sets the beginning-of-sentence id (1, <c>^</c>).</summary>
    public int BosId { get; set; } = 1;

    /// <summary>Gets or sets the end-of-sentence id (2, <c>$</c>).</summary>
    public int EosId { get; set; } = 2;

    /// <summary>Gets or sets whether the padding id follows every id (true).</summary>
    public bool InterspersePad { get; set; } = true;
}
