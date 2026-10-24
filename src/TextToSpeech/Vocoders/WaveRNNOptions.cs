namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for WaveRNN (Kalchbrenner et al. 2018): a single-layer recurrent vocoder with a dual softmax over
/// the coarse and fine bytes of 16-bit samples, optionally sparsified by magnitude pruning.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's WaveRNN-896 (§2, §5): a state of 896 units split into coarse and fine halves; 24 kHz
/// 16-bit audio; training on 960-sample sequences with full back-propagation through time. Weight pruning (§3) is off
/// (<see cref="SparsityTarget"/> 0); when on, the recurrent matrices are pruned by magnitude every 500 steps from
/// step 1,000 over 200,000 steps along <c>z = Z(1 − (1 − (t − t₀)/S)³)</c>, optionally in 16×1 or 4×4 blocks.
/// </para>
/// <para>What the paper leaves open: it conditions on linguistic features and pitch without saying how, so the mel
/// spectrogram (Tacotron 2's 80 bands, 50 ms window, 12.5 ms hop, 125 Hz–7.6 kHz) is upsampled by a transposed
/// convolutional network as in WaveNet and projected into the gates; the optimizer follows fatchord/WaveRNN (Adam at
/// 1e-4, batch 32, gradient norm clipped to 4).</para>
/// <para><b>For Beginners:</b> These options configure the WaveRNN model. Default values follow the original paper settings.</para>
/// </remarks>
public class WaveRNNOptions : VocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public WaveRNNOptions(WaveRNNOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        RnnDim = other.RnnDim;
        MaxGradientNorm = other.MaxGradientNorm;
        UpsampleScales = (int[])other.UpsampleScales.Clone();
        SequenceSamples = other.SequenceSamples;
        WindowSize = other.WindowSize;
        MelMinFrequency = other.MelMinFrequency;
        MelMaxFrequency = other.MelMaxFrequency;
        SparsityTarget = other.SparsityTarget;
        PruneBlockRows = other.PruneBlockRows;
        PruneBlockColumns = other.PruneBlockColumns;
        PruningStartStep = other.PruningStartStep;
        PruningSteps = other.PruningSteps;
        PruningInterval = other.PruningInterval;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Creates the paper's WaveRNN-896 configuration.</summary>
    public WaveRNNOptions()
    {
        SampleRate = 24000;
        MelChannels = 80;
        HopSize = 300;
        FftSize = 2048;
        LearningRate = 1e-4;
        WeightDecay = 0.0;
    }

    /// <summary>Gets or sets the state size (896), split into coarse and fine halves.</summary>
    public int RnnDim { get; set; } = 896;

    /// <summary>Gets or sets the global gradient-norm clip (4).</summary>
    public double MaxGradientNorm { get; set; } = 4.0;

    /// <summary>Gets or sets the conditioning upsampler's factors (5, 5, 4, 3; their product is the hop).</summary>
    public int[] UpsampleScales { get; set; } = [5, 5, 4, 3];

    /// <summary>Gets or sets the samples of a training sequence (960; rounded down to whole frames).</summary>
    public int SequenceSamples { get; set; } = 960;

    /// <summary>Gets or sets the analysis window of the input features (1200: 50 ms).</summary>
    public int WindowSize { get; set; } = 1200;

    /// <summary>Gets or sets the lowest mel frequency (125 Hz).</summary>
    public double MelMinFrequency { get; set; } = 125;

    /// <summary>Gets or sets the highest mel frequency (7.6 kHz).</summary>
    public double MelMaxFrequency { get; set; } = 7600;

    /// <summary>Gets or sets the final fraction Z of pruned recurrent weights (0: the dense WaveRNN).</summary>
    public double SparsityTarget { get; set; }

    /// <summary>Gets or sets the rows of a pruning block (1; 16 for the 16×1 structure, 4 for 4×4).</summary>
    public int PruneBlockRows { get; set; } = 1;

    /// <summary>Gets or sets the columns of a pruning block (1; 4 for the 4×4 structure).</summary>
    public int PruneBlockColumns { get; set; } = 1;

    /// <summary>Gets or sets the step t₀ at which pruning begins (1,000).</summary>
    public int PruningStartStep { get; set; } = 1000;

    /// <summary>Gets or sets the pruning steps S over which sparsity reaches the target (200,000).</summary>
    public int PruningSteps { get; set; } = 200000;

    /// <summary>Gets or sets the steps between mask updates (500).</summary>
    public int PruningInterval { get; set; } = 500;

    /// <summary>Gets or sets the seed of the synthesis draws.</summary>
    public int SamplingSeed { get; set; }
}
