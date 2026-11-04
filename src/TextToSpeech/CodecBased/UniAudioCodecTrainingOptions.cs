namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which independently trained part of UniAudio a sample updates.</summary>
public enum UniAudioTrainingComponent
{
    /// <summary>The global and local causal language models, with the universal codec frozen.</summary>
    LanguageModel,
    /// <summary>The universal codec and its six mel discriminators.</summary>
    UniversalCodec,
}

/// <summary>The released universal codec's adversarial-pretraining configuration.</summary>
public sealed class UniAudioCodecTrainingOptions
{
    public UniAudioCodecTrainingOptions() { }

    public UniAudioCodecTrainingOptions(UniAudioCodecTrainingOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        Bandwidths = (double[])other.Bandwidths.Clone();
        DiscriminatorHops = (int[])other.DiscriminatorHops.Clone();
        DiscriminatorChannels = (int[])other.DiscriminatorChannels.Clone();
        FullFftSizes = (int[])other.FullFftSizes.Clone();
        FullWindowSizes = (int[])other.FullWindowSizes.Clone();
        FullHopSizes = (int[])other.FullHopSizes.Clone();
        SubFftSizes = (int[])other.SubFftSizes.Clone();
        SubWindowSizes = (int[])other.SubWindowSizes.Clone();
        SubHopSizes = (int[])other.SubHopSizes.Clone();
        Subbands = other.Subbands;
        SegmentSize = other.SegmentSize;
        FeatureMatchingWeight = other.FeatureMatchingWeight;
        FullStftWeight = other.FullStftWeight;
        SubStftWeight = other.SubStftWeight;
        CommitmentWeight = other.CommitmentWeight;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        EpochDecay = other.EpochDecay;
        Epoch = other.Epoch;
        CompletedSteps = other.CompletedSteps;
        AdversarialStartStep = other.AdversarialStartStep;
    }

    /// <summary>Training bandwidths in kbps; v1's eight-stage codec uses the first five values.</summary>
    public double[] Bandwidths { get; set; } = [0.5, 1, 1.5, 2, 4, 6];
    /// <summary>Critic STFT hop lengths; each FFT/window is four times its hop.</summary>
    public int[] DiscriminatorHops { get; set; } = [32, 64, 128, 256, 512, 1024];
    /// <summary>Each critic's final hidden width (paper Appendix E).</summary>
    public int[] DiscriminatorChannels { get; set; } = [64, 128, 256, 512, 512, 512];
    public int[] FullFftSizes { get; set; } = [512, 1024, 2048];
    public int[] FullWindowSizes { get; set; } = [480, 960, 1200];
    public int[] FullHopSizes { get; set; } = [120, 240, 300];
    public int[] SubFftSizes { get; set; } = [128, 256, 256];
    public int[] SubWindowSizes { get; set; } = [80, 120, 200];
    public int[] SubHopSizes { get; set; } = [20, 40, 50];
    /// <summary>PQMF subbands for the sub-band spectral objective (six).</summary>
    public int Subbands { get; set; } = 6;
    /// <summary>Samples in the released training segment (32,000).</summary>
    public int SegmentSize { get; set; } = 32000;
    public double FeatureMatchingWeight { get; set; } = 20;
    public double FullStftWeight { get; set; } = 1;
    public double SubStftWeight { get; set; } = 1;
    public double CommitmentWeight { get; set; } = 1;
    /// <summary>AdamW rate for both codec and critic; betas are (0.8, 0.99), epsilon 1e-6.</summary>
    public double LearningRate { get; set; } = 2e-4;
    public double WeightDecay { get; set; } = 0.01;
    /// <summary>The released exponential learning-rate multiplier per completed codec epoch.</summary>
    public double EpochDecay { get; set; } = 0.999;
    /// <summary>Completed codec epochs, retained with the training configuration when resuming.</summary>
    public int Epoch { get; set; }
    /// <summary>Completed generator/discriminator step pairs.</summary>
    public long CompletedSteps { get; set; }
    /// <summary>The generator includes adversarial losses after this step (zero in the release).</summary>
    public long AdversarialStartStep { get; set; }
}
