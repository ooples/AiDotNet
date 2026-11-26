namespace AiDotNet.Models.Options;

/// <summary>
/// Controls how <c>AiModelBuilder.Build(samples)</c> trains a model on its own sample-level objective
/// (an <see cref="Interfaces.ISampleObjectiveModel{T, TSample}"/>, such as a text-to-speech model trained on
/// <see cref="TextToSpeech.TtsTrainingSample{T}"/>).
/// </summary>
/// <remarks>
/// <para>
/// Every setting has a default that is right for most runs, so configuring these options is optional: by default
/// the builder trains for <see cref="Epochs"/> epochs, reshuffles the samples every epoch, holds out
/// <see cref="ValidationFraction"/> of them to measure the objective on data the model did not train on, and stops
/// once that held-out objective has not improved for <see cref="EarlyStoppingPatience"/> epochs. Set a value to
/// change one behaviour; set <see cref="ValidationFraction"/> or <see cref="EarlyStoppingPatience"/> to 0 to turn
/// that behaviour off.
/// </para>
/// <para>
/// <see cref="ModelOptions.Seed"/> fixes the shuffle and the held-out split. When it is <c>null</c> the builder uses
/// a fixed seed, unless <c>AllowNondeterminism()</c> was called.
/// </para>
/// <para><b>For Beginners:</b> An epoch is one pass over all your training samples. The held-out samples are a
/// small part of your data the model never trains on; its loss on them tells you whether it is learning to speak
/// (the loss falls) or only memorising your samples (the training loss falls while the held-out loss rises), and
/// early stopping ends training when it stops improving.</para>
/// </remarks>
public class SampleObjectiveTrainingOptions : ModelOptions
{
    /// <summary>Initializes a new instance with the default settings.</summary>
    public SampleObjectiveTrainingOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
    public SampleObjectiveTrainingOptions(SampleObjectiveTrainingOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Epochs = other.Epochs;
        ShuffleEachEpoch = other.ShuffleEachEpoch;
        ValidationFraction = other.ValidationFraction;
        EarlyStoppingPatience = other.EarlyStoppingPatience;
        RestoreBestModel = other.RestoreBestModel;
    }

    /// <summary>Gets or sets the maximum number of passes over the training samples.</summary>
    /// <value>At least 1; defaults to 100.</value>
    public int Epochs { get; set; } = 100;

    /// <summary>Gets or sets whether the training samples are visited in a new random order every epoch.</summary>
    /// <value>Defaults to <c>true</c>, so consecutive steps do not see the same utterances in the same order.</value>
    public bool ShuffleEachEpoch { get; set; } = true;

    /// <summary>
    /// Gets or sets the fraction of the samples held out to measure the training objective after every epoch.
    /// </summary>
    /// <value>
    /// In [0, 1); defaults to 0.1. Rounded down to whole samples, so fewer than 10 samples hold none out, and the
    /// training set always keeps at least one sample. 0 trains on every sample.
    /// </value>
    public double ValidationFraction { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets how many epochs may pass without the held-out objective improving before training stops.
    /// </summary>
    /// <value>Defaults to 10. 0 turns early stopping off. It has no effect when no samples are held out.</value>
    public int EarlyStoppingPatience { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether the trained model is returned with the parameters of the epoch whose held-out objective
    /// was lowest, rather than those of the last epoch.
    /// </summary>
    /// <value>
    /// Defaults to <c>true</c>, as the builder's input/target path returns the model of its best step: when early
    /// stopping ends training, the last <see cref="EarlyStoppingPatience"/> epochs were the ones that did not improve.
    /// It has no effect when no samples are held out. Keeping the best parameters holds one copy of them in memory.
    /// </value>
    public bool RestoreBestModel { get; set; } = true;

    /// <summary>Throws when a setting is out of range.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A setting is out of range.</exception>
    internal void Validate()
    {
        if (Epochs < 1)
            throw new ArgumentOutOfRangeException(nameof(Epochs), Epochs, "Train for at least one epoch.");
        if (double.IsNaN(ValidationFraction) || ValidationFraction < 0 || ValidationFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(ValidationFraction), ValidationFraction,
                "The held-out fraction lies in [0, 1).");
        if (EarlyStoppingPatience < 0)
            throw new ArgumentOutOfRangeException(nameof(EarlyStoppingPatience), EarlyStoppingPatience,
                "Patience is a number of epochs, 0 to turn early stopping off.");
    }
}
