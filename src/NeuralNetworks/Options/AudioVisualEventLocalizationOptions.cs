using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the AudioVisualEventLocalizationNetwork.
/// </summary>
public class AudioVisualEventLocalizationOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AudioVisualEventLocalizationOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090.
    /// </para>
    /// </remarks>
    public AudioVisualEventLocalizationOptions()
    {
        EmbeddingDimension = 512; // DEFAULT_EMBEDDING_DIM
        TemporalResolution = 0.1; // DEFAULT_TEMPORAL_RESOLUTION
        NumEncoderLayers = 6;
        AudioEmbeddingFullyConnectedWidth = VGGishAudioEmbedding<double>.PaperFullyConnectedWidth;
        AudioEmbeddingSize = VGGishAudioEmbedding<double>.PaperEmbeddingSize;
    }

    /// <summary>Copies the timing, optimizer rate, audio widths and inherited configuration.</summary>
    /// <param name="other">The source options.</param>
    /// <exception cref="ArgumentNullException">The source is null.</exception>
    public AudioVisualEventLocalizationOptions(AudioVisualEventLocalizationOptions other) : base(other)
    {
        LearningRate = other.LearningRate;
        TemporalResolution = other.TemporalResolution;
        AudioEmbeddingFullyConnectedWidth = other.AudioEmbeddingFullyConnectedWidth;
        AudioEmbeddingSize = other.AudioEmbeddingSize;
    }

    /// <summary>
    /// Gets or sets the learning rate for gradient descent parameter updates. Default: 0.001.
    /// </summary>
    /// <value>A finite positive step scale; defaults to 0.001, preserving the previous property initializer.</value>
    /// <remarks><para><b>For Beginners:</b> This scales native training updates. Larger steps
    /// change weights faster but can make optimization unstable; prediction does not train the model.</para></remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the temporal resolution.
    /// </summary>
    /// <value>A positive interval in seconds; defaults to 0.1, preserving the former DEFAULT_TEMPORAL_RESOLUTION constant.</value>
    /// <remarks><para><b>For Beginners:</b> This describes the spacing of event time bins.
    /// Smaller intervals distinguish more closely timed events; callers must provide aligned audio/video inputs.</para></remarks>
    public double TemporalResolution { get; set; }

    /// <summary>
    /// Gets or sets the audio embedding fully connected width.
    /// </summary>
    /// <value>A positive feature count; defaults to <see cref="VGGishAudioEmbedding{T}.PaperFullyConnectedWidth"/>, the existing VGGish projection width.</value>
    /// <remarks><para><b>For Beginners:</b> This sizes the native audio projection's internal
    /// fully connected representation. Wider projections allocate more weights before the audio embedding is produced.</para></remarks>
    public int AudioEmbeddingFullyConnectedWidth { get; set; }

    /// <summary>
    /// Gets or sets the audio embedding size.
    /// </summary>
    /// <value>A positive feature count; defaults to <see cref="VGGishAudioEmbedding{T}.PaperEmbeddingSize"/>, the existing VGGish output width.</value>
    /// <remarks><para><b>For Beginners:</b> Each audio segment is represented by this many
    /// values before it is combined with visual features. This is distinct from the projection's internal width.</para></remarks>
    public int AudioEmbeddingSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.None);
        Require(TemporalResolution, nameof(TemporalResolution));
        Require(AudioEmbeddingFullyConnectedWidth, nameof(AudioEmbeddingFullyConnectedWidth));
        Require(AudioEmbeddingSize, nameof(AudioEmbeddingSize));
        Require(LearningRate, nameof(LearningRate));
        // The actual dual-stream factory supports zero optional encoder-attention layers.
        if (NumEncoderLayers < 0)
            throw new ArgumentException($"{GetType().Name}.{nameof(NumEncoderLayers)} must be non-negative.", OptionsParameterName);
        // Temporal and fusion attention always use eight heads, even with zero encoders.
        if (EmbeddingDimension % 8 != 0)
            throw new ArgumentException($"{GetType().Name}.{nameof(EmbeddingDimension)} must be divisible by the fixed attention head count (8).", OptionsParameterName);
    }
}
