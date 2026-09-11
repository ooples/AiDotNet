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
        EmbeddingDimension = 512; // DEFAULT_EMBEDDING_DIM
        TemporalResolution = 0.1; // DEFAULT_TEMPORAL_RESOLUTION
        NumEncoderLayers = 6;
        AudioEmbeddingFullyConnectedWidth = VGGishAudioEmbedding<double>.PaperFullyConnectedWidth;
        AudioEmbeddingSize = VGGishAudioEmbedding<double>.PaperEmbeddingSize;
        EmbeddingDimension = 512; // DEFAULT_EMBEDDING_DIM
        TemporalResolution = 0.1; // DEFAULT_TEMPORAL_RESOLUTION
        NumEncoderLayers = 6;
        AudioEmbeddingFullyConnectedWidth = VGGishAudioEmbedding<double>.PaperFullyConnectedWidth;
        AudioEmbeddingSize = VGGishAudioEmbedding<double>.PaperEmbeddingSize;
    }

    /// <summary>
    /// Gets or sets the learning rate for gradient descent parameter updates. Default: 0.001.
    /// </summary>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the temporal resolution.
    /// </summary>
    public double TemporalResolution { get; set; }

    /// <summary>
    /// Gets or sets the audio embedding fully connected width.
    /// </summary>
    public int AudioEmbeddingFullyConnectedWidth { get; set; }

    /// <summary>
    /// Gets or sets the audio embedding size.
    /// </summary>
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
