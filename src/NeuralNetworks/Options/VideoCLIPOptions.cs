using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the VideoCLIPNeuralNetwork.
/// </summary>
public class VideoCLIPOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VideoCLIPOptions"/> class carrying
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
    public VideoCLIPOptions()
    {
        NumFrames = 8;
        FrameRate = 1.0;
        EmbeddingDimension = 512;
        MaxSequenceLength = 77;
        ImageSize = 224;
        Channels = 3;
        PatchSize = 16;
        VocabSize = 49408;
        VisionDim = 768;
        TextHiddenDim = 512;
        NumFrameEncoderLayers = 12;
        NumTemporalLayers = 4;
        NumTextLayers = 12;
        NumHeads = 12;
        TemporalAggregation = TemporalAggregationType.TemporalTransformer;
    }


    /// <summary>
    /// Gets or sets the number of supplied video frames selected for each clip embedding.
    /// </summary>
    /// <value>A positive count of frames per clip. Defaults to 8, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The encoder selects frames uniformly from the list you provide.
    /// If there are too few, it repeats the last frame. More frames require more encoding work;
    /// this count does not decode a video file or determine the source video's playback speed.</para>
    /// </remarks>
    public int NumFrames { get; set; }

    /// <summary>
    /// Gets or sets the nominal sampling rate associated with the supplied video frames.
    /// </summary>
    /// <value>A rate in frames per second. Defaults to 1.0, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> 1.0 describes one sampled frame per second of source video.
    /// The current model exposes this as metadata; it does not resample frames or change playback speed.
    /// Frame selection uses <see cref="NumFrames"/> and the supplied list, so callers prepare the sampling cadence.</para>
    /// </remarks>
    public double FrameRate { get; set; }

    /// <summary>
    /// Gets or sets the number of features in each native text-token representation.
    /// </summary>
    /// <value>The native text width in features. Defaults to 512, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each caption token is represented by this many numbers inside
    /// the native text encoder. This is distinct from the shared output embedding size.
    /// The setting does not resize a text encoder loaded from an ONNX graph.</para>
    /// </remarks>
    public int TextHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the number of native transformer blocks applied to each selected frame.
    /// </summary>
    /// <value>A count of per-frame encoder blocks. Defaults to 12, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These blocks extract features from each still image before
    /// frames are combined. Adding blocks increases native network depth and work;
    /// it does not change the layers stored in a loaded ONNX graph.</para>
    /// </remarks>
    public int NumFrameEncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the number of native temporal-transformer blocks that combine frame features.
    /// </summary>
    /// <value>A count of temporal encoder blocks. Defaults to 4, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These blocks process information across frames rather than
    /// within one image when temporal-transformer aggregation is used. They are native network
    /// settings, not a way to replace the temporal architecture in an ONNX video encoder.</para>
    /// </remarks>
    public int NumTemporalLayers { get; set; }

    /// <summary>
    /// Gets or sets the number of native text-transformer blocks used to encode a caption.
    /// </summary>
    /// <value>A count of text encoder blocks. Defaults to 12, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These blocks let caption tokens exchange information before
    /// projection into the shared video-text embedding. More blocks add native computation;
    /// the depth of a loaded ONNX text encoder remains defined by that graph.</para>
    /// </remarks>
    public int NumTextLayers { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.Text | ValidationRequirements.PatchGeometry);
        Require(NumFrames, nameof(NumFrames));
        Require(FrameRate, nameof(FrameRate));
        Require(VisionDim, nameof(VisionDim));
        Require(TextHiddenDim, nameof(TextHiddenDim));
        Require(NumFrameEncoderLayers, nameof(NumFrameEncoderLayers));
        Require(NumTemporalLayers, nameof(NumTemporalLayers));
        Require(NumTextLayers, nameof(NumTextLayers));
        Require(NumHeads, nameof(NumHeads));
        Require(VocabSize, nameof(VocabSize));
        if (!Enum.IsDefined(typeof(TemporalAggregationType), TemporalAggregation))
            throw new ArgumentException($"{GetType().Name}.{nameof(TemporalAggregation)} is not a defined aggregation mode.", OptionsParameterName);
    }

    internal void ValidateOnnx()
    {
        ValidateInputs(InputValidationRequirements.Text | InputValidationRequirements.Image);
        Require(Channels, nameof(Channels));
        Require(NumFrames, nameof(NumFrames));
        var defaults = new VideoCLIPOptions();
        RequireNativeDefaultForOnnx(PatchSize, defaults.PatchSize, nameof(PatchSize));
        RequireNativeDefaultForOnnx(VocabSize, defaults.VocabSize, nameof(VocabSize));
        RequireNativeDefaultForOnnx(VisionDim, defaults.VisionDim, nameof(VisionDim));
        RequireNativeDefaultForOnnx(TextHiddenDim, defaults.TextHiddenDim, nameof(TextHiddenDim));
        RequireNativeDefaultForOnnx(NumFrameEncoderLayers, defaults.NumFrameEncoderLayers, nameof(NumFrameEncoderLayers));
        RequireNativeDefaultForOnnx(NumTemporalLayers, defaults.NumTemporalLayers, nameof(NumTemporalLayers));
        RequireNativeDefaultForOnnx(NumTextLayers, defaults.NumTextLayers, nameof(NumTextLayers));
        RequireNativeDefaultForOnnx(NumHeads, defaults.NumHeads, nameof(NumHeads));
        RequireNativeDefaultForOnnx(TemporalAggregation, defaults.TemporalAggregation, nameof(TemporalAggregation));
        RequireNativeDefaultForOnnx(HiddenDim, defaults.HiddenDim, nameof(HiddenDim));
        RequireNativeDefaultForOnnx(NumEncoderLayers, defaults.NumEncoderLayers, nameof(NumEncoderLayers));
        RequireNativeDefaultForOnnx(VisionLayers, defaults.VisionLayers, nameof(VisionLayers));
    }

    /// <summary>
    /// Gets or sets the temporal aggregation.
    /// </summary>
    public TemporalAggregationType TemporalAggregation { get; set; }
}
