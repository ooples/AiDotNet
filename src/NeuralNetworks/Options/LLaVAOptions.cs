using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the LLaVANeuralNetwork.
/// </summary>
public class LLaVAOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LLaVAOptions"/> class carrying
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
    public LLaVAOptions()
    {
        EmbeddingDimension = 4096;
        MaxSequenceLength = 2048;
        ImageSize = 336;
        Channels = 3;
        PatchSize = 14;
        VocabSize = 32000;
        VisionDim = 1024;
        VisionLayers = 24;
        NumLmLayers = 32;
        NumHeads = 16;
        LanguageModelBackbone = LanguageModelBackbone.LLaMA;
        VisionEncoderType = "clip-vit-l";
    }


    /// <summary>
    /// Gets or sets the number of native language-model decoder blocks.
    /// </summary>
    /// <value>A count of decoder blocks. Defaults to 32, preserving the previous implementation's constructor default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These blocks process text conditioned on the projected image
    /// features. More blocks deepen the native language model and increase its computation.
    /// This option does not change the architecture of a language model loaded from ONNX.</para>
    /// </remarks>
    public int NumLmLayers { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.Text | ValidationRequirements.PatchGeometry);
        Require(VocabSize, nameof(VocabSize));
        Require(VisionDim, nameof(VisionDim));
        Require(VisionLayers, nameof(VisionLayers));
        Require(NumLmLayers, nameof(NumLmLayers));
        Require(NumHeads, nameof(NumHeads));
        if (string.IsNullOrWhiteSpace(VisionEncoderType))
            throw new ArgumentException($"{GetType().Name}.{nameof(VisionEncoderType)} must identify a vision encoder.", OptionsParameterName);
    }

    internal void ValidateOnnx()
    {
        ValidateInputs(InputValidationRequirements.Text | InputValidationRequirements.Image);
        var defaults = new LLaVAOptions();
        RequireNativeDefaultForOnnx(Channels, defaults.Channels, nameof(Channels));
        RequireNativeDefaultForOnnx(PatchSize, defaults.PatchSize, nameof(PatchSize));
        RequireNativeDefaultForOnnx(VocabSize, defaults.VocabSize, nameof(VocabSize));
        RequireNativeDefaultForOnnx(VisionDim, defaults.VisionDim, nameof(VisionDim));
        RequireNativeDefaultForOnnx(VisionLayers, defaults.VisionLayers, nameof(VisionLayers));
        RequireNativeDefaultForOnnx(NumLmLayers, defaults.NumLmLayers, nameof(NumLmLayers));
        RequireNativeDefaultForOnnx(NumHeads, defaults.NumHeads, nameof(NumHeads));
        if (string.IsNullOrWhiteSpace(VisionEncoderType))
            throw new ArgumentException($"{GetType().Name}.{nameof(VisionEncoderType)} must identify a vision encoder.", OptionsParameterName);
    }

    /// <summary>
    /// Gets or sets the language model backbone.
    /// </summary>
    public LanguageModelBackbone LanguageModelBackbone { get; set; }

    /// <summary>
    /// Gets or sets the vision encoder type.
    /// </summary>
    public string VisionEncoderType { get; set; }
}
