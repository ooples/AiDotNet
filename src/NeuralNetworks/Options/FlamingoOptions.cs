using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the FlamingoNeuralNetwork.
/// </summary>
public class FlamingoOptions : VisionLanguageModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FlamingoOptions"/> class carrying
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
    public FlamingoOptions()
    {
        EmbeddingDimension = 768;
        MaxSequenceLength = 2048;
        ImageSize = 224;
        PatchSize = 14;
        NumPerceiverTokens = 64;
        MaxImagesInContext = 5;
        Channels = 3;
        VisionDim = 1024;
        LmHiddenDim = 2048;
        VisionLayers = 24;
        NumLmLayers = 32;
        NumHeads = 16;
        VocabSize = 32000;
        NumPerceiverLayers = 6;
        LearningRate = 1e-3;
        LanguageModelBackbone = LanguageModelBackbone.Chinchilla;
    }


    /// <summary>
    /// Gets or sets the num perceiver tokens.
    /// </summary>
    public int NumPerceiverTokens { get; set; }

    /// <summary>
    /// Gets or sets the max images in context.
    /// </summary>
    public int MaxImagesInContext { get; set; }

    /// <summary>
    /// Gets or sets the lm hidden dim.
    /// </summary>
    public int LmHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the number of language-model transformer layers.
    /// </summary>
    /// <value>At least 4; defaults to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The native architecture inserts an image cross-attention
    /// gate once per four language layers. A shorter stack has no gate and cannot condition
    /// its generated text on the image, so validation rejects it.</para>
    /// </remarks>
    public int NumLmLayers { get; set; }

    /// <summary>
    /// Gets or sets the num perceiver layers.
    /// </summary>
    public int NumPerceiverLayers { get; set; }

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    public double LearningRate { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(ValidationRequirements.Text | ValidationRequirements.PatchGeometry);
        Require(LearningRate, nameof(LearningRate));
        Require(NumPerceiverTokens, nameof(NumPerceiverTokens));
        Require(MaxImagesInContext, nameof(MaxImagesInContext));
        Require(VisionDim, nameof(VisionDim));
        Require(LmHiddenDim, nameof(LmHiddenDim));
        Require(VisionLayers, nameof(VisionLayers));
        Require(NumHeads, nameof(NumHeads));
        Require(VocabSize, nameof(VocabSize));
        Require(NumPerceiverLayers, nameof(NumPerceiverLayers));
        if (NumLmLayers < 4)
        {
            throw new ArgumentException(
                $"{GetType().Name}.{nameof(NumLmLayers)} must be at least 4 so the language model contains gated cross-attention to its image features.",
                OptionsParameterName);
        }
    }

    internal void ValidateOnnx()
    {
        ValidateInputs(InputValidationRequirements.Text | InputValidationRequirements.Image);
        Require(VisionDim, nameof(VisionDim));
        Require(MaxImagesInContext, nameof(MaxImagesInContext));
        var defaults = new FlamingoOptions();
        RequireNativeDefaultForOnnx(Channels, defaults.Channels, nameof(Channels));
        RequireNativeDefaultForOnnx(PatchSize, defaults.PatchSize, nameof(PatchSize));
        RequireNativeDefaultForOnnx(VocabSize, defaults.VocabSize, nameof(VocabSize));
        RequireNativeDefaultForOnnx(LmHiddenDim, defaults.LmHiddenDim, nameof(LmHiddenDim));
        RequireNativeDefaultForOnnx(VisionLayers, defaults.VisionLayers, nameof(VisionLayers));
        RequireNativeDefaultForOnnx(NumLmLayers, defaults.NumLmLayers, nameof(NumLmLayers));
        RequireNativeDefaultForOnnx(NumHeads, defaults.NumHeads, nameof(NumHeads));
        RequireNativeDefaultForOnnx(NumPerceiverLayers, defaults.NumPerceiverLayers, nameof(NumPerceiverLayers));
        RequireNativeDefaultForOnnx(NumPerceiverTokens, defaults.NumPerceiverTokens, nameof(NumPerceiverTokens));
        RequireNativeDefaultForOnnx(LearningRate, defaults.LearningRate, nameof(LearningRate));
    }

    /// <summary>
    /// Gets or sets the language model backbone.
    /// </summary>
    public LanguageModelBackbone LanguageModelBackbone { get; set; }
}
