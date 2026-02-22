namespace AiDotNet.VisionLanguage.Editing;

/// <summary>
/// Configuration options for Emu Edit: precise image editing via recognition and generation tasks.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the EmuEdit model. Default values follow the original paper settings.</para>
/// </remarks>
public class EmuEditOptions : EditingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EmuEditOptions(EmuEditOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        DecoderDim = other.DecoderDim;
        NumVisionLayers = other.NumVisionLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxSequenceLength = other.MaxSequenceLength;
        MaxGenerationLength = other.MaxGenerationLength;
        DropoutRate = other.DropoutRate;
        ArchitectureType = other.ArchitectureType;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        OutputImageSize = other.OutputImageSize;
        NumDiffusionSteps = other.NumDiffusionSteps;
        GuidanceScale = other.GuidanceScale;
        EnablePreciseEditing = other.EnablePreciseEditing;
    }

    public EmuEditOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to use recognition-guided precise editing.</summary>
    public bool EnablePreciseEditing { get; set; } = true;
}
