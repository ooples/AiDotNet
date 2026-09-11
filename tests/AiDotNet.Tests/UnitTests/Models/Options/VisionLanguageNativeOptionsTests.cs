using System;
using System.Collections.Generic;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

public class VisionLanguageNativeOptionsTests
{
    public enum ModelFamily
    {
        VisionMamba,
        AudioVisualCorrespondence,
        AudioVisualEventLocalization,
        UnifiedMultimodal,
        Blip,
        Blip2,
        Flamingo,
        LLaVA,
        Gpt4Vision,
        ImageBind,
        VideoClip,
        Clip
    }

    public enum VisionDimension
    {
        ImageHeight,
        ImageWidth,
        PatchSize,
        Channels,
        ModelDimension,
        NumLayers,
        NumClasses,
        StateDimension
    }

    public enum AudioVisualDimension
    {
        EmbeddingDimension,
        AudioEmbeddingFullyConnectedWidth,
        AudioEmbeddingSize
    }

    public static IEnumerable<object[]> Families()
    {
        foreach (ModelFamily family in Enum.GetValues(typeof(ModelFamily)))
            yield return new object[] { family };
    }

    public static IEnumerable<object[]> PatchFamilies()
    {
        yield return new object[] { ModelFamily.Blip };
        yield return new object[] { ModelFamily.Blip2 };
        yield return new object[] { ModelFamily.Flamingo };
        yield return new object[] { ModelFamily.LLaVA };
        yield return new object[] { ModelFamily.Gpt4Vision };
        yield return new object[] { ModelFamily.ImageBind };
        yield return new object[] { ModelFamily.VideoClip };
    }

    public static IEnumerable<object[]> InvalidVisionDimensions()
    {
        foreach (VisionDimension dimension in Enum.GetValues(typeof(VisionDimension)))
        {
            yield return new object[] { dimension, 0 };
            yield return new object[] { dimension, -1 };
        }
    }

    [Theory]
    [MemberData(nameof(Families))]
    public void ShippedDefaults_ValidateWithoutAllocatingModels(ModelFamily family)
    {
        CreateValidator(family)();
    }

    [Theory]
    [MemberData(nameof(PatchFamilies))]
    public void PatchGeometry_RejectsZeroBeforeDivision(ModelFamily family)
    {
        var (options, validate) = CreatePatchOptions(family);
        validate();
        options.PatchSize = 0;
        AssertInvalid(validate, options.GetType(), nameof(options.PatchSize));
    }

    [Theory]
    [MemberData(nameof(PatchFamilies))]
    public void PatchGeometry_RejectsNegativePatch(ModelFamily family)
    {
        var (options, validate) = CreatePatchOptions(family);
        validate();
        options.PatchSize = -1;
        AssertInvalid(validate, options.GetType(), nameof(options.PatchSize));
    }

    [Theory]
    [MemberData(nameof(PatchFamilies))]
    public void PatchGeometry_PreservesEachModelsCropOrExactTilingContract(ModelFamily family)
    {
        var (options, validate) = CreatePatchOptions(family);
        validate();
        options.ImageSize = 31;
        options.PatchSize = 8;
        if (family == ModelFamily.Blip2)
            AssertInvalid(validate, options.GetType(), nameof(options.ImageSize));
        else
            validate();
    }

    [Theory]
    [MemberData(nameof(PatchFamilies))]
    public void PatchGeometry_RejectsAnImageWithNoCompletePatch(ModelFamily family)
    {
        var (options, validate) = CreatePatchOptions(family);
        validate();
        options.ImageSize = 4;
        options.PatchSize = 8;
        AssertInvalid(validate, options.GetType(), nameof(options.ImageSize));
    }

    [Theory]
    [MemberData(nameof(PatchFamilies))]
    public void PatchGeometry_AcceptsOnePatchAndExactTiling(ModelFamily family)
    {
        var (options, validate) = CreatePatchOptions(family);
        options.ImageSize = 8;
        options.PatchSize = 8;
        validate();
        options.ImageSize = 32;
        validate();
    }

    [Fact]
    public void Flamingo_DefaultPatchMatchesItsNativeArchitecture()
    {
        Assert.Equal(14, new FlamingoOptions().PatchSize);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    public void Flamingo_RejectsAStackWithoutGatedCrossAttention(int layers)
    {
        var options = new FlamingoOptions();
        options.Validate();
        options.NumLmLayers = layers;
        AssertInvalid(options.Validate, typeof(FlamingoOptions), nameof(options.NumLmLayers));
    }

    [Fact]
    public void Flamingo_AcceptsTheFirstCompleteCrossAttentionInterval()
    {
        new FlamingoOptions { NumLmLayers = 4 }.Validate();
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Flamingo_RejectsInvalidConsumedLearningRate(double rate)
    {
        var options = new FlamingoOptions();
        options.Validate();
        options.LearningRate = rate;
        AssertInvalid(options.Validate, typeof(FlamingoOptions), nameof(options.LearningRate));
    }

    [Fact]
    public void VisionMamba_DoesNotRequireLanguageOnlyDimensions()
    {
        var options = new VisionMambaOptions { VocabSize = 0, MaxSequenceLength = 0 };
        options.Validate();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void VisionMamba_InvalidClassCountNamesTheClassCount(int classes)
    {
        var options = new VisionMambaOptions();
        options.Validate();
        options.NumClasses = classes;
        AssertInvalid(options.Validate, typeof(VisionMambaOptions), nameof(options.NumClasses));
    }

    [Theory]
    [MemberData(nameof(InvalidVisionDimensions))]
    public void VisionMamba_ValidatesEachConsumedDimensionIndependently(VisionDimension dimension, int value)
    {
        var options = new VisionMambaOptions();
        options.Validate();
        string property;
        switch (dimension)
        {
            case VisionDimension.ImageHeight: options.ImageHeight = value; property = nameof(options.ImageHeight); break;
            case VisionDimension.ImageWidth: options.ImageWidth = value; property = nameof(options.ImageWidth); break;
            case VisionDimension.PatchSize: options.PatchSize = value; property = nameof(options.PatchSize); break;
            case VisionDimension.Channels: options.Channels = value; property = nameof(options.Channels); break;
            case VisionDimension.ModelDimension: options.ModelDimension = value; property = nameof(options.ModelDimension); break;
            case VisionDimension.NumLayers: options.NumLayers = value; property = nameof(options.NumLayers); break;
            case VisionDimension.NumClasses: options.NumClasses = value; property = nameof(options.NumClasses); break;
            case VisionDimension.StateDimension: options.StateDimension = value; property = nameof(options.StateDimension); break;
            default: throw new ArgumentOutOfRangeException(nameof(dimension));
        }
        AssertInvalid(options.Validate, typeof(VisionMambaOptions), property);
    }

    [Theory]
    [InlineData(17, 16, nameof(VisionMambaOptions.ImageHeight))]
    [InlineData(16, 17, nameof(VisionMambaOptions.ImageWidth))]
    public void VisionMamba_RectangularTilingNamesTheInvalidSide(int height, int width, string property)
    {
        var options = new VisionMambaOptions { ImageHeight = height, ImageWidth = width, PatchSize = 4 };
        AssertInvalid(options.Validate, typeof(VisionMambaOptions), property);
    }

    [Fact]
    public void AudioVisualOptions_DoNotRequireImageOrTextTokenGeometry()
    {
        new AudioVisualCorrespondenceOptions { ImageSize = 0, MaxSequenceLength = 0, PatchSize = 0, Channels = 0 }.Validate();
        new AudioVisualEventLocalizationOptions { ImageSize = 0, MaxSequenceLength = 0, PatchSize = 0, Channels = 0 }.Validate();
        new UnifiedMultimodalNetworkOptions { ImageSize = 0, PatchSize = 0, Channels = 0 }.Validate();
    }

    [Fact]
    public void AudioVisualEventLocalization_PreservesZeroEncoderLayerSupport()
    {
        new AudioVisualEventLocalizationOptions { NumEncoderLayers = 0 }.Validate();
    }

    [Theory]
    [InlineData(AudioVisualDimension.EmbeddingDimension, 0)]
    [InlineData(AudioVisualDimension.EmbeddingDimension, -1)]
    [InlineData(AudioVisualDimension.AudioEmbeddingFullyConnectedWidth, 0)]
    [InlineData(AudioVisualDimension.AudioEmbeddingFullyConnectedWidth, -1)]
    [InlineData(AudioVisualDimension.AudioEmbeddingSize, 0)]
    [InlineData(AudioVisualDimension.AudioEmbeddingSize, -1)]
    public void AudioVisualEventLocalization_ValidatesConsumedWidths(AudioVisualDimension dimension, int value)
    {
        var options = new AudioVisualEventLocalizationOptions();
        options.Validate();
        string property;
        switch (dimension)
        {
            case AudioVisualDimension.EmbeddingDimension:
                options.EmbeddingDimension = value;
                property = nameof(options.EmbeddingDimension);
                break;
            case AudioVisualDimension.AudioEmbeddingFullyConnectedWidth:
                options.AudioEmbeddingFullyConnectedWidth = value;
                property = nameof(options.AudioEmbeddingFullyConnectedWidth);
                break;
            case AudioVisualDimension.AudioEmbeddingSize:
                options.AudioEmbeddingSize = value;
                property = nameof(options.AudioEmbeddingSize);
                break;
            default: throw new ArgumentOutOfRangeException(nameof(dimension));
        }
        AssertInvalid(options.Validate, typeof(AudioVisualEventLocalizationOptions), property);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void AudioVisualEventLocalization_RejectsInvalidTimingAndLearningRate(double value)
    {
        var options = new AudioVisualEventLocalizationOptions();
        options.Validate();
        options.TemporalResolution = value;
        AssertInvalid(options.Validate, typeof(AudioVisualEventLocalizationOptions), nameof(options.TemporalResolution));
        options.TemporalResolution = 0.1;
        options.LearningRate = value;
        AssertInvalid(options.Validate, typeof(AudioVisualEventLocalizationOptions), nameof(options.LearningRate));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(9)]
    public void AudioVisualEventLocalization_RejectsWidthsIncompatibleWithItsFixedHeads(int width)
    {
        var options = new AudioVisualEventLocalizationOptions { EmbeddingDimension = width, NumEncoderLayers = 0 };
        AssertInvalid(options.Validate, typeof(AudioVisualEventLocalizationOptions), nameof(options.EmbeddingDimension));
    }

    [Fact]
    public void AudioVisualEventLocalization_RejectsNegativeDepthButAcceptsTheMinimumWidth()
    {
        var options = new AudioVisualEventLocalizationOptions { EmbeddingDimension = 8, NumEncoderLayers = 0 };
        options.Validate();
        options.NumEncoderLayers = -1;
        AssertInvalid(options.Validate, typeof(AudioVisualEventLocalizationOptions), nameof(options.NumEncoderLayers));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Correspondence_RejectsInvalidConsumedSampleRate(int sampleRate)
    {
        var options = new AudioVisualCorrespondenceOptions();
        options.Validate();
        options.AudioSampleRate = sampleRate;
        AssertInvalid(options.Validate, typeof(AudioVisualCorrespondenceOptions), nameof(options.AudioSampleRate));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Correspondence_RejectsInvalidConsumedFrameRate(double frameRate)
    {
        var options = new AudioVisualCorrespondenceOptions();
        options.Validate();
        options.VideoFrameRate = frameRate;
        AssertInvalid(options.Validate, typeof(AudioVisualCorrespondenceOptions), nameof(options.VideoFrameRate));
    }

    [Fact]
    public void Unified_PreservesZeroDepthAndRequiresItsActualTextContext()
    {
        var options = new UnifiedMultimodalNetworkOptions { NumTransformerLayers = 0 };
        options.Validate();
        options.NumTransformerLayers = -1;
        AssertInvalid(options.Validate, typeof(UnifiedMultimodalNetworkOptions), nameof(options.NumTransformerLayers));
        options.NumTransformerLayers = 0;
        options.MaxSequenceLength = 0;
        AssertInvalid(options.Validate, typeof(UnifiedMultimodalNetworkOptions), nameof(options.MaxSequenceLength));
    }

    private static void AssertInvalid(Action validate, Type optionsType, string property)
    {
        var error = Assert.ThrowsAny<ArgumentException>(validate);
        Assert.Equal("options", error.ParamName);
        Assert.Contains(optionsType.Name + "." + property, error.Message);
    }

    private static Action CreateValidator(ModelFamily family) => family switch
    {
        ModelFamily.VisionMamba => new VisionMambaOptions().Validate,
        ModelFamily.AudioVisualCorrespondence => new AudioVisualCorrespondenceOptions().Validate,
        ModelFamily.AudioVisualEventLocalization => new AudioVisualEventLocalizationOptions().Validate,
        ModelFamily.UnifiedMultimodal => new UnifiedMultimodalNetworkOptions().Validate,
        _ => CreatePatchOrClipValidator(family)
    };

    private static Action CreatePatchOrClipValidator(ModelFamily family)
    {
        if (family == ModelFamily.Clip)
            return new ClipOptions().Validate;
        return CreatePatchOptions(family).Validate;
    }

    private static (VisionLanguageModelOptions Options, Action Validate) CreatePatchOptions(ModelFamily family)
    {
        switch (family)
        {
            case ModelFamily.Blip:
                var blip = new BlipOptions();
                return (blip, blip.Validate);
            case ModelFamily.Blip2:
                var blip2 = new Blip2Options();
                return (blip2, blip2.Validate);
            case ModelFamily.Flamingo:
                var flamingo = new FlamingoOptions();
                return (flamingo, flamingo.Validate);
            case ModelFamily.LLaVA:
                var llava = new LLaVAOptions();
                return (llava, llava.Validate);
            case ModelFamily.Gpt4Vision:
                var gpt4Vision = new Gpt4VisionOptions();
                return (gpt4Vision, gpt4Vision.Validate);
            case ModelFamily.ImageBind:
                var imageBind = new ImageBindOptions();
                return (imageBind, imageBind.Validate);
            case ModelFamily.VideoClip:
                var videoClip = new VideoCLIPOptions();
                return (videoClip, videoClip.Validate);
            default:
                throw new ArgumentOutOfRangeException(nameof(family), family, "This family has no native patch encoder.");
        }
    }
}
