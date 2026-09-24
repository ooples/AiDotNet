using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

public sealed class VisionLanguageNativeValidationContractTests
{
    public static IEnumerable<object[]> InvalidDimensions()
    {
        // Names here identify reflected public properties; typeof/nameof bind every case to
        // the actual options API. No string chooses a production validation policy.
        var properties = new (Type Type, string[] Properties)[]
        {
            (typeof(BlipOptions), new[] { nameof(BlipOptions.VocabSize), nameof(BlipOptions.HiddenDim),
                nameof(BlipOptions.NumEncoderLayers), nameof(BlipOptions.NumDecoderLayers), nameof(BlipOptions.NumHeads), nameof(BlipOptions.MlpDim) }),
            (typeof(ImageBindOptions), new[] { nameof(ImageBindOptions.AudioSampleRate), nameof(ImageBindOptions.AudioMaxDuration),
                nameof(ImageBindOptions.ImuTimesteps), nameof(ImageBindOptions.NumVideoFrames), nameof(ImageBindOptions.VocabSize),
                nameof(ImageBindOptions.HiddenDim), nameof(ImageBindOptions.NumEncoderLayers), nameof(ImageBindOptions.NumHeads) }),
            (typeof(LLaVAOptions), new[] { nameof(LLaVAOptions.VocabSize), nameof(LLaVAOptions.VisionDim),
                nameof(LLaVAOptions.VisionLayers), nameof(LLaVAOptions.NumLmLayers), nameof(LLaVAOptions.NumHeads) }),
            (typeof(VideoCLIPOptions), new[] { nameof(VideoCLIPOptions.NumFrames), nameof(VideoCLIPOptions.VisionDim),
                nameof(VideoCLIPOptions.TextHiddenDim), nameof(VideoCLIPOptions.NumFrameEncoderLayers), nameof(VideoCLIPOptions.NumTemporalLayers),
                nameof(VideoCLIPOptions.NumTextLayers), nameof(VideoCLIPOptions.NumHeads), nameof(VideoCLIPOptions.VocabSize) }),
            (typeof(FlamingoOptions), new[] { nameof(FlamingoOptions.NumPerceiverTokens), nameof(FlamingoOptions.MaxImagesInContext),
                nameof(FlamingoOptions.VisionDim), nameof(FlamingoOptions.LmHiddenDim), nameof(FlamingoOptions.VisionLayers),
                nameof(FlamingoOptions.NumLmLayers), nameof(FlamingoOptions.NumHeads), nameof(FlamingoOptions.VocabSize), nameof(FlamingoOptions.NumPerceiverLayers) }),
            (typeof(Gpt4VisionOptions), new[] { nameof(Gpt4VisionOptions.ContextWindowSize), nameof(Gpt4VisionOptions.MaxImagesPerRequest),
                nameof(Gpt4VisionOptions.HiddenDim), nameof(Gpt4VisionOptions.VisionDim), nameof(Gpt4VisionOptions.VisionLayers),
                nameof(Gpt4VisionOptions.NumLmLayers), nameof(Gpt4VisionOptions.NumHeads), nameof(Gpt4VisionOptions.VocabSize) }),
            (typeof(Blip2Options), new[] { nameof(Blip2Options.QformerHiddenDim), nameof(Blip2Options.VisionDim),
                nameof(Blip2Options.LmHiddenDim), nameof(Blip2Options.NumQformerLayers), nameof(Blip2Options.NumQueryTokens),
                nameof(Blip2Options.NumHeads), nameof(Blip2Options.NumLmDecoderLayers), nameof(Blip2Options.VocabSize) })
        };
        foreach (var group in properties)
            foreach (string property in group.Properties)
                foreach (int invalid in new[] { 0, -1 })
                    yield return new object[] { group.Type, property, invalid };
    }

    [Theory]
    [MemberData(nameof(InvalidDimensions))]
    public void ConsumedNativeDimensionsAreRejectedBeforeLayerConstruction(Type optionsType, string propertyName, int value)
    {
        object options = Activator.CreateInstance(optionsType) ?? throw new InvalidOperationException("Options construction failed.");
        InvokeValidate(options); // Every unchanged default remains valid.
        Property(optionsType, propertyName).SetValue(options, value);
        AssertInvalid(options, propertyName);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void VideoClip_NominalFrameRateMustStillBeFiniteAndPositive(double value) =>
        AssertInvalid(new VideoCLIPOptions { FrameRate = value }, nameof(VideoCLIPOptions.FrameRate));

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Llava_RejectsMissingVisionEncoderIdentifier(string? value)
    {
        var options = new LLaVAOptions();
        Property(typeof(LLaVAOptions), nameof(LLaVAOptions.VisionEncoderType)).SetValue(options, value);
        AssertInvalid(options, nameof(LLaVAOptions.VisionEncoderType));
    }

    [Fact]
    public void VideoClip_RejectsAnUndefinedTemporalAggregation() =>
        AssertInvalid(new VideoCLIPOptions { TemporalAggregation = (AiDotNet.Enums.TemporalAggregationType)int.MaxValue },
            nameof(VideoCLIPOptions.TemporalAggregation));

    private static PropertyInfo Property(Type type, string name) => type.GetProperty(name)
        ?? throw new InvalidOperationException($"Missing actual property {type.Name}.{name}.");

    private static void InvokeValidate(object options)
    {
        var method = options.GetType().GetMethod("Validate", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("Missing actual options validator.");
        method.Invoke(options, Array.Empty<object>());
    }

    private static void AssertInvalid(object options, string propertyName)
    {
        var invocation = Assert.Throws<TargetInvocationException>(() => InvokeValidate(options));
        var exception = Assert.IsType<ArgumentException>(invocation.InnerException);
        Assert.Equal("options", exception.ParamName);
        Assert.Contains(propertyName, exception.Message);
    }
}
