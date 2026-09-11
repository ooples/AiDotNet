using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class VisionLanguageNativeConstructorValidationTests
{
    public VisionLanguageNativeConstructorValidationTests() => TestModuleInitializer.EnsureInitialized();
    public enum ModelCase { Blip, Blip2, Flamingo, Gpt4Vision, ImageBind, Llava, VideoClip }

    [Theory]
    [InlineData(ModelCase.Blip, nameof(BlipOptions.VocabSize))]
    [InlineData(ModelCase.Blip2, nameof(Blip2Options.NumQueryTokens))]
    [InlineData(ModelCase.Flamingo, nameof(FlamingoOptions.NumPerceiverTokens))]
    [InlineData(ModelCase.Gpt4Vision, nameof(Gpt4VisionOptions.ContextWindowSize))]
    [InlineData(ModelCase.ImageBind, nameof(ImageBindOptions.AudioSampleRate))]
    [InlineData(ModelCase.Llava, nameof(LLaVAOptions.VisionEncoderType))]
    [InlineData(ModelCase.VideoClip, nameof(VideoCLIPOptions.NumFrames))]
    public void ActualNativeConstructorsRejectInvalidConfigurationBeforeBuildingLayers(ModelCase modelCase, string property)
    {
        var architecture = new NeuralNetworkArchitecture<float>(InputType.ThreeDimensional,
            NeuralNetworkTaskType.ImageClassification, inputDepth: 3, inputHeight: 16, inputWidth: 16, outputSize: 4);
        var exception = Assert.Throws<ArgumentException>(() =>
        {
            using IDisposable model = modelCase switch
            {
                ModelCase.Blip => new BlipNeuralNetwork<float>(architecture, new BlipOptions { VocabSize = 0 }),
                ModelCase.Blip2 => new Blip2NeuralNetwork<float>(architecture, new Blip2Options { NumQueryTokens = 0 }),
                ModelCase.Flamingo => new FlamingoNeuralNetwork<float>(architecture, new FlamingoOptions { NumPerceiverTokens = 0 }),
                ModelCase.Gpt4Vision => new Gpt4VisionNeuralNetwork<float>(architecture,
                    ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), new Gpt4VisionOptions { ContextWindowSize = 0 }),
                ModelCase.ImageBind => new ImageBindNeuralNetwork<float>(architecture, new ImageBindOptions { AudioSampleRate = 0 }),
                ModelCase.Llava => new LLaVANeuralNetwork<float>(architecture, new LLaVAOptions { VisionEncoderType = " " }),
                ModelCase.VideoClip => new VideoCLIPNeuralNetwork<float>(architecture, new VideoCLIPOptions { NumFrames = 0 }),
                _ => throw new ArgumentOutOfRangeException(nameof(modelCase))
            };
        });
        Assert.Equal("options", exception.ParamName);
        Assert.Contains(property, exception.Message);
    }
}
