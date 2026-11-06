using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.VisionLanguage.Generative;
using AiDotNet.VisionLanguage.InstructionTuned;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

public class NativePerceiverTokenInputTests
{
    [Theory]
    [InlineData("OpenFlamingo")]
    [InlineData("IDEFICS")]
    [InlineData("IDEFICS2")]
    [InlineData("IDEFICS3")]
    [InlineData("MPLUGOwl")]
    [InlineData("MPLUGOwl2")]
    [InlineData("MPLUGOwl3")]
    [InlineData("QwenVL")]
    [InlineData("Qwen2VL")]
    [InlineData("Qwen25VL")]
    [InlineData("Qwen3VL")]
    public void NativeTokens_PredictAndTrainingDoNotApplyImageNormalization(string name)
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 4);
        using NeuralNetworkBase<double> model = name switch
        {
            "OpenFlamingo" => new OpenFlamingo<double>(architecture, Bounded(new OpenFlamingoOptions())),
            "IDEFICS" => new IDEFICS<double>(architecture, Bounded(new IDEFICSOptions())),
            "IDEFICS2" => new IDEFICS2<double>(architecture, Bounded(new IDEFICS2Options())),
            "IDEFICS3" => new IDEFICS3<double>(architecture, Bounded(new IDEFICS3Options())),
            "MPLUGOwl" => new MPLUGOwl<double>(architecture, Bounded(new MPLUGOwlOptions())),
            "MPLUGOwl2" => new MPLUGOwl2<double>(architecture, Bounded(new MPLUGOwl2Options())),
            "MPLUGOwl3" => new MPLUGOwl3<double>(architecture, Bounded(new MPLUGOwl3Options())),
            "QwenVL" => new QwenVL<double>(architecture, Bounded(new QwenVLOptions())),
            "Qwen2VL" => new Qwen2VL<double>(architecture, Bounded(new Qwen2VLOptions())),
            "Qwen25VL" => new Qwen25VL<double>(architecture, Bounded(new Qwen25VLOptions())),
            "Qwen3VL" => new Qwen3VL<double>(architecture, Bounded(new Qwen3VLOptions())),
            _ => throw new ArgumentOutOfRangeException(nameof(name))
        };
        var input = new Tensor<double>(new[] { 1, 4, 8 });
        for (int i = 0; i < input.Length; i++) input[i] = 0.1 + 0.01 * i;
        var output = model.Predict(input);
        Assert.Equal(new[] { 1, 4, 8 }, output.Shape.ToArray());
        var before = model.GetParameters().ToArray();
        model.Train(input, new Tensor<double>(output.Shape.ToArray()));
        Assert.False(before.SequenceEqual(model.GetParameters().ToArray()), name + " must update its vision stack.");
        Assert.All(model.Predict(input).ToArray(), value => Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
    }

    private static TOptions Bounded<TOptions>(TOptions options) where TOptions : ModelOptions
    {
        var type = options.GetType();
        foreach (string property in new[] { "VisionDim", "DecoderDim", "PerceiverDim", "AbstractorDim", "ResamplerDim" })
            type.GetProperty(property)?.SetValue(options, 8);
        foreach (string property in new[] { "NumVisionLayers", "NumDecoderLayers", "NumPerceiverLayers", "NumAbstractorLayers", "NumResamplerLayers" })
            type.GetProperty(property)?.SetValue(options, 1);
        foreach (string property in new[] { "NumHeads", "NumPerceiverHeads", "NumAbstractorHeads", "NumResamplerHeads" })
            type.GetProperty(property)?.SetValue(options, 2);
        type.GetProperty("DropoutRate")!.SetValue(options, 0.0);
        return options;
    }
}
