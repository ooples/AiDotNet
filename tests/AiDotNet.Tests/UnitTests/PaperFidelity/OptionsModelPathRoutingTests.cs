using System;
using System.IO;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Generation;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class OptionsModelPathRoutingTests
{
    private static string MissingModel() => Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".onnx");

    private static NeuralNetworkArchitecture<double> Architecture() => new(
        InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 320, outputSize: 16);

    [Fact]
    public void W2vBert_UsesTheOptionsModelPath()
    {
        string path = MissingModel();
        Assert.Throws<FileNotFoundException>(() => new W2vBert<double>(Architecture(), new W2vBertOptions { ModelPath = path }));
    }

    [Fact]
    public void Conformer_UsesTheOptionsModelPath()
    {
        string path = MissingModel();
        Assert.Throws<FileNotFoundException>(() => new Conformer<double>(Architecture(), new ConformerOptions { ModelPath = path }));
    }

    [Fact]
    public void SpeechTokenizer_UsesTheOptionsModelPath()
    {
        string path = MissingModel();
        Assert.Throws<FileNotFoundException>(() => new SpeechTokenizer<double>(Architecture(), new SpeechTokenizerOptions { ModelPath = path }));
    }
}
