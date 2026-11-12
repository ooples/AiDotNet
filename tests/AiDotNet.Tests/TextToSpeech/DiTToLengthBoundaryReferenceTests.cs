using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class DiTToLengthBoundaryReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public void ExpectedRemaining_HonorsAZeroLengthPosterior(int promptLength)
    {
        using var document = Read();
        var options = Options();
        var predictor = new DiTToLengthPredictor<double>(AiDotNetEngine.Current,
            new List<LayerBase<double>>(), options, new Random(7));
        Configure(predictor, options, Posterior(document.RootElement, 0));
        Assert.Equal(0, predictor.ExpectedRemaining(new Tensor<double>(new[] { 3, 8 }),
            promptLength == 0 ? null : new Tensor<double>(new[] { promptLength, 8 })));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void ActualSynthesis_AddsPromptLengthToThePredictedRemainingLength(int index)
    {
        using var document = Read();
        var root = document.RootElement;
        var example = root.GetProperty("synthesis_boundaries")[index];
        int promptLength = example.GetProperty("prompt_length").GetInt32();
        int selected = example.GetProperty("selected").GetInt32();
        var posterior = Posterior(root, selected);
        Assert.Equal(selected, posterior.GetProperty("expected_remaining").GetInt32());
        var options = Options();
        using var inference = new NoGradScope<double>();
        var model = new DiTToTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        model.SetTrainingMode(false);
        Configure(model.LengthPredictor!, options, posterior);
        var prompt = promptLength == 0 ? null : new Tensor<double>(new[] { promptLength, options.LatentDim });
        var waveform = model.SynthesizeFromEncoding(new Tensor<double>(new[] { 3, options.TextHiddenDim }), prompt);
        int samplesPerLatent = options.MelVaeDownsampleFactor * options.BigVganUpsampleRates.Aggregate(1, (a, b) => a * b);
        Assert.Equal(example.GetProperty("expected_total").GetInt32() * samplesPerLatent, waveform.Length);
    }

    private static JsonElement Posterior(JsonElement root, int selected) => root.GetProperty("boundary_posteriors")
        .EnumerateArray().Single(value => value.GetProperty("selected").GetInt32() == selected);

    private static void Configure(DiTToLengthPredictor<double> predictor, DiTToTTSOptions options, JsonElement reference)
        => TorchParameters.Linear(AiDotNetEngine.Current, predictor.Classifier, options.LengthHiddenDim,
            options.MaxLatentLength + 1, new double[(options.MaxLatentLength + 1) * options.LengthHiddenDim],
            reference.GetProperty("logits").EnumerateArray().Select(x => x.GetDouble()).ToArray());

    private static DiTToTTSOptions Options()
    {
        var options = DiTToTTSPaperTests.TinyOptions();
        options.MaxLatentLength = 8;
        options.LengthTopK = 3;
        options.BigVganUpsampleRates = new[] { 2, 2 };
        options.BigVganUpsampleKernels = new[] { 4, 4 };
        return options;
    }

    private static JsonDocument Read() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "ditto_length_reference.json")));
}
