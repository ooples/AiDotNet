using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class VoiceCraftGenerationReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void CompleteGenerationControlFlow_MatchesOriginalAuthorSampler(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "voicecraft_generation_reference.json")));
        var expected = document.RootElement.GetProperty("cases")[index];
        var options = new VoiceCraftOptions
        {
            LLMDim = 8, NumHeads = 2, NumLLMLayers = 1, FeedForwardDim = 16,
            TextVocabularySize = 16, CodebookSize = 8, NumCodebooks = 4,
            TextEmbeddingDropout = 0, AudioEmbeddingDropout = 0,
            TextPositionDropout = 0, AudioPositionDropout = 0, TransformerDropout = 0,
            Temperature = 1, SilenceTokens = new[] { 1 },
            StopSilenceRepetition = expected.GetProperty("repetition").GetInt32(),
            TopK = expected.GetProperty("top_k").GetInt32(),
            TopP = expected.GetProperty("top_p").GetDouble()
        };
        var layers = new List<LayerBase<double>>();
        var model = new VoiceCraftNetwork<double>(AiDotNetEngine.Current, layers, options);
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var heads = ((DenseLayer<double> Hidden, DenseLayer<double> Output)[])typeof(VoiceCraftNetwork<double>)
            .GetField("_predictionHeads", flags)!.GetValue(model)!;
        var logits = expected.GetProperty("logits").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        for (int q = 0; q < options.NumCodebooks; q++)
            TorchParameters.Linear(AiDotNetEngine.Current, heads[q].Output, options.CodebookSize / 2,
                options.AudioVocabularySize, new double[options.CodebookSize / 2 * options.AudioVocabularySize],
                logits.Skip(q * options.AudioVocabularySize).Take(options.AudioVocabularySize).ToArray());
        foreach (var layer in layers) layer.SetTrainingMode(false);
        var arguments = new List<object?>
        {
            new[] { 1 }, new[] { Enumerable.Repeat(options.EmptyToken, options.NumCodebooks).ToArray() },
            null, expected.GetProperty("minimum").GetInt32(), expected.GetProperty("maximum").GetInt32(),
            new FixedUniform(expected.GetProperty("uniform").GetDouble())
        };
        var generate = typeof(VoiceCraftNetwork<double>).GetMethod("Generate", flags)!;
        if (generate.GetParameters().Length == 7) arguments.Add(expected.GetProperty("tts").GetBoolean());
        var actual = (int[,])generate.Invoke(model, arguments.ToArray())!;
        Assert.Equal(expected.GetProperty("frames").GetInt32(), actual.GetLength(1));
        for (int q = 0; q < options.NumCodebooks; q++)
            Assert.Equal(expected.GetProperty("codes")[q].EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                Enumerable.Range(0, actual.GetLength(1)).Select(frame => actual[q, frame]).ToArray());
        var delayed = typeof(VoiceCraftNetwork<double>).GetMethod("GenerateDelayed", flags);
        Assert.NotNull(delayed);
        var sequence = ((IReadOnlyList<int[]> Rows, int Frames))delayed!.Invoke(model, arguments.ToArray())!;
        Assert.Equal(expected.GetProperty("frames").GetInt32(), sequence.Frames);
        Assert.Equal(expected.GetProperty("rows").GetArrayLength(), sequence.Rows.Count);
        for (int row = 0; row < sequence.Rows.Count; row++)
            Assert.Equal(expected.GetProperty("rows")[row].EnumerateArray().Select(x => x.GetInt32()).ToArray(), sequence.Rows[row]);
    }

    private sealed class FixedUniform : Random
    {
        private readonly double _value;
        internal FixedUniform(double value) => _value = value;
        public override double NextDouble() => _value;
    }
}
