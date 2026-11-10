using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.VoiceCloning;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Chatterbox Turbo/Nano against resemble-ai/chatterbox (tools/reference-data/chatterbox_turbo_reference.py): the
/// GPT-2 T3's logits, embeddings, next-token loss and one SGD step; the mean-flow decoder's field and basic_euler;
/// tts_turbo.punc_norm; with <c>CHATTERBOX_TURBO_DIR</c>, the release's GPT-2 tokenization.</summary>
public class ChatterboxTurboReferenceTests
{
    private static JObject Fixture() => WhisperSpeechT2SReferenceTests.Load("chatterbox_turbo_reference.json");
    private static double[] Values(JToken token) => token.Select(v => (double)v).ToArray();

    private static void Close(double[] expected, Tensor<double> actual, string what, double tolerance = 1e-9)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual.GetFlat(i):R}.");
    }

    [Fact]
    public void PunctuationNormalization_MatchesTurboPuncNorm()
    {
        foreach (var row in Fixture()["punc_norm"]!)
            Assert.Equal((string)row["normalized"]!, Chatterbox<double>.TurboNormalizeText((string)row["text"]!));
    }

    [Fact]
    public void T3_LogitsLossAndSgdStep_MatchTheGpt2Release()
    {
        var f = (JObject)Fixture()["t3"]!;
        var options = new ChatterboxOptions
        {
            Variant = ChatterboxVariant.Turbo, VocabSize = 32, SpeechVocabularySize = 20, LLMDim = 16, NumLLMLayers = 2, NumHeads = 2,
            KeyValueHeads = 2, LmFeedForwardDim = 64, MaxTextLength = 16, MaxSpeechTokens = 16, T3SpeakerEmbeddingDim = 6,
            TextStartToken = 30, TextStopToken = 31, SpeechStartToken = 18, SpeechStopToken = 19,
        };
        var layers = new List<LayerBase<double>>();
        var model = new ChatterboxT3<double>(AiDotNetEngine.Current, layers, options, new Random(1));
        foreach (var layer in layers) layer.SetTrainingMode(false);
        var weights = (JObject)f["weights"]!;
        var shapes = (JObject)f["weight_shapes"]!;
        model.LoadTorchWeights(TorchTensorSource.From((name, shape) =>
        {
            var values = Values(weights[name] ?? throw new InvalidDataException($"No tensor {name}."));
            var stored = shapes[name]!.Select(v => (int)v).ToArray();
            if (name == "tfmr.wpe.weight")
            {
                // The tiny GPT-2 has 64 positions; the release table has 8196. Rows past 64 are never read.
                Assert.Equal(new[] { ChatterboxT3<double>.Gpt2Positions, stored[1] }, shape);
                Array.Resize(ref values, shape[0] * shape[1]);
                return values;
            }
            if (shape.Length > 0) Assert.Equal(stored, shape);
            return values;
        }));
        var condition = new ChatterboxT3Condition<double>(new Tensor<double>(new[] { 6 }, new Vector<double>(Values(f["speaker"]!))),
            Values(f["prompt"]!).Select(v => (int)v).ToArray(), 0);
        int[] text = Values(f["text"]!).Select(v => (int)v).ToArray(), speech = Values(f["speech"]!).Select(v => (int)v).ToArray();
        Tensor<double> Ids(int[] ids) => new(new[] { 1, ids.Length }, new Vector<double>(ids.Select(v => (double)v).ToArray()));

        Close(Values(f["embeddings"]!), model.PrepareEmbeddings(condition, Ids(text), Ids(speech), false), "embeddings");
        var logits = model.TeacherForcedLogits(text, speech, condition);
        Close(Values(f["text_logits"]!), logits.Text, "text logits");
        Close(Values(f["speech_logits"]!), logits.Speech, "speech logits");

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        IReadOnlyDictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var loss = model.Loss(text.Skip(1).Take(text.Length - 2).ToArray(), speech.Skip(1).Take(speech.Length - 2).ToArray(), condition);
            Assert.True(Math.Abs((double)f["next_token_loss"]! - loss[0]) < 1e-9, $"loss {loss[0]:R}");
            gradients = tape.ComputeGradients(loss, sources: parameters);
        }
        foreach (var parameter in parameters)
        {
            if (!gradients.TryGetValue(parameter, out var gradient)) continue;
            for (int i = 0; i < parameter.Length; i++) parameter.SetFlat(i, parameter.GetFlat(i) - .01 * gradient.GetFlat(i));
            AiDotNetEngine.Current.InvalidatePersistentTensor(parameter);
        }
        logits = model.TeacherForcedLogits(text, speech, condition);
        Close(Values(f["after_sgd_text"]!), logits.Text, "text logits after the step");
        Close(Values(f["after_sgd_speech"]!), logits.Speech, "speech logits after the step");
    }

    [Fact]
    public void MeanFlowDecoder_FieldAndBasicEuler_MatchTheRelease()
    {
        var f = (JObject)Fixture()["decoder"]!;
        var c = f["config"]!;
        var decoder = new MatchaDecoder<double>(AiDotNetEngine.Current, (int)c["in_channels"]!, (int)c["out_channels"]!,
            c["channels"]!.Select(v => (int)v).ToArray(), (double)c["dropout"]!, (int)c["attention_head_dim"]!, (int)c["num_heads"]!,
            (int)c["num_mid_blocks"]!, blocksPerLevel: (int)c["n_blocks"]!, feedForward: MatchaFeedForward.Gelu, biasMask: true,
            causal: true, meanFlow: true);
        var state = (JObject)f["state"]!;
        decoder.LoadTorchWeights(TorchTensorSource.From((string name) => state[name] is { } v ? Values(v) : null), "");
        int mel = (int)f["mel"]!, frames = (int)f["frames"]!;
        double[] x = Values(f["x"]!), mu = Values(f["mu"]!), spks = Values(f["spks"]!), cond = Values(f["cond"]!);
        Tensor<double> Stacked(double[] state_)
        {
            var stacked = new Tensor<double>(new[] { 1, 4 * mel, frames });
            for (int ch = 0; ch < mel; ch++)
                for (int t = 0; t < frames; t++)
                {
                    stacked[0, ch, t] = state_[ch * frames + t];
                    stacked[0, mel + ch, t] = mu[ch * frames + t];
                    stacked[0, 2 * mel + ch, t] = spks[ch];
                    stacked[0, 3 * mel + ch, t] = cond[ch * frames + t];
                }
            return stacked;
        }
        var mask = new Tensor<double>(new[] { frames });
        for (int i = 0; i < frames; i++) mask[i] = 1;
        double[] ChannelsFirst(Tensor<double> field)                                          // [frames, mel] -> [mel, frames]
        {
            var values = new double[mel * frames];
            for (int ch = 0; ch < mel; ch++)
                for (int t = 0; t < frames; t++) values[ch * frames + t] = field[t, ch];
            return values;
        }
        var fieldValues = ChannelsFirst(decoder.EstimateChannels(Stacked(x), (double)f["t"]!, mask, 0, (double)f["r"]!));
        Close(Values(f["field"]!), new Tensor<double>(new[] { fieldValues.Length }, new Vector<double>(fieldValues)), "field");

        // basic_euler over linspace(0, 1, 3): each step reads its end time.
        var y = (double[])x.Clone();
        double[] span = { 0, 0.5, 1 };
        for (int s = 0; s < 2; s++)
        {
            var velocity = ChannelsFirst(decoder.EstimateChannels(Stacked(y), span[s], mask, 0, span[s + 1]));
            for (int i = 0; i < y.Length; i++) y[i] += (span[s + 1] - span[s]) * velocity[i];
        }
        Close(Values(f["sample"]!), new Tensor<double>(new[] { y.Length }, new Vector<double>(y)), "basic_euler");
        Assert.Throws<ArgumentException>(() => decoder.EstimateChannels(Stacked(x), 0.3, mask));
    }

    [Fact]
    public void Tokenize_MatchesTheReleaseGpt2Tokenizer()
    {
        string directory = ChatterboxTurboTokenizerFixture.DirectoryPath;
        var options = ChatterboxFixture.Options();
        options.Variant = ChatterboxVariant.Turbo;
        options.TokenizerPath = directory;
        var model = new Chatterbox<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(AiDotNet.Enums.InputType.OneDimensional,
            AiDotNet.Enums.NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);
        foreach (var row in Fixture()["tokens"]!)
            Assert.Equal(row["ids"]!.Values<int>().ToArray(), model.Tokenize(Chatterbox<double>.TurboNormalizeText((string)row["text"]!)));
    }
}
