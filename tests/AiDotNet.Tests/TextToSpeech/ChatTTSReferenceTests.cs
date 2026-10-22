using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// ChatTTS against 2noise/ChatTTS's own modules (<c>tools/reference-data/chattts_reference.py</c>): the embeddings with a
/// speaker applied, Hugging Face's <c>LlamaModel</c> hidden states, the weight-normalized text and code heads, the
/// training objective before and after an SGD step, the DVAE's codes (vector-quantize-pytorch 1.14's grouped residual
/// FSQ) and decoded mel, and the hidden-state Decoder's mel.
/// </summary>
public class ChatTTSReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "chattts_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    private static (ChatTTS<double> Model, JObject Fixture) Load()
    {
        var f = Reference();
        var stack = f["stack"]!;
        int hidden = (int)f["hidden_size"]!, dvae = (int)f["dvae_dim"]!, stackHidden = (int)stack["hidden"]!, layers = (int)stack["n_layer"]!,
            bottleneck = (int)stack["bn_dim"]!;
        var options = new ChatTTSOptions
        {
            HiddenDim = hidden, NumHeads = (int)f["heads"]!, NumDecoderLayers = (int)f["layers"]!, IntermediateSize = (int)f["intermediate"]!,
            FsqLevels = f["levels"]!.Select(v => (int)v).ToArray(), FsqGroups = (int)f["groups"]!, FsqResiduals = (int)f["residuals"]!,
            DvaeDim = dvae, DvaeEncoderHidden = stackHidden, DvaeDecoderHidden = stackHidden, DvaeLayers = layers, DvaeBottleneck = bottleneck,
            DecoderDim = hidden / 2, DecoderHidden = stackHidden, DecoderLayers = layers, DecoderBottleneck = bottleneck,
            VocosDim = 8, VocosIntermediate = 16, VocosLayers = 1,
        };
        var model = new ChatTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), options);

        string path = Path.Combine(Path.GetTempPath(), $"chattts_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            Func<string, int[], double[]> Reader(string prefix) => (name, shape) =>
            {
                var tensor = weights[prefix + name];
                Assert.Equal(shape.Aggregate(1, (a, b) => a * b), tensor.Length);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            };
            (model.GptNetwork ?? throw new InvalidOperationException("native model")).LoadTorchWeights(Reader("gpt."), prefix: "");
            model.LoadEmbedWeights(Reader("embed."));
            (model.DvaeEncoder ?? throw new InvalidOperationException("native model")).LoadTorchWeights(Reader("dvae."), "");
            (model.DvaeDecoder ?? throw new InvalidOperationException("native model")).LoadTorchWeights(Reader("dvae."), "");
            (model.HiddenStateDecoder ?? throw new InvalidOperationException("native model")).LoadTorchWeights(Reader("decoder."), "");
        }
        finally
        {
            File.Delete(path);
        }
        return (model, f);
    }

    private static void AssertClose(JToken expected, Tensor<double> actual, string what)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(values[i])),
                $"{what}[{i}]: reference {values[i]:R}, port {actual[i]:R}.");
    }

    private static (List<int[]> Rows, List<bool> TextMask) Sequence(JObject f)
    {
        var text = f["text"]!.Select(v => (int)v).ToList();
        var frames = f["frames"]!.Select(r => r.Select(v => (int)v).ToArray()).ToList();
        var rows = text.Select(t => new[] { t, t, t, t }).Concat(frames).Concat(new[] { new[] { 9, 9, 9, 9 } }).ToList();
        var mask = text.Select(_ => true).Concat(Enumerable.Repeat(false, frames.Count + 1)).ToList();
        return (rows, mask);
    }

    private static Tensor<double> Vector(JToken values)
    {
        var data = values.Select(v => (double)v).ToArray();
        return new Tensor<double>(new[] { data.Length }, new Vector<double>(data));
    }

    private static TtsTrainingSample<double> Sample(JObject f)
    {
        var text = f["text"]!.Select(v => (double)v).ToArray();
        var frames = f["frames"]!.Select(r => r.Select(v => (double)v).ToArray()).ToArray();
        var codes = new Tensor<double>(new[] { frames.Length, 4 });
        for (int t = 0; t < frames.Length; t++) for (int q = 0; q < 4; q++) codes[t, q] = frames[t][q];
        return new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { text.Length }, new Vector<double>(text)),
            CodecTokens = codes,
            SpeakerReference = Vector(f["speaker"]!),
        };
    }

    [Fact(Timeout = 120000)]
    public async Task GptEmbeddingsHiddenStatesAndHeads_MatchTheReference()
    {
        await Task.Yield();
        var (model, f) = Load();
        var (rows, mask) = Sequence(f);
        var speaker = Vector(f["speaker"]!);
        AssertClose(f["embeddings"]!, model.Embed(rows, mask, speaker), "embeddings");
        var hidden = model.Hidden(rows, mask, speaker);
        AssertClose(f["hidden"]!, hidden, "hidden");
        AssertClose(f["text_logits"]!, model.TextLogits(hidden), "text logits");
        var codeLogits = Enumerable.Range(0, 4).Select(q => model.CodeLogits(q, hidden)).ToArray();
        AssertClose(f["code_logits"]!, AiDotNetEngine.Current.TensorConcatenate(codeLogits, 0), "code logits");
    }

    [Fact(Timeout = 120000)]
    public async Task Objective_AndOneSgdStep_MatchTheReference()
    {
        await Task.Yield();
        var (model, f) = Load();
        var objective = model.BuildObjective(Sample(f));
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.GptLayers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Tensor<double> loss;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            loss = objective();
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double expected = (double)f["loss"]!;
        Assert.True(Math.Abs(expected - loss[0]) <= 1e-9 * expected, $"loss: reference {expected:R}, port {loss[0]:R}.");
        double rate = (double)f["learning_rate"]!;
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= rate * g[i];
            AiDotNetEngine.Current.InvalidatePersistentTensor(p);
        }
        double after = (double)f["loss_after_sgd"]!;
        double actual = model.BuildObjective(Sample(f))()[0];
        Assert.True(Math.Abs(after - actual) <= 1e-9 * after, $"loss after the step: reference {after:R}, port {actual:R}.");
    }

    [Fact(Timeout = 120000)]
    public async Task DvaeCodesAndMels_MatchTheReference()
    {
        await Task.Yield();
        var (model, f) = Load();
        var codes = model.EncodeCodes(Vector(f["wave"]!));
        var expected = f["codes"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        Assert.Equal(expected.Length, codes.Length);
        for (int t = 0; t < expected.Length; t++) Assert.Equal(expected[t], codes[t]);
        AssertClose(f["dvae_mel"]!, model.MelFromCodes(codes), "DVAE mel");

        var (rows, mask) = Sequence(f);
        AssertClose(f["hidden_mel"]!, model.MelFromHidden(model.Hidden(rows, mask, Vector(f["speaker"]!))), "hidden-state mel");
    }
}
