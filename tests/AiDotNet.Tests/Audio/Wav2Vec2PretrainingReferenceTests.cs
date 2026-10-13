using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Foundations;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// wav2vec 2.0 pre-training against transformers' Wav2Vec2ForPreTraining
/// (tools/reference-data/wav2vec2_pretraining_reference.py): the contrastive and diversity losses for a fixed mask and
/// fixed distractors, and their sum after one SGD step.
/// </summary>
public class Wav2Vec2PretrainingReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "wav2vec2_pretraining_reference.json";
        string relative = Path.Combine("Audio", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    [Fact(Timeout = 120000)]
    public async Task ContrastiveAndDiversityLosses_AndOneSgdStep_MatchTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        var layers = new List<LayerBase<double>>();
        var net = new Wav2Vec2Network<double>(AiDotNetEngine.Current, layers, new Wav2Vec2Configuration
        {
            ConvDims = new[] { 8, 8, 8 }, ConvKernels = new[] { 10, 3, 2 }, ConvStrides = new[] { 5, 2, 2 }, HiddenSize = 16, Layers = 2,
            Heads = 2, IntermediateSize = 32, PositionalConvKernel = 8, PositionalConvGroups = 4,
        });
        var pretraining = new Wav2Vec2Pretraining<double>(AiDotNetEngine.Current, layers, new Wav2Vec2PretrainingConfiguration
        {
            Groups = 2, EntriesPerGroup = 8, CodevectorDim = 8, ProjectedDim = 6, Negatives = 4,
        }, featureDim: 8, hiddenSize: 16);
        var state = f.GetProperty("state");
        double[] Read(string name, int[] shape)
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        }
        net.LoadTorchWeights((name, shape) => Read($"wav2vec2.{name}", shape));
        pretraining.LoadTorchWeights(Read);

        int frames = f.GetProperty("frames").GetInt32();
        var maskRows = f.GetProperty("mask").EnumerateArray().Select(r => r.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();
        var mask = new bool[2, frames];
        for (int b = 0; b < 2; b++)
            for (int t = 0; t < frames; t++) mask[b, t] = maskRows[b][t] == 1;
        var negativeRows = f.GetProperty("negatives").EnumerateArray()
            .Select(b => b.EnumerateArray().Select(t => t.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray()).ToArray();
        var negatives = new int[2, frames, 4];
        for (int b = 0; b < 2; b++)
            for (int t = 0; t < frames; t++)
                for (int j = 0; j < 4; j++) negatives[b, t, j] = negativeRows[b][t][j];
        var audio = new Tensor<double>(new[] { 2, 400 }, new Vector<double>(Values(f.GetProperty("audio"))));

        var engine = AiDotNetEngine.Current;
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Wav2Vec2Pretraining<double>.Losses losses;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var output = net.Forward(audio, training: false, new Random(0), mask: mask);
            losses = pretraining.Loss(output, negatives, training: false, new Random(0));
            // The reference's loss: contrastive + 0.1 · diversity (fairseq's feature penalty is not part of it).
            var referenceLoss = engine.TensorAdd(losses.Contrastive, engine.TensorMultiplyScalar(losses.Diversity, 0.1));
            grads = tape.ComputeGradients(referenceLoss, sources: parameters);
        }
        Assert.True(Math.Abs(f.GetProperty("contrastive").GetDouble() - losses.Contrastive[0]) < 1e-9,
            $"contrastive {f.GetProperty("contrastive").GetDouble()} vs {losses.Contrastive[0]}");
        Assert.True(Math.Abs(f.GetProperty("diversity").GetDouble() - losses.Diversity[0]) < 1e-9,
            $"diversity {f.GetProperty("diversity").GetDouble()} vs {losses.Diversity[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.1 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        var after = pretraining.Loss(net.Forward(audio, training: false, new Random(0), mask: mask), negatives, training: false, new Random(0));
        double afterLoss = after.Contrastive[0] + 0.1 * after.Diversity[0];
        Assert.True(Math.Abs(f.GetProperty("loss_after_sgd").GetDouble() - afterLoss) < 1e-8,
            $"after SGD {f.GetProperty("loss_after_sgd").GetDouble()} vs {afterLoss}");
    }
}
