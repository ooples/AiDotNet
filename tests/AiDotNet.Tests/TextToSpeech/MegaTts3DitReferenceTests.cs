using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MegaTTS 3's latent diffusion transformer against the release's own code (tools/reference-data/megatts3_dit_reference.py)
/// on a tiny model: the phone encoding, one CFG-combined velocity, the training prediction and target, and inference with
/// the sway schedule and amo sampler on the reference's recorded draws.
/// </summary>
public class MegaTts3DitReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "megatts3_dit_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
    private static int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();

    private static int[,] Rows(params int[][] rows)
    {
        var a = new int[rows.Length, rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[i].Length; j++) a[i, j] = rows[i][j];
        return a;
    }

    private static void Close(JToken expected, Tensor<double> actual, string what)
    {
        var e = Doubles(expected);
        Assert.Equal(e.Length, actual.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
    }

    [Fact(Timeout = 300000)]
    public async Task EncodingVelocitySamplingAndTraining_MatchTheRelease()
    {
        await Task.Yield();
        var f = Reference();
        var engine = AiDotNetEngine.Current;
        int width = (int)f["width"]!, latent = (int)f["latent"]!;
        var dit = new MegaTts3Dit<double>(engine, new List<LayerBase<double>>(), new MegaTts3DitConfiguration
        {
            Width = width, Layers = (int)f["layers"]!, Heads = (int)f["heads"]!, LatentChannels = latent, LocalConditionDim = (int)f["local"]!,
            ContextMaskDim = (int)f["ctx_mask_dim"]!, PhoneEncoderLayers = (int)f["layers"]!, PhoneEncoderHeads = (int)f["heads"]!,
        });
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        dit.LoadTorch(engine, name =>
        {
            used.Add(name);
            return Doubles(state[name] ?? throw new KeyNotFoundException(name));
        });
        Assert.Empty(state.Properties().Select(p => p.Name).Where(n => !used.Contains(n)));

        using var _ = new NoGradScope<double>();
        var phone = Ints(f["phone"]!);
        var tone = Ints(f["tone"]!);
        var mel2ph = Ints(f["mel2ph"]!);
        Close(f["x_ling_phone"]!, dit.EncodePhones(Rows(phone), Rows(tone)), "phone encoding");

        // The release's three-way CFG batch: speaker and text, text, neither (cfg mask tokens 301 and 31).
        var masked = Enumerable.Repeat(301, phone.Length).ToArray();
        var maskedTone = Enumerable.Repeat(31, tone.Length).ToArray();
        var linguistic = dit.LinguisticFeatures(Rows(phone, phone, masked), Rows(tone, tone, maskedTone), Rows(mel2ph, mel2ph, mel2ph));
        int frames = linguistic.Shape[1];
        var prompt = Doubles(f["latent_prompt"]!);
        int promptFrames = prompt.Length / latent;
        var context = new Tensor<double>(new[] { 3, frames, latent });
        var contextMask = new Tensor<double>(new[] { 3, frames, 1 });
        for (int b = 0; b < 3; b++)
            for (int t = 0; t < promptFrames; t++)
            {
                contextMask[b, t, 0] = 1;
                // The prompt latent conditions the first row only (ctx_feature[1:] = 0).
                if (b == 0)
                    for (int ch = 0; ch < latent; ch++) context[b, t, ch] = prompt[t * latent + ch];
            }
        var local = dit.LocalCondition(context, contextMask);
        var probe = new Tensor<double>(new[] { 1, frames, latent }, new Vector<double>(Doubles(f["x_probe"]!)));
        Close(f["velocity"]!, dit.GuidedVelocity(probe, contextMask, local, linguistic, 0.3, 1.6, 2.5), "velocity");

        var draws = f["draws"]!.Select(d => Doubles(d)).ToList();
        int next = 0;
        var generated = dit.Sample(contextMask, local, linguistic, 4, 1.6, 2.5, shape => new Tensor<double>(shape, new Vector<double>(draws[next++])));
        Assert.Equal(draws.Count, next);
        Close(f["generated"]!, generated, "generated");

        // Training: xt = t x1 + (1 − t) x0 masked over the prompt, the velocity at t, the target x1 − x0.
        var x0 = Doubles(f["x0"]!);
        var x1 = Doubles(f["x1"]!);
        const double time = 0.37;
        var trainMask = new Tensor<double>(new[] { 1, frames, 1 });
        var trainContext = new Tensor<double>(new[] { 1, frames, latent });
        var noisy = new Tensor<double>(new[] { 1, frames, latent });
        var target = new Tensor<double>(new[] { 1, frames, latent });
        for (int t = 0; t < frames; t++)
        {
            if (t < 6) trainMask[0, t, 0] = 1;
            for (int ch = 0; ch < latent; ch++)
            {
                int i = t * latent + ch;
                if (t < 6) trainContext[0, t, ch] = x1[i];
                noisy[0, t, ch] = time * x1[i] + (1 - time) * x0[i];
                target[0, t, ch] = x1[i] - x0[i];
            }
        }
        var trainLinguistic = dit.LinguisticFeatures(Rows(phone), Rows(tone), Rows(mel2ph));
        var trainPred = dit.Velocity(noisy, trainMask, dit.LocalCondition(trainContext, trainMask), trainLinguistic, dit.TimeEmbedding(new[] { time }));
        Close(f["train_pred"]!, trainPred, "training prediction");
        Close(f["train_target"]!, target, "training target");
    }
}
