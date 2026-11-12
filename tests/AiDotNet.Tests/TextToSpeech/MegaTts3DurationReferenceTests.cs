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
/// MegaTTS 3's duration predictor against the release's own code (tools/reference-data/megatts3_duration_reference.py)
/// on a tiny model: the prompt pass teacher-forced on the prompt's durations, the continuation from its incremental state
/// (each step's prediction and the rounded durations), and a training forward under the causal mask.
/// </summary>
public class MegaTts3DurationReferenceTests
{
    [Fact(Timeout = 300000)]
    public async Task PromptContinuationAndTrainingForward_MatchTheRelease()
    {
        await Task.Yield();
        const string fileName = "megatts3_duration_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
        int[] Ints(string key) => f[key]!.Select(v => (int)v).ToArray();
        var engine = AiDotNetEngine.Current;
        var model = new MegaTts3Duration<double>(engine, new List<LayerBase<double>>(), new MegaTts3DurationConfiguration
        {
            EncoderWidth = 16, EncoderFilter = 32, EncoderHeads = 2, EncoderLayers = 2, DecoderWidth = 32, DecoderLayers = 2,
        });
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        model.LoadTorch(engine, name =>
        {
            used.Add(name);
            return Doubles(state[name]!);
        });
        Assert.Empty(state.Properties().Select(p => p.Name).Where(n => !used.Contains(n) && !n.StartsWith("char_")));

        void Close(JToken expected, IReadOnlyList<double> actual, string what)
        {
            var e = Doubles(expected);
            Assert.Equal(e.Length, actual.Count);
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
        }
        using var _ = new NoGradScope<double>();
        var context = Ints("ctx_dur");
        int p = context.Length;
        var reference = model.EncodeLinguistic(Ints("ph_ref"), Ints("tone_ref"));
        var cache = new MegaTts3DurationCache<double>();
        model.Infer(reference, cache, context, null, Enumerable.Range(0, p).ToArray());
        var phones = Ints("ph_pred");
        var (durations, steps) = model.Infer(model.EncodeLinguistic(phones, Ints("tone_pred")), cache, null, context[p - 1],
            Enumerable.Range(p, phones.Length).ToArray());
        Close(f["steps"]!, steps, "steps");
        Assert.Equal(Ints("raw"), durations);
        var previous = new[] { 129 }.Concat(context.Take(p - 1)).ToArray();
        Close(f["full"]!, model.Forward(reference, previous, Enumerable.Range(0, p).ToArray()).ToArray(), "full");
    }
}
