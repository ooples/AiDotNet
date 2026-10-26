using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MegaTTS 3's inference glue against the release's own functions (tools/reference-data/megatts3_pipeline_reference.py):
/// make_dur_prompt's duration tokens, dur_pred's frame-to-phone map for a first and a final chunk (given the disturbance's
/// draws), and prepare_inputs_for_dit's three classifier-free-guidance rows, latent context and mask.
/// </summary>
public class MegaTts3PipelineReferenceTests
{
    internal static MegaTTS3Options TinyOptions() => new()
    {
        HiddenDim = 16, DitLayers = 1, DitHeads = 2, LatentChannels = 4, LocalConditionDim = 8, ContextMaskDim = 4,
        PhoneEncoderLayers = 1, PhoneEncoderHeads = 2,
        DurationEncoderWidth = 16, DurationEncoderFilter = 32, DurationEncoderHeads = 2, DurationEncoderLayers = 2, DurationWidth = 32,
        DurationLayers = 2, DurationHeads = 8,
        AlignerWidth = 16, AlignerHeads = 2, AlignerLayers = 1,
        G2pVocabSize = 1200, G2pHidden = 32, G2pIntermediate = 64, G2pLayers = 1, G2pHeads = 4, G2pKeyValueHeads = 2,
        // The release's 960 samples a latent frame (the 0.4 s loudness block needs speech of real length), few channels.
        VaeEncoderRatios = new[] { 8, 10, 12 }, VaeEncoderFilters = 2, VaeEncoderDimension = 16, VaeUpsamplerChannels = 8, VaeLatentUpsampling = 4,
        VaeGeneratorFilters = 2, VaeGeneratorRatios = new[] { 5, 4, 4, 3 },
    };

    private static JObject Load(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    [Fact(Timeout = 300000)]
    public async Task DurationsAndTransformerInputs_MatchTheRelease()
    {
        await Task.Yield();
        var f = Load("megatts3_pipeline_reference.json");
        var duration = (JObject)Load("megatts3_duration_reference.json")["state"]!;
        int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
        var model = new MegaTTS3<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), TinyOptions());
        model.Duration.LoadTorch(AiDotNetEngine.Current, name => duration[name]!.Select(v => (double)v).ToArray());

        using var _ = new NoGradScope<double>();
        var mel2phRef = Ints(f["mel2ph_ref"]!);
        var latentValues = f["latent"]!.Select(v => (double)v).ToArray();
        var latent = new Tensor<double>(new[] { 1, latentValues.Length / 4, 4 }, new Vector<double>(latentValues));
        var prompt = model.PromptFrom(Ints(f["ph_ref"]!), Ints(f["tone_ref"]!), mel2phRef, latent, 0);
        Assert.Equal(Ints(f["ctx"]!), prompt.DurationTokens);

        var draws = f["draws"]!.SelectMany(d => d.Select(v => (double)v)).ToArray();
        int next = 0;
        var chunks = (JArray)f["chunks"]!;
        for (int i = 0; i < chunks.Count; i++)
        {
            var c = chunks[i];
            var phones = Ints(c["ph"]!);
            var tones = Ints(c["tone"]!);
            var mel2ph = model.PredictMel2Ph(prompt, phones, tones, i == 0, i == chunks.Count - 1, () => draws[next++]);
            Assert.Equal(Ints(c["mel2ph"]!), mel2ph);
            var (ph, tone, map, context, mask) = model.DitInputs(prompt, mel2ph, phones, tones);
            Assert.Equal(c["phone"]!.SelectMany(r => r.Select(v => (int)v)), ph.Cast<int>());
            Assert.Equal(c["tone_rows"]!.SelectMany(r => r.Select(v => (int)v)), tone.Cast<int>());
            Assert.Equal(c["dur"]!.SelectMany(r => r.Select(v => (int)v)), map.Cast<int>());
            Assert.Equal(c["ctx_mask"]!.Select(v => (double)v), mask.ToArray());
            var expected = c["lat_ctx"]!.Select(v => (double)v).ToArray();
            var actual = context.ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (int k = 0; k < expected.Length; k++) Assert.True(Math.Abs(expected[k] - actual[k]) <= 1e-12, $"lat_ctx[{k}]");
        }
        Assert.Equal(draws.Length, next);
    }
}
