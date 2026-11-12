using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// OuteTTS end to end against the release's own code (tools/reference-data/outetts_model_reference.py), a tiny Qwen3
/// model with OuteTTS-1.0-0.6B's tokenizer and a tiny DAC hopping 320 samples: the completion prompt's tokens, greedy
/// generation with the windowed repetition penalty, the completion-only training loss on the bundled speaker's training
/// prompt, and a speaker profile from timed words.
/// </summary>
public class OuteTTSModelTests
{
    private static JObject Reference()
    {
        const string fileName = "outetts_model_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static OuteTTS<double> Model(JObject f, int maxLength)
    {
        var lm = f["lm_config"]!;
        var dac = f["dac_config"]!;
        var model = new OuteTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            new OuteTTSOptions
            {
                Backbone = OuteTTSBackbone.Qwen3_0_6B, VocabSize = (int)lm["vocab_size"]!, HiddenDim = (int)lm["hidden_size"]!, Layers = (int)lm["num_hidden_layers"]!,
                Heads = (int)lm["num_attention_heads"]!, KeyValueHeads = (int)lm["num_key_value_heads"]!, HeadDim = (int)lm["head_dim"]!,
                Intermediate = (int)lm["intermediate_size"]!, RmsEpsilon = (double)lm["rms_norm_eps"]!, RopeTheta = (double)lm["rope_theta"]!, RopeScalingFactor = 0,
                QueryKeyNorm = true, MaxLength = maxLength, Temperature = 0,
                CodecEncoderDim = (int)dac["encoder_dim"]!, CodecEncoderRates = dac["encoder_rates"]!.Select(v => (int)v).ToArray(), CodecLatentDim = (int)dac["latent_dim"]!,
                CodecDecoderDim = (int)dac["decoder_dim"]!, CodecDecoderRates = dac["decoder_rates"]!.Select(v => (int)v).ToArray(), CodecCodebookDim = (int)dac["codebook_dim"]!,
                AlignerMels = 80, AlignerWidth = 16, AlignerHeads = 2, AlignerEncoderLayers = 1, AlignerDecoderLayers = 1,
            });
        var rows = f["embedding_rows"]!.Select(v => (int)v).ToArray();
        var bytes = Convert.FromBase64String((string)f["embedding_values"]!);
        int width = (int)lm["hidden_size"]!, vocab = (int)lm["vocab_size"]!;
        var embedding = new double[vocab * width];
        for (int r = 0; r < rows.Length; r++)
            for (int c = 0; c < width; c++) embedding[rows[r] * width + c] = BitConverter.ToSingle(bytes, 4 * (r * width + c));
        var state = (JObject)f["lm_state"]!;
        model.LanguageModel.LoadTorchWeights((name, shape) =>
            name == "model.embed_tokens.weight" ? embedding : state[name]!.Select(v => (double)v).ToArray());
        var dacState = (JObject)f["dac_state"]!;
        model.LoadCodec(name => dacState[name]?.Select(v => (double)v).ToArray(), dacState.Properties().Select(p => p.Name));
        return model;
    }

    [Fact(Timeout = 600000)]
    public async Task GreedyGeneration_MatchesTheRelease()
    {
        await Task.Yield();
        var f = Reference();
        var promptIds = f["prompt_ids"]!.Select(v => (int)v).ToArray();
        var expected = f["generated"]!.Select(v => (int)v).ToList();
        var model = Model(f, promptIds.Length + expected.Count);
        Assert.Equal(promptIds, model.Tokenizer.Encode(OuteTTSPrompts.CompletionPrompt("The weather is lovely today.", OuteTTSSpeaker.DefaultEnglishFemale())));
        Assert.Equal(expected, model.Generate(promptIds, new Random(0)));
    }

    [Fact(Timeout = 600000)]
    public async Task TrainingLoss_MatchesTheRelease()
    {
        await Task.Yield();
        var f = Reference();
        var model = Model(f, 8192);
        var ids = f["training_ids"]!.Select(v => (int)v).ToArray();
        Assert.Equal(ids, model.Tokenizer.Encode(OuteTTSPrompts.TrainingPrompt(OuteTTSSpeaker.DefaultEnglishFemale())));
        int header = (int)f["header_length"]!;
        var labels = ids.Select((id, i) => i < header ? -100 : id).ToArray();
        double loss;
        using (new NoGradScope<double>()) loss = model.Loss(ids, labels)[0];
        double reference = (double)f["loss"]!;
        Assert.True(Math.Abs(reference - loss) <= 1e-9 * Math.Max(1, Math.Abs(reference)), $"loss {reference:R} vs {loss:R}");
    }

    [Fact(Timeout = 600000)]
    public async Task SpeakerFromWords_MatchesCreateSpeakerFromDict()
    {
        await Task.Yield();
        var f = Reference();
        var model = Model(f, 8192);
        var bytes = Convert.FromBase64String((string)f["audio"]!);
        var audio = Enumerable.Range(0, bytes.Length / 8).Select(i => BitConverter.ToDouble(bytes, 8 * i)).ToArray();
        var words = f["words"]!.Select(w => ((string)w[0]!, (double)w[1]!, (double)w[2]!)).ToList();
        var speaker = model.SpeakerFromWords(audio, "Hello there, how are you?", words);
        var expected = OuteTTSSpeaker.FromJson(new JObject(((JObject)f["speaker"]!).Properties().Concat(new[] { new JProperty("interface_version", 3) })).ToString());
        Assert.Equal(expected.Text, speaker.Text);
        Assert.Equal(expected.GlobalFeatures.Energy, speaker.GlobalFeatures.Energy);
        Assert.Equal(expected.GlobalFeatures.SpectralCentroid, speaker.GlobalFeatures.SpectralCentroid);
        Assert.Equal(expected.GlobalFeatures.Pitch, speaker.GlobalFeatures.Pitch);
        Assert.Equal(expected.Words.Count, speaker.Words.Count);
        for (int i = 0; i < expected.Words.Count; i++)
        {
            var e = expected.Words[i];
            var a = speaker.Words[i];
            Assert.Equal(e.Word, a.Word);
            Assert.Equal(e.Duration, a.Duration);
            Assert.Equal(e.C1, a.C1);
            Assert.Equal(e.C2, a.C2);
            Assert.Equal((e.Features.Energy, e.Features.SpectralCentroid, e.Features.Pitch), (a.Features.Energy, a.Features.SpectralCentroid, a.Features.Pitch));
        }
    }
}
