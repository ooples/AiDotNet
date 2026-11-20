using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Release-contract tests for Fun-CosyVoice3-0.5B-2512.</summary>
public class CosyVoice3ReferenceTests
{
    [Fact]
    public void FrozenEncoderPaths_SurviveOptionsCopy()
    {
        var original = new CosyVoice3Options { SpeechTokenizerPath = "speech_tokenizer_v3.onnx", SpeakerEncoderOnnxPath = "campplus.onnx" };
        var copy = new CosyVoice3Options(original);
        Assert.Equal(original.SpeechTokenizerPath, copy.SpeechTokenizerPath);
        Assert.Equal(original.SpeakerEncoderOnnxPath, copy.SpeakerEncoderOnnxPath);
    }

    [Fact]
    public void TextTokenizer_UsesQwen2BaseAndOfficialCosyVoice3Ids()
    {
        var tokenizer = QwenTokenizer.CosyVoice3;
        // The checkpoint keeps Qwen's 151,936-row embedding table; the released tokenizer assigns 151,924 ids and
        // deliberately leaves the remaining rows unreachable.
        Assert.Equal(151924, tokenizer.VocabularySize);
        Assert.True(new CosyVoice3Options().VocabSize >= tokenizer.VocabularySize);
        Assert.Equal(151646, tokenizer.AddedTokenId("<|endofprompt|>"));
        Assert.Equal(new[] { 151646 }, tokenizer.Encode("<|endofprompt|>"));
        Assert.Single(tokenizer.Encode("[AA1]"));
        Assert.Single(tokenizer.Encode("[iǎng]"));
    }

    [Fact]
    public void LanguageModel_UsesSpeechSpecialRowsAndBiasFree6761WayContract()
    {
        var configuration = new CosyVoice2LmConfiguration(new LlamaConfiguration
        {
            VocabSize = 32,
            Hidden = 8,
            Layers = 1,
            Heads = 2,
            KeyValueHeads = 1,
            Intermediate = 16,
            RmsEpsilon = 1e-6,
            RopeTheta = 1_000_000,
            AttentionBias = true,
            IncludeHead = false,
        }, SpeechVocabulary: 5, SpeechSpecialTokens: 200, SpecialsInSpeechEmbedding: true, BiasFreeDecoder: true);
        var layers = new List<LayerBase<double>>();
        var lm = new CosyVoice2Lm<double>(AiDotNetEngine.Current, layers, configuration, new Random(1));

        Assert.Equal(6, lm.EndToken);
        Assert.Equal(8, lm.FillToken);
        Assert.NotNull(lm.BiasFreeHead);
        Assert.Throws<InvalidOperationException>(() => lm.LlmEmbedding);
        Assert.Equal(205 * 8, lm.SpeechEmbedding.GetParameters().Length);

        var requested = new HashSet<string>(StringComparer.Ordinal);
        lm.LoadTorchWeights((name, shape) =>
        {
            requested.Add(name);
            return new double[shape.Aggregate(1, (a, b) => a * b)];
        }, "");
        Assert.Contains("llm_decoder.weight", requested);
        Assert.Contains("speech_embedding.weight", requested);
        Assert.DoesNotContain("llm_decoder.bias", requested);
        Assert.DoesNotContain("llm_embedding.weight", requested);

        var (_, targets) = lm.TrainingSequence(new[] { 1, 2 }, new[] { 3, 4 }, new Random(2));
        Assert.Equal(lm.EndToken, targets[^1]);
    }

    [Fact]
    public void DiTFlow_RepeatsEachTokenTwice_AndRunsTrainingAndInference()
    {
        var options = new CosyVoice3Options
        {
            CodebookSize = 8,
            MelChannels = 4,
            SpeakerEmbeddingDim = 4,
            DiTInputDim = 4,
            PreLookaheadChannels = 8,
            DiTDim = 16,
            DiTDepth = 1,
            DiTHeads = 2,
            DiTHeadDim = 8,
            DiTFeedForwardMultiplier = 2,
            DiTTimesteps = 2,
            StaticChunkSize = 2,
            DropoutRate = 0,
        };
        var layers = new List<LayerBase<double>>();
        var flow = new CosyVoice3Flow<double>(AiDotNetEngine.Current, layers, options, new Random(3));
        var encoded = flow.Encode(new[] { 1, 2, 3 });
        Assert.Equal(new[] { 6, 4 }, encoded.Shape.ToArray());

        var mel = new Tensor<double>(new[] { 1, 4, 6 });
        var speaker = new Tensor<double>(new[] { 4 });
        speaker[0] = 1;
        var loss = flow.TrainingLoss(new[] { 1, 2, 3 }, mel, speaker, training: false, new Random(4));
        Assert.Single(loss.ToArray());
        Assert.True(TtsPaperNumerics.IsFinite(loss[0]));

        var noise = new Tensor<double>(new[] { 1, 4, 6 });
        var generated = flow.Inference(new[] { 1, 2, 3 }, Array.Empty<int>(), null, speaker, noise);
        Assert.Equal(new[] { 1, 4, 6 }, generated.Shape.ToArray());
    }

    [Fact]
    public void CausalHiFT_RequestsOfficialCausalKernelShapes_AndPreservesF0Length()
    {
        var configuration = new HiftConfiguration(
            InChannels: 4, BaseChannels: 8, Harmonics: 2, SampleRate: 24000, NsfAlpha: 0.1, NsfSigma: 0.003,
            VoicedThreshold: 10, UpsampleRates: new[] { 2 }, UpsampleKernels: new[] { 4 }, FftSize: 4, HopSize: 1,
            ResblockKernels: new[] { 3 }, ResblockDilations: new[] { new[] { 1 } }, SourceKernels: new[] { 3 },
            SourceDilations: new[] { new[] { 1 } }, LeakySlope: 0.1, AudioLimit: 0.99, F0Channels: 8,
            Causal: true, ConvPreLookRight: 4);
        var layers = new List<LayerBase<double>>();
        var hift = new HiftGenerator<double>(AiDotNetEngine.Current, layers, configuration, new Random(5));
        var shapes = new Dictionary<string, int[]>(StringComparer.Ordinal);
        hift.LoadTorchWeights((name, shape) =>
        {
            shapes[name] = shape;
            return new double[shape.Aggregate(1, (a, b) => a * b)];
        });
        Assert.Equal(new[] { 8, 4, 4 }, shapes["f0_predictor.condnet.0.parametrizations.weight.original1"]);
        Assert.Equal(new[] { 8, 4, 5 }, shapes["conv_pre.parametrizations.weight.original1"]);
        Assert.Equal(new[] { 4, 8, 4 }, shapes["ups.0.parametrizations.weight.original1"]);

        var mel = new Tensor<double>(new[] { 1, 4, 3 });
        Assert.Equal(3, hift.PredictF0(mel).Length);
    }

    // CosyVoice's S3 tokenizer reads the log-mel's frame count. Carried in T, as before, a Half model sent 2,049 frames
    // as 2,048 and 4,095 as 4,096 (Half is exact only to 2,048), and the codes it read back above 2,048 were rounded the
    // same way. The count now reaches the graph's declared int32 or int64 input exactly, and int64 codes come back exactly.
    [Theory]
    [InlineData(2049, "int32")]
    [InlineData(4095, "int64")]
    public void SpeechTokenizer_InHalf_GetsTheExactFrameCount(int frames, string lengthType)
    {
        var logMel = new Tensor<Half>(new[] { 128, frames });
        var (features, lengths) = CosyVoiceBase<Half>.SpeechTokenizerInputs(new[] { "feats", "feats_length" }, logMel);
        Assert.Equal(new[] { 1, 128, frames }, features["feats"].Shape.ToArray());
        Assert.Equal(new[] { frames }, lengths["feats_length"].ToArray());

        var declared = new List<AiDotNet.Interfaces.IOnnxTensorInfo>
        {
            new AiDotNet.Onnx.OnnxTensorInfo { Name = "feats", ElementType = "float", Shape = new[] { 1, 128, -1 } },
            new AiDotNet.Onnx.OnnxTensorInfo { Name = "feats_length", ElementType = lengthType, Shape = new[] { 1 } },
        };
        var onnx = AiDotNet.Onnx.OnnxModel<Half>.BuildInputs(declared, features, lengths).Single(v => v.Name == "feats_length");
        long sent = lengthType == "int32" ? onnx.AsTensor<int>()[0] : onnx.AsTensor<long>()[0];
        Assert.Equal(frames, sent);

        var codes = Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("speech_tokens",
            new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(new long[] { 2049, 4095, 6560 }, new[] { 1, 3 }));
        var read = AiDotNet.Onnx.OnnxModel<Half>.IntegersOf(codes);
        Assert.NotNull(read);
        Assert.Equal(new[] { 2049, 4095, 6560 }, read?.ToArray());
    }
}
