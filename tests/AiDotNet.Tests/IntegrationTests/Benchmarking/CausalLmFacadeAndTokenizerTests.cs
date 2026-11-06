using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.ModelLoading.Pretrained;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Configuration;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Benchmarking;

/// <summary>
/// End-to-end unit and integration tests for Hugging Face ByteLevel BPE loading,
/// <see cref="AiModelBuilder{T,TInput,TOutput}"/> tokenizer configuration and auto-discovery,
/// incremental KV-cached <c>PredictNextWord</c> / <c>GenerateText</c> on <c>AiModelResult</c>,
/// and <see cref="BenchmarkSuite.LAMBADA"/> + <see cref="BenchmarkSuite.Perplexity"/> benchmarking.
/// </summary>
public sealed class CausalLmFacadeAndTokenizerTests : IDisposable
{
    private readonly string _tempDir;

    public CausalLmFacadeAndTokenizerTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "aidn_causal_lm_tests_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            try
            {
                Directory.Delete(_tempDir, recursive: true);
            }
            catch
            {
                // Best-effort temp cleanup.
            }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task HuggingFaceTokenizerLoader_QwenStyleByteLevelBpe_HandlesSentinelMaxLengthAddedTokensAndRoundTrip()
    {
        await Task.Yield();
        WriteQwenStyleByteLevelTokenizer(_tempDir);

        var tokenizer = AutoTokenizer.FromPretrained(_tempDir);
        string eosSpecial = "<|" + "endoftext|>";
        string imEndSpecial = "<|" + "im_end|>";

        var encoded = tokenizer.Encode(" Aidan walked.");
        // ByteLevel causal BPE must NOT wrap sequences in [CLS]/[SEP].
        Assert.DoesNotContain("[CLS]", encoded.Tokens);
        Assert.DoesNotContain("[SEP]", encoded.Tokens);

        // " Aidan" splits across two BPE subwords ("ĠAid" + "an"), followed by "Ġwalked" and ".".
        Assert.Equal(4, encoded.TokenIds.Count);
        Assert.Equal(" Aidan walked.", tokenizer.Decode(encoded.TokenIds));

        // Multi-byte UTF-8 round-trip (" café — 123") must decode losslessly.
        const string utf8Sample = " café — 123";
        var utf8Encoded = tokenizer.Encode(utf8Sample);
        Assert.Equal(utf8Sample, tokenizer.Decode(utf8Encoded.TokenIds));

        // Individual digits rule splits "123" into three separate digit tokens.
        var digitEncoded = tokenizer.Encode("123");
        Assert.Equal(3, digitEncoded.TokenIds.Count);

        // Whole-unit special tokens from added_tokens match atomically and skip when requested.
        var withSpecial = tokenizer.Encode("Hello" + imEndSpecial + eosSpecial);
        int imEndId = tokenizer.Vocabulary.GetTokenId(imEndSpecial);
        int eosId = tokenizer.Vocabulary.GetTokenId(eosSpecial);
        Assert.Contains(imEndId, withSpecial.TokenIds);
        Assert.Contains(eosId, withSpecial.TokenIds);
        Assert.Equal("Hello", tokenizer.Decode(withSpecial.TokenIds, skipSpecialTokens: true));
    }

    [Fact(Timeout = 60000)]
    public async Task PretrainedLoader_OnnxSource_AutoDiscoversAncestorTokenizerJson()
    {
        await Task.Yield();
        WriteQwenStyleByteLevelTokenizer(_tempDir);
        string onnxSubdir = Path.Combine(_tempDir, "onnx");
        Directory.CreateDirectory(onnxSubdir);
        string fakeOnnxPath = Path.Combine(onnxSubdir, "model.onnx");
        File.WriteAllBytes(fakeOnnxPath, new byte[] { 0x08, 0x07 });

        var source = PretrainedSource.Onnx(fakeOnnxPath);
        var discovered = PretrainedLoader<float>.TryLoadTokenizer(source);

        Assert.NotNull(discovered);
        var encoded = discovered!.Encode(" Aidan");
        Assert.Equal(" Aidan", discovered.Decode(encoded.TokenIds));
    }

    [Fact(Timeout = 60000)]
    public async Task AiModelResult_PredictNextWord_And_GenerateText_DriveIncrementalKvCache_And_StopAtWordBoundary()
    {
        await Task.Yield();
        WriteQwenStyleByteLevelTokenizer(_tempDir);
        var tokenizer = AutoTokenizer.FromPretrained(_tempDir);

        int idG_Aid = tokenizer.Encode(" Aid").TokenIds.Single();
        int idAn = tokenizer.Encode("an").TokenIds.Single();
        int idG_Walked = tokenizer.Encode(" walked").TokenIds.Single();
        int idPeriod = tokenizer.Encode(".").TokenIds.Single();
        int idG_Don = tokenizer.Encode(" don").TokenIds.Single();
        int idContractionT = tokenizer.Encode("'t").TokenIds.Single();

        var promptMetIds = tokenizer.Encode("The hero met").TokenIds;
        var promptPleaseIds = tokenizer.Encode("Please").TokenIds;

        var transitions = new Dictionary<int, int>
        {
            [promptMetIds[promptMetIds.Count - 1]] = idG_Aid,
            [idG_Aid] = idAn,
            [idAn] = idG_Walked,
            [idG_Walked] = idPeriod,
            [promptPleaseIds[promptPleaseIds.Count - 1]] = idG_Don,
            [idG_Don] = idContractionT,
            [idContractionT] = idG_Walked
        };

        var stubModel = new DeterministicIncrementalCausalLm(tokenizer.VocabularySize, transitions);
        var result = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(stubModel)
            .ConfigureTokenizer(new TokenizationConfig { PretrainedModelOrPath = _tempDir })
            .ConfigureInferenceOptimizations(new InferenceOptimizationConfig { EnableKVCache = true })
            .BuildForInference();

        // 1. Multi-subword proper noun ("ĠAid" + "an" -> "Aidan"), stopping before "Ġwalked".
        string word = result.PredictNextWord("The hero met", maxSubwordTokens: 8);
        Assert.Equal("Aidan", word);
        Assert.Equal(1, stubModel.StartSequenceCalls);
        Assert.Equal(2, stubModel.AppendTokenCalls); // Appended "ĠAid", then "an" -> saw "Ġwalked" and stopped!
        Assert.Equal(0, stubModel.FullRefeedCalls);

        // 2. Contraction ("Ġdon" + "'t" -> "don't"), stopping before "Ġwalked".
        stubModel.ResetCounters();
        string contraction = result.PredictNextWord("Please", maxSubwordTokens: 8);
        Assert.Equal("don't", contraction);
        Assert.Equal(1, stubModel.StartSequenceCalls);
        Assert.Equal(2, stubModel.AppendTokenCalls);
        Assert.Equal(0, stubModel.FullRefeedCalls);

        // 3. GenerateText drives incremental KV-cache and honors stopSequences.
        stubModel.ResetCounters();
        string continuation = result.GenerateText(
            "The hero met",
            maxNewTokens: 6,
            temperature: 0.0,
            stopSequences: new[] { "." });
        Assert.Equal(" Aidan walked", continuation);
        Assert.Equal(1, stubModel.StartSequenceCalls);
        Assert.Equal(3, stubModel.AppendTokenCalls);
        Assert.Equal(0, stubModel.FullRefeedCalls);
    }

    [Fact(Timeout = 60000)]
    public async Task EvaluateBenchmarksAsync_Lambada_And_Perplexity_ReportExactMetrics_And_EnforceDisjointSlices()
    {
        WriteQwenStyleByteLevelTokenizer(_tempDir);
        var tokenizer = AutoTokenizer.FromPretrained(_tempDir);

        int idG_Aid = tokenizer.Encode(" Aid").TokenIds.Single();
        int idAn = tokenizer.Encode("an").TokenIds.Single();
        int idG_Walked = tokenizer.Encode(" walked").TokenIds.Single();

        var prefix1Ids = tokenizer.Encode("The hero met").TokenIds;
        var prefix2Ids = tokenizer.Encode("Everyone cheered for").TokenIds;

        const double targetLogitBoost = 2.0;
        var transitions = new Dictionary<int, int>
        {
            // Passage 1 ("The hero met Aidan") -> predicts "ĠAid" + "an" + "Ġwalked" -> "Aidan" (exact match)
            [prefix1Ids[prefix1Ids.Count - 1]] = idG_Aid,
            [idG_Aid] = idAn,
            [idAn] = idG_Walked,
            // Passage 2 ("Everyone cheered for Bob") -> predicts "Ġwalked" -> "walked" != "Bob" (mismatch)
            [prefix2Ids[prefix2Ids.Count - 1]] = idG_Walked
        };

        var corpusText = " Aidan walked don't.";
        var corpusIds = tokenizer.Encode(corpusText).TokenIds;
        // Make every adjacent token transition in corpusText deterministic with logit = targetLogitBoost (2.0) vs 0.0 elsewhere
        // so analytical cross-entropy and perplexity are exact!
        var seqTransitions = new Dictionary<int, int>(transitions);
        for (int i = 0; i + 1 < corpusIds.Count; i++)
        {
            seqTransitions[corpusIds[i]] = corpusIds[i + 1];
        }

        var stubModel = new DeterministicIncrementalCausalLm(
            tokenizer.VocabularySize,
            seqTransitions,
            winningLogit: (float)targetLogitBoost);

        var benchOptions = new BenchmarkingOptions
        {
            Suites = new[] { BenchmarkSuite.LAMBADA, BenchmarkSuite.Perplexity },
            FailurePolicy = BenchmarkFailurePolicy.FailFast,
            LanguageModel = new CausalLmBenchmarkOptions
            {
                LambadaPassages = new[]
                {
                    "The hero met Aidan",
                    "Everyone cheered for Bob"
                },
                PerplexityCorpusText = corpusText,
                ContextWindowLength = 64,
                EvaluationTokenStart = 0,
                WarmupItems = 0
            }
        };

        var result = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(stubModel)
            .ConfigureTokenizer(tokenizer)
            .ConfigureBenchmarking(benchOptions)
            .BuildForInference();

        var report = await result.EvaluateBenchmarksAsync();
        Assert.Equal(2, report.Suites.Count);

        var lambada = report.Suites.Single(s => s.Suite == BenchmarkSuite.LAMBADA);
        Assert.Equal(BenchmarkExecutionStatus.Succeeded, lambada.Status);
        Assert.Equal(2.0, GetMetric(lambada, BenchmarkMetric.TotalEvaluated), precision: 6);
        Assert.Equal(0.5, GetMetric(lambada, BenchmarkMetric.Accuracy), precision: 6);
        Assert.Equal(0.5, GetMetric(lambada, BenchmarkMetric.SubwordTop1Accuracy), precision: 6);
        Assert.True(GetMetric(lambada, BenchmarkMetric.MedianTimePerItemMilliseconds) >= 0.0);
        Assert.True(GetMetric(lambada, BenchmarkMetric.P95TimePerItemMilliseconds) >= 0.0);

        var pplSuite = report.Suites.Single(s => s.Suite == BenchmarkSuite.Perplexity);
        Assert.Equal(BenchmarkExecutionStatus.Succeeded, pplSuite.Status);

        // Analytical NLL when winning target has logit = 2.0 and (V - 1) tokens have logit = 0.0:
        // logZ = 2.0 + ln(1 + (V - 1) * exp(-2.0)), nll = logZ - 2.0 = ln(1 + (V - 1) * exp(-2.0))
        double expectedNll = Math.Log(1.0 + (tokenizer.VocabularySize - 1) * Math.Exp(-targetLogitBoost));
        double expectedPpl = Math.Exp(expectedNll);
        Assert.Equal(corpusIds.Count - 1, (int)GetMetric(pplSuite, BenchmarkMetric.EvaluatedTokens));
        Assert.Equal(1.0, GetMetric(pplSuite, BenchmarkMetric.SubwordTop1Accuracy), precision: 6);
        Assert.Equal(expectedNll, GetMetric(pplSuite, BenchmarkMetric.CrossEntropyLoss), precision: 4);
        Assert.Equal(expectedPpl, GetMetric(pplSuite, BenchmarkMetric.Perplexity), precision: 2);

        // Verify strict calibration/eval disjointness check rejects any overlapping token range!
        var overlappingOptions = new BenchmarkingOptions
        {
            Suites = new[] { BenchmarkSuite.Perplexity },
            FailurePolicy = BenchmarkFailurePolicy.FailFast,
            LanguageModel = new CausalLmBenchmarkOptions
            {
                PerplexityCorpusText = corpusText,
                EvaluationTokenStart = 1,
                EvaluationTokenCount = 3,
                CalibrationTokenRange = (0, 2) // Overlaps token index 1!
            }
        };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => result.EvaluateBenchmarksAsync(overlappingOptions));
        Assert.Contains("overlap", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 120000)]
    public async Task OnnxCausalLm_WhenExternalCheckpointConfigured_LoadsExternalDataAutoDiscoversTokenizerAndPredictsNextWord()
    {
        await Task.Yield();
        string? onnxPath = Environment.GetEnvironmentVariable("AIDOTNET_QWEN_ONNX_PATH");
        if (string.IsNullOrWhiteSpace(onnxPath) || !File.Exists(onnxPath))
        {
            return;
        }

        var result = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(PretrainedSource.Onnx(onnxPath))
            .ConfigureInferenceOptimizations(new InferenceOptimizationConfig { EnableKVCache = true })
            .BuildForInference();

        try
        {
            string word = result.PredictNextWord("The capital of France is", maxSubwordTokens: 8);
            Assert.Equal("Paris", word);
        }
        finally
        {
            if (result.Model is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }

    private static double GetMetric(BenchmarkSuiteReport suite, BenchmarkMetric metric)
        => suite.Metrics.Single(m => m.Metric == metric).Value;

    private static void WriteQwenStyleByteLevelTokenizer(string directory)
    {
        var vocab = new JObject();
        var printable = new List<int>();
        for (int b = 33; b <= 126; b++) printable.Add(b);
        for (int b = 161; b <= 172; b++) printable.Add(b);
        for (int b = 174; b <= 255; b++) printable.Add(b);

        var printableSet = new HashSet<int>(printable);
        int extra = 0;
        var byteToChar = new char[256];
        for (int b = 0; b < 256; b++)
        {
            if (printableSet.Contains(b))
            {
                byteToChar[b] = (char)b;
            }
            else
            {
                byteToChar[b] = (char)(256 + extra);
                extra++;
            }
        }

        int nextId = 0;
        for (int b = 0; b < 256; b++)
        {
            vocab[byteToChar[b].ToString()] = nextId++;
        }

        string g = byteToChar[32].ToString(); // 'Ġ'
        var mergePairs = new (string Left, string Right)[]
        {
            (g, "A"),
            (g + "A", "i"),
            (g + "Ai", "d"),
            ("a", "n"),
            (g, "w"),
            (g + "w", "a"),
            (g + "wa", "l"),
            (g + "wal", "k"),
            (g + "walk", "e"),
            (g + "walke", "d"),
            (g, "d"),
            (g + "d", "o"),
            (g + "do", "n"),
            ("'", "t")
        };

        var mergesJson = new JArray();
        foreach (var (left, right) in mergePairs)
        {
            vocab[left + right] = nextId++;
            mergesJson.Add(new JArray(left, right));
        }

        string eosToken = "<|" + "endoftext|>";
        string imEndToken = "<|" + "im_end|>";
        int eosId = nextId++;
        int imEndId = nextId++;

        var tokenizerJson = new JObject
        {
            ["version"] = "1.0",
            ["added_tokens"] = new JArray
            {
                new JObject
                {
                    ["id"] = eosId,
                    ["content"] = eosToken,
                    ["single_word"] = false,
                    ["lstrip"] = false,
                    ["rstrip"] = false,
                    ["normalized"] = false,
                    ["special"] = true
                },
                new JObject
                {
                    ["id"] = imEndId,
                    ["content"] = imEndToken,
                    ["single_word"] = false,
                    ["lstrip"] = false,
                    ["rstrip"] = false,
                    ["normalized"] = false,
                    ["special"] = true
                }
            },
            ["pre_tokenizer"] = new JObject
            {
                ["type"] = "Sequence",
                ["pretokenizers"] = new JArray
                {
                    new JObject
                    {
                        ["type"] = "Digits",
                        ["individual_digits"] = true
                    },
                    new JObject
                    {
                        ["type"] = "ByteLevel",
                        ["add_prefix_space"] = false,
                        ["trim_offsets"] = true,
                        ["use_regex"] = true
                    }
                }
            },
            ["decoder"] = new JObject
            {
                ["type"] = "ByteLevel",
                ["add_prefix_space"] = true,
                ["trim_offsets"] = true,
                ["use_regex"] = true
            },
            ["model"] = new JObject
            {
                ["type"] = "BPE",
                ["vocab"] = vocab,
                ["merges"] = mergesJson
            }
        };

        // Include Hugging Face's 1e30 sentinel model_max_length and AddedToken object entries.
        string tokenizerConfigRaw =
            "{\n" +
            "  \"tokenizer_class\": \"Qwen2Tokenizer\",\n" +
            "  \"add_bos_token\": false,\n" +
            "  \"model_max_length\": 1000000000000000019884624838656,\n" +
            "  \"eos_token\": { \"__type\": \"AddedToken\", \"content\": \"" + eosToken + "\", \"special\": true },\n" +
            "  \"pad_token\": { \"__type\": \"AddedToken\", \"content\": \"" + eosToken + "\", \"special\": true }\n" +
            "}";

        File.WriteAllText(Path.Combine(directory, "tokenizer.json"), tokenizerJson.ToString(Formatting.Indented));
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), tokenizerConfigRaw);
    }

    private sealed class DeterministicIncrementalCausalLm
        : IFullModel<float, Tensor<float>, Tensor<float>>,
          ILanguageModel<float>,
          IIncrementalCausalLanguageModel<float>
    {
        private readonly Dictionary<int, int> _transitions;
        private readonly float _winningLogit;
        private int _lastToken;

        public DeterministicIncrementalCausalLm(
            int vocabularySize,
            Dictionary<int, int> transitions,
            float winningLogit = 10.0f)
        {
            VocabularySize = vocabularySize;
            _transitions = new Dictionary<int, int>(transitions);
            _winningLogit = winningLogit;
        }

        public int VocabularySize { get; }
        public int MaxContextTokens => 2048;
        public int MaxGenerationTokens => 128;

        public int StartSequenceCalls { get; private set; }
        public int AppendTokenCalls { get; private set; }
        public int FullRefeedCalls { get; private set; }

        public void ResetCounters()
        {
            StartSequenceCalls = 0;
            AppendTokenCalls = 0;
            FullRefeedCalls = 0;
        }

        public void ResetCache()
        {
            _lastToken = 0;
        }

        public Vector<float> StartSequence(IReadOnlyList<int> promptTokenIds)
        {
            StartSequenceCalls++;
            _lastToken = promptTokenIds[promptTokenIds.Count - 1];
            return BuildLogits(_lastToken);
        }

        public Vector<float> AppendToken(int tokenId)
        {
            AppendTokenCalls++;
            _lastToken = tokenId;
            return BuildLogits(_lastToken);
        }

        public Vector<float> NextTokenLogits(IReadOnlyList<int> tokenIds)
        {
            FullRefeedCalls++;
            return BuildLogits(tokenIds[tokenIds.Count - 1]);
        }

        public Tensor<float> ForwardLogits(Tensor<float> tokenIds)
        {
            var shape = tokenIds.Shape;
            int batch = shape.Length == 2 ? shape[0] : 1;
            int seqLen = shape[shape.Length - 1];
            var outTensor = new Tensor<float>(new[] { batch, seqLen, VocabularySize });

            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    int tok = shape.Length == 2 ? (int)Math.Round(tokenIds[b, s]) : (int)Math.Round(tokenIds[s]);
                    if (_transitions.TryGetValue(tok, out int next) && next >= 0 && next < VocabularySize)
                    {
                        outTensor[b, s, next] = _winningLogit;
                    }
                }
            }

            return outTensor;
        }

        public Tensor<float> Predict(Tensor<float> input) => ForwardLogits(input);

        public void Train(Tensor<float> input, Tensor<float> expectedOutput)
            => throw new NotSupportedException();

        public ModelMetadata<float> GetModelMetadata()
            => new ModelMetadata<float> { Name = "DeterministicIncrementalCausalLm", Description = "Test stub" };

        public ILossFunction<float> DefaultLossFunction => new MeanSquaredErrorLoss<float>();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) => File.WriteAllBytes(filePath, Array.Empty<byte>());
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Empty<int>();
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }
        public bool IsFeatureUsed(int featureIndex) => true;
        public Dictionary<string, float> GetFeatureImportance() => new Dictionary<string, float>();

        public IFullModel<float, Tensor<float>, Tensor<float>> Clone()
            => new DeterministicIncrementalCausalLm(VocabularySize, _transitions, _winningLogit);

        public IFullModel<float, Tensor<float>, Tensor<float>> DeepCopy() => Clone();

        public long ParameterCount => 0;
        public bool SupportsParameterInitialization => false;
        public Vector<float> GetParameters() => new Vector<float>(0);
        public Vector<float> SanitizeParameters(Vector<float> parameters) => parameters;
        public void SetParameters(Vector<float> parameters) { }
        public IFullModel<float, Tensor<float>, Tensor<float>> WithParameters(Vector<float> parameters) => this;
        public void Dispose() { }

        private Vector<float> BuildLogits(int token)
        {
            var arr = new float[VocabularySize];
            if (_transitions.TryGetValue(token, out int next) && next >= 0 && next < VocabularySize)
            {
                arr[next] = _winningLogit;
            }
            return new Vector<float>(arr);
        }
    }
}
