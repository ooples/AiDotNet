using System.Diagnostics;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.TextToSpeech;

/// <summary>
/// Architecture and regression tests for Bark's semantic, coarse, fine, and codec pipeline.
/// The tiny configuration changes widths only; stage topology and causal contracts remain identical
/// to the released checkpoint configuration.
/// </summary>
public sealed class BarkModelTests
{
    [Fact]
    public async Task DefaultOptions_MatchReleasedBarkCheckpointTopology()
    {
        await Task.Yield();

        var options = new BarkOptions();

        Assert.Equal(24_000, options.SampleRate);
        Assert.Equal(8, options.NumCodebooks);
        Assert.Equal(1_024, options.CodebookSize);
        Assert.Equal(75, options.CodecFrameRate);
        Assert.Equal(2, options.NumberOfCoarseCodebooks);
        Assert.Equal(256, options.CoarseSemanticContextLength);
        Assert.Equal(129_600, options.Semantic.InputVocabularySize);
        Assert.Equal(10_048, options.Semantic.OutputVocabularySize);
        Assert.Equal(12_096, options.Coarse.InputVocabularySize);
        Assert.Equal(1_056, options.Fine.InputVocabularySize);
        Assert.All(new[] { options.Semantic, options.Coarse, options.Fine }, stage =>
        {
            Assert.Equal(1_024, stage.HiddenSize);
            Assert.Equal(24, stage.NumberOfLayers);
            Assert.Equal(16, stage.NumberOfHeads);
            Assert.Equal(4_096, stage.FeedForwardSize);
        });
        Assert.True(options.Semantic.IsCausal);
        Assert.True(options.Coarse.IsCausal);
        Assert.False(options.Fine.IsCausal);

        var generation = new BarkGenerationOptions();
        Assert.Equal(0.7, generation.SemanticSampling.Temperature);
        Assert.Equal(0.7, generation.CoarseSampling.Temperature);
        Assert.Equal(0.5, generation.FineSampling.Temperature);
        Assert.All(
            new[] { generation.SemanticSampling, generation.CoarseSampling, generation.FineSampling },
            sampling => Assert.Equal(50, sampling.TopK));
    }

    [Fact]
    public async Task DefaultConstruction_IsAllocationLightAndReportsStructuralCapacity()
    {
        await Task.Yield();
        long before = GC.GetAllocatedBytesForCurrentThread();
        using var model = new BarkModel<float>();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(model.EstimatedBarkTransformerParameterCount > 1_000_000_000L);
        Assert.True(allocated < 128L * 1024 * 1024,
            $"Metadata-only Bark construction allocated {allocated / (1024.0 * 1024.0):F1} MiB; weights must remain lazy.");
        Assert.Equal(8, model.NumberOfCodebooks);
        Assert.Equal(1_024, model.CodebookSize);
    }

    [Fact]
    public async Task SemanticForward_CompletesQuicklyWithFaithfulTinyTopology()
    {
        await Task.Yield();
        using var model = CreateTinyModel();
        var stopwatch = Stopwatch.StartNew();

        var logits = model.PredictSemanticLogits([1, 2, 3, 4]);

        stopwatch.Stop();
        Assert.Equal(new[] { 1, 4, 33 }, logits.Shape);
        Assert.True(stopwatch.Elapsed < TimeSpan.FromSeconds(10),
            $"Tiny Bark semantic forward took {stopwatch.Elapsed}; this is a compute-path regression, not a test-timeout issue.");
        AssertFinite(logits);
    }

    [Fact]
    public async Task SemanticCache_IncrementalLogitsMatchFullCausalForward()
    {
        await Task.Yield();
        using var model = CreateTinyModel();

        _ = model.BeginSemanticSequence([1, 2, 3]);
        Vector<float> cached = model.AppendSemanticToken(4);
        Assert.True(model.IsSemanticCacheActive);
        Tensor<float> full = model.PredictSemanticLogits([1, 2, 3, 4]);

        int finalPosition = full.Shape[1] - 1;
        for (int token = 0; token < cached.Length; token++)
            Assert.Equal(full[0, finalPosition, token], cached[token], precision: 4);
    }

    [Fact]
    public async Task SemanticGeneration_UsesBoundedSlidingCacheAndValidVocabulary()
    {
        await Task.Yield();
        using var model = CreateTinyModel();

        var tokens = model.GenerateSemanticTokens(
            [1, 2],
            new BarkGenerationOptions
            {
                MaxSemanticTokens = 3,
                AllowEarlyStop = false,
                SemanticSampling = AiDotNet.NeuralNetworks.Generation.SamplingOptions.Greedy,
            });

        Assert.Equal(3, tokens.Count);
        Assert.All(tokens, token => Assert.InRange(token, 0, 27));
        Assert.True(model.IsSemanticCacheActive);
    }

    [Fact]
    public async Task AllStages_ProduceConnectedPaperContractShapes()
    {
        await Task.Yield();
        using var model = CreateTinyModel();
        var generation = new BarkGenerationOptions
        {
            MaxSemanticTokens = 2,
            AllowEarlyStop = false,
            SemanticSampling = AiDotNet.NeuralNetworks.Generation.SamplingOptions.Greedy,
            CoarseSampling = AiDotNet.NeuralNetworks.Generation.SamplingOptions.Greedy,
            FineSampling = AiDotNet.NeuralNetworks.Generation.SamplingOptions.Greedy,
        };

        var semantic = model.GenerateSemanticTokens([1, 2], generation);
        var coarse = model.GenerateCoarseTokens([1, 2], generation);
        var fine = model.GenerateFineTokens(coarse);
        var audio = model.DecodeAudioTokens(fine);

        Assert.InRange(semantic.Count, 0, 2);
        Assert.Equal(2, coarse.GetLength(0));
        Assert.Equal(8, fine.GetLength(0));
        Assert.Equal(coarse.GetLength(1), fine.GetLength(1));
        for (int codebook = 0; codebook < coarse.GetLength(0); codebook++)
            for (int frame = 0; frame < coarse.GetLength(1); frame++)
                Assert.Equal(coarse[codebook, frame], fine[codebook, frame]);
        Assert.Equal(new[] { 1, fine.GetLength(1) }, audio.Shape);
        AssertFinite(audio);
    }

    [Fact]
    public async Task ParameterLifecycle_IsAutomatedAndCloneIsIndependent()
    {
        await Task.Yield();
        using var model = CreateTinyModel();
        _ = model.PredictSemanticLogits([1, 2]);
        long count = model.ParameterCount;
        Vector<float> parameters = model.GetParameters();
        using var clone = model.Clone();

        Assert.True(count > 0);
        Assert.Equal(count, parameters.Length);
        Assert.Equal(count, clone.ParameterCount);
        Assert.NotSame(model, clone);

        var changed = new Vector<float>(parameters.ToArray());
        changed[0] += 0.25f;
        clone.SetParameters(changed);
        Assert.NotEqual(model.GetParameters()[0], clone.GetParameters()[0]);
    }

    [Fact]
    public async Task InvalidConnectedConfiguration_FailsWithActionableMessage()
    {
        await Task.Yield();
        var options = BarkOptions.TinyForTests();
        options.Fine.IsCausal = true;

        var error = Assert.Throws<ArgumentException>(() => new BarkModel<float>(options));

        Assert.Contains("fine transformer must be bidirectional", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task GenerateAsync_ObservesCancellationBeforeMaterializingWeights()
    {
        await Task.Yield();
        using var model = CreateTinyModel();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => model.GenerateAsync([1, 2], cancellationToken: cancellation.Token));
    }

    private static BarkModel<float> CreateTinyModel()
    {
        var options = BarkOptions.TinyForTests();
        return new BarkModel<float>(options, new TestCodec(options));
    }

    private static void AssertFinite(Tensor<float> tensor)
    {
        for (int index = 0; index < tensor.Length; index++)
            Assert.True(float.IsFinite(tensor[index]), $"Tensor value {index} is not finite.");
    }

    private sealed class TestCodec : IAudioCodec<float>
    {
        internal TestCodec(BarkOptions options)
        {
            SampleRate = options.SampleRate;
            NumQuantizers = options.NumCodebooks;
            CodebookSize = options.CodebookSize;
            TokenFrameRate = options.CodecFrameRate;
        }

        public int SampleRate { get; }
        public int NumQuantizers { get; }
        public int CodebookSize { get; }
        public int TokenFrameRate { get; }

        public int[,] Encode(Tensor<float> audio)
            => new int[NumQuantizers, Math.Max(1, audio.Shape[^1] / Math.Max(1, SampleRate / TokenFrameRate))];

        public async Task<int[,]> EncodeAsync(Tensor<float> audio, CancellationToken cancellationToken = default)
        {
            await Task.Yield();
            cancellationToken.ThrowIfCancellationRequested();
            return Encode(audio);
        }

        public Tensor<float> Decode(int[,] tokens) => new([1, tokens.GetLength(1)]);

        public async Task<Tensor<float>> DecodeAsync(int[,] tokens, CancellationToken cancellationToken = default)
        {
            await Task.Yield();
            cancellationToken.ThrowIfCancellationRequested();
            return Decode(tokens);
        }

        public Tensor<float> EncodeEmbeddings(Tensor<float> audio) => audio;
        public Tensor<float> DecodeEmbeddings(Tensor<float> embeddings) => embeddings;
        public double GetBitrate(int? numQuantizers = null)
            => (numQuantizers ?? NumQuantizers) * TokenFrameRate * Math.Log2(CodebookSize);
    }
}
