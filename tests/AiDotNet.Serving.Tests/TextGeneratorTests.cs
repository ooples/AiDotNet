using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Interfaces;
using AiDotNet.Serving.Engine;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for the beginner-facing generation facade: <see cref="TextGenerator{T}"/>, the
/// <see cref="ServingRunnerFactory"/> runner selection, and the paged fast-path adapter. Uses deterministic
/// counter models so exact output can be asserted.
/// </summary>
public class TextGeneratorTests
{
    private const int Vocab = 128;

    private sealed class CounterLm : ICausalLmModel<double>
    {
        public int VocabularySize => Vocab;
        public int? EosTokenId => null;
        public Tensor<double> ForwardLogits(Tensor<double> tokenIds)
        {
            int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
            int last = (int)Math.Round(Convert.ToDouble(tokenIds[0, n - 1]));
            var t = new Tensor<double>(new[] { 1, n, Vocab });
            t[0, n - 1, (last + 1) % Vocab] = 1.0;
            return t;
        }
    }

    // Fast-path counter: implements the paged runner capability; ignores KV (stateless counter).
    private sealed class CounterPagedRunner : ICausalLmRunner<double>
    {
        public int VocabularySize => Vocab;
        public int NumLayers => 1;
        public int NumKvHeads => 1;
        public int HeadDim => 1;
        public int BlockSize => 16;
        public int PrefillCalls { get; private set; }
        public int DecodeCalls { get; private set; }

        public Tensor<double> Prefill(
            IReadOnlyList<IReadOnlyList<int>> tokenIdsPerSequence,
            IReadOnlyList<SequenceKvLayout> layouts,
            IReadOnlyList<int> tokenCounts)
        {
            PrefillCalls++;
            var t = new Tensor<double>(new[] { tokenIdsPerSequence.Count, Vocab });
            for (int i = 0; i < tokenIdsPerSequence.Count; i++)
            {
                var seq = tokenIdsPerSequence[i];
                int last = seq[seq.Count - 1];
                t[i, (last + 1) % Vocab] = 1.0;
            }
            return t;
        }

        public Tensor<double> DecodeStep(IReadOnlyList<int> lastTokenIds, IReadOnlyList<SequenceKvLayout> layouts)
        {
            DecodeCalls++;
            var t = new Tensor<double>(new[] { lastTokenIds.Count, Vocab });
            for (int i = 0; i < lastTokenIds.Count; i++)
                t[i, (lastTokenIds[i] + 1) % Vocab] = 1.0;
            return t;
        }

        public void CopyBlocks(IReadOnlyList<BlockCopy> copies) { }
    }

    // Minimal char-level tokenizer: char <-> (int)char; no EOS.
    private sealed class CharTokenizer : IGenerationTokenizer
    {
        public int EosTokenId => -1;
        public IReadOnlyList<int> Encode(string text) => text.Select(c => (int)c).ToArray();
        public string Decode(IReadOnlyList<int> tokenIds) => new string(tokenIds.Select(id => (char)id).ToArray());
    }

    private static SamplingParameters Greedy(int maxTokens) => new() { Temperature = 0.0, MaxTokens = maxTokens };

    [Fact]
    public void RecomputePath_GeneratesDeterministicIds()
    {
        using var gen = new TextGenerator<double>(new CounterLm());
        var ids = gen.Generate(new[] { 5 }, Greedy(4));
        Assert.Equal(new[] { 6, 7, 8, 9 }, ids);
    }

    [Fact]
    public void FastPath_IsSelected_WhenModelImplementsPagedRunner()
    {
        var runner = new CounterPagedRunner();
        var selection = ServingRunnerFactory.Create<double>(runner);
        Assert.IsType<PagedRunnerAdapter<double>>(selection.Runner);

        using var gen = new TextGenerator<double>(runner);
        var ids = gen.Generate(new[] { 40 }, Greedy(3));
        Assert.Equal(new[] { 41, 42, 43 }, ids);
        Assert.True(runner.PrefillCalls >= 1);
        Assert.True(runner.DecodeCalls >= 1);
    }

    [Fact]
    public void StringOverload_RoundTripsThroughTokenizer()
    {
        using var gen = new TextGenerator<double>(new CounterLm(), new CharTokenizer());
        // 'A' = 65 -> 66,67,68 = "BCD"
        Assert.Equal("BCD", gen.Generate("A", Greedy(3)));
    }

    [Fact]
    public void StringOverload_WithoutTokenizer_Throws()
    {
        using var gen = new TextGenerator<double>(new CounterLm());
        Assert.Throws<InvalidOperationException>(() => gen.Generate("hi", Greedy(2)));
    }

    [Fact]
    public void UnsupportedModel_ThrowsClearError()
    {
        var ex = Assert.Throws<NotSupportedException>(() => ServingRunnerFactory.Create<double>("not a model"));
        Assert.Contains("cannot generate text", ex.Message);
    }

    [Fact]
    public void EmptyPrompt_Throws()
    {
        using var gen = new TextGenerator<double>(new CounterLm());
        Assert.Throws<ArgumentException>(() => gen.Generate(Array.Empty<int>(), Greedy(2)));
    }

    [Fact]
    public void RepeatedGenerate_OnSameGenerator_IsConsistent()
    {
        using var gen = new TextGenerator<double>(new CounterLm());
        var first = gen.Generate(new[] { 10 }, Greedy(3));
        var second = gen.Generate(new[] { 10 }, Greedy(3));
        Assert.Equal(first, second);
        Assert.Equal(new[] { 11, 12, 13 }, first);
    }
}
