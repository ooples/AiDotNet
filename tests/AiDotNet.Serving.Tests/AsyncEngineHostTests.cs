using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Serving.Engine;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for <see cref="AsyncEngineHost{T}"/> — the background pump + concurrent async submission over the
/// continuous-batching engine. Uses a deterministic counter runner so concurrent results can be asserted
/// exactly.
/// </summary>
public class AsyncEngineHostTests
{
    private const int Vocab = 100;

    private sealed class CounterRunner : IServingModelRunner<double>
    {
        public int VocabularySize => Vocab;
        public IReadOnlyList<Vector<double>> Execute(IReadOnlyList<SequenceExecution<double>> batch)
        {
            var result = new List<Vector<double>>(batch.Count);
            foreach (var exec in batch)
            {
                int last = exec.AllTokenIds[exec.AllTokenIds.Count - 1];
                var row = new double[Vocab];
                row[(last + 1) % Vocab] = 1.0;
                result.Add(new Vector<double>(row));
            }
            return result;
        }
    }

    private static AsyncEngineHost<double> NewHost()
        => new(new ContinuousBatchingEngine<double>(new CounterRunner()));

    private static SamplingParameters Greedy(int maxTokens) => new() { Temperature = 0.0, MaxTokens = maxTokens };

    [Fact]
    public async Task GenerateAsync_ReturnsDeterministicSequence()
    {
        using var host = NewHost();
        var ids = await host.GenerateAsync(new[] { 5 }, Greedy(4));
        Assert.Equal(new[] { 6, 7, 8, 9 }, ids);
    }

    [Fact]
    public async Task ConcurrentRequests_AllReturnCorrectOutputs()
    {
        using var host = NewHost();
        var tasks = Enumerable.Range(0, 32)
            .Select(i => host.GenerateAsync(new[] { i }, Greedy(5)))
            .ToArray();

        var results = await Task.WhenAll(tasks);

        for (int i = 0; i < 32; i++)
        {
            var expected = Enumerable.Range(1, 5).Select(k => (i + k) % Vocab).ToArray();
            Assert.Equal(expected, results[i]);
        }
    }

    [Fact]
    public async Task StreamAsync_EmitsCumulativeUpdates_EndingFinished()
    {
        using var host = NewHost();
        var updates = new List<GenerationUpdate>();
        await foreach (var u in host.StreamAsync(new[] { 20 }, Greedy(3)))
            updates.Add(u);

        Assert.True(updates.Count >= 1);
        var last = updates[updates.Count - 1];
        Assert.True(last.IsFinished);
        Assert.Equal("length", last.FinishReason);
        Assert.Equal(new[] { 21, 22, 23 }, last.TokenIds);
        // Token counts are non-decreasing across updates (cumulative).
        for (int i = 1; i < updates.Count; i++)
            Assert.True(updates[i].TokenIds.Count >= updates[i - 1].TokenIds.Count);
    }

    [Fact]
    public async Task Cancellation_AbortsRequest()
    {
        using var host = NewHost();
        using var cts = new CancellationTokenSource();
        // Long generation; cancel after the first update.
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var u in host.StreamAsync(new[] { 1 }, Greedy(100000), cts.Token))
            {
                cts.Cancel();
            }
        });

        // After a moment the engine should have drained the aborted request.
        for (int i = 0; i < 50 && host.GetStatistics().RunningSequences > 0; i++) await Task.Delay(10);
        Assert.Equal(0, host.GetStatistics().RunningSequences);
    }
}
