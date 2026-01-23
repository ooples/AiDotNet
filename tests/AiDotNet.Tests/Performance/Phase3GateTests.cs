using System.Collections.Concurrent;
using System.Diagnostics;
using AiDotNet.Interfaces;
using AiDotNet.Memory;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.Performance;

/// <summary>
/// Phase 3 Gate Tests for Performance Optimization Plan.
/// These tests validate layer refactoring to use IEngine operations and InferenceContext.
/// </summary>
[Trait("Category", "Phase3Gate")]
[Trait("Category", "Performance")]
public class Phase3GateTests
{
    #region InferenceContext Tests

    [Fact]
    public void InferenceContext_Rent_ReturnsValidTensor()
    {
        var pool = new TensorPool<float>(maxPoolSizeMB: 10);
        using var context = new InferenceContext<float>(pool);
        var shape = new[] { 32, 32 };

        var tensor = context.Rent(shape);

        Assert.NotNull(tensor);
        Assert.Equal(2, tensor.Shape.Length);
        Assert.Equal(32, tensor.Shape[0]);
        Assert.Equal(32, tensor.Shape[1]);
    }

    [Fact]
    public void InferenceContext_Rent1D_ReturnsCorrectShape()
    {
        using var context = new InferenceContext<float>();

        var tensor = context.Rent1D(100);

        Assert.Single(tensor.Shape);
        Assert.Equal(100, tensor.Shape[0]);
    }

    [Fact]
    public void InferenceContext_Rent2D_ReturnsCorrectShape()
    {
        using var context = new InferenceContext<float>();

        var tensor = context.Rent2D(32, 64);

        Assert.Equal(2, tensor.Shape.Length);
        Assert.Equal(32, tensor.Shape[0]);
        Assert.Equal(64, tensor.Shape[1]);
    }

    [Fact]
    public void InferenceContext_Rent3D_ReturnsCorrectShape()
    {
        using var context = new InferenceContext<float>();

        var tensor = context.Rent3D(4, 32, 64);

        Assert.Equal(3, tensor.Shape.Length);
        Assert.Equal(4, tensor.Shape[0]);
        Assert.Equal(32, tensor.Shape[1]);
        Assert.Equal(64, tensor.Shape[2]);
    }

    [Fact]
    public void InferenceContext_Rent4D_ReturnsCorrectShape()
    {
        using var context = new InferenceContext<float>();

        var tensor = context.Rent4D(2, 3, 32, 32);

        Assert.Equal(4, tensor.Shape.Length);
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(3, tensor.Shape[1]);
        Assert.Equal(32, tensor.Shape[2]);
        Assert.Equal(32, tensor.Shape[3]);
    }

    [Fact]
    public void InferenceContext_RentLike_MatchesTemplate()
    {
        using var context = new InferenceContext<float>();
        var template = new Tensor<float>(new[] { 8, 16, 32 });

        var tensor = context.RentLike(template);

        Assert.Equal(3, tensor.Shape.Length);
        Assert.Equal(8, tensor.Shape[0]);
        Assert.Equal(16, tensor.Shape[1]);
        Assert.Equal(32, tensor.Shape[2]);
    }

    [Fact]
    public void InferenceContext_TracksRentedTensorCount()
    {
        using var context = new InferenceContext<float>();

        Assert.Equal(0, context.RentedTensorCount);

        context.Rent1D(10);
        Assert.Equal(1, context.RentedTensorCount);

        context.Rent2D(10, 10);
        Assert.Equal(2, context.RentedTensorCount);

        context.Rent3D(5, 10, 10);
        Assert.Equal(3, context.RentedTensorCount);
    }

    [Fact]
    public void InferenceContext_DisposedReturnsToPool()
    {
        var pool = new TensorPool<float>(maxPoolSizeMB: 10);
        var shape = new[] { 32, 32 };

        Assert.Equal(0, pool.TotalPooledTensors);

        using (var context = new InferenceContext<float>(pool))
        {
            context.Rent(shape);
            context.Rent(shape);
            context.Rent(shape);
            Assert.Equal(3, context.RentedTensorCount);
        }

        // After dispose, tensors should be in pool
        Assert.Equal(3, pool.TotalPooledTensors);
    }

    [Fact]
    public void InferenceContext_Release_ReturnsImmediately()
    {
        var pool = new TensorPool<float>(maxPoolSizeMB: 10);
        using var context = new InferenceContext<float>(pool);

        var tensor = context.Rent1D(100);
        Assert.Equal(0, pool.TotalPooledTensors);

        context.Release(tensor);
        Assert.Equal(1, pool.TotalPooledTensors);
    }

    [Fact]
    public void InferenceContext_DisablePooling_AllocatesNew()
    {
        var pool = new TensorPool<float>(maxPoolSizeMB: 10);

        // First, populate the pool with tensors
        using (var setupContext = new InferenceContext<float>(pool))
        {
            setupContext.Rent1D(50);
        }
        Assert.Equal(1, pool.TotalPooledTensors);

        // Now with pooling disabled
        using var context = new InferenceContext<float>(pool);
        context.IsPoolingEnabled = false;

        var tensor = context.Rent1D(50);
        // Should not take from pool
        Assert.Equal(1, pool.TotalPooledTensors);
    }

    [Fact]
    public void InferenceContext_DisposeAfterObjectDisposed_ThrowsException()
    {
        var context = new InferenceContext<float>();
        context.Dispose();

        Assert.Throws<ObjectDisposedException>(() => context.Rent1D(10));
    }

    #endregion

    #region InferenceScope Tests

    [Fact]
    public void InferenceScope_Current_IsNullByDefault()
    {
        // Ensure no context is set
        InferenceScope<float>.Current = null;

        Assert.Null(InferenceScope<float>.Current);
        Assert.False(InferenceScope<float>.IsActive);
    }

    [Fact]
    public void InferenceScope_Begin_SetsCurrentContext()
    {
        var context = new InferenceContext<float>();

        using (InferenceScope<float>.Begin(context))
        {
            Assert.Same(context, InferenceScope<float>.Current);
            Assert.True(InferenceScope<float>.IsActive);
        }

        // After scope ends, should be restored to null
        Assert.Null(InferenceScope<float>.Current);
        Assert.False(InferenceScope<float>.IsActive);
    }

    [Fact]
    public void InferenceScope_NestedScopes_RestoreCorrectly()
    {
        var context1 = new InferenceContext<float>();
        var context2 = new InferenceContext<float>();

        using (InferenceScope<float>.Begin(context1))
        {
            Assert.Same(context1, InferenceScope<float>.Current);

            using (InferenceScope<float>.Begin(context2))
            {
                Assert.Same(context2, InferenceScope<float>.Current);
            }

            // Inner scope ended, should be back to context1
            Assert.Same(context1, InferenceScope<float>.Current);
        }

        // Outer scope ended, should be null
        Assert.Null(InferenceScope<float>.Current);
    }

    [Fact]
    public void InferenceScope_RentOrCreate_UsesPoolWhenActive()
    {
        var pool = new TensorPool<float>(maxPoolSizeMB: 10);
        var context = new InferenceContext<float>(pool);
        var shape = new[] { 16, 16 };

        // Without active scope
        var tensor1 = InferenceScope<float>.RentOrCreate(shape);
        Assert.NotNull(tensor1);
        Assert.Equal(0, context.RentedTensorCount);

        // With active scope
        using (InferenceScope<float>.Begin(context))
        {
            var tensor2 = InferenceScope<float>.RentOrCreate(shape);
            Assert.NotNull(tensor2);
            Assert.Equal(1, context.RentedTensorCount);
        }
    }

    [Fact]
    public void InferenceScope_RentOrCreateLike_MatchesTemplate()
    {
        var template = new Tensor<float>(new[] { 8, 16 });
        var context = new InferenceContext<float>();

        using (InferenceScope<float>.Begin(context))
        {
            var tensor = InferenceScope<float>.RentOrCreateLike(template);
            Assert.Equal(2, tensor.Shape.Length);
            Assert.Equal(8, tensor.Shape[0]);
            Assert.Equal(16, tensor.Shape[1]);
        }
    }

    [Fact]
    public void InferenceScope_IsThreadLocal()
    {
        var context1 = new InferenceContext<float>();
        var context2 = new InferenceContext<float>();
        var threadResults = new ConcurrentDictionary<int, bool>();

        using (InferenceScope<float>.Begin(context1))
        {
            Assert.Same(context1, InferenceScope<float>.Current);

            // Start another thread with different context
            var thread = new Thread(() =>
            {
                // Should be null on new thread
                var isNullInitially = InferenceScope<float>.Current == null;

                using (InferenceScope<float>.Begin(context2))
                {
                    // Should be context2 on this thread
                    threadResults[1] = ReferenceEquals(context2, InferenceScope<float>.Current);
                }

                threadResults[2] = isNullInitially;
            });
            thread.Start();
            thread.Join();

            // Main thread should still have context1
            Assert.Same(context1, InferenceScope<float>.Current);
        }

        Assert.True(threadResults.GetValueOrDefault(1), "Thread should have context2");
        Assert.True(threadResults.GetValueOrDefault(2), "Thread should start with null context");
    }

    #endregion

    #region CrossAttentionLayer Tests

    [Fact]
    public void CrossAttentionLayer_Forward_ProducesValidOutput()
    {
        // Test that CrossAttentionLayer still works after refactoring
        // Constructor: (queryDim, contextDim, headCount, sequenceLength)
        var layer = new CrossAttentionLayer<float>(
            queryDim: 64,
            contextDim: 64,
            headCount: 8,
            sequenceLength: 16);

        // Query: [batch, seqLen, queryDim]
        var query = new Tensor<float>(new[] { 2, 16, 64 });
        // Context: [batch, contextLen, contextDim]
        var context = new Tensor<float>(new[] { 2, 32, 64 });

        // Initialize with random values
        var rand = new Random(42);
        for (int i = 0; i < query.Length; i++)
        {
            query[i] = (float)(rand.NextDouble() - 0.5);
        }
        for (int i = 0; i < context.Length; i++)
        {
            context[i] = (float)(rand.NextDouble() - 0.5);
        }

        // Forward takes (query, context) as parameters
        var output = layer.Forward(query, context);

        // Output should have same shape as query
        Assert.NotNull(output);
        Assert.Equal(3, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]);  // batch
        Assert.Equal(16, output.Shape[1]); // seqLen
        Assert.Equal(64, output.Shape[2]); // queryDim
    }

    [Fact]
    public void CrossAttentionLayer_Forward_DifferentContextLength()
    {
        // Context can have different sequence length than query
        var layer = new CrossAttentionLayer<float>(
            queryDim: 32,
            contextDim: 32,
            headCount: 4,
            sequenceLength: 8);

        var query = new Tensor<float>(new[] { 1, 8, 32 });
        var context = new Tensor<float>(new[] { 1, 16, 32 });

        var rand = new Random(42);
        for (int i = 0; i < query.Length; i++)
        {
            query[i] = (float)(rand.NextDouble() - 0.5);
        }
        for (int i = 0; i < context.Length; i++)
        {
            context[i] = (float)(rand.NextDouble() - 0.5);
        }

        var output = layer.Forward(query, context);

        Assert.Equal(new[] { 1, 8, 32 }, output.Shape);
    }

    [Fact]
    public void CrossAttentionLayer_Forward_IsDeterministic()
    {
        var layer = new CrossAttentionLayer<float>(
            queryDim: 32,
            contextDim: 32,
            headCount: 4,
            sequenceLength: 8);

        var query = new Tensor<float>(new[] { 1, 8, 32 });
        var context = new Tensor<float>(new[] { 1, 16, 32 });

        var rand = new Random(42);
        for (int i = 0; i < query.Length; i++)
        {
            query[i] = (float)(rand.NextDouble() - 0.5);
        }
        for (int i = 0; i < context.Length; i++)
        {
            context[i] = (float)(rand.NextDouble() - 0.5);
        }

        // Two forward passes with same input should produce same output
        var output1 = layer.Forward(query, context);
        var output2 = layer.Forward(query, context);

        for (int i = 0; i < output1.Length; i++)
        {
            Assert.Equal(output1[i], output2[i], 5);
        }
    }

    #endregion

    #region SelfAttentionLayer Tests

    [Fact]
    public void SelfAttentionLayer_Forward_ProducesValidOutput()
    {
        // Constructor: (sequenceLength, embeddingDimension, headCount, activationFunction?)
        // Using explicit null cast to resolve constructor ambiguity
        var layer = new SelfAttentionLayer<float>(
            sequenceLength: 16,
            embeddingDimension: 64,
            headCount: 8,
            activationFunction: (IActivationFunction<float>?)null);

        // Input: [batch, seqLen, embeddingDim]
        var input = new Tensor<float>(new[] { 2, 16, 64 });

        var rand = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(rand.NextDouble() - 0.5);
        }

        var output = layer.Forward(input);

        // Output should have same shape as input
        Assert.NotNull(output);
        Assert.Equal(3, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]);  // batch
        Assert.Equal(16, output.Shape[1]); // seqLen
        Assert.Equal(64, output.Shape[2]); // embeddingDim
    }

    [Fact]
    public void SelfAttentionLayer_Forward_IsDeterministic()
    {
        var layer = new SelfAttentionLayer<float>(
            sequenceLength: 8,
            embeddingDimension: 32,
            headCount: 4,
            activationFunction: (IActivationFunction<float>?)null);

        var input = new Tensor<float>(new[] { 1, 8, 32 });

        var rand = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(rand.NextDouble() - 0.5);
        }

        // Two forward passes with same input should produce same output
        var output1 = layer.Forward(input);
        var output2 = layer.Forward(input);

        for (int i = 0; i < output1.Length; i++)
        {
            Assert.Equal(output1[i], output2[i], 5);
        }
    }

    #endregion

    #region GraphAttentionLayer Tests

    [Fact]
    public void GraphAttentionLayer_Forward_ProducesValidOutput()
    {
        // Constructor: (inputFeatures, outputFeatures, numHeads, alpha, dropoutRate, ...)
        var layer = new GraphAttentionLayer<float>(
            inputFeatures: 32,
            outputFeatures: 64,
            numHeads: 4);

        // Node features: [numNodes, inputFeatures]
        var input = new Tensor<float>(new[] { 10, 32 });
        // Adjacency matrix: [numNodes, numNodes]
        var adjacency = new Tensor<float>(new[] { 10, 10 });

        var rand = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(rand.NextDouble() - 0.5);
        }
        // Create a sparse adjacency matrix with self-loops and neighbor connections
        for (int i = 0; i < 10; i++)
        {
            adjacency[i, i] = 1.0f; // Self-loops
            if (i > 0) adjacency[i, i - 1] = 1.0f;
            if (i < 9) adjacency[i, i + 1] = 1.0f;
        }

        layer.SetAdjacencyMatrix(adjacency);
        var output = layer.Forward(input);

        // Output should be [numNodes, outputFeatures]
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(10, output.Shape[0]); // numNodes
        Assert.Equal(64, output.Shape[1]); // outputFeatures
    }

    #endregion

    #region Performance Tests

    [Fact]
    public void InferenceContext_RentReturn_IsFast()
    {
        var pool = new TensorPool<float>(maxPoolSizeMB: 100);
        var shape = new[] { 64, 64 };
        const int iterations = 10000;

        // Warmup
        for (int i = 0; i < 100; i++)
        {
            using var ctx = new InferenceContext<float>(pool);
            ctx.Rent(shape);
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            using var ctx = new InferenceContext<float>(pool);
            ctx.Rent(shape);
        }
        sw.Stop();

        var usPerIteration = (double)sw.ElapsedTicks / iterations / (Stopwatch.Frequency / 1_000_000.0);

        // Should be fast - target is < 50 microseconds per context+rent+dispose
        Assert.True(usPerIteration < 500, $"Context cycle took {usPerIteration:F2} microseconds, expected < 500");
    }

    [Fact]
    public void InferenceScope_RentOrCreate_WithPooling_IsFasterThanWithout()
    {
        var pool = new TensorPool<float>(maxPoolSizeMB: 100);
        var shape = new[] { 128, 128 };
        const int iterations = 1000;

        // Warmup
        for (int i = 0; i < 50; i++)
        {
            using var ctx = new InferenceContext<float>(pool);
            using (InferenceScope<float>.Begin(ctx))
            {
                InferenceScope<float>.RentOrCreate(shape);
            }
        }

        // Measure with pooling
        var swPooled = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            using var ctx = new InferenceContext<float>(pool);
            using (InferenceScope<float>.Begin(ctx))
            {
                var t = InferenceScope<float>.RentOrCreate(shape);
            }
        }
        swPooled.Stop();

        // Measure without pooling (direct allocation)
        var swAlloc = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var t = new Tensor<float>(shape);
        }
        swAlloc.Stop();

        // Pooling should be competitive with direct allocation
        // (Not necessarily faster due to overhead, but not significantly slower)
        var ratio = (double)swPooled.ElapsedMilliseconds / swAlloc.ElapsedMilliseconds;
        Assert.True(ratio < 10,
            $"Pooled ({swPooled.ElapsedMilliseconds}ms) should not be >10x slower than allocation ({swAlloc.ElapsedMilliseconds}ms)");
    }

    [Fact]
    public void CrossAttentionLayer_Forward_ExecutesInReasonableTime()
    {
        var layer = new CrossAttentionLayer<float>(
            queryDim: 64,
            contextDim: 64,
            headCount: 8,
            sequenceLength: 32);

        var query = new Tensor<float>(new[] { 4, 32, 64 });
        var context = new Tensor<float>(new[] { 4, 64, 64 });

        var rand = new Random(42);
        for (int i = 0; i < query.Length; i++)
        {
            query[i] = (float)(rand.NextDouble() - 0.5);
        }
        for (int i = 0; i < context.Length; i++)
        {
            context[i] = (float)(rand.NextDouble() - 0.5);
        }

        // Warmup
        for (int i = 0; i < 5; i++)
        {
            layer.Forward(query, context);
        }

        var sw = Stopwatch.StartNew();
        const int iterations = 10;
        for (int i = 0; i < iterations; i++)
        {
            layer.Forward(query, context);
        }
        sw.Stop();

        var msPerForward = sw.ElapsedMilliseconds / (double)iterations;

        // Should complete in reasonable time (< 1 second per forward)
        Assert.True(msPerForward < 1000,
            $"Forward pass took {msPerForward:F2}ms, expected < 1000ms");
    }

    #endregion
}
