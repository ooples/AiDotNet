using System.Threading.Tasks;
namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

/// <summary>
/// Integration tests for attention layer implementations testing any-rank tensor support,
/// forward/backward passes, multi-input scenarios, and cloning.
/// </summary>
public class AttentionLayersIntegrationTests
{
    #region AttentionLayer Tests

    [Fact(Timeout = 120000)]
    public async Task AttentionLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [batch, features]
        int inputSize = 64;
        int attentionSize = 32;
        var layer = new AttentionLayer<float>(attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input [batch, seq, features]
        int inputSize = 64;
        int attentionSize = 32;
        var layer = new AttentionLayer<float>(attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, 10, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }


    [Fact(Timeout = 120000)]
    public async Task AttentionLayer_CrossAttention_ProducesValidOutput()
    {
        // Arrange - cross-attention with separate query and key/value inputs
        int inputSize = 64;
        int attentionSize = 32;
        var layer = new AttentionLayer<float>(attentionSize, (IActivationFunction<float>?)null);
        var query = CreateRandomTensor<float>([2, 8, inputSize]);
        var keyValue = CreateRandomTensor<float>([2, 12, inputSize]);

        // Act
        var output = layer.Forward(query, keyValue);

        // Assert
        Assert.Equal([2, 8, inputSize], output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionLayer_MaskedAttention_ProducesValidOutput()
    {
        // Arrange - attention with mask
        int inputSize = 32;
        int attentionSize = 16;
        var layer = new AttentionLayer<float>(attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, 6, inputSize]);
        // Mask shape: [batch, queryLen, keyLen]
        var mask = CreateMaskTensor([2, 6, 6]);

        // Act
        var output = layer.Forward(input, mask);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 32;
        int attentionSize = 16;
        var original = new AttentionLayer<float>(attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([4, inputSize]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (AttentionLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape.ToArray(), clonedOutput.Shape.ToArray());
    }

    #endregion

    #region SelfAttentionLayer Tests

    [Fact(Timeout = 120000)]
    public async Task SelfAttentionLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [seqLen, embedDim]
        int seqLen = 16;
        int embedDim = 64;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 4, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task SelfAttentionLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input [batch, seqLen, embedDim]
        int seqLen = 16;
        int embedDim = 64;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 8, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task SelfAttentionLayer_ForwardPass_4D_ProducesValidOutput()
    {
        // Arrange - 4D input [batch1, batch2, seqLen, embedDim]
        int seqLen = 8;
        int embedDim = 32;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 4, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, 3, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }


    [Fact(Timeout = 120000)]
    public async Task SelfAttentionLayer_MultiHeadConfiguration_Works()
    {
        // Arrange - embedDim must be divisible by headCount
        int seqLen = 16;
        int embedDim = 96; // 96 / 12 = 8
        int headCount = 12;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task SelfAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int seqLen = 8;
        int embedDim = 32;
        var original = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 4, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (SelfAttentionLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape.ToArray(), clonedOutput.Shape.ToArray());
    }

    #endregion

    #region MultiHeadAttentionLayer Tests

    [Fact(Timeout = 120000)]
    public async Task MultiHeadAttentionLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [seqLen, embedDim]
        int seqLen = 16;
        int embedDim = 64;
        var layer = new MultiHeadAttentionLayer<float>(8, (embedDim) / (8));
        var input = CreateRandomTensor<float>([seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task MultiHeadAttentionLayer_ForwardPass_3D_ProducesValidOutput()
    {
        // Arrange - 3D input [batch, seqLen, embedDim]
        int seqLen = 16;
        int embedDim = 64;
        var layer = new MultiHeadAttentionLayer<float>(8, (embedDim) / (8));
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task MultiHeadAttentionLayer_ForwardPass_5D_ProducesValidOutput()
    {
        // Arrange - 5D input [batch1, batch2, batch3, seqLen, embedDim]
        int seqLen = 4;
        int embedDim = 32;
        var layer = new MultiHeadAttentionLayer<float>(4, (embedDim) / (4));
        var input = CreateRandomTensor<float>([2, 2, 2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }


    [Fact(Timeout = 120000)]
    public async Task MultiHeadAttentionLayer_CrossAttention_ProducesValidOutput()
    {
        // Arrange - cross-attention with separate query and key/value
        int seqLen = 8;
        int embedDim = 32;
        var layer = new MultiHeadAttentionLayer<float>(4, (embedDim) / (4));
        var query = CreateRandomTensor<float>([2, seqLen, embedDim]);
        var keyValue = CreateRandomTensor<float>([2, 12, embedDim]);

        // Act
        var output = layer.Forward(query, keyValue);

        // Assert
        Assert.Equal([2, seqLen, embedDim], output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task MultiHeadAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int seqLen = 8;
        int embedDim = 32;
        var original = new MultiHeadAttentionLayer<float>(4, (embedDim) / (4));
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (MultiHeadAttentionLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape.ToArray(), clonedOutput.Shape.ToArray());
    }

    #endregion

    #region FlashAttentionLayer Tests

    [Fact(Timeout = 120000)]
    public async Task FlashAttentionLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D input [seqLen, embedDim]
        int seqLen = 8;
        int embedDim = 32;
        int headCount = 4;
        var layer = new FlashAttentionLayer<float>(seqLen, embedDim, headCount);
        var input = CreateRandomTensor<float>([seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task FlashAttentionLayer_ForwardPass_4D_ProducesValidOutput()
    {
        // Arrange - 4D input [batch1, batch2, seqLen, embedDim]
        int seqLen = 6;
        int embedDim = 24;
        int headCount = 4;
        var layer = new FlashAttentionLayer<float>(seqLen, embedDim, headCount);
        var input = CreateRandomTensor<float>([2, 3, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    /// <summary>
    /// Regression test: FlashAttentionLayer must propagate non-zero gradients to ALL
    /// of its registered trainable parameters (Q/K/V/O projection weights + output bias).
    ///
    /// <para>This guards against the FA-vs-MHA degenerate-output bug observed in the
    /// HarmonicEngine consumer at dModel=128/L=2/ctx=64/10KB WikiText-2: FA arm produced
    /// top1=0% / top5=100% / ppl=V=256 uniform-output, indicating that gradient flow
    /// through Engine.FlashAttention to the layer's projection weights was broken.</para>
    ///
    /// <para>If this test fails, the layer's Forward composition is using Engine ops that
    /// don't propagate gradients on the autodiff tape, OR Engine.FlashAttention itself
    /// fails to record a tape op for the registered parameters' downstream computation.</para>
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task FlashAttentionLayer_GradientFlow_ReachesAllRegisteredParameters()
    {
        await Task.Yield();

        // Arrange — same shapes as a tiny transformer training block.
        int batchSize = 2;
        int seqLen = 8;
        int embedDim = 16;
        int headCount = 2;

        var layer = new FlashAttentionLayer<double>(seqLen, embedDim, headCount);
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor<double>([batchSize, seqLen, embedDim]);

        var trainable = layer as ITrainableLayer<double>;
        Assert.NotNull(trainable);

        // Act — forward through a tape, then random-projection scalar loss.
        using var tape = new GradientTape<double>();
        var output = layer.Forward(input);

        // Pull params AFTER Forward (lazy-init layers reassign their tensor refs).
        var trainableParams = trainable.GetTrainableParameters();
        Assert.True(trainableParams.Count > 0, "FlashAttentionLayer must expose trainable parameters.");

        var projection = CreateRandomTensor<double>(output.Shape.ToArray());
        var elementwise = AiDotNetEngine.Current.TensorMultiply(output, projection);
        var allAxes = new int[elementwise.Shape.Length];
        for (int i = 0; i < allAxes.Length; i++) allAxes[i] = i;
        var loss = AiDotNetEngine.Current.ReduceSum(elementwise, allAxes, keepDims: false);

        var grads = tape.ComputeGradients(loss, trainableParams);

        // Assert — EVERY registered parameter must have a non-zero gradient.
        // Per-parameter check (not just "some param got a grad") because the bug
        // could leave one weight tensor stranded outside the tape graph.
        int paramsWithGrad = 0;
        int paramsWithNonZeroGrad = 0;
        var failed = new System.Text.StringBuilder();
        for (int p = 0; p < trainableParams.Count; p++)
        {
            var param = trainableParams[p];
            if (!grads.TryGetValue(param, out var grad) || grad is null)
            {
                failed.AppendLine($"Param {p} (shape [{string.Join(",", param.Shape)}]): no gradient recorded on tape.");
                continue;
            }
            paramsWithGrad++;
            bool anyNonZero = false;
            for (int i = 0; i < grad.Length; i++)
            {
                if (Math.Abs(grad[i]) > 1e-9)
                {
                    anyNonZero = true;
                    break;
                }
            }
            if (!anyNonZero)
            {
                failed.AppendLine($"Param {p} (shape [{string.Join(",", param.Shape)}]): gradient is all-zero (length={grad.Length}).");
                continue;
            }
            paramsWithNonZeroGrad++;
        }

        Assert.True(
            paramsWithNonZeroGrad == trainableParams.Count,
            $"FlashAttentionLayer gradient flow broken. {paramsWithNonZeroGrad}/{trainableParams.Count} " +
            $"registered params received non-zero gradients (got grad: {paramsWithGrad}). " +
            $"Failures:\n{failed}");
    }

    /// <summary>
    /// Regression test: a tiny model built around FlashAttentionLayer must train
    /// (loss decreases by at least 5% over 20 SGD steps on a random target).
    ///
    /// <para>This reproduces the HarmonicEngine PathB FlashAttention sanity failure
    /// (top1=0% / top5=100% / ppl=V on 10KB WikiText-2) at a tiny scale that runs in
    /// under a second. If the layer trains here, the HE consumer test will also train.</para>
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task FlashAttentionLayer_TrainsViaTape_LossDecreases()
    {
        await Task.Yield();

        // Arrange — single layer + scalar SGD loop on a random regression target.
        int batchSize = 2;
        int seqLen = 8;
        int embedDim = 16;
        int headCount = 2;
        const double LearningRate = 0.01;
        const int Steps = 20;

        var layer = new FlashAttentionLayer<double>(seqLen, embedDim, headCount);
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor<double>([batchSize, seqLen, embedDim]);
        // Fixed random target — the layer must learn to map input → target.
        var target = CreateRandomTensor<double>([batchSize, seqLen, embedDim]);

        var trainable = (ITrainableLayer<double>)layer;

        double LossAt()
        {
            // No tape — pure forward pass for loss observation.
            var fwd = layer.Forward(input);
            double sse = 0.0;
            for (int i = 0; i < fwd.Length; i++)
            {
                double d = fwd[i] - target[i];
                sse += d * d;
            }
            return sse;
        }

        double initialLoss = LossAt();

        // Manual SGD on the layer's registered parameters via tape gradients.
        for (int step = 0; step < Steps; step++)
        {
            using var tape = new GradientTape<double>();
            var output = layer.Forward(input);
            var trainableParams = trainable.GetTrainableParameters();

            // Loss = sum((output - target)^2). Engine ops only — all tape-tracked.
            var diff = AiDotNetEngine.Current.TensorSubtract(output, target);
            var sq = AiDotNetEngine.Current.TensorMultiply(diff, diff);
            var allAxes = new int[sq.Shape.Length];
            for (int i = 0; i < allAxes.Length; i++) allAxes[i] = i;
            var loss = AiDotNetEngine.Current.ReduceSum(sq, allAxes, keepDims: false);

            var grads = tape.ComputeGradients(loss, trainableParams);

            // Apply SGD step in-place on each registered parameter.
            for (int p = 0; p < trainableParams.Count; p++)
            {
                var param = trainableParams[p];
                if (!grads.TryGetValue(param, out var grad) || grad is null) continue;
                var pSpan = param.Data.Span;
                var gSpan = grad.Data.Span;
                for (int i = 0; i < pSpan.Length; i++)
                {
                    pSpan[i] = pSpan[i] - LearningRate * gSpan[i];
                }
            }
        }

        double finalLoss = LossAt();

        Assert.True(
            finalLoss < initialLoss * 0.95,
            $"FlashAttentionLayer failed to train. Initial loss = {initialLoss:F6}, " +
            $"final loss after {Steps} SGD steps = {finalLoss:F6} (must be < {initialLoss * 0.95:F6}). " +
            "This indicates that gradient flow through Engine.FlashAttention to the layer's " +
            "Q/K/V/O projection weights is broken — the parameters are not being updated.");
    }

    /// <summary>
    /// Sanity test: FlashAttentionLayer should train AT LEAST as well as MultiHeadAttentionLayer
    /// on the same regression task. FA is documented as a drop-in replacement; loss reduction
    /// magnitudes should be within an order of magnitude.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task FlashAttentionLayer_VsMultiHeadAttention_TrainsComparably()
    {
        await Task.Yield();

        int batchSize = 2;
        int seqLen = 8;
        int embedDim = 16;
        int headCount = 2;
        const double LearningRate = 0.01;
        const int Steps = 20;

        // Shared input + target across both arms.
        var input = CreateRandomTensor<double>([batchSize, seqLen, embedDim]);
        var target = CreateRandomTensor<double>([batchSize, seqLen, embedDim]);

        double TrainLayerAndGetRatio(LayerBase<double> layer)
        {
            layer.SetTrainingMode(true);
            var trainable = (ITrainableLayer<double>)layer;
            double Loss()
            {
                var fwd = layer.Forward(input);
                double sse = 0.0;
                for (int i = 0; i < fwd.Length; i++)
                {
                    double d = fwd[i] - target[i];
                    sse += d * d;
                }
                return sse;
            }
            double initial = Loss();
            for (int step = 0; step < Steps; step++)
            {
                using var tape = new GradientTape<double>();
                var output = layer.Forward(input);
                var trainableParams = trainable.GetTrainableParameters();
                var diff = AiDotNetEngine.Current.TensorSubtract(output, target);
                var sq = AiDotNetEngine.Current.TensorMultiply(diff, diff);
                var allAxes = new int[sq.Shape.Length];
                for (int i = 0; i < allAxes.Length; i++) allAxes[i] = i;
                var loss = AiDotNetEngine.Current.ReduceSum(sq, allAxes, keepDims: false);
                var grads = tape.ComputeGradients(loss, trainableParams);
                for (int p = 0; p < trainableParams.Count; p++)
                {
                    var param = trainableParams[p];
                    if (!grads.TryGetValue(param, out var grad) || grad is null) continue;
                    var pSpan = param.Data.Span;
                    var gSpan = grad.Data.Span;
                    for (int i = 0; i < pSpan.Length; i++)
                    {
                        pSpan[i] = pSpan[i] - LearningRate * gSpan[i];
                    }
                }
            }
            double final = Loss();
            return final / initial;
        }

        // FlashAttention arm
        var faLayer = new FlashAttentionLayer<double>(seqLen, embedDim, headCount);
        double faRatio = TrainLayerAndGetRatio(faLayer);

        // MultiHeadAttention arm — MHA ctor takes (headCount, headDimension), so
        // headDimension = embedDim/headCount to keep the same total embedding width.
        var mhaLayer = new MultiHeadAttentionLayer<double>(headCount, embedDim / headCount);
        double mhaRatio = TrainLayerAndGetRatio(mhaLayer);

        // Both must reduce loss; FA must be within 10x of MHA's reduction quality
        // (FA cannot regress wildly versus the documented drop-in replacement target).
        Assert.True(faRatio < 1.0, $"FlashAttention did not decrease loss (final/initial = {faRatio:F4}).");
        Assert.True(mhaRatio < 1.0, $"MultiHeadAttention did not decrease loss (final/initial = {mhaRatio:F4}).");
        Assert.True(
            faRatio < mhaRatio * 10.0,
            $"FlashAttention loss-reduction ({faRatio:F4}) is more than 10x worse than MultiHeadAttention ({mhaRatio:F4}). " +
            "FlashAttention is documented as a drop-in replacement; this gap indicates a backward-pass bug.");
    }


    #endregion

    #region CrossAttentionLayer Tests

    [Fact(Timeout = 120000)]
    public async Task CrossAttentionLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int queryDim = 64;
        int contextDim = 64;
        int seqLen = 16;
        var layer = new CrossAttentionLayer<float>(queryDim, contextDim, headCount: 8, sequenceLength: seqLen);
        var query = CreateRandomTensor<float>([2, seqLen, queryDim]);
        var context = CreateRandomTensor<float>([2, seqLen, contextDim]);

        // Act
        var output = layer.Forward(query, context);

        // Assert
        Assert.Equal([2, seqLen, queryDim], output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task CrossAttentionLayer_ForwardPass_DifferentContextLength_ProducesValidOutput()
    {
        // Arrange - context can have different sequence length
        int queryDim = 32;
        int contextDim = 32;
        int querySeqLen = 8;
        int contextSeqLen = 16;
        var layer = new CrossAttentionLayer<float>(queryDim, contextDim, headCount: 4, sequenceLength: querySeqLen);
        var query = CreateRandomTensor<float>([2, querySeqLen, queryDim]);
        var context = CreateRandomTensor<float>([2, contextSeqLen, contextDim]);

        // Act
        var output = layer.Forward(query, context);

        // Assert
        Assert.Equal([2, querySeqLen, queryDim], output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }


    [Fact(Timeout = 120000)]
    public async Task CrossAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int dim = 32;
        int seqLen = 8;
        var original = new CrossAttentionLayer<float>(dim, dim, headCount: 4, sequenceLength: seqLen);
        var query = CreateRandomTensor<float>([2, seqLen, dim]);
        var context = CreateRandomTensor<float>([2, seqLen, dim]);
        var originalOutput = original.Forward(query, context);

        // Act
        var cloned = (CrossAttentionLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(query, context);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape.ToArray(), clonedOutput.Shape.ToArray());
    }

    #endregion

    #region GraphAttentionLayer Tests

    [Fact(Timeout = 120000)]
    public async Task GraphAttentionLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - graph attention with node features and adjacency matrix
        int inputFeatures = 32;
        int outputFeatures = 16;
        int numNodes = 10;
        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 4);
        var nodeFeatures = CreateRandomTensor<float>([numNodes, inputFeatures]);
        var adjacency = CreateRandomAdjacencyMatrix(numNodes);
        layer.SetAdjacencyMatrix(adjacency);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.Equal([numNodes, outputFeatures], output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task GraphAttentionLayer_ForwardPass_BatchedInput_ProducesValidOutput()
    {
        // Arrange - batched graph input
        int inputFeatures = 32;
        int outputFeatures = 16;
        int batchSize = 2;
        int numNodes = 8;
        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 2);
        var nodeFeatures = CreateRandomTensor<float>([batchSize, numNodes, inputFeatures]);
        var adjacency = CreateRandomAdjacencyMatrix(numNodes);
        layer.SetAdjacencyMatrix(adjacency);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.Equal([batchSize, numNodes, outputFeatures], output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }


    [Fact(Timeout = 120000)]
    public async Task GraphAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputFeatures = 16;
        int outputFeatures = 8;
        int numNodes = 6;
        var original = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 2);
        var nodeFeatures = CreateRandomTensor<float>([numNodes, inputFeatures]);
        var adjacency = CreateRandomAdjacencyMatrix(numNodes);
        original.SetAdjacencyMatrix(adjacency);
        var originalOutput = original.Forward(nodeFeatures);

        // Act
        var cloned = (GraphAttentionLayer<float>)original.Clone();
        cloned.SetAdjacencyMatrix(adjacency);
        var clonedOutput = cloned.Forward(nodeFeatures);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(originalOutput.Shape.ToArray(), clonedOutput.Shape.ToArray());
    }

    #endregion

    #region Edge Cases

    [Fact(Timeout = 120000)]
    public async Task AttentionLayer_SingleBatch_Works()
    {
        // Arrange
        int inputSize = 32;
        int attentionSize = 16;
        var layer = new AttentionLayer<float>(attentionSize, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([1, 4, inputSize]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task SelfAttentionLayer_SingleHead_Works()
    {
        // Arrange - single attention head
        int seqLen = 8;
        int embedDim = 32;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 1, (IActivationFunction<float>?)null);
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task MultiHeadAttentionLayer_LargeHeadCount_Works()
    {
        // Arrange - many attention heads
        int seqLen = 8;
        int embedDim = 64;
        int headCount = 16; // 64 / 16 = 4 per head
        var layer = new MultiHeadAttentionLayer<float>(headCount, (embedDim) / (headCount));
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task GraphAttentionLayer_SingleHead_Works()
    {
        // Arrange
        int inputFeatures = 16;
        int outputFeatures = 8;
        int numNodes = 4;
        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 1);
        var nodeFeatures = CreateRandomTensor<float>([numNodes, inputFeatures]);
        var adjacency = CreateRandomAdjacencyMatrix(numNodes);
        layer.SetAdjacencyMatrix(adjacency);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionLayer_AuxiliaryLoss_Works()
    {
        // Arrange
        int inputSize = 32;
        int attentionSize = 16;
        var layer = new AttentionLayer<float>(attentionSize, (IActivationFunction<float>?)null);
        layer.UseAuxiliaryLoss = true;
        var input = CreateRandomTensor<float>([2, 8, inputSize]);

        // Act
        var output = layer.Forward(input);
        var auxLoss = layer.ComputeAuxiliaryLoss();

        // Assert
        Assert.False(ContainsNaN(output));
        Assert.False(float.IsNaN(auxLoss));
    }

    [Fact(Timeout = 120000)]
    public async Task SelfAttentionLayer_AuxiliaryLoss_Works()
    {
        // Arrange
        int seqLen = 8;
        int embedDim = 32;
        var layer = new SelfAttentionLayer<float>(seqLen, embedDim, headCount: 4, (IActivationFunction<float>?)null);
        layer.UseAuxiliaryLoss = true;
        var input = CreateRandomTensor<float>([2, seqLen, embedDim]);

        // Act
        var output = layer.Forward(input);
        var auxLoss = layer.ComputeAuxiliaryLoss();

        // Assert
        Assert.False(ContainsNaN(output));
        Assert.False(float.IsNaN(auxLoss));
    }

    #endregion

    #region Helper Methods

    private static Tensor<T> CreateRandomTensor<T>(int[] shape) where T : struct, IComparable<T>
    {
        var tensor = new Tensor<T>(shape);
        var random = new Random(42);

        for (int i = 0; i < tensor.Length; i++)
        {
            double value = random.NextDouble() * 2 - 1; // [-1, 1]
            tensor[i] = (T)Convert.ChangeType(value, typeof(T));
        }

        return tensor;
    }

    private static Tensor<float> CreateMaskTensor(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(42);

        // Create a causal mask (lower triangular)
        for (int b = 0; b < shape[0]; b++)
        {
            for (int i = 0; i < shape[1]; i++)
            {
                for (int j = 0; j < shape[2]; j++)
                {
                    // 0 for attended positions, -inf for masked positions
                    tensor[new int[] { b, i, j }] = j <= i ? 0f : float.NegativeInfinity;
                }
            }
        }

        return tensor;
    }

    private static Tensor<float> CreateRandomAdjacencyMatrix(int numNodes)
    {
        var tensor = new Tensor<float>([numNodes, numNodes]);
        var random = new Random(42);

        // Create a random adjacency matrix (sparse, symmetric)
        for (int i = 0; i < numNodes; i++)
        {
            tensor[new int[] { i, i }] = 1f; // Self-loops
            for (int j = i + 1; j < numNodes; j++)
            {
                float edge = random.NextDouble() > 0.5 ? 1f : 0f;
                tensor[new int[] { i, j }] = edge;
                tensor[new int[] { j, i }] = edge; // Symmetric
            }
        }

        return tensor;
    }

    private static bool ContainsNaN<T>(Tensor<T> tensor) where T : struct, IComparable<T>
    {
        foreach (var value in tensor.ToArray())
        {
            if (value is float f && float.IsNaN(f)) return true;
            if (value is double d && double.IsNaN(d)) return true;
        }
        return false;
    }

    #endregion
}
