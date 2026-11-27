using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.CreditAssignment;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// The training-only frame-alignment heads of <see cref="BlockwiseAnchorLocalErrorCreditRule{T}"/>: they change only the
/// intermediate anchors' local objectives, they are trained by the rule itself, and a configuration they cannot serve
/// fails loudly instead of being ignored.
/// </summary>
public sealed class BlockwiseAnchorFrameAlignmentTests
{
    private const int Width = 16;
    private const int Vocab = 48;

    private static (NeuralNetwork<float> net, Tensor<float> tokens, Tensor<float> targets) BuildLm(int blocks, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var emb = new EmbeddingLayer<float>(Vocab, Width);
        var layers = new List<ILayer<float>> { emb };
        for (int b = 0; b < blocks; b++)
            layers.Add(new DenseLayer<float>(Width, activationFunction: new TanhActivation<float>()));
        layers.Add(new RMSNormalizationLayer<float>(Width));
        layers.Add(new DenseLayer<float>(Vocab, activationFunction: new IdentityActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: Width,
            outputSize: Vocab,
            layers: layers)
        {
            RandomSeed = seed
        };

        var net = new NeuralNetwork<float>(arch, lossFunction: new CrossEntropyWithLogitsLoss<float>());
        var tokens = new Tensor<float>(new[] { 2, 4 });
        var targets = new Tensor<float>(new[] { 2, 4 });
        for (int i = 0; i < tokens.Length; i++)
        {
            tokens.SetFlat(i, rng.Next(Vocab));
            targets.SetFlat(i, rng.Next(Vocab));
        }
        _ = net.Predict(tokens);
        return (net, tokens, targets);
    }

    private static BlockwiseAnchorLocalErrorCreditRule<float> Plain() =>
        new(readoutLayerCount: 2, anchorStride: 4, maxCandidates: 16, localWeight: 0.8,
            detachReadoutOnHiddenBlocks: true, useDepthProgressiveWeight: true, streamBlocksSequentially: true, seed: 123);

    private static BlockwiseAnchorLocalErrorCreditRule<float> WithHeads(int hiddenDimension = Width, int maxAnchorBlocks = 2) =>
        BlockwiseAnchorLocalErrorCreditRule<float>.CreateWithFrameAlignment(
            readoutLayerCount: 2, anchorStride: 4, maxCandidates: 16, localWeight: 0.8,
            detachReadoutOnHiddenBlocks: true, useDepthProgressiveWeight: true, streamBlocksSequentially: true, seed: 123,
            auxiliaryProjectionRank: 4, hiddenDimension: hiddenDimension, maxAnchorBlocks: maxAnchorBlocks);

    // Per hidden layer (0 = embedding), the slice of the flat gradient that belongs to its parameters.
    private static float[][] PerLayer(NeuralNetwork<float> net, Vector<float> grad, int layerCount)
    {
        var offsets = new Dictionary<Tensor<float>, int>(TensorReferenceComparer<Tensor<float>>.Instance);
        int offset = 0;
        foreach (var chunk in net.GetParameterChunks())
        {
            if (chunk is null || chunk.Length == 0) continue;
            offsets[chunk] = offset;
            offset += chunk.Length;
        }
        var result = new float[layerCount][];
        for (int layer = 0; layer < layerCount; layer++)
        {
            var values = new List<float>();
            foreach (var p in ((ITrainableLayer<float>)net.Layers[layer]).GetTrainableParameters())
                for (int i = 0; i < p.Length; i++) values.Add(grad[offsets[p] + i]);
            result[layer] = values.ToArray();
        }
        return result;
    }

    [Fact]
    public void Heads_ChangeOnlyTheIntermediateBlock_AndAreTrainedByTheRule()
    {
        // 7 dense blocks + embedding = 8 hidden layers, stride 4: block 0 = layers 0..3 anchored at 3 (intermediate, head
        // slot 0), block 1 = layers 4..7 anchored at the terminal output (no head).
        var (netPlain, tokens, targets) = BuildLm(blocks: 7, seed: 77);
        var (netHeads, _, _) = BuildLm(blocks: 7, seed: 77);
        netHeads.SetParameters(netPlain.GetParameters());

        var heads = WithHeads();
        Assert.Empty(Plain().AuxiliaryParameters);
        Assert.Equal(4, heads.AuxiliaryParameters.Count);
        Assert.All(heads.AuxiliaryParameters, p => Assert.Equal(new[] { Width, 4 }, p.Shape.ToArray()));
        var before = heads.AuxiliaryParameters.Select(p => p.ToArray()).ToArray();

        var loss = new CrossEntropyWithLogitsLoss<float>();
        netPlain.SetCreditRule(Plain());
        netHeads.SetCreditRule(heads);
        var gPlain = PerLayer(netPlain, netPlain.ComputeGradients(tokens, targets, loss), 8);
        var gHeads = PerLayer(netHeads, netHeads.ComputeGradients(tokens, targets, loss), 8);

        // Block 0 trains on the head-adapted anchor loss, so its gradients move; block 1 never sees a head.
        double block0Diff = Enumerable.Range(0, 4).Sum(l => gPlain[l].Zip(gHeads[l], (a, b) => Math.Abs(a - b)).Max());
        Assert.True(block0Diff > 1e-7, $"The heads did not change block 0's gradients (max diff sum {block0Diff}).");
        for (int l = 4; l < 8; l++)
            for (int i = 0; i < gPlain[l].Length; i++)
                Assert.True(Math.Abs(gPlain[l][i] - gHeads[l][i]) <= 1e-6f * Math.Max(1f, Math.Abs(gPlain[l][i])),
                    $"Block 1 layer {l} gradient {i} changed under the heads: {gPlain[l][i]} vs {gHeads[l][i]}.");

        // The rule's own step trained slot 0's two matrices and left the unused slot 1 alone.
        var after = heads.AuxiliaryParameters.Select(p => p.ToArray()).ToArray();
        for (int k = 0; k < 2; k++)
            Assert.True(before[k].Zip(after[k], (a, b) => Math.Abs(a - b)).Max() > 0f, $"Slot 0 head matrix {k} was not trained.");
        for (int k = 2; k < 4; k++)
            Assert.Equal(before[k], after[k]);
    }

    [Fact]
    public void AnIntermediateBlockWithoutAHead_Throws()
    {
        // 11 dense blocks + embedding = 12 hidden layers: intermediate anchors at layers 3 (slot 0) and 7 (slot 1).
        var (net, tokens, targets) = BuildLm(blocks: 11, seed: 5);
        net.SetCreditRule(WithHeads(maxAnchorBlocks: 1));
        var ex = Assert.ThrowsAny<Exception>(() => net.ComputeGradients(tokens, targets, new CrossEntropyWithLogitsLoss<float>()));
        Assert.Contains("maxAnchorBlocks", Flatten(ex));
    }

    [Fact]
    public void HiddenStatesOfAnotherWidth_Throw()
    {
        var (net, tokens, targets) = BuildLm(blocks: 7, seed: 5);
        net.SetCreditRule(WithHeads(hiddenDimension: Width / 2));
        var ex = Assert.ThrowsAny<Exception>(() => net.ComputeGradients(tokens, targets, new CrossEntropyWithLogitsLoss<float>()));
        Assert.Contains("width 8", Flatten(ex));
    }

    [Theory]
    [InlineData(-1, Width, 8, 1e-3, "auxiliaryProjectionRank")]
    [InlineData(4, 0, 8, 1e-3, "hiddenDimension")]
    [InlineData(4, Width, 0, 1e-3, "maxAnchorBlocks")]
    [InlineData(4, Width, 8, 0.0, "auxiliaryLearningRate")]
    [InlineData(4, Width, 8, double.NaN, "auxiliaryLearningRate")]
    public void InvalidHeadSettings_Throw(int rank, int hidden, int maxBlocks, double lr, string parameter)
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            BlockwiseAnchorLocalErrorCreditRule<float>.CreateWithFrameAlignment(
                2, 4, 16, 0.8, true, true, true, 1, rank, hidden, maxBlocks, lr));
        Assert.Equal(parameter, ex.ParamName);
    }

    private static string Flatten(Exception ex)
    {
        var text = new System.Text.StringBuilder();
        for (Exception? e = ex; e is not null; e = e.InnerException) text.Append(e.Message).Append(" | ");
        return text.ToString();
    }
}
