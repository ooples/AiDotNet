using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.CreditAssignment;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

public sealed class DeepScaleTrack2UpstreamTests
{
    private static (NeuralNetwork<float> net, Tensor<float> tokens, Tensor<float> targets) BuildLm(
        int blocks,
        int width,
        int vocab,
        bool tiedHead,
        int seed,
        IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>? optimizer = null)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var emb = new EmbeddingLayer<float>(vocab, width);
        var layers = new List<ILayer<float>> { emb };
        for (int b = 0; b < blocks; b++)
        {
            layers.Add(new DenseLayer<float>(width, activationFunction: new TanhActivation<float>()));
        }
        layers.Add(new RMSNormalizationLayer<float>(width));
        layers.Add(tiedHead
            ? new TiedEmbeddingHeadLayer<float>(emb)
            : new DenseLayer<float>(vocab, activationFunction: new IdentityActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: width,
            outputSize: vocab,
            layers: layers)
        {
            RandomSeed = seed
        };

        var net = new NeuralNetwork<float>(arch, optimizer: optimizer, lossFunction: new CrossEntropyWithLogitsLoss<float>());
        int batch = 2, seq = 4;
        var tokens = new Tensor<float>(new[] { batch, seq });
        var targets = new Tensor<float>(new[] { batch, seq });
        for (int i = 0; i < tokens.Length; i++)
        {
            tokens.SetFlat(i, rng.Next(vocab));
            targets.SetFlat(i, rng.Next(vocab));
        }

        // Materialize any lazy state cleanly before tests inspect parameters.
        _ = net.Predict(tokens);
        return (net, tokens, targets);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(7)]
    public void HouseholderSubspaceLayer_IsExactOrthogonalIsometryAndMatchesNumericalGradients(int numReflectors)
    {
        const int dim = 12;
        var fwd = new HouseholderSubspaceLayer<double>(dim, numReflectors, transpose: false);
        var inv = new HouseholderSubspaceLayer<double>(dim, numReflectors, transpose: true);

        // Perturb reflectors away from unit norm to prove Q^T Q = I holds without unit-norm constraints.
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(100 + numReflectors);
        for (int i = 0; i < fwd.Reflectors.Length; i++)
        {
            double val = (rng.NextDouble() * 2.0 - 1.0) * 2.5;
            if (Math.Abs(val) < 0.05) val = 0.25;
            fwd.Reflectors.SetFlat(i, val);
            inv.Reflectors.SetFlat(i, val);
        }

        var x3d = new Tensor<double>(new[] { 2, 3, dim });
        double xNormSq = 0.0;
        for (int i = 0; i < x3d.Length; i++)
        {
            double v = (rng.NextDouble() * 2.0 - 1.0) * 1.7;
            x3d.SetFlat(i, v);
            xNormSq += v * v;
        }

        var y = fwd.Forward(x3d);
        Assert.Equal(x3d._shape, y._shape);

        double yNormSq = 0.0;
        for (int i = 0; i < y.Length; i++)
            yNormSq += y.GetFlat(i) * y.GetFlat(i);
        Assert.True(Math.Abs(xNormSq - yNormSq) <= 1e-11 * Math.Max(1.0, xNormSq),
            $"Householder chain failed norm preservation: ||x||^2={xNormSq}, ||Qx||^2={yNormSq}");

        var reconstructed = inv.Forward(y);
        for (int i = 0; i < x3d.Length; i++)
        {
            Assert.True(Math.Abs(x3d.GetFlat(i) - reconstructed.GetFlat(i)) <= 1e-11,
                $"Q^T Q x != x at index {i}: expected {x3d.GetFlat(i)}, got {reconstructed.GetFlat(i)}");
        }

        // Finite-difference gradient check wrt both input and reflectors.
        var x2d = new Tensor<double>(new[] { 3, dim });
        var wTarget = new Tensor<double>(new[] { 3, dim });
        for (int i = 0; i < x2d.Length; i++)
        {
            x2d.SetFlat(i, (rng.NextDouble() * 2.0 - 1.0));
            wTarget.SetFlat(i, (rng.NextDouble() * 2.0 - 1.0));
        }

        double EvalLoss()
        {
            var outTensor = fwd.Forward(x2d);
            double sum = 0.0;
            for (int i = 0; i < outTensor.Length; i++)
            {
                double diff = outTensor.GetFlat(i) - wTarget.GetFlat(i);
                sum += 0.5 * diff * diff;
            }
            return sum;
        }

        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var outTensor = fwd.Forward(x2d);
            var diff = AiDotNetEngine.Current.TensorSubtract(outTensor, wTarget);
            var sq = AiDotNetEngine.Current.TensorMultiply(diff, diff);
            var loss = AiDotNetEngine.Current.TensorMultiplyScalar(
                AiDotNetEngine.Current.ReduceSum(sq, null), 0.5);
            grads = tape.ComputeGradients(loss, sources: new[] { x2d, fwd.Reflectors });
        }

        const double eps = 1e-6;
        var gx = grads[x2d];
        for (int i = 0; i < x2d.Length; i++)
        {
            double orig = x2d.GetFlat(i);
            x2d.SetFlat(i, orig + eps);
            double lp = EvalLoss();
            x2d.SetFlat(i, orig - eps);
            double lm = EvalLoss();
            x2d.SetFlat(i, orig);
            double num = (lp - lm) / (2.0 * eps);
            Assert.True(Math.Abs(gx.GetFlat(i) - num) <= 1e-6 * Math.Max(1.0, Math.Abs(num)),
                $"dL/dx mismatch at {i}: analytic={gx.GetFlat(i)}, numeric={num}");
        }

        var gr = grads[fwd.Reflectors];
        for (int i = 0; i < fwd.Reflectors.Length; i++)
        {
            double orig = fwd.Reflectors.GetFlat(i);
            fwd.Reflectors.SetFlat(i, orig + eps);
            double lp = EvalLoss();
            fwd.Reflectors.SetFlat(i, orig - eps);
            double lm = EvalLoss();
            fwd.Reflectors.SetFlat(i, orig);
            double num = (lp - lm) / (2.0 * eps);
            Assert.True(Math.Abs(gr.GetFlat(i) - num) <= 1e-6 * Math.Max(1.0, Math.Abs(num)),
                $"dL/dReflectors mismatch at {i}: analytic={gr.GetFlat(i)}, numeric={num}");
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BlockwiseAnchorLocalError_TrainsEveryIntraBlockLayer_IsolatesBlocks_AndSequentialMatchesSingleTape(bool tiedHead)
    {
        // 7 dense blocks + 1 embedding = 8 hidden trainable layers (k = 0..7), anchorStride = 4 ->
        // Block 0 = k in {0,1,2,3} anchored at k=3; Block 1 = k in {4,5,6,7} anchored at terminal k=7.
        var (netSeq, tokens, targets) = BuildLm(blocks: 7, width: 16, vocab: 48, tiedHead: tiedHead, seed: 77);
        var (netSingle, _, _) = BuildLm(blocks: 7, width: 16, vocab: 48, tiedHead: tiedHead, seed: 77);
        netSingle.SetParameters(netSeq.GetParameters());

        var loss = new CrossEntropyWithLogitsLoss<float>();
        var seqRule = new BlockwiseAnchorLocalErrorCreditRule<float>(
            readoutLayerCount: 2,
            anchorStride: 4,
            maxCandidates: 16,
            localWeight: 0.8,
            detachReadoutOnHiddenBlocks: true,
            useDepthProgressiveWeight: true,
            streamBlocksSequentially: true,
            seed: 123);
        var singleRule = new BlockwiseAnchorLocalErrorCreditRule<float>(
            readoutLayerCount: 2,
            anchorStride: 4,
            maxCandidates: 16,
            localWeight: 0.8,
            detachReadoutOnHiddenBlocks: true,
            useDepthProgressiveWeight: true,
            streamBlocksSequentially: false,
            seed: 123);

        netSeq.SetCreditRule(seqRule);
        netSingle.SetCreditRule(singleRule);

        var gradSeq = netSeq.ComputeGradients(tokens, targets, loss);
        var gradSingle = netSingle.ComputeGradients(tokens, targets, loss);
        Assert.Equal(gradSingle.Length, gradSeq.Length);

        for (int i = 0; i < gradSeq.Length; i++)
        {
            Assert.False(float.IsNaN(gradSeq[i]) || float.IsInfinity(gradSeq[i]));
            Assert.True(Math.Abs(gradSeq[i] - gradSingle[i]) <= 1e-4f * Math.Max(1f, Math.Abs(gradSingle[i])),
                $"Sequential vs single-tape block gradient mismatch at {i}: seq={gradSeq[i]}, single={gradSingle[i]}");
        }

        // Verify ALL 8 hidden layers (0..7) received non-zero gradients (no orphaned intermediate layers).
        int offset = 0;
        var perChunkNorms = new Dictionary<Tensor<float>, double>(TensorReferenceComparer<Tensor<float>>.Instance);
        var perChunkValues = new Dictionary<Tensor<float>, float[]>(TensorReferenceComparer<Tensor<float>>.Instance);
        foreach (var chunk in netSeq.GetParameterChunks())
        {
            if (chunk is null || chunk.Length == 0) continue;
            var arr = new float[chunk.Length];
            double normSq = 0.0;
            for (int i = 0; i < chunk.Length; i++)
            {
                arr[i] = gradSeq[offset + i];
                normSq += (double)arr[i] * arr[i];
            }
            perChunkNorms[chunk] = Math.Sqrt(normSq);
            perChunkValues[chunk] = arr;
            offset += chunk.Length;
        }

        for (int layerIdx = 0; layerIdx < 8; layerIdx++)
        {
            var layerParams = ((ITrainableLayer<float>)netSeq.Layers[layerIdx]).GetTrainableParameters();
            double layerGradNorm = layerParams.Sum(p => perChunkNorms[p]);
            Assert.True(layerGradNorm > 1e-6,
                $"Hidden layer {layerIdx} received zero gradient under BlockwiseAnchorLocalError (norm={layerGradNorm}).");
        }

        // Perturb Block 1 layer 5 weights (downstream of anchor k=3) and verify Block 0 non-shared layers {1,2,3}
        // have IDENTICALLY unchanged gradients (proving zero gradient crosses the anchor StopGradient boundary).
        var block1Param = ((ITrainableLayer<float>)netSeq.Layers[5]).GetTrainableParameters()[0];
        for (int i = 0; i < block1Param.Length; i++)
            block1Param.SetFlat(i, block1Param.GetFlat(i) * 2.5f + 0.1f);

        // Reset rule seed so candidate subset matches step 1 for exact Block-0 invariance check.
        netSeq.SetCreditRule(new BlockwiseAnchorLocalErrorCreditRule<float>(
            readoutLayerCount: 2, anchorStride: 4, maxCandidates: 16, localWeight: 0.8,
            detachReadoutOnHiddenBlocks: true, useDepthProgressiveWeight: true,
            streamBlocksSequentially: true, seed: 123));
        var gradAfterPerturb = netSeq.ComputeGradients(tokens, targets, loss);

        int off2 = 0;
        var afterValues = new Dictionary<Tensor<float>, float[]>(TensorReferenceComparer<Tensor<float>>.Instance);
        foreach (var chunk in netSeq.GetParameterChunks())
        {
            if (chunk is null || chunk.Length == 0) continue;
            var arr = new float[chunk.Length];
            for (int i = 0; i < chunk.Length; i++) arr[i] = gradAfterPerturb[off2 + i];
            afterValues[chunk] = arr;
            off2 += chunk.Length;
        }

        for (int layerIdx = 1; layerIdx <= 3; layerIdx++)
        {
            foreach (var p in ((ITrainableLayer<float>)netSeq.Layers[layerIdx]).GetTrainableParameters())
            {
                var before = perChunkValues[p];
                var after = afterValues[p];
                for (int i = 0; i < before.Length; i++)
                {
                    Assert.True(Math.Abs(before[i] - after[i]) <= 1e-6f,
                        $"Cross-block gradient leaked into layer {layerIdx}[{i}]: before={before[i]}, after={after[i]}");
                }
            }
        }
    }

    [Fact]
    public void CheckpointedFusedReadoutStreaming_PreservesFusedHead_AndMatchesUncheckpointedAdam8BitUpdates()
    {
        var optPlain = new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 1e-2,
                BlockSize = 64,
                Min8BitSize = 0,
                WeightDecay = 0.05,
                UseDecoupledWeightDecay = true,
            });
        var optCkpt = new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 1e-2,
                BlockSize = 64,
                Min8BitSize = 0,
                WeightDecay = 0.05,
                UseDecoupledWeightDecay = true,
            });

        var (plain, tokens, targets) = BuildLm(blocks: 5, width: 16, vocab: 48, tiedHead: true, seed: 91, optimizer: optPlain);
        var (ckpt, _, _) = BuildLm(blocks: 5, width: 16, vocab: 48, tiedHead: true, seed: 91, optimizer: optCkpt);
        ckpt.SetParameters(plain.GetParameters());

        plain.FastApproxGradClip = true;
        ckpt.FastApproxGradClip = true;
        plain.StreamingTraining = StreamingTrainingMode.ForceOn;
        ckpt.StreamingTraining = StreamingTrainingMode.ForceOn;
        ckpt.EnableGradientCheckpointing(segmentSize: 2);
        Assert.True(ckpt.IsGradientCheckpointingEnabled);

        for (int step = 0; step < 2; step++)
        {
            long fusedBefore = DenseLayer<float>.FusedReadoutLossCount;
            ckpt.Train(tokens, targets);
            long fusedAfter = DenseLayer<float>.FusedReadoutLossCount;
            Assert.True(fusedAfter > fusedBefore,
                $"Step {step}: expected TiedEmbeddingHeadLayer fused readout loss to remain active under EnableGradientCheckpointing.");

            plain.Train(tokens, targets);
        }

        var pPlain = plain.GetParameters();
        var pCkpt = ckpt.GetParameters();
        Assert.Equal(pPlain.Length, pCkpt.Length);
        for (int i = 0; i < pPlain.Length; i++)
        {
            Assert.True(Math.Abs(pPlain[i] - pCkpt[i]) <= 1e-4f * Math.Max(1f, Math.Abs(pPlain[i])),
                $"Checkpointed streaming Adam8Bit diverged at param {i}: plain={pPlain[i]}, ckpt={pCkpt[i]}");
        }
    }

    [Fact]
    public void Adam8BitOptimizer_DecoupledWeightDecay_ShrinksZeroGradientParameter()
    {
        var (net, _, _) = BuildLm(blocks: 1, width: 16, vocab: 32, tiedHead: false, seed: 105);
        var opt = new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(net,
            new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.1,
                WeightDecay = 0.2,
                UseDecoupledWeightDecay = true,
                Min8BitSize = 0,
                BlockSize = 64,
            });

        var w = new Tensor<float>(new[] { 128 });
        var gZero = new Tensor<float>(new[] { 128 });
        for (int i = 0; i < w.Length; i++) w.SetFlat(i, 2.0f);

        var grads = new Dictionary<Tensor<float>, Tensor<float>>(TensorReferenceComparer<Tensor<float>>.Instance)
        {
            [w] = gZero
        };
        var ctx = new TapeStepContext<float>(new[] { w }, grads, loss: 0f);

        opt.Step(ctx);
        // With g == 0 and decoupled weight decay: w_new = w * (1 - lr * wd) = 2.0 * (1 - 0.02) = 1.96.
        for (int i = 0; i < w.Length; i++)
        {
            Assert.True(Math.Abs(w.GetFlat(i) - 1.96f) <= 1e-5f,
                $"Expected decoupled Adam8Bit weight decay -> 1.96f, got {w.GetFlat(i)}");
        }
    }
}
