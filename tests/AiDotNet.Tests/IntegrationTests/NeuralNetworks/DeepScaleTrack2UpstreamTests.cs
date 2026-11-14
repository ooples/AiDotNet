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

    [Fact]
    public void HouseholderSubspaceLayer_SmallReflectorStillReflects_AndZeroReflectorIsIdentity()
    {
        // H = I - 2 u u^T / (u^T u) does not depend on the length of u, so a reflector of length 1e-10 must still map
        // [1, 0] to [-1, 0]. A zero reflector has no direction and leaves the input unchanged.
        var input = new Tensor<double>(new[] { 1, 2 });
        input.SetFlat(0, 1.0);

        var small = new HouseholderSubspaceLayer<double>(2, 1, transpose: false);
        small.Reflectors.SetFlat(0, 1e-10);
        small.Reflectors.SetFlat(1, 0.0);
        var reflected = small.Forward(input);
        Assert.Equal(-1.0, reflected.GetFlat(0), 12);
        Assert.Equal(0.0, reflected.GetFlat(1), 12);

        var zero = new HouseholderSubspaceLayer<double>(2, 1, transpose: false);
        zero.Reflectors.SetFlat(0, 0.0);
        zero.Reflectors.SetFlat(1, 0.0);
        var unchanged = zero.Forward(input);
        Assert.Equal(1.0, unchanged.GetFlat(0), 12);
        Assert.Equal(0.0, unchanged.GetFlat(1), 12);

        // Also verify Compact WY (k > 1) when zero reflectors are interleaved with tiny (1e-10) and normal reflectors:
        // H_0 = small x-reflection ([1e-10, 0, 0]), H_1 = zero (identity), H_2 = y-reflection ([0, 2.5, 0]), H_3 = zero (identity).
        var input3 = new Tensor<double>(new[] { 1, 3 });
        input3.SetFlat(0, 1.0);
        input3.SetFlat(1, 2.0);
        input3.SetFlat(2, 3.0);

        var mixedChain = new HouseholderSubspaceLayer<double>(3, 4, transpose: false);
        for (int i = 0; i < mixedChain.Reflectors.Length; i++) mixedChain.Reflectors.SetFlat(i, 0.0);
        mixedChain.Reflectors.SetFlat(0 * 3 + 0, 1e-10);
        mixedChain.Reflectors.SetFlat(2 * 3 + 1, 2.5);
        var mixedOut = mixedChain.Forward(input3);
        Assert.Equal(-1.0, mixedOut.GetFlat(0), 12);
        Assert.Equal(-2.0, mixedOut.GetFlat(1), 12);
        Assert.Equal(3.0, mixedOut.GetFlat(2), 12);
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

    [Theory]
    [InlineData(1, false)]
    [InlineData(4, false)]
    [InlineData(4, true)]
    public void HouseholderSubspaceLayer_FloatZeroAndTinyReflectors_HaveFiniteGradients_AndTransposeMatchesFiniteDifference(
        int numReflectors,
        bool transpose)
    {
        const int dim = 6;
        var layer = new HouseholderSubspaceLayer<float>(dim, numReflectors, transpose: transpose);
        for (int i = 0; i < layer.Reflectors.Length; i++)
        {
            layer.Reflectors.SetFlat(i, 0.0f);
        }

        if (numReflectors >= 2)
        {
            // Row 0: identically zero; Row 1: sub-floor (1e-20f); Row 2: small exact reflector (1e-6f); Row 3: normal.
            layer.Reflectors.SetFlat(1 * dim + 0, 1e-20f);
            layer.Reflectors.SetFlat(2 * dim + 1, 1e-6f);
            layer.Reflectors.SetFlat(3 * dim + 2, 1.75f);
            layer.Reflectors.SetFlat(3 * dim + 3, -0.5f);
        }

        var x = new Tensor<float>(new[] { 2, dim });
        for (int i = 0; i < x.Length; i++)
        {
            x.SetFlat(i, (i + 1) * 0.25f);
        }

        Dictionary<Tensor<float>, Tensor<float>> grads;
        Tensor<float> y;
        using (var tape = new GradientTape<float>())
        {
            y = layer.Forward(x);
            var weights = new Tensor<float>(y._shape);
            for (int i = 0; i < weights.Length; i++)
            {
                weights.SetFlat(i, 0.3f * (i + 1));
            }
            var loss = AiDotNetEngine.Current.ReduceSum(
                AiDotNetEngine.Current.TensorMultiply(y, weights), null);
            grads = tape.ComputeGradients(loss, sources: new[] { x, layer.Reflectors });
        }

        for (int i = 0; i < y.Length; i++)
        {
            float v = y.GetFlat(i);
            Assert.False(float.IsNaN(v) || float.IsInfinity(v), $"Forward produced non-finite value at {i}: {v}");
        }

        var gx = grads[x];
        for (int i = 0; i < gx.Length; i++)
        {
            float g = gx.GetFlat(i);
            Assert.False(float.IsNaN(g) || float.IsInfinity(g), $"dL/dx non-finite at {i}: {g}");
        }

        var gr = grads[layer.Reflectors];
        for (int i = 0; i < gr.Length; i++)
        {
            float g = gr.GetFlat(i);
            Assert.False(float.IsNaN(g) || float.IsInfinity(g), $"dL/dReflectors non-finite at {i}: {g}");
        }

        // Empty batch [0, dim] must preserve shape cleanly.
        var emptyInput = new Tensor<float>(new[] { 0, dim });
        var emptyOut = layer.Forward(emptyInput);
        Assert.Equal(emptyInput._shape, emptyOut._shape);

        // Verify transpose finite-difference accuracy in double precision as well.
        if (transpose)
        {
            var invDbl = new HouseholderSubspaceLayer<double>(dim, numReflectors, transpose: true);
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(515);
            for (int i = 0; i < invDbl.Reflectors.Length; i++)
            {
                invDbl.Reflectors.SetFlat(i, (rng.NextDouble() * 2.0 - 1.0) + 0.3);
            }
            var xd = new Tensor<double>(new[] { 2, dim });
            for (int i = 0; i < xd.Length; i++)
            {
                xd.SetFlat(i, rng.NextDouble() * 2.0 - 1.0);
            }

            double EvalTransposeLoss()
            {
                var o = invDbl.Forward(xd);
                double s = 0.0;
                for (int i = 0; i < o.Length; i++)
                {
                    double v = o.GetFlat(i);
                    s += 0.5 * (i + 1) * v * v;
                }
                return s;
            }

            Dictionary<Tensor<double>, Tensor<double>> dblGrads;
            using (var tape = new GradientTape<double>())
            {
                var o = invDbl.Forward(xd);
                var scale = new Tensor<double>(o._shape);
                for (int i = 0; i < scale.Length; i++) scale.SetFlat(i, 0.5 * (i + 1));
                var loss = AiDotNetEngine.Current.ReduceSum(
                    AiDotNetEngine.Current.TensorMultiply(scale, AiDotNetEngine.Current.TensorMultiply(o, o)), null);
                dblGrads = tape.ComputeGradients(loss, sources: new[] { invDbl.Reflectors });
            }

            const double eps = 1e-6;
            var grDbl = dblGrads[invDbl.Reflectors];
            for (int i = 0; i < invDbl.Reflectors.Length; i++)
            {
                double orig = invDbl.Reflectors.GetFlat(i);
                invDbl.Reflectors.SetFlat(i, orig + eps);
                double lp = EvalTransposeLoss();
                invDbl.Reflectors.SetFlat(i, orig - eps);
                double lm = EvalTransposeLoss();
                invDbl.Reflectors.SetFlat(i, orig);
                double num = (lp - lm) / (2.0 * eps);
                Assert.True(Math.Abs(grDbl.GetFlat(i) - num) <= 1e-6 * Math.Max(1.0, Math.Abs(num)),
                    $"Transpose dL/dReflectors mismatch at {i}: analytic={grDbl.GetFlat(i)}, numeric={num}");
            }
        }
    }

    [Fact]
    public void HouseholderSubspaceLayer_ConcurrentTapeBackward_HasZeroMaskGradIndexContention()
    {
        const int dim = 8;
        const int numReflectors = 4;

        static (double[] gx, double[] gr) RunSingleBackward(double[] xVals, double[] rVals, bool transpose)
        {
            var layer = new HouseholderSubspaceLayer<double>(dim, numReflectors, transpose: transpose);
            for (int i = 0; i < rVals.Length; i++) layer.Reflectors.SetFlat(i, rVals[i]);
            var x = new Tensor<double>(new[] { 3, dim });
            for (int i = 0; i < xVals.Length; i++) x.SetFlat(i, xVals[i]);

            using var tape = new GradientTape<double>();
            var y = layer.Forward(x);
            var loss = AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(y, y), null);
            // Also evaluate linear projection so gradients are non-zero (||Qx||^2 == ||x||^2 has zero reflector grad).
            var linWeights = new Tensor<double>(y._shape);
            for (int i = 0; i < linWeights.Length; i++) linWeights.SetFlat(i, 0.1 * (i + 1));
            var total = AiDotNetEngine.Current.TensorAdd(
                loss,
                AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(y, linWeights), null));
            var grads = tape.ComputeGradients(total, sources: new[] { x, layer.Reflectors });
            return (grads[x].ToArray(), grads[layer.Reflectors].ToArray());
        }

        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(909);
        var baseX = Enumerable.Range(0, 3 * dim).Select(_ => rng.NextDouble() * 2.0 - 1.0).ToArray();
        var baseR = Enumerable.Range(0, numReflectors * dim).Select(_ => rng.NextDouble() * 2.0 - 1.0 + 0.2).ToArray();
        var (refGxF, refGrF) = RunSingleBackward(baseX, baseR, transpose: false);
        var (refGxT, refGrT) = RunSingleBackward(baseX, baseR, transpose: true);

        System.Threading.Tasks.Parallel.For(0, 8, worker =>
        {
            for (int iter = 0; iter < 24; iter++)
            {
                bool tr = ((worker + iter) & 1) == 1;
                var (gx, gr) = RunSingleBackward(baseX, baseR, tr);
                var expGx = tr ? refGxT : refGxF;
                var expGr = tr ? refGrT : refGrF;
                for (int i = 0; i < expGx.Length; i++)
                {
                    Assert.True(Math.Abs(gx[i] - expGx[i]) <= 1e-12,
                        $"Concurrent worker {worker} iter {iter} gx[{i}] mismatch: got {gx[i]}, expected {expGx[i]}");
                }
                for (int i = 0; i < expGr.Length; i++)
                {
                    Assert.True(Math.Abs(gr[i] - expGr[i]) <= 1e-12,
                        $"Concurrent worker {worker} iter {iter} gr[{i}] mismatch: got {gr[i]}, expected {expGr[i]}");
                }
            }
        });
    }

    [Fact]
    public void LocalCreditAndStreamingRetention_PreservePublishedGradientsAcrossArenaRecycleAndDisposeCleanly()
    {
        // Width 260 -> DenseLayer weight has 260 * 260 = 67,600 >= 65,536 scalars (TensorArena large-array threshold).
        var (net, tokens, targets) = BuildLm(blocks: 2, width: 260, vocab: 32, tiedHead: true, seed: 313);
        try
        {
            net.SetCreditRule(new BlockwiseAnchorLocalErrorCreditRule<float>(
                readoutLayerCount: 2,
                anchorStride: 1,
                maxCandidates: 16,
                localWeight: 0.7,
                detachReadoutOnHiddenBlocks: true,
                useDepthProgressiveWeight: true,
                streamBlocksSequentially: true,
                seed: 42));

            net.Train(tokens, targets);

            // Recycle thread-static TensorArena buffers with poison values to verify published layer gradients
            // never alias arena-returned backing arrays after Train() exits.
            using (var arena = AiDotNet.Tensors.Helpers.TensorArena.Create())
            {
                var rented = AiDotNet.Tensors.Helpers.TensorAllocator.Rent<float>(new[] { 260, 260 });
                rented.Fill(98765.0f);
            }

            var denseGrads = ((LayerBase<float>)net.Layers[1]).GetParameterGradients();
            Assert.True(denseGrads.Length >= 67600);
            double normSq = 0.0;
            for (int i = 0; i < denseGrads.Length; i++)
            {
                float g = denseGrads[i];
                Assert.False(float.IsNaN(g) || float.IsInfinity(g));
                Assert.NotEqual(98765.0f, g);
                normSq += (double)g * g;
            }
            Assert.True(normSq > 0.0, "Expected non-zero published gradients on large dense layer.");
        }
        finally
        {
            net.Dispose();
        }

        // Also verify multi-step StreamingTrainingMode.ForceOn retention reuses snapshots and disposes cleanly.
        var (streamNet, sTokens, sTargets) = BuildLm(blocks: 2, width: 16, vocab: 32, tiedHead: true, seed: 414);
        try
        {
            streamNet.StreamingTraining = StreamingTrainingMode.ForceOn;
            streamNet.FastApproxGradClip = true;
            for (int step = 0; step < 3; step++)
            {
                streamNet.Train(sTokens, sTargets);
                var g = ((LayerBase<float>)streamNet.Layers[1]).GetParameterGradients();
                Assert.True(g.Length > 0);
                Assert.All(g.ToArray(), v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
            }
        }
        finally
        {
            streamNet.Dispose();
        }
    }
}
