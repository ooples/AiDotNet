using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
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
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Facade;

public sealed class BaseClassCachingAndCompilationFacadeTests
{
    private static Tensor<float> CreateSeededTensor(int[] shape, int seed, float scale = 0.5f)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<float>(shape);
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        }
        tensor.IncrementVersion();
        return tensor;
    }

    private static float MaxAbsDiff(Tensor<float> a, Tensor<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        var sa = a.AsSpan();
        var sb = b.AsSpan();
        float maxDiff = 0f;
        for (int i = 0; i < sa.Length; i++)
        {
            // A NaN or infinity on either side makes the difference non-finite, which compares false against any
            // threshold and would otherwise leave maxDiff at zero; a parity comparison must reject it outright.
            Assert.True(!float.IsNaN(sa[i]) && !float.IsInfinity(sa[i]) && !float.IsNaN(sb[i]) && !float.IsInfinity(sb[i]),
                $"Parity comparison saw a non-finite value at element {i}: {sa[i]} vs {sb[i]}.");
            float d = Math.Abs(sa[i] - sb[i]);
            if (d > maxDiff) maxDiff = d;
        }
        return maxDiff;
    }

    private static float L2Norm(Tensor<float> t)
    {
        var s = t.AsSpan();
        double sumSq = 0.0;
        for (int i = 0; i < s.Length; i++)
        {
            sumSq += (double)s[i] * s[i];
        }
        return (float)Math.Sqrt(sumSq);
    }

    private static NeuralNetwork<float> BuildClassifierNetwork(
        int inputDim,
        int hiddenDim,
        int numClasses,
        int hiddenLayers,
        int seed,
        bool includeHouseholder = false)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var layers = new List<ILayer<float>>();
        for (int i = 0; i < hiddenLayers; i++)
        {
            layers.Add(new DenseLayer<float>(hiddenDim, activationFunction: new TanhActivation<float>()));
            if (includeHouseholder)
            {
                layers.Add(new HouseholderSubspaceLayer<float>(hiddenDim, numReflectors: 4, transpose: (i & 1) == 1));
            }
        }
        layers.Add(new DenseLayer<float>(numClasses, activationFunction: new IdentityActivation<float>()));

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: inputDim,
            outputSize: numClasses,
            layers: layers);

        var net = new NeuralNetwork<float>(arch, lossFunction: new CrossEntropyLoss<float>());
        _ = net.Predict(new Tensor<float>(new[] { 1, inputDim }));
        var p = net.GetParameters();
        for (int i = 0; i < p.Length; i++)
        {
            p[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.15);
        }
        net.SetParameters(p);
        return net;
    }

    private static (Tensor<float> X, Tensor<float> Y) CreateClassificationDataset(
        int rows,
        int inputDim,
        int numClasses,
        int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var x = new Tensor<float>(new[] { rows, inputDim });
        var y = new Tensor<float>(new[] { rows, numClasses });
        for (int r = 0; r < rows; r++)
        {
            int cls = r % numClasses;
            y[r, cls] = 1.0f;
            for (int c = 0; c < inputDim; c++)
            {
                float signal = (c % numClasses) == cls ? 1.25f : -0.25f;
                x[r, c] = signal + (float)((rng.NextDouble() * 2.0 - 1.0) * 0.15);
            }
        }
        return (x, y);
    }

    [Theory]
    [InlineData(32, 1)]
    [InlineData(32, 2)]
    [InlineData(32, 4)]
    [InlineData(32, 8)]
    [InlineData(192, 1)]
    [InlineData(192, 4)]
    public void HouseholderSubspaceLayer_InferenceCache_MatchesTapeGraphWithin1e6_AndPreservesIsometry(int dimension, int numReflectors)
    {
        var layerFwd = new HouseholderSubspaceLayer<float>(dimension, numReflectors, transpose: false);
        var layerTrans = new HouseholderSubspaceLayer<float>(dimension, numReflectors, transpose: true);
        layerTrans.SetParameters(layerFwd.GetParameters());

        var input3D = CreateSeededTensor(new[] { 3, 5, dimension }, seed: 100 + dimension + numReflectors);

        // Warm and cached inference evaluations (exercises 1-GEMM full-Q for d=32 and 2-GEMM Compact WY for d=192).
        var infer1 = layerFwd.Forward(input3D);
        var infer2 = layerFwd.Forward(input3D);
        Assert.True(MaxAbsDiff(infer1, infer2) <= 1e-7f);

        // Compare against tape-tracked training path (bypasses derived-parameter cache).
        Tensor<float> tapedFwd;
        using (var tape = new GradientTape<float>())
        {
            tapedFwd = layerFwd.Forward(input3D);
        }

        float fwdParity = MaxAbsDiff(infer1, tapedFwd);
        Assert.True(fwdParity <= 1e-6f, $"Forward cached vs tape parity {fwdParity:E3} exceeded 1e-6 for D={dimension}, k={numReflectors}.");

        // Orthogonality: ||Q x||_2 == ||x||_2 and Q^T(Q(x)) == x.
        Assert.InRange(Math.Abs(L2Norm(infer1) - L2Norm(input3D)), 0f, 1e-4f * Math.Max(1f, L2Norm(input3D)));
        var roundTrip = layerTrans.Forward(infer1);
        float invDiff = MaxAbsDiff(roundTrip, input3D);
        Assert.True(invDiff <= 5e-6f, $"Inverse round-trip error {invDiff:E3} exceeded 5e-6 for D={dimension}, k={numReflectors}.");
    }

    [Fact]
    public void HouseholderSubspaceLayer_SharedReflectors_StaticApplyChain_And_InPlaceOptimizerInvalidation()
    {
        const int d = 48;
        const int k = 4;
        var engine = AiDotNetEngine.Current;
        var sharedReflectors = CreateSeededTensor(new[] { k, d }, seed: 4242);
        var x = CreateSeededTensor(new[] { 6, d }, seed: 4343);

        // Alternate forward and transpose on the SAME shared reflector tensor to ensure dual-slot static cache never thrashes.
        var yFwd1 = HouseholderSubspaceLayer<float>.ApplyChain(engine, x, sharedReflectors, transpose: false);
        var yTrans1 = HouseholderSubspaceLayer<float>.ApplyChain(engine, yFwd1, sharedReflectors, transpose: true);
        var yFwd2 = HouseholderSubspaceLayer<float>.ApplyChain(engine, x, sharedReflectors, transpose: false);
        var yTrans2 = HouseholderSubspaceLayer<float>.ApplyChain(engine, yFwd2, sharedReflectors, transpose: true);

        Assert.True(MaxAbsDiff(yFwd1, yFwd2) == 0f);
        Assert.True(MaxAbsDiff(yTrans1, x) <= 5e-6f);
        Assert.True(MaxAbsDiff(yTrans2, x) <= 5e-6f);

        // Mutate sharedReflectors in place and verify static cache rebuilds immediately.
        sharedReflectors.SetFlat(0, sharedReflectors.GetFlat(0) + 1.75f);
        var yAfterMutation = HouseholderSubspaceLayer<float>.ApplyChain(engine, x, sharedReflectors, transpose: false);
        Assert.True(MaxAbsDiff(yAfterMutation, yFwd1) > 1e-3f);

        Tensor<float> yAfterMutationTape;
        using (new GradientTape<float>())
        {
            yAfterMutationTape = HouseholderSubspaceLayer<float>.ApplyChain(engine, x, sharedReflectors, transpose: false);
        }
        Assert.True(MaxAbsDiff(yAfterMutation, yAfterMutationTape) <= 1e-6f);

        // Verify instance cache invalidation across AdamWOptimizer.Step and EnableDerivedWeightCaching=false toggle.
        var layer = new HouseholderSubspaceLayer<float>(d, k, transpose: false);
        var beforeStepOut = layer.Forward(x);

        var opt = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 0.05 });
        using (var tape = new GradientTape<float>())
        {
            var outTrain = layer.Forward(x);
            var loss = engine.ReduceSum(engine.TensorMultiply(outTrain, outTrain), new[] { 0, 1 }, keepDims: false);
            var grads = tape.ComputeGradients(loss, layer.GetTrainableParameters());
            opt.Step(new TapeStepContext<float>(layer.GetTrainableParameters(), grads, loss.AsSpan()[0]));
        }

        var afterStepCached = layer.Forward(x);
        Assert.True(MaxAbsDiff(beforeStepOut, afterStepCached) > 1e-4f, "LayerBase derived cache failed to invalidate after AdamWOptimizer.Step.");

        DerivedTensorCacheSettings.SetThreadOverride(false);
        try
        {
            var uncachedOut = layer.Forward(x);
            Assert.True(MaxAbsDiff(afterStepCached, uncachedOut) <= 1e-6f);
        }
        finally
        {
            DerivedTensorCacheSettings.SetThreadOverride(null);
        }
    }

    [Fact]
    public void DerivedWeightLayers_RefreshCachedWeightsAfterInPlaceUpdates()
    {
        var engine = AiDotNetEngine.Current;

        // 1. SelfAttentionLayer: verify fused QKV inference weights refresh after in-place AdamW step.
        var attn = new SelfAttentionLayer<float>(
            sequenceLength: 4,
            embeddingDimension: 16,
            headCount: 4,
            activationFunction: (IActivationFunction<float>?)null);
        attn.SetTrainingMode(false);
        var attnIn = CreateSeededTensor(new[] { 2, 4, 16 }, seed: 701);
        var attnBefore = attn.Forward(attnIn);

        var adamW = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 0.05 });
        using (var tape = new GradientTape<float>())
        {
            var pred = attn.Forward(attnIn);
            var target = CreateSeededTensor(new[] { 2, 4, 16 }, seed: 702);
            var diff = engine.TensorSubtract(pred, target);
            var loss = engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 0, 1, 2 }, keepDims: false);
            var grads = tape.ComputeGradients(loss, attn.GetTrainableParameters());
            adamW.Step(new TapeStepContext<float>(attn.GetTrainableParameters(), grads, loss.AsSpan()[0]));
        }

        var attnAfterCached = attn.Forward(attnIn);
        Assert.True(MaxAbsDiff(attnBefore, attnAfterCached) > 1e-4f, "SelfAttentionLayer served stale fused QKV weights after in-place optimizer step.");

        Tensor<float> attnAfterTape;
        using (new GradientTape<float>())
        {
            attnAfterTape = attn.Forward(attnIn);
        }
        Assert.True(MaxAbsDiff(attnAfterCached, attnAfterTape) <= 1e-5f);

        // 2. WeightNormConv1DLayer + NormedConv1DLayer (Weight & Spectral) + HyperbolicLinearLayer.
        var wnConv = new WeightNormConv1DLayer<float>(inChannels: 4, outChannels: 6, kernelSize: 3, dilation: 1, padding: 1);
        wnConv.SetTrainingMode(false);
        var convIn = CreateSeededTensor(new[] { 2, 4, 8 }, seed: 703);
        var wnOut1 = wnConv.Forward(convIn);
        var wnParams = wnConv.GetParameters();
        for (int i = 0; i < wnParams.Length; i++) wnParams[i] *= 1.35f;
        wnConv.SetParameters(wnParams);
        var wnOut2 = wnConv.Forward(convIn);
        Assert.True(MaxAbsDiff(wnOut1, wnOut2) > 1e-4f);

        var specConv = new NormedConv1DLayer<float>(
            inChannels: 4, outChannels: 6, kernelSize: 3, stride: 1, dilation: 1, groups: 1, padding: 1,
            transposed: false, normalization: ConvolutionNormalization.Spectral);
        specConv.SetTrainingMode(false);
        var specOut1 = specConv.Forward(convIn);
        var specOut2 = specConv.Forward(convIn);
        Assert.True(MaxAbsDiff(specOut1, specOut2) == 0f);

        var hyp = new HyperbolicLinearLayer<float>(inputFeatures: 8, outputFeatures: 6);
        hyp.SetTrainingMode(false);
        var hypIn = CreateSeededTensor(new[] { 3, 8 }, seed: 704, scale: 0.25f);
        var hypOut1 = hyp.Forward(hypIn);
        var hypParams = hyp.GetParameters();
        for (int i = 0; i < hypParams.Length; i++) hypParams[i] += 0.1f;
        hyp.SetParameters(hypParams);
        var hypOut2 = hyp.Forward(hypIn);
        Assert.True(MaxAbsDiff(hypOut1, hypOut2) > 1e-4f);
    }

    [Fact]
    public void DevicePredicatedMoERouting_MatchesHostSparseRouting_WithoutHostSyncUnderGraphCapture()
    {
        var moeFfn = new MoEFeedForwardLayer<float>(hiddenSize: 16, ffnDim: 32, numExperts: 4, topK: 2);
        moeFfn.Materialize();
        var x = CreateSeededTensor(new[] { 2, 5, 16 }, seed: 808);

        var hostSparseOut = moeFfn.Forward(x);
        Tensor<float> devicePredicatedOut;
        using (GraphCaptureCompatibility.Enter())
        {
            devicePredicatedOut = moeFfn.Forward(x);
        }

        Assert.True(MaxAbsDiff(hostSparseOut, devicePredicatedOut) <= 1e-5f);
    }

    [Fact]
    public void CompilationOptions_Validate_Presets_And_BucketResolution_WorkAsExpected()
    {
        var def = CompilationOptions.Default;
        def.Validate();
        Assert.Equal(ExecutionMode.AutoCompiled, def.Mode);
        Assert.Equal(ComputePrecisionTier.FP32, def.PrecisionTier);
        Assert.True(def.EnableHybridFusionTileIr);
        Assert.True(def.EnableCudaGraphTrees);
        Assert.True(def.EnableMultiBucketShapeRouting);
        Assert.True(def.EnableSegmentedBlockGraphs);
        Assert.True(def.EnableGpuResidentOptimizerScalars);
        Assert.True(def.EnableDevicePredicatedDynamicRouting);
        Assert.True(def.EnableDerivedWeightCaching);
        Assert.True(def.AllowExternalVendorLibraries);
        Assert.False(def.PreferSelfContainedDirectPtx);

        Assert.Equal(16, def.ResolveSequenceBucket(9));
        Assert.Equal(128, def.ResolveSequenceBucket(128));
        Assert.Equal(8192, def.ResolveSequenceBucket(5000));
        Assert.Equal(8, def.ResolveBatchBucket(5));

        var purePtx = CompilationOptions.PureDirectPtx;
        purePtx.Validate();
        Assert.False(purePtx.AllowExternalVendorLibraries);
        Assert.True(purePtx.PreferSelfContainedDirectPtx);

        var eager = CompilationOptions.Eager;
        eager.Validate();
        Assert.Equal(ExecutionMode.Eager, eager.Mode);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            var bad = new CompilationOptions { CudaGraphWarmupIterations = -1 };
            bad.Validate();
        });

        Assert.Throws<ArgumentException>(() =>
        {
            var unsorted = new CompilationOptions { SequenceLengthBuckets = new[] { 32, 16 } };
            unsorted.Validate();
        });
    }

    [Fact]
    public async Task AiModelBuilder_ConfigureCompilation_WorksAcrossBuildAsync_BuildForInference_And_Streaming_ForBackpropAndLocalCredit()
    {
        const int rows = 24;
        const int inputDim = 10;
        const int hiddenDim = 16;
        const int numClasses = 4;
        var (trainX, trainY) = CreateClassificationDataset(rows, inputDim, numClasses, seed: 901);

        // 1. Default out-of-the-box BuildAsync (CompilationOptions.Default active with zero ConfigureCompilation call).
        var defaultModel = BuildClassifierNetwork(inputDim, hiddenDim, numClasses, hiddenLayers: 2, seed: 910, includeHouseholder: true);
        var defaultBefore = defaultModel.GetParameters().Clone();
        var defaultOpt = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(
            defaultModel,
            new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.02,
                MaxIterations = 2,
                BatchSize = 8,
            });

        var defaultResult = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(defaultModel)
            .ConfigureOptimizer(defaultOpt)
            .ConfigureLossFunction(new CrossEntropyLoss<float>())
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(trainX, trainY))
            .BuildAsync();

        Assert.Equal(ExecutionMode.AutoCompiled, defaultResult.CompilationOptions.Mode);
        Assert.True(defaultOpt.ModelOwnStepCount > 0, "Expected AdamWOptimizer to execute via TryModelOwnStep.");
        var defaultAfter = defaultModel.GetParameters();
        double deltaSq = 0.0;
        for (int i = 0; i < defaultAfter.Length; i++)
        {
            double diff = defaultAfter[i] - defaultBefore[i];
            deltaSq += diff * diff;
        }
        Assert.True(deltaSq > 1e-8);
        var predDefault = defaultResult.Predict(trainX);
        Assert.Equal(rows, predDefault.Shape[0]);
        Assert.Equal(numClasses, predDefault.Shape[1]);

        // 2. BuildForInference with PureDirectPtx + TF32 tier.
        var inferResult = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(defaultModel)
            .ConfigureCompilation(new CompilationOptions(CompilationOptions.PureDirectPtx)
            {
                PrecisionTier = ComputePrecisionTier.TF32,
            })
            .BuildForInference();

        Assert.False(inferResult.CompilationOptions.AllowExternalVendorLibraries);
        Assert.True(inferResult.CompilationOptions.PreferSelfContainedDirectPtx);
        Assert.Equal(ComputePrecisionTier.TF32, inferResult.CompilationOptions.PrecisionTier);
        var predPurePtx = inferResult.Predict(trainX);
        Assert.True(MaxAbsDiff(predDefault, predPurePtx) <= 1e-5f);

        // 3. Streaming + TiedReadoutLocalErrorCreditRule + ConfigureCompilation(MaxAutotune).
        var streamModel = BuildClassifierNetwork(inputDim, hiddenDim, numClasses, hiddenLayers: 3, seed: 920, includeHouseholder: true);
        var streamBefore = streamModel.GetParameters().Clone();
        var streamLoader = new StreamingDataLoader<float, Tensor<float>, Tensor<float>>(
            rows,
            (idx, _) =>
            {
                var xi = new Tensor<float>(new[] { inputDim });
                var yi = new Tensor<float>(new[] { numClasses });
                for (int c = 0; c < inputDim; c++) xi[c] = trainX[idx, c];
                for (int c = 0; c < numClasses; c++) yi[c] = trainY[idx, c];
                return Task.FromResult((xi, yi));
            },
            batchSize: 8,
            prefetchCount: 1,
            numWorkers: 1);

        var streamOpt = new LionOptimizer<float, Tensor<float>, Tensor<float>>(
            streamModel,
            new LionOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.005,
                MaxIterations = 2,
                BatchSize = 8,
            });

        var streamResult = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(streamModel)
            .ConfigureOptimizer(streamOpt)
            .ConfigureCreditRule(new TiedReadoutLocalErrorCreditRule<float>(readoutLayerCount: 1))
            .ConfigureCompilation(CompilationOptions.MaxAutotune)
            .ConfigureLossFunction(new CrossEntropyLoss<float>())
            .ConfigureDataLoader(streamLoader)
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                MaxSteps = 6,
                Seed = 42,
            })
            .BuildAsync();

        Assert.Equal(ExecutionMode.AutoCompiled, streamResult.CompilationOptions.Mode);
        Assert.True(streamOpt.TapeStepCount >= 4, $"Expected streaming LionOptimizer to step >= 4 batches, got {streamOpt.TapeStepCount}.");
        var streamAfter = streamModel.GetParameters();
        double streamDeltaSq = 0.0;
        for (int i = 0; i < streamAfter.Length; i++)
        {
            Assert.False(float.IsNaN(streamAfter[i]) || float.IsInfinity(streamAfter[i]));
            double d = streamAfter[i] - streamBefore[i];
            streamDeltaSq += d * d;
        }
        Assert.True(streamDeltaSq > 1e-8);
    }

    [Fact]
    public async Task NewlyWiredFacadeOptimizers_ExecuteViaModelOwnStep_WithoutWithParametersClone()
    {
        var (trainX, trainY) = CreateClassificationDataset(rows: 16, inputDim: 8, numClasses: 4, seed: 1001);
        var factories = new (string Name, Func<NeuralNetwork<float>, GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>>> Create)[]
        {
            ("SGD_NoMomentum", net => new StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(
                net, new StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 0.02, InitialMomentum = 0.0, UseAdaptiveMomentum = false, BatchSize = 8, MaxIterations = 2,
                })),
            ("RMSProp", net => new RootMeanSquarePropagationOptimizer<float, Tensor<float>, Tensor<float>>(
                net, new RootMeanSquarePropagationOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 0.005, InitialMomentum = 0.0, UseAdaptiveMomentum = false, BatchSize = 8, MaxIterations = 2,
                })),
            ("Adagrad", net => new AdagradOptimizer<float, Tensor<float>, Tensor<float>>(
                net, new AdagradOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 0.02, BatchSize = 8, MaxIterations = 2,
                })),
            ("Adafactor", net => new AdafactorOptimizer<float, Tensor<float>, Tensor<float>>(
                net, new AdafactorOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 0.01, UseRelativeStepSize = false, EnableGradientClipping = true, BatchSize = 8, MaxIterations = 2,
                })),
            ("ScheduleFreeAdamW", net => new ScheduleFreeAdamWOptimizer<float, Tensor<float>, Tensor<float>>(
                net, new ScheduleFreeAdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 0.01, BatchSize = 8, MaxIterations = 2,
                })),
            ("DAdaptAdam", net => new DAdaptAdamOptimizer<float, Tensor<float>, Tensor<float>>(
                net, new DAdaptAdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 1.0, BatchSize = 8, MaxIterations = 2,
                })),
            ("ScaledAdam", net => new ScaledAdamOptimizer<float, Tensor<float>, Tensor<float>>(
                net, new ScaledAdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 0.01, EnableGradientClipping = true, BatchSize = 8, MaxIterations = 2,
                })),
            ("Rprop", net => new RpropOptimizer<float, Tensor<float>, Tensor<float>>(
                net, new RpropOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    MaxIterations = 2,
                })),
        };

        foreach (var (name, makeOpt) in factories)
        {
            var net = BuildClassifierNetwork(inputDim: 8, hiddenDim: 12, numClasses: 4, hiddenLayers: 2, seed: 505);
            var before = net.GetParameters().Clone();
            var opt = makeOpt(net);

            var res = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(net)
                .ConfigureOptimizer(opt)
                .ConfigureCompilation(CompilationOptions.Eager)
                .ConfigureLossFunction(new CrossEntropyLoss<float>())
                .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(trainX, trainY))
                .BuildAsync();

            Assert.True(opt.ModelOwnStepCount > 0, $"{name} declined TryModelOwnStep: {opt.LastModelStepDeclineReason}");
            var after = net.GetParameters();
            double delta = 0.0;
            for (int i = 0; i < after.Length; i++)
            {
                Assert.False(float.IsNaN(after[i]) || float.IsInfinity(after[i]), $"{name} produced non-finite parameter.");
                double d = after[i] - before[i];
                delta += d * d;
            }
            Assert.True(delta > 0.0, $"{name} did not update parameters.");
            Assert.NotNull(res.Predict(trainX));
        }
    }
    [Fact]
    public void ConfigureJitCompilation_ReEnable_RestoresCompilation_AfterDisabledOptOut()
    {
        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
        var view = (AiDotNet.Configuration.IConfiguredView<float, Tensor<float>, Tensor<float>>)builder;

        builder.ConfigureJitCompilation(AiDotNet.Configuration.JitCompilationConfig.Disabled);
        Assert.False(view.ConfiguredJitCompilation?.Enabled ?? true);

        // The legacy opt-out must not leave Eager behind once JIT is explicitly enabled again.
        builder.ConfigureJitCompilation(AiDotNet.Configuration.JitCompilationConfig.Default);
        Assert.True(view.ConfiguredJitCompilation?.Enabled ?? false);
    }

    [Fact]
    public void ConfigureCompilation_TemporaryEagerOrFusionOptOut_DoesNotOverwriteExplicitJitSettings()
    {
        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
        var view = (AiDotNet.Configuration.IConfiguredView<float, Tensor<float>, Tensor<float>>)builder;
        builder.ConfigureJitCompilation(new AiDotNet.Configuration.JitCompilationConfig
        {
            Enabled = true,
            EnableDataflowFusion = true,
            EnableConstantFolding = true,
        });

        builder.ConfigureCompilation(CompilationOptions.Eager);
        Assert.False(view.ConfiguredJitCompilation?.Enabled ?? true);
        builder.ConfigureCompilation(CompilationOptions.Default);
        Assert.True(view.ConfiguredJitCompilation?.Enabled ?? false);

        builder.ConfigureCompilation(new CompilationOptions { EnableHybridFusionTileIr = false });
        Assert.False(view.ConfiguredJitCompilation?.EnableDataflowFusion ?? true);
        builder.ConfigureCompilation(CompilationOptions.Default);
        Assert.True(view.ConfiguredJitCompilation?.EnableDataflowFusion ?? false);
    }

    [Fact]
    public void CompilationOptions_CopyConstructor_PreservesEmptyBuckets_SoValidationRejectsThem()
    {
        var empty = new CompilationOptions { SequenceLengthBuckets = Array.Empty<int>() };
        Assert.Throws<ArgumentException>(() => empty.Validate());

        var copy = new CompilationOptions(empty);
        Assert.Empty(copy.SequenceLengthBuckets);
        Assert.Throws<ArgumentException>(() => copy.Validate());
        Assert.Throws<ArgumentException>(() =>
            new AiModelBuilder<float, Tensor<float>, Tensor<float>>().ConfigureCompilation(empty));

        // Valid arrays are still independent copies.
        var valid = new CompilationOptions { BatchSizeBuckets = new[] { 1, 4 } };
        var validCopy = new CompilationOptions(valid);
        valid.BatchSizeBuckets[0] = 2;
        Assert.Equal(new[] { 1, 4 }, validCopy.BatchSizeBuckets);
    }

    [Fact]
    public void DerivedTensorCache_RejectsNonContiguousSources_BeforeBuilding()
    {
        var a = CreateSeededTensor(new[] { 2, 2 }, seed: 11);
        var c = CreateSeededTensor(new[] { 2, 2 }, seed: 12);
        var engine = AiDotNetEngine.Current;
        int builds = 0;
        Tensor<float> Build(IEngine e, Tensor<float>[] sources) { builds++; return sources[0].Clone(); }

        var single = new DerivedTensorCache<float>();
        Assert.Throws<ArgumentException>(() => single.GetOrBuild(engine, Build, a, source1: null, source2: c));
        var pair = new DerivedTensorPairCache<float>(registerPersistent: false);
        Assert.Throws<ArgumentException>(() =>
            pair.GetOrBuild(engine, (e, s) => (Build(e, s), null), a, source1: null, source2: null, source3: c));
        Assert.Equal(0, builds);

        // Contiguous lists keep working.
        Assert.NotNull(single.GetOrBuild(engine, Build, a, c));
        Assert.Equal(1, builds);
    }

    [Fact]
    public void DerivedTensorCache_ReleasedSlot_IsTerminal_AndNeverRebuilds()
    {
        var a = CreateSeededTensor(new[] { 2, 2 }, seed: 13);
        var engine = AiDotNetEngine.Current;
        int builds = 0;
        Tensor<float> Build(IEngine e, Tensor<float>[] sources) { builds++; return sources[0].Clone(); }

        var single = new DerivedTensorCache<float>();
        single.GetOrBuild(engine, Build, a);
        single.Release();
        Assert.Throws<ObjectDisposedException>(() => single.GetOrBuild(engine, Build, a));

        var pair = new DerivedTensorPairCache<float>(registerPersistent: false);
        pair.GetOrBuild(engine, (e, s) => (Build(e, s), null), a);
        pair.Release();
        Assert.Throws<ObjectDisposedException>(() => pair.GetOrBuild(engine, (e, s) => (Build(e, s), null), a));
        Assert.Equal(2, builds);

        // Invalidate (a weight change) is not terminal: the slot rebuilds.
        var live = new DerivedTensorCache<float>();
        live.GetOrBuild(engine, Build, a);
        live.Invalidate();
        live.GetOrBuild(engine, Build, a);
        Assert.Equal(4, builds);
    }

    [Fact]
    public void MixtureOfExperts_DevicePredicatedTopK_AuxiliaryLossCountsEverySelectedExpert()
    {
        const int inputSize = 6;
        const int outputSize = 4;
        const int numExperts = 3;
        const int topK = 2;
        var experts = new List<ILayer<float>>
        {
            new DenseLayer<float>(outputSize),
            new DenseLayer<float>(outputSize),
            new DenseLayer<float>(outputSize),
        };
        var router = new DenseLayer<float>(numExperts);
        var moe = new MixtureOfExpertsLayer<float>(experts, router, [inputSize], [outputSize], topK: topK);
        moe.UseAuxiliaryLoss = true;
        moe.SetTrainingMode(true);
        var x = CreateSeededTensor(new[] { 16, inputSize }, seed: 1201, scale: 1.0f);
        _ = moe.Forward(x);

        // Sharpen the router so retained top-2 weights fall below the dense branch's 0.01 threshold. Device-predicated
        // routing skips the host index readback; its Top-K count must still see both selected experts on every token.
        var routerParams = router.GetParameters();
        for (int i = 0; i < routerParams.Length; i++) routerParams[i] *= 200f;
        router.SetParameters(routerParams);

        using (GraphCaptureCompatibility.Enter())
        {
            _ = moe.Forward(x);
        }
        float deviceLoss = moe.ComputeAuxiliaryLoss();

        var weights = (Tensor<float>?)typeof(MixtureOfExpertsLayer<float>)
            .GetField("_lastRoutingWeights", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
            ?.GetValue(moe);
        Assert.NotNull(weights);
        int rows = weights.Shape[0];
        var tokenCounts = new double[numExperts];
        var probMass = new double[numExperts];
        bool anySmallRetained = false;
        for (int r = 0; r < rows; r++)
        {
            int selected = 0;
            for (int e = 0; e < numExperts; e++)
            {
                float w = weights[r, e];
                probMass[e] += w;
                if (w > 0f)
                {
                    tokenCounts[e]++;
                    selected++;
                    if (w <= 0.01f) anySmallRetained = true;
                }
            }
            Assert.Equal(topK, selected);
        }
        Assert.True(anySmallRetained, "Precondition: some retained Top-K weight must be at or below the dense 0.01 threshold.");

        double tokenTotal = 0, massTotal = 0;
        for (int e = 0; e < numExperts; e++) { tokenTotal += tokenCounts[e]; massTotal += probMass[e]; }
        double expected = 0;
        for (int e = 0; e < numExperts; e++) expected += (tokenCounts[e] / tokenTotal) * (probMass[e] / massTotal);
        expected *= numExperts;

        Assert.True(Math.Abs(expected - deviceLoss) <= 1e-4 * Math.Max(1.0, Math.Abs(expected)),
            $"Device-predicated Top-K auxiliary loss {deviceLoss} does not count every selected expert (expected {expected}).");
    }

    [Fact]
    public void MoEFeedForward_DevicePredicatedRouting_IgnoresNonFiniteOutputsOfUnselectedExperts()
    {
        var moe = new MoEFeedForwardLayer<double>(hiddenSize: 2, ffnDim: 2, numExperts: 2, topK: 1);
        moe.Materialize();
        var x = new Tensor<double>(new[] { 1, 2 });
        x[0, 0] = 1.0;
        x[0, 1] = 1.0;

        var logits = moe.Router.Forward(x);
        int unselected = logits[0, 0] >= logits[0, 1] ? 1 : 0;
        foreach (var projection in new[] { moe.ExpertGate(unselected), moe.ExpertUp(unselected) })
        {
            var huge = projection.GetParameters();
            for (int i = 0; i < huge.Length; i++) huge[i] = 1e200;
            projection.SetParameters(huge);
        }

        var host = moe.Forward(x);
        Tensor<double> device;
        using (GraphCaptureCompatibility.Enter())
        {
            device = moe.Forward(x);
        }

        for (int i = 0; i < host.Length; i++)
        {
            double h = host.GetFlat(i), d = device.GetFlat(i);
            Assert.False(double.IsNaN(h) || double.IsInfinity(h), $"host routing produced a non-finite output at {i}");
            Assert.False(double.IsNaN(d) || double.IsInfinity(d),
                $"device-predicated routing leaked an unselected expert's non-finite output at {i}: {d}");
            Assert.True(Math.Abs(h - d) <= 1e-9 * Math.Max(1.0, Math.Abs(h)), $"routing paths disagree at {i}: {h} vs {d}");
        }
    }

    [Fact]
    public void LocalObjectiveCredit_LazyHiddenLayers_MaterializeBeforeTheRecordedHiddenPass()
    {
        const int inputDim = 6, width = 8, classes = 3;
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(width, activationFunction: new TanhActivation<float>()),
            new DenseLayer<float>(width, activationFunction: new TanhActivation<float>()),
            new DenseLayer<float>(classes, activationFunction: new IdentityActivation<float>()),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: inputDim,
            outputSize: classes,
            layers: layers) { RandomSeed = 77 };
        var net = new NeuralNetwork<float>(arch, lossFunction: new CrossEntropyLoss<float>());
        Assert.False(((LayerBase<float>)net.Layers[0]).IsInitialized,
            "Precondition: the hidden DenseLayer must still be lazy so the first local-credit step materializes it.");

        var (x, y) = CreateClassificationDataset(rows: 12, inputDim, classes, seed: 1301);
        using var result = LocalObjectiveGradientComputer<float>.Compute(
            net, x, y, new TiedReadoutLocalErrorCreditRule<float>(readoutLayerCount: 1),
            new CrossEntropyLoss<float>(), (prediction, target) => target);

        Assert.Equal(2, result.HiddenLayers);
        for (int layer = 0; layer < 2; layer++)
        {
            foreach (var parameter in ((ITrainableLayer<float>)net.Layers[layer]).GetTrainableParameters())
            {
                Assert.True(result.Gradients.TryGetValue(parameter, out var grad) && grad is not null,
                    $"hidden layer {layer} received no local-objective gradient on its first (materializing) step");
                Assert.True(L2Norm(grad) > 0f, $"hidden layer {layer} local-objective gradient is identically zero");
            }
        }
    }

    [Fact]
    public async Task ScheduleFreeAdamW_AveragedParameters_ReturnsTheTapeAverage_InTheModelsParameterOrder()
    {
        var (trainX, trainY) = CreateClassificationDataset(rows: 16, inputDim: 8, numClasses: 4, seed: 1401);
        var net = BuildClassifierNetwork(8, 12, 4, hiddenLayers: 1, seed: 1402);
        var opt = new ScheduleFreeAdamWOptimizer<float, Tensor<float>, Tensor<float>>(
            net,
            new ScheduleFreeAdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.02,
                MaxIterations = 3,
                BatchSize = 8,
            });

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(net)
            .ConfigureOptimizer(opt)
            .ConfigureLossFunction(new CrossEntropyLoss<float>())
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(trainX, trainY))
            .BuildAsync();

        Assert.True(opt.ModelOwnStepCount > 0, $"ScheduleFreeAdamW declined TryModelOwnStep: {opt.LastModelStepDeclineReason}");
        var y = net.GetParameters().Clone();
        var averaged = opt.AveragedParameters();

        Assert.Equal(y.Length, averaged.Length);
        double gapSq = 0.0;
        for (int i = 0; i < averaged.Length; i++)
        {
            Assert.False(float.IsNaN(averaged[i]) || float.IsInfinity(averaged[i]));
            double d = averaged[i] - y[i];
            gapSq += d * d;
        }
        Assert.True(gapSq > 0.0, "AveragedParameters returned the gradient point y, not the running average x.");

        // Reading the average leaves the model's parameters untouched.
        var after = net.GetParameters();
        for (int i = 0; i < y.Length; i++) Assert.Equal(y[i], after[i]);
    }
}
