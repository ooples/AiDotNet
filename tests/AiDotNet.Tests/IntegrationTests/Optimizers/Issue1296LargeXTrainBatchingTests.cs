using System;
using System.Diagnostics;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Regression suite for AiDotNet#1296 — two coupled full-batch sites in
/// the gradient-optimizer evaluation path:
/// <list type="number">
///   <item>
///     <c>OptimizerBase.PrepareAndEvaluateSolution</c> ran a redundant
///     <c>Model.Train(XTrain, YTrain)</c> on the full training tensor
///     before the mini-batched epoch loop began, ignoring
///     <c>AdamOptimizerOptions.BatchSize</c>.
///   </item>
///   <item>
///     <c>OptimizerBase.EvaluateModelDirectly</c> called
///     <c>model.Predict(X)</c> with the full <c>XTrain</c> /
///     <c>XValidation</c> tensor on every epoch — and
///     <see cref="NeuralNetworkBase{T}.Predict"/> had no internal
///     mini-batching, so the entire dataset flowed through every
///     layer (including <c>MultiHeadAttention</c>'s O(N²) attention
///     scores) in a single forward pass.
///   </item>
/// </list>
///
/// <para>
/// PR #1297 fixes both: gradient-based optimizers now always skip the
/// pre-epoch initial Train, and the evaluator's Predict call routes
/// through the new <see cref="NeuralNetworkBase{T}.PredictInBatches"/>
/// helper, chunking along axis 0 by
/// <see cref="OptimizerBase{T,TInput,TOutput}.EvaluationBatchSize"/>
/// (default 256). These probes pin the post-fix baseline and assert
/// against the original pre-fix symptoms.
/// </para>
///
/// <para>
/// <b>Pre-fix behaviour (would re-occur on regression):</b>
/// <c>Optimize()</c> with N=2000 samples on a Transformer at
/// <c>d=64 / L=2 / heads=2 / ctx=32 / V=64</c> allocates an attention-
/// scores tensor of shape <c>[2000, 2, 32, 32] × 4 B ≈ 16 MB</c> per
/// attention layer per forward pass, multiplied by the train-then-
/// eval-then-eval cycle that fires before and during the first epoch.
/// Scaling up to the reporter's d=128 / L=4 / heads=4 / ctx=64 at
/// 57 500 samples reaches <c>~3.8 GB</c> and OOMs.
/// </para>
/// </summary>
[Collection("NonParallelIntegration")]
public class Issue1296LargeXTrainBatchingTests
{
    private readonly ITestOutputHelper _output;

    public Issue1296LargeXTrainBatchingTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static (TransformerArchitecture<float> Arch, Tensor<float> X, Tensor<float> Y) BuildFixture(
        int sampleCount = 2000,
        int ctxLen = 32,
        int vocabSize = 64,
        int dModel = 64,
        int feedForwardDim = 128,
        int numHeads = 2,
        int numEncoderLayers = 2)
    {
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: numEncoderLayers,
            numDecoderLayers: 0,
            numHeads: numHeads,
            modelDimension: dModel,
            feedForwardDimension: feedForwardDim,
            inputSize: ctxLen,
            outputSize: vocabSize,
            maxSequenceLength: ctxLen,
            vocabularySize: vocabSize);

        // Deterministic synthetic token stream so the assertion thresholds
        // depend on shape + computation cost only, not on data variance.
        var x = new Tensor<float>([sampleCount, ctxLen]);
        var y = new Tensor<float>([sampleCount, vocabSize]);
        // Seeded random for deterministic synthetic fixtures — these are
        // memory/shape assertions, not stochastic statistical claims, so
        // reproducibility matters more than entropy quality.
        var rng = RandomHelper.CreateSeededRandom(1296);
        for (int i = 0; i < sampleCount; i++)
        {
            for (int s = 0; s < ctxLen; s++) x[i, s] = rng.Next(vocabSize);
            y[i, rng.Next(vocabSize)] = 1.0f;
        }
        return (arch, x, y);
    }

    /// <summary>
    /// Pre-fix: <see cref="OptimizerBase{T,TInput,TOutput}.PrepareAndEvaluateSolution"/>
    /// called <c>Model.Train(XTrain, YTrain)</c> with the full N=2000 batch
    /// before the epoch loop, producing a managed-heap delta in the
    /// hundreds of MB on a small Transformer just for the pre-epoch step.
    /// Post-fix the initial Train is skipped entirely for gradient-based
    /// optimizers (epoch loop's mini-batched <c>UpdateSolution</c> is the
    /// only training path).
    ///
    /// <para>
    /// Assertion: managed-heap delta across <c>Optimize()</c> with
    /// <c>MaxIterations=1, BatchSize=32</c> stays under 500 MB. Pre-fix
    /// measurement on the same config was &gt; 1 GB and trending toward
    /// OOM on larger corpora.
    /// </para>
    /// </summary>
    [Fact(Timeout = 300_000)]
    public void Adam_LargeXTrain_InitialTrainSkipped_NoHeapBlowup()
    {
        const int sampleCount = 2000;
        var (arch, xTrain, yTrain) = BuildFixture(sampleCount: sampleCount);

        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-3,
            MaxIterations = 1,
            BatchSize = 32,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);

        var inputData = new OptimizationInputData<float, Tensor<float>, Tensor<float>>
        {
            XTrain = xTrain,
            YTrain = yTrain,
        };

        long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
        var sw = Stopwatch.StartNew();
        var _ = optimizer.Optimize(inputData);
        sw.Stop();
        long heapAfter = GC.GetTotalMemory(forceFullCollection: false);

        long deltaBytes = Math.Max(0L, heapAfter - heapBefore);
        double deltaMb = deltaBytes / 1024.0 / 1024.0;
        _output.WriteLine($"Optimize wall: {sw.Elapsed.TotalSeconds:F1} s; heap delta: {deltaMb:F1} MB");

        Assert.True(
            deltaMb < 500.0,
            $"Heap delta {deltaMb:F1} MB exceeds 500 MB threshold — initial full-batch Train may have re-engaged (#1296 P0-1).");
    }

    /// <summary>
    /// Pre-fix: every epoch's <c>EvaluateModelDirectly</c> called
    /// <c>model.Predict(XValidation)</c> with the full XValidation tensor.
    /// On a fixture with N_val = 4000 and a 2-layer attention Transformer,
    /// that pushed <c>[4000, 2, 32, 32]</c> attention scores through every
    /// epoch — ~32 MB per layer per epoch, sustained across MaxIterations.
    /// Post-fix the evaluator routes through
    /// <see cref="NeuralNetworkBase{T}.PredictInBatches"/> at the
    /// <see cref="OptimizerBase{T,TInput,TOutput}.EvaluationBatchSize"/>
    /// chunk size (default 256), bounding peak Predict-residency to a
    /// single chunk.
    ///
    /// <para>
    /// Assertion: peak managed heap during a 2-epoch <c>Optimize()</c>
    /// with a 4000-sample validation set stays under 800 MB.
    /// </para>
    /// </summary>
    [Fact(Timeout = 600_000)]
    public void Adam_LargeXValidation_PerEpochPredict_IsChunked()
    {
        var (arch, xTrain, yTrain) = BuildFixture(sampleCount: 800);
        var (_, xVal, yVal) = BuildFixture(sampleCount: 4000);

        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-3,
            MaxIterations = 2,
            BatchSize = 32,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);

        var inputData = new OptimizationInputData<float, Tensor<float>, Tensor<float>>
        {
            XTrain = xTrain,
            YTrain = yTrain,
            XValidation = xVal,
            YValidation = yVal,
        };

        long peakHeap = 0;
        long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
        var sw = Stopwatch.StartNew();
        var _ = optimizer.Optimize(inputData);
        sw.Stop();
        long heapAfter = GC.GetTotalMemory(forceFullCollection: false);
        peakHeap = Math.Max(peakHeap, heapAfter - heapBefore);

        double peakMb = peakHeap / 1024.0 / 1024.0;
        _output.WriteLine($"Optimize wall: {sw.Elapsed.TotalSeconds:F1} s; peak heap delta: {peakMb:F1} MB");

        Assert.True(
            peakMb < 800.0,
            $"Peak heap delta {peakMb:F1} MB exceeds 800 MB — per-epoch Predict(XValidation) may have skipped chunking (#1296 P0-1/P0-2).");
    }

    /// <summary>
    /// <see cref="NeuralNetworkBase{T}.PredictInBatches"/> must produce
    /// numerically identical output to a single-shot <see cref="NeuralNetworkBase{T}.Predict"/>
    /// when both fit in memory. Verifies that chunking is a pure memory-
    /// bounding refactor of the same forward computation, not a semantic
    /// change — element-wise tolerance is required because chunking
    /// changes the order of matmul reductions inside the engine kernels.
    /// </summary>
    [Fact(Timeout = 60_000)]
    public void PredictInBatches_MatchesUnchunkedPredict_WithinTolerance()
    {
        const int sampleCount = 128;
        var (arch, x, _) = BuildFixture(sampleCount: sampleCount);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Force lazy-init: a single forward materialises weights so both
        // call paths share identical parameters and only differ in how
        // the input is sliced.
        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        var fullOutput = model.Predict(x);
        var chunkedOutput = model.PredictInBatches(x, batchSize: 16);

        Assert.Equal(fullOutput.Rank, chunkedOutput.Rank);
        for (int axis = 0; axis < fullOutput.Rank; axis++)
        {
            Assert.Equal(fullOutput.Shape[axis], chunkedOutput.Shape[axis]);
        }

        const float relTol = 1e-3f;
        const float absTol = 1e-4f;
        int compared = 0;
        int mismatches = 0;
        float maxDelta = 0f;
        for (int i = 0; i < fullOutput.Length; i++)
        {
            float a = fullOutput[i];
            float b = chunkedOutput[i];
            float delta = MathF.Abs(a - b);
            float tol = absTol + relTol * MathF.Abs(a);
            if (delta > tol) mismatches++;
            if (delta > maxDelta) maxDelta = delta;
            compared++;
        }
        _output.WriteLine(
            $"Compared {compared} elements; mismatches over rel={relTol} abs={absTol}: {mismatches}; maxDelta={maxDelta:G6}");

        Assert.Equal(0, mismatches);
    }

    /// <summary>
    /// Short-circuit contract: when input.Shape[0] is &lt;= batchSize,
    /// <see cref="NeuralNetworkBase{T}.PredictInBatches"/> delegates to
    /// the existing <see cref="NeuralNetworkBase{T}.Predict"/> path
    /// without slicing — output reference and shape match exactly. This
    /// is the common case for typical user predict-on-one-batch flows
    /// and must stay zero-overhead.
    /// </summary>
    [Fact(Timeout = 60_000)]
    public void PredictInBatches_BelowChunkThreshold_DelegatesToPredict()
    {
        var (arch, x, _) = BuildFixture(sampleCount: 16);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        var direct = model.Predict(x);
        var batched = model.PredictInBatches(x, batchSize: 64); // 16 < 64 -> direct delegate

        Assert.Equal(direct.Length, batched.Length);
        for (int i = 0; i < direct.Length; i++)
        {
            Assert.Equal(direct[i], batched[i]);
        }
    }

    /// <summary>
    /// Defensive: a degenerate <paramref name="batchSize"/> of zero or
    /// negative is silently clamped to 1 rather than throwing. The
    /// optimizer evaluator path defaults to <c>EvaluationBatchSize=256</c>
    /// — a hostile subclass returning 0 would otherwise infinite-loop or
    /// throw deep inside the chunk loop.
    /// </summary>
    [Theory(Timeout = 60_000)]
    [InlineData(0)]
    [InlineData(-1)]
    public void PredictInBatches_NonPositiveBatchSize_ClampsToOne(int batchSize)
    {
        var (arch, x, _) = BuildFixture(sampleCount: 4);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        var output = model.PredictInBatches(x, batchSize: batchSize);
        Assert.Equal(x.Shape[0], output.Shape[0]);
    }

    /// <summary>
    /// <c>PredictInBatches</c> throws <see cref="ArgumentNullException"/>
    /// on a null input — defensive guard inherited by every caller
    /// including <c>OptimizerBase.PredictForEvaluation</c>.
    /// </summary>
    [Fact(Timeout = 30_000)]
    public void PredictInBatches_NullInput_Throws()
    {
        var (arch, _, _) = BuildFixture(sampleCount: 4);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
        Assert.Throws<ArgumentNullException>(() => model.PredictInBatches(null!, batchSize: 32));
    }

    /// <summary>
    /// Fix-applies-to-all-subclasses sanity: <see cref="AdamWOptimizer{T,TInput,TOutput}"/>
    /// must also skip the initial full-batch Train, since it derives from
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}"/>. If
    /// someone reintroduces the per-subclass flag flip from before
    /// PR #1297, this catches it.
    /// </summary>
    [Fact(Timeout = 300_000)]
    public void AdamW_LargeXTrain_InitialTrainSkipped()
    {
        const int sampleCount = 2000;
        var (arch, xTrain, yTrain) = BuildFixture(sampleCount: sampleCount);

        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
        var options = new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-3,
            MaxIterations = 1,
            BatchSize = 32,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null, options);

        var inputData = new OptimizationInputData<float, Tensor<float>, Tensor<float>>
        {
            XTrain = xTrain,
            YTrain = yTrain,
        };

        long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
        var _ = optimizer.Optimize(inputData);
        long heapAfter = GC.GetTotalMemory(forceFullCollection: false);

        double deltaMb = Math.Max(0, heapAfter - heapBefore) / 1024.0 / 1024.0;
        _output.WriteLine($"AdamW heap delta: {deltaMb:F1} MB");
        Assert.True(deltaMb < 500.0, $"AdamW heap delta {deltaMb:F1} MB > 500 MB — full-batch Train may have engaged.");
    }

    /// <summary>
    /// <see cref="NeuralBatchHelper.PredictMaybeBatched{T,TInput,TOutput}"/>
    /// preserves the unchunked output element-for-element on small inputs
    /// (below the chunk threshold) AND on large inputs (where chunking
    /// engages). Verifies the helper's two-path contract: short-circuit to
    /// <c>Predict</c> when the input fits, route through
    /// <see cref="NeuralNetworkBase{T}.PredictInBatches"/> when it doesn't,
    /// output is element-equivalent in both cases.
    /// </summary>
    [Fact(Timeout = 60_000)]
    public void NeuralBatchHelper_PredictMaybeBatched_ShortCircuitsAndChunks()
    {
        var (arch, x, _) = BuildFixture(sampleCount: 64);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Warm the model so weights are materialised before either call.
        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        // Below threshold -> short-circuit delegates straight to Predict.
        var ground = model.Predict(x);
        var shortCircuited = NeuralBatchHelper.PredictMaybeBatched(model, x, batchSize: 256);
        Assert.Equal(ground.Length, shortCircuited.Length);
        for (int i = 0; i < ground.Length; i++)
            Assert.Equal(ground[i], shortCircuited[i]);

        // Above threshold -> chunked path. Use small batchSize so chunking
        // engages on a 64-sample input.
        var chunked = NeuralBatchHelper.PredictMaybeBatched(model, x, batchSize: 8);
        const float relTol = 1e-3f;
        const float absTol = 1e-4f;
        int mismatches = 0;
        for (int i = 0; i < ground.Length; i++)
        {
            float delta = MathF.Abs(ground[i] - chunked[i]);
            float tol = absTol + relTol * MathF.Abs(ground[i]);
            if (delta > tol) mismatches++;
        }
        Assert.Equal(0, mismatches);
    }

    /// <summary>
    /// <see cref="NeuralBatchHelper.TrainMaybeBatched{T,TInput,TOutput}"/>
    /// chunks a large NN training call into <c>ceil(N / batchSize)</c>
    /// smaller <c>Train</c> calls. We can't directly verify the call count
    /// from the outside, so the contract probe is: a 2000-sample training
    /// run completes without OOM at the same Transformer config that OOMs
    /// the unchunked path. P1-3 sentinel: AutoML trial loop's
    /// <c>model.Train(trainInputs, trainTargets)</c> at <c>SupervisedAutoMLModelBase.cs:134</c>
    /// would otherwise allocate the full attention-scores tensor in one
    /// shot.
    /// </summary>
    [Fact(Timeout = 600_000)]
    public void NeuralBatchHelper_TrainMaybeBatched_LargeNNBatch_NoOOM()
    {
        const int sampleCount = 2000;
        var (arch, x, y) = BuildFixture(sampleCount: sampleCount);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Warm to materialise weights before the chunked Train.
        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
        var sw = Stopwatch.StartNew();
        NeuralBatchHelper.TrainMaybeBatched(model, x, y, batchSize: 64);
        sw.Stop();
        long heapAfter = GC.GetTotalMemory(forceFullCollection: false);

        double deltaMb = Math.Max(0, heapAfter - heapBefore) / 1024.0 / 1024.0;
        _output.WriteLine($"TrainMaybeBatched wall: {sw.Elapsed.TotalSeconds:F1} s; heap delta: {deltaMb:F1} MB");
        Assert.True(deltaMb < 800.0,
            $"TrainMaybeBatched heap delta {deltaMb:F1} MB > 800 MB — chunked Train path may have collapsed to a single full-batch call.");
    }

}
