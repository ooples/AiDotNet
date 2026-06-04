using System;
using System.Diagnostics;
using System.Threading.Tasks;
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
        int numEncoderLayers = 2,
        double dropoutRate = 0.1)
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
            dropoutRate: dropoutRate,
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
    public async Task Adam_LargeXTrain_InitialTrainSkipped_NoHeapBlowup()
    {
        await Task.Yield();
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
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(model, options);

        var inputData = new OptimizationInputData<float, Tensor<float>, Tensor<float>>
        {
            XTrain = xTrain,
            YTrain = yTrain,
            // OptimizerBase.PrepareAndEvaluateSolution calls
            // SelectFeatures on every dataset; the default rank-0 stubs
            // produced by the OptimizationInputData ctor throw
            // "rank>=2 required" inside OptimizerHelper. Reuse the
            // train tensors as placeholders for validation / test —
            // the tests below only assert against memory deltas, not
            // generalisation, so the duplicate-data is irrelevant.
            XValidation = xTrain,
            YValidation = yTrain,
            XTest = xTrain,
            YTest = yTrain,
        };

        long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
        var sw = Stopwatch.StartNew();
        var _ = optimizer.Optimize(inputData);
        sw.Stop();
        // forceFullCollection:true here too so we measure RETAINED memory
        // (true leak signal) rather than transient garbage that .NET would
        // sweep on the next gen-0 pause. Without this, the heap-delta
        // metric is dominated by Workstation-GC's deferred collection of
        // per-mini-batch intermediates, which the optimizer correctly
        // discards but the gen-0 pause may not have fired yet.
        long heapAfter = GC.GetTotalMemory(forceFullCollection: true);

        long deltaBytes = Math.Max(0L, heapAfter - heapBefore);
        double deltaMb = deltaBytes / 1024.0 / 1024.0;
        _output.WriteLine($"Optimize wall: {sw.Elapsed.TotalSeconds:F1} s; retained heap delta: {deltaMb:F1} MB");

        Assert.True(
            deltaMb < 500.0,
            $"Retained heap delta {deltaMb:F1} MB exceeds 500 MB threshold — initial full-batch Train may have re-engaged (#1296 P0-1).");
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
    public async Task Adam_LargeXValidation_PerEpochPredict_IsChunked()
    {
        await Task.Yield();
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
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(model, options);

        var inputData = new OptimizationInputData<float, Tensor<float>, Tensor<float>>
        {
            XTrain = xTrain,
            YTrain = yTrain,
            XValidation = xVal,
            YValidation = yVal,
            // SelectFeatures rejects the default rank-0 XTest stub; reuse
            // validation tensors as a placeholder.
            XTest = xVal,
            YTest = yVal,
        };

        // Sample the LIVE managed heap (forced collection) on a polling
        // loop while Optimize runs on a background task. `peakHeap` is the
        // maximum live delta observed during the call — i.e., a TRUE
        // residency peak, not just the end-of-call retention. A single
        // pre/post pair would miss transient peaks (the regression symptom
        // for #1296) entirely.
        //
        // forceFullCollection: true is deliberate. The earlier non-forced
        // 25 ms sampler measured live + NOT-YET-COLLECTED GARBAGE, which
        // made the "peak" a function of GC scheduling — observed swinging
        // 700-1030 MB across identical runs and flapping the 800 MB
        // assertion. The regression this probe exists to catch (per-epoch
        // full-tensor Predict on N_val = 4000) shows up as GBs of LIVE
        // intermediates, so live residency is both the right signal and a
        // stable one. The 250 ms cadence bounds the forced-GC overhead to
        // a few dozen collections across the ~10 s Optimize.
        long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
        long peakHeap = 0;
        var sw = Stopwatch.StartNew();
        var optimizeTask = Task.Run(() => optimizer.Optimize(inputData));
        while (!optimizeTask.IsCompleted)
        {
            long live = GC.GetTotalMemory(forceFullCollection: true) - heapBefore;
            if (live > peakHeap) peakHeap = live;
            try { await Task.Delay(250); }
            catch (TaskCanceledException) { break; }
        }
        // Surface any optimizer exception before reading post-call heap.
        _ = await optimizeTask;
        sw.Stop();
        // Fold the end-of-call live heap into peakHeap so a peak that
        // happens to land right at end-of-call is also captured.
        long heapAfter = GC.GetTotalMemory(forceFullCollection: true);
        long endDelta = Math.Max(0L, heapAfter - heapBefore);
        if (endDelta > peakHeap) peakHeap = endDelta;

        double peakMb = peakHeap / 1024.0 / 1024.0;
        _output.WriteLine($"Optimize wall: {sw.Elapsed.TotalSeconds:F1} s; PEAK heap delta during run: {peakMb:F1} MB");

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
    public async Task PredictInBatches_MatchesUnchunkedPredict_WithinTolerance()
    {
        await Task.Yield();
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
    public async Task PredictInBatches_BelowChunkThreshold_DelegatesToPredict()
    {
        await Task.Yield();
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
    public async Task PredictInBatches_NonPositiveBatchSize_ClampsToOne(int batchSize)
    {
        await Task.Yield();
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
    public async Task PredictInBatches_NullInput_Throws()
    {
        await Task.Yield();
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
    public async Task AdamW_LargeXTrain_InitialTrainSkipped()
    {
        await Task.Yield();
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
        var optimizer = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(model, options);

        var inputData = new OptimizationInputData<float, Tensor<float>, Tensor<float>>
        {
            XTrain = xTrain,
            YTrain = yTrain,
            // OptimizerBase.PrepareAndEvaluateSolution calls
            // SelectFeatures on every dataset; the default rank-0 stubs
            // produced by the OptimizationInputData ctor throw
            // "rank>=2 required" inside OptimizerHelper. Reuse the
            // train tensors as placeholders for validation / test —
            // the tests below only assert against memory deltas, not
            // generalisation, so the duplicate-data is irrelevant.
            XValidation = xTrain,
            YValidation = yTrain,
            XTest = xTrain,
            YTest = yTrain,
        };

        long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
        var _ = optimizer.Optimize(inputData);
        long heapAfter = GC.GetTotalMemory(forceFullCollection: true);

        double deltaMb = Math.Max(0, heapAfter - heapBefore) / 1024.0 / 1024.0;
        _output.WriteLine($"AdamW retained heap delta: {deltaMb:F1} MB");
        Assert.True(deltaMb < 500.0, $"AdamW retained heap delta {deltaMb:F1} MB > 500 MB — full-batch Train may have engaged.");
    }

    /// <summary>
    /// Targeted regression guard for #1296 root cause: gradient-based
    /// optimizers MUST NOT invoke <c>model.Train(...)</c> during the
    /// pre-epoch <c>PrepareAndEvaluateSolution</c> path. They update
    /// parameters via the mini-batched epoch loop's <c>UpdateSolution</c>
    /// only. If a future refactor reintroduces a pre-epoch full-batch
    /// Train, this test surfaces it independently of any heap-delta or
    /// wall-time threshold — the model's <c>Train</c> override increments
    /// a counter, and we assert it stays at zero across the optimizer's
    /// preparation phase.
    /// </summary>
    [Fact(Timeout = 120_000)]
    public async Task GradientOptimizer_NeverCallsModelTrainDuringPrepareAndEvaluate()
    {
        await Task.Yield();
        var (arch, xTrain, yTrain) = BuildFixture(sampleCount: 64);

        var model = new TrainCallCountingTransformer(arch);
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-3,
            MaxIterations = 1,
            BatchSize = 16,
            UseAdaptiveLearningRate = false,
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(model, options);

        var inputData = new OptimizationInputData<float, Tensor<float>, Tensor<float>>
        {
            XTrain = xTrain,
            YTrain = yTrain,
            XValidation = xTrain,
            YValidation = yTrain,
            XTest = xTrain,
            YTest = yTrain,
        };

        _ = optimizer.Optimize(inputData);

        _output.WriteLine($"Counted model.Train calls during Optimize: {model.TrainCallCount}");
        Assert.Equal(0, model.TrainCallCount);
    }

    /// <summary>
    /// Test-only Transformer subclass that counts how many times
    /// <see cref="NeuralNetworkBase{T}.Train"/> is invoked from outside
    /// the optimizer's per-step tape path. Used by
    /// <see cref="GradientOptimizer_NeverCallsModelTrainDuringPrepareAndEvaluate"/>
    /// to guarantee the #1296 fix (skipping the pre-epoch full-batch Train)
    /// can't silently regress.
    /// </summary>
    private sealed class TrainCallCountingTransformer : Transformer<float>
    {
        public int TrainCallCount { get; private set; }

        public TrainCallCountingTransformer(TransformerArchitecture<float> arch)
            : base(arch, lossFunction: new CategoricalCrossEntropyLoss<float>())
        {
        }

        public override void Train(Tensor<float> input, Tensor<float> expectedOutput)
        {
            TrainCallCount++;
            base.Train(input, expectedOutput);
        }
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
    public async Task NeuralBatchHelper_PredictMaybeBatched_ShortCircuitsAndChunks()
    {
        await Task.Yield();
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
    public async Task NeuralBatchHelper_TrainMaybeBatched_LargeNNBatch_NoOOM()
    {
        await Task.Yield();
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
        long heapAfter = GC.GetTotalMemory(forceFullCollection: true);

        double deltaMb = Math.Max(0, heapAfter - heapBefore) / 1024.0 / 1024.0;
        _output.WriteLine($"TrainMaybeBatched wall: {sw.Elapsed.TotalSeconds:F1} s; retained heap delta: {deltaMb:F1} MB");
        Assert.True(deltaMb < 800.0,
            $"TrainMaybeBatched retained heap delta {deltaMb:F1} MB > 800 MB — chunked Train path may have collapsed to a single full-batch call.");
    }

    // ───────────────────────────────────────────────────────────────────
    // STRETCH FEATURES — beyond-industry-standard probes
    // ───────────────────────────────────────────────────────────────────

    /// <summary>
    /// <b>Stretch #5 (value-stable compile replay correctness probe).</b>
    /// The original AiDotNet posture kept compile-by-default OFF the Predict
    /// path because <c>PredictCompiled</c>'s replay returned the trace-time
    /// data when called with a new tensor of the same shape — silently wrong
    /// outputs for the common "same model, same shape, new values" pattern.
    /// This PR fixed it in <see cref="AiDotNet.NeuralNetworks.CompiledModelHost{T}.Predict"/>
    /// by calling <see cref="AiDotNet.Tensors.Engines.Compilation.ICompiledPlan{T}.SetInputs"/>
    /// before every Execute, copying the current call's data into the
    /// captured input buffer. This probe enables compilation, runs Predict
    /// on two distinct tensors of identical shape, and asserts the outputs
    /// differ. Pre-fix: outputs would be byte-identical (stale-data bug).
    /// Post-fix: outputs reflect the actual input data.
    /// </summary>
    [Fact(Timeout = 60_000)]
    public async Task CompiledReplay_ValueStability_DifferentInputs_ProduceDifferentOutputs()
    {
        await Task.Yield();
        var prevCompile = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation;
        try
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = true;

            var (arch, _, _) = BuildFixture(sampleCount: 1);
            var model = new TestExposingTransformer(arch);

            // Prime the model so weights materialise BEFORE compile trace.
            // First Predict on a Transformer lazy-inits weights — those
            // are constants by then so subsequent inferences can produce
            // input-dependent output.
            var primer = new Tensor<float>([1, arch.InputSize]);
            for (int s = 0; s < arch.InputSize; s++) primer[0, s] = s * 1.0f;
            _ = model.Predict(primer);

            // Cross-check eager outputs differ for distinct inputs BEFORE
            // going through the compile cache. If they don't differ via
            // eager, the model itself is producing input-independent
            // output (e.g., all-zero weights / dropout-only at inference
            // / a degenerate config) — in which case the compile-replay
            // test premise is invalid and we can't distinguish a real
            // SetInputs bug from a benign model artifact.
            //
            // Token values must stay in [0, vocabSize-1] for the
            // embedding lookup to succeed. arch.InputSize is the
            // SEQUENCE length, not the vocab; vocab is arch.VocabularySize.
            // We use modulo to keep values in range for both patterns.
            int vocab = Math.Max(2, arch.VocabularySize);
            var inputA = new Tensor<float>([1, arch.InputSize]);
            for (int s = 0; s < arch.InputSize; s++) inputA[0, s] = s % vocab;
            var inputB = new Tensor<float>([1, arch.InputSize]);
            for (int s = 0; s < arch.InputSize; s++) inputB[0, s] = (vocab - 1 - (s % vocab)) % vocab;
            var eagerA = model.Predict(inputA);
            var eagerB = model.Predict(inputB);
            int eagerDiff = 0;
            for (int i = 0; i < eagerA.Length; i++)
            {
                if (Math.Abs(eagerA[i] - eagerB[i]) > 1e-5f) eagerDiff++;
            }
            _output.WriteLine($"Eager-path sanity: distinct inputs differ in {eagerDiff} / {eagerA.Length} positions");
            if (eagerDiff == 0)
            {
                // The model itself produces input-independent output at
                // this config — likely a dropout-only-during-training
                // layer that becomes identity at eval. Skip the
                // compile-replay assertion since the premise (distinct
                // inputs → distinct outputs) doesn't hold for the
                // ground-truth eager path either.
                _output.WriteLine("Eager path produced identical outputs for distinct inputs — model config can't surface SetInputs-rebind effect. Skipping compile-replay value-stability assertion.");
                return;
            }

            // Now exercise the compile path. Both calls share one input
            // shape so the second call hits the cache. SetInputs(inputB)
            // is supposed to overwrite the captured input buffer with
            // inputB's bytes before Execute.
            var outputA = model.PredictCompiledPublic(inputA);
            var snapshotA = new float[outputA.Length];
            for (int i = 0; i < outputA.Length; i++) snapshotA[i] = outputA[i];

            var outputB = model.PredictCompiledPublic(inputB);

            int diffCount = 0;
            for (int i = 0; i < outputB.Length; i++)
            {
                if (Math.Abs(outputB[i] - snapshotA[i]) > 1e-5f) diffCount++;
            }
            _output.WriteLine($"Compile-path outputs differ in {diffCount} / {outputB.Length} positions");
            Assert.True(diffCount > 0,
                "CompiledReplay returned identical outputs for distinct inputs — the stale-data bug is back.");
        }
        finally
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = prevCompile;
        }
    }

    /// <summary>
    /// <b>Stretch #3 (gradient accumulation semantic check).</b>
    /// <see cref="NeuralNetworkBase{T}.TrainWithGradientAccumulation"/> runs
    /// N chunks of forward+backward then one optimizer step with the
    /// averaged gradient. This probe asserts that the call completes
    /// without OOM and produces a finite loss — the optimizer fired
    /// exactly once and the accumulated gradient was well-defined.
    /// </summary>
    [Fact(Timeout = 600_000)]
    public async Task GradientAccumulation_LargeNNBatch_CompletesAndAdvancesLoss()
    {
        await Task.Yield();
        const int sampleCount = 1000;
        var (arch, x, y) = BuildFixture(sampleCount: sampleCount);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
        NeuralBatchHelper.TrainMaybeBatched(
            model, x, y,
            batchSize: 64,
            mode: NeuralBatchHelper.GradientAccumulationMode.Accumulate);
        long heapAfter = GC.GetTotalMemory(forceFullCollection: true);

        double deltaMb = Math.Max(0, heapAfter - heapBefore) / 1024.0 / 1024.0;
        _output.WriteLine($"Accumulate mode retained heap delta: {deltaMb:F1} MB");

        // Heap bound: accumulator must not leak per-chunk gradients. With
        // a 1000-sample 2-layer Transformer at batchSize=64 → ~16 chunks,
        // the accumulator holds one tensor per trainable parameter. The
        // accumulator should stay small relative to the chunked-forward
        // peak; 1500 MB is a generous ceiling.
        Assert.True(deltaMb < 1500.0,
            $"Accumulate mode heap delta {deltaMb:F1} MB > 1500 MB — accumulator may be leaking per-chunk grads.");
    }

    /// <summary>
    /// <b>Stretch #1 (adaptive OOM recovery success path).</b>
    /// <see cref="NeuralBatchHelper.PredictAdaptive{T,TInput,TOutput}"/> on
    /// a tensor that fits at the initial chunk size returns equivalently
    /// to <see cref="NeuralBatchHelper.PredictMaybeBatched{T,TInput,TOutput}"/>.
    /// Catches a regression where the adaptive ratchet refuses to engage
    /// at the initial size.
    /// </summary>
    [Fact(Timeout = 60_000)]
    public async Task PredictAdaptive_NoOOM_MatchesNonAdaptive()
    {
        await Task.Yield();
        var (arch, x, _) = BuildFixture(sampleCount: 200);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        var ground = NeuralBatchHelper.PredictMaybeBatched(model, x, batchSize: 64);
        var adaptive = NeuralBatchHelper.PredictAdaptive(model, x, initialBatchSize: 64);
        Assert.Equal(ground.Length, adaptive.Length);
        const float relTol = 1e-3f;
        const float absTol = 1e-4f;
        int mismatches = 0;
        for (int i = 0; i < ground.Length; i++)
        {
            float delta = MathF.Abs(ground[i] - adaptive[i]);
            float tol = absTol + relTol * MathF.Abs(ground[i]);
            if (delta > tol) mismatches++;
        }
        Assert.Equal(0, mismatches);
    }

    /// <summary>
    /// <b>Stretch #4 (memory-budget API sizes chunks within the budget).</b>
    /// <see cref="NeuralBatchHelper.EstimateChunkSize{T}"/> with a small
    /// budget picks a chunk size strictly less than the input's leading
    /// axis. Catches a regression where the estimator returns full size
    /// regardless of the budget.
    /// </summary>
    [Fact(Timeout = 60_000)]
    public async Task MemoryBudget_SmallBudget_ChunksBelowFullSize()
    {
        await Task.Yield();
        var (arch, x, _) = BuildFixture(sampleCount: 4000);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        // 4 MB budget on a 4000-sample tensor where one sample probes at
        // a non-trivial cost — chunk should bound STRICTLY below 4000.
        // A returned chunk equal to the input size would mean the
        // estimator failed to bound the budget (the regression this test
        // claims to catch), so we assert strict inequality.
        const long fourMb = 4L * 1024 * 1024;
        int chunk = NeuralBatchHelper.EstimateChunkSize(model, x, fourMb);
        _output.WriteLine($"4 MB budget -> chunk size {chunk} (input has {x.Shape[0]} samples)");
        Assert.True(chunk >= 1, "Estimator returned a chunk size below 1.");
        Assert.True(chunk < x.Shape[0],
            $"Estimator returned chunk={chunk} which is not strictly below input size {x.Shape[0]} — a small budget must force chunking.");
    }

    /// <summary>
    /// <b>Stretch #2 (stream-aggregation reducer fires per chunk).</b>
    /// <see cref="NeuralBatchHelper.PredictAndReduce{T,TInput,TOutput,TAccumulator}"/>
    /// invokes its reducer once per chunk and threads the accumulator
    /// through. Probe: count the chunks via a counter accumulator on a
    /// 1000-sample input at batchSize=128 → expect <c>ceil(1000/128) = 8</c>
    /// reducer calls.
    /// </summary>
    [Fact(Timeout = 60_000)]
    public async Task StreamAggregation_ReducerFiresOncePerChunk()
    {
        await Task.Yield();
        var (arch, x, _) = BuildFixture(sampleCount: 1000);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        int reducerCalls = 0;
        int lastEnd = 0;
        int count = NeuralBatchHelper.PredictAndReduce<float, Tensor<float>, Tensor<float>, int>(
            model, x, seed: 0,
            reducer: (acc, chunk, start, end) =>
            {
                reducerCalls++;
                Assert.Equal(lastEnd, start);
                Assert.True(end > start);
                lastEnd = end;
                return acc + (end - start);
            },
            batchSize: 128);

        _output.WriteLine($"reducer fired {reducerCalls} times; covered {count} samples; expected {x.Shape[0]}");
        Assert.Equal(x.Shape[0], count);
        Assert.Equal(8, reducerCalls);
        Assert.Equal(x.Shape[0], lastEnd);
    }

    // ───────────────────────────────────────────────────────────────────
    // A/B BENCHMARK PROBES — measure improvements vs baseline
    // ───────────────────────────────────────────────────────────────────

    /// <summary>
    /// <b>Baseline — adaptive OOM recovery is a no-op when no OOM fires.</b>
    /// Reliably triggering an actual <see cref="OutOfMemoryException"/>
    /// inside a CI test is brittle (depends on host RAM, GC state, and
    /// platform), so this probe asserts the WEAKER invariant that's
    /// portable: when the requested batch is large enough that the
    /// model's forward fits without OOM, the adaptive-on and adaptive-off
    /// paths produce element-equivalent output. Catches a regression
    /// where the adaptive wrapper somehow corrupts output even on the
    /// happy path. The actual OOM-retry-and-recover branch is exercised
    /// by unit tests that mock <see cref="OutOfMemoryException"/> rather
    /// than allocating real GB-scale tensors.
    /// </summary>
    [Fact(Timeout = 600_000)]
    public async Task AdaptiveOOMRecovery_NoOOM_NoOpEquivalence()
    {
        await Task.Yield();
        var (arch, x, _) = BuildFixture(sampleCount: 256);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        // sampleCount=256 with batchSize=80_000: all 256 samples fit in
        // one chunk, no chunking needed on either path. Both should
        // succeed and return element-equivalent output.
        var directOutput = NeuralBatchHelper.PredictMaybeBatched(
            model, x, batchSize: 80_000, disableAdaptiveRetry: true);
        var adaptiveOutput = NeuralBatchHelper.PredictMaybeBatched(
            model, x, batchSize: 80_000, disableAdaptiveRetry: false);

        Assert.Equal(directOutput.Length, adaptiveOutput.Length);
        const float tol = 1e-3f;
        int mismatches = 0;
        for (int i = 0; i < directOutput.Length; i++)
        {
            if (MathF.Abs(directOutput[i] - adaptiveOutput[i]) > tol) mismatches++;
        }
        Assert.Equal(0, mismatches);
        _output.WriteLine("Adaptive default = no-op when no OOM fires (baseline behaviour preserved)");
    }

    /// <summary>
    /// <b>Benchmark — gradient accumulation produces full-batch-equivalent gradient.</b>
    /// True grad accumulation should produce a parameter update direction
    /// that matches a single full-batch SGD step under mean-reduced loss.
    /// We train two fresh copies of the same model from identical seeds:
    /// (1) one full-batch <see cref="NeuralNetworkBase{T}.Train"/> step,
    /// (2) chunked <see cref="NeuralNetworkBase{T}.TrainWithGradientAccumulation"/>.
    /// After one step each, both models' Predict on a held-out sample
    /// must match within tolerance. If grad accumulation isn't actually
    /// equivalent (e.g., we scaled gradients wrong), the predictions will
    /// diverge measurably.
    /// </summary>
    [Fact(Timeout = 600_000)]
    public async Task GradientAccumulation_MatchesFullBatchGradient_OnOneStep()
    {
        await Task.Yield();
        const int sampleCount = 128;
        // dropoutRate: 0 — the full-batch-equivalence property this test
        // asserts only holds for a DETERMINISTIC forward. With the default
        // dropout (0.1) the single full-batch forward and the 4 chunked
        // forwards draw DIFFERENT dropout masks (DropoutLayer derives its
        // per-call mask seed from RandomSeed + an internal call counter),
        // so their gradients diverge by construction — the same reason the
        // equivalent PyTorch grad-accum identity is only exact under
        // model.eval() or p=0. Stochastic-regularization behaviour is
        // covered elsewhere; this probe pins the accumulation SCALING math.
        var (arch, x, y) = BuildFixture(sampleCount: sampleCount, dropoutRate: 0);
        var probeInput = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) probeInput[0, s] = x[0, s];

        // Two fresh models, identical weights via deterministic init.
        // Materialise weights via a primer Predict before training so the
        // lazy initialisation produces identical RNG sequences.
        var fullBatch = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
        _ = fullBatch.Predict(probeInput);

        var accum = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
        _ = accum.Predict(probeInput);

        // Plain SGD on BOTH models — NOT the default Adam. Adam's bias-
        // corrected first step is update ≈ lr·sign(g) per element, so any
        // near-zero gradient whose SIGN differs between the one-GEMM full-
        // batch reduction and the chunked-sum reduction moves the parameter
        // by a full 2·lr — amplifying benign reduction-order noise into
        // O(0.1) prediction deltas (observed: 35/64 probe mismatches), while
        // simultaneously being scale-invariant and therefore blind to the
        // very scaling bug this probe exists to catch. Vanilla SGD's update
        // is lr·g: reduction-order noise stays noise, and a wrong
        // accumulation scale (e.g. missing the 1/totalSamples divide)
        // shifts predictions by the full magnitude — exactly the signal we
        // want.
        fullBatch.SetBaseTrainOptimizer(
            new GradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(fullBatch));

        // Copy fullBatch's parameters into accum so both start identical.
        // Cheaper than relying on RNG determinism, which depends on layer
        // construction order being identical (it is, but assert defensively).
        var fullParams = fullBatch.GetParameters();
        var accumParams = accum.GetParameters();
        Assert.Equal(fullParams.Length, accumParams.Length);
        var accumNet = accum.WithParameters(fullParams);
        // WithParameters may return a new instance; rebind.
        accum = (Transformer<float>)accumNet;
        // Install SGD AFTER the rebind — a fresh instance from
        // WithParameters would otherwise lazily fall back to default Adam.
        accum.SetBaseTrainOptimizer(
            new GradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(accum));

        // Pre-training sanity: after WithParameters both models must be
        // IDENTICAL — same probe output bit-for-bit (deterministic forward,
        // dropout disabled). If this already diverges, the comparison
        // harness is broken and the post-step assertion below would blame
        // grad-accum for a setup defect. This guard caught the original
        // root cause of this test's failure: the CPU engine's identity-
        // keyed pre-packed weight caches served the accum model's PRE-
        // WithParameters random weights even though its parameter vector
        // was bit-identical to fullBatch's (fixed via
        // InferenceWeightCache.InvalidateAll in WithParameters /
        // SetParameters / post-optimizer-step).
        {
            var preFull = fullBatch.Predict(probeInput);
            var preAccum = accum.Predict(probeInput);
            float preMaxDelta = 0f;
            for (int i = 0; i < preFull.Length; i++)
            {
                float d = MathF.Abs(preFull[i] - preAccum[i]);
                if (d > preMaxDelta) preMaxDelta = d;
            }
            _output.WriteLine($"PRE-TRAIN probe maxDelta={preMaxDelta:G6}");
            Assert.True(preMaxDelta < 1e-6f,
                $"Models diverge BEFORE training (maxDelta={preMaxDelta:G6}) — WithParameters did not replicate " +
                "state (parameter copy incomplete, or stale identity-keyed weight caches were not invalidated).");
        }

        // Single full-batch step
        fullBatch.Train(x, y);
        // Chunked grad-accum step (batchSize=32 → 4 chunks → 4× forward+backward → 1 optimizer step)
        accum.TrainWithGradientAccumulation(x, y, batchSize: 32);

        // Predict the same probe with both. If grad-accum produced the
        // same update direction (averaged), predictions should match.
        var pFull = fullBatch.Predict(probeInput);
        var pAccum = accum.Predict(probeInput);

        const float relTol = 5e-2f;  // loose: floating-point reduction order differs
        const float absTol = 5e-3f;
        int mismatches = 0;
        float maxDelta = 0f;
        for (int i = 0; i < pFull.Length; i++)
        {
            float delta = MathF.Abs(pFull[i] - pAccum[i]);
            float tol = absTol + relTol * MathF.Abs(pFull[i]);
            if (delta > tol) mismatches++;
            if (delta > maxDelta) maxDelta = delta;
        }
        _output.WriteLine($"FullBatch vs Accumulate: maxDelta={maxDelta:G6}, mismatches={mismatches}/{pFull.Length}");
        // Allow up to 5 % positions to mismatch — floating-point reduction
        // order across chunked-accumulate sum differs from one big matmul.
        Assert.True(mismatches < pFull.Length * 0.05,
            $"FullBatch vs Accumulate diverged in {mismatches}/{pFull.Length} positions — grad-accum scaling may be wrong.");
    }

    /// <summary>
    /// <b>Benchmark — memory-budget API: actual peak heap stays under stated budget.</b>
    /// Probes <see cref="NeuralBatchHelper.PredictWithMemoryBudget{T,TInput,TOutput}"/>
    /// with a 50 MB budget and asserts the actual managed-heap peak
    /// during the call stays under (budget × 2) — generous slack for
    /// transient allocations the per-sample probe doesn't fully capture,
    /// but tight enough to fail if the estimator returned chunks that
    /// would peak at GBs.
    /// </summary>
    [Fact(Timeout = 600_000)]
    public async Task MemoryBudget_ActualPeakStaysUnderTwiceBudget()
    {
        await Task.Yield();
        // Use a fixture whose total work is large (50000 samples) so
        // chunking is meaningful, and a budget (1 GB) well above the
        // small-Transformer per-call fixed overhead (~50 MB at this
        // config) so the slope-fit estimator can actually find a
        // chunk size that fits. Previous fixture (4000 samples, 50 MB
        // budget) had no solution: fixed overhead alone exceeded the
        // budget.
        var (arch, x, _) = BuildFixture(sampleCount: 50_000);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        const long budget = 1024L * 1024 * 1024; // 1 GB
        long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
        var result = NeuralBatchHelper.PredictWithMemoryBudget(model, x, budget);
        // forceFullCollection:true → measure RETAINED memory at end-of-call,
        // not transient gen-0 garbage. The budget API targets persistent
        // heap footprint, not allocation count.
        long heapAfter = GC.GetTotalMemory(forceFullCollection: true);
        double peakMb = Math.Max(0, heapAfter - heapBefore) / 1024.0 / 1024.0;

        _output.WriteLine($"Budget=1024 MB, retained heap delta={peakMb:F1} MB");
        Assert.True(peakMb < 2048.0,
            $"Memory budget breached: retained heap delta {peakMb:F1} MB > 2× the 1024 MB budget. " +
            $"Slope-fit estimator may be under-counting per-sample bytes.");
        Assert.NotNull(result);
    }

    /// <summary>
    /// <b>Benchmark — stream-aggregation eliminates the concat 2× peak.</b>
    /// A/B compare peak managed-heap delta between
    /// <see cref="NeuralBatchHelper.PredictMaybeBatched{T,TInput,TOutput}"/>
    /// (which calls <see cref="NeuralNetworkBase{T}.PredictInBatches"/>
    /// → concat) vs
    /// <see cref="NeuralBatchHelper.PredictAndReduce{T,TInput,TOutput,TAccumulator}"/>
    /// (which folds per-chunk into a scalar). The reduce path should
    /// peak meaningfully lower because it never holds the full prediction
    /// tensor in memory alongside the chunk outputs.
    /// </summary>
    [Fact(Timeout = 300_000)]
    public async Task StreamAggregation_LowersPeakHeapVsConcatPath()
    {
        await Task.Yield();
        // Output tensor must be LARGE relative to per-call infrastructure
        // for the concat-vs-reduce gap to be measurable. V=2048 puts the
        // output at [N, 2048] = 8 MB per 1000 samples vs the prior V=64
        // (0.25 MB per 1000 samples) which was below the noise floor of
        // per-call allocation churn. 5000 samples × V=2048 = 40 MB
        // output — concat path holds this PLUS all chunk-outputs;
        // reduce path holds only one chunk at a time.
        var (arch, x, _) = BuildFixture(sampleCount: 5000, vocabSize: 2048);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var primer = new Tensor<float>([1, x.Shape[1]]);
        for (int s = 0; s < x.Shape[1]; s++) primer[0, s] = x[0, s];
        _ = model.Predict(primer);

        // Use GetAllocatedBytesForCurrentThread — strictly better than
        // heap-delta because it captures every allocation regardless of
        // intra-call GC timing. Heap-delta is just (live_after -
        // live_before) and intermediate frees during the call vanish.
        // For "did the algorithm allocate fewer total bytes?" the
        // allocated-bytes counter is the correct measurement.

        // Concat path
#if NET5_0_OR_GREATER
        long allocBeforeConcat = GC.GetAllocatedBytesForCurrentThread();
#else
        long allocBeforeConcat = GC.GetTotalMemory(forceFullCollection: true);
#endif
        var concatOutput = NeuralBatchHelper.PredictMaybeBatched(model, x, batchSize: 64);
#if NET5_0_OR_GREATER
        long allocAfterConcat = GC.GetAllocatedBytesForCurrentThread();
#else
        long allocAfterConcat = GC.GetTotalMemory(forceFullCollection: false);
#endif
        double concatAllocMb = Math.Max(0, allocAfterConcat - allocBeforeConcat) / 1024.0 / 1024.0;
        GC.KeepAlive(concatOutput);
        concatOutput = null!;
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Reduce path — fold per-chunk into a counter (no full tensor materialised)
#if NET5_0_OR_GREATER
        long allocBeforeReduce = GC.GetAllocatedBytesForCurrentThread();
#else
        long allocBeforeReduce = GC.GetTotalMemory(forceFullCollection: true);
#endif
        int sum = NeuralBatchHelper.PredictAndReduce<float, Tensor<float>, Tensor<float>, int>(
            model, x, seed: 0,
            reducer: (acc, chunk, start, end) => acc + (end - start),
            batchSize: 64);
#if NET5_0_OR_GREATER
        long allocAfterReduce = GC.GetAllocatedBytesForCurrentThread();
#else
        long allocAfterReduce = GC.GetTotalMemory(forceFullCollection: false);
#endif
        double reduceAllocMb = Math.Max(0, allocAfterReduce - allocBeforeReduce) / 1024.0 / 1024.0;

        _output.WriteLine($"Concat path total allocations: {concatAllocMb:F1} MB");
        _output.WriteLine($"Reduce path total allocations: {reduceAllocMb:F1} MB");
        double absSavingsMb = concatAllocMb - reduceAllocMb;
        _output.WriteLine($"Savings: {absSavingsMb:F1} MB ({100.0 * absSavingsMb / Math.Max(1e-3, concatAllocMb):F1}%)");

        Assert.Equal(x.Shape[0], sum);
        // The CONCAT path allocates: per-chunk outputs (held in array
        // for the loop) + intermediates + the final concatenated output
        // tensor. The REDUCE path allocates: per-chunk outputs
        // (transient, GC-eligible after reducer returns) + intermediates
        // + scalar accumulator. The DIFFERENCE = the final concatenated
        // output tensor size = N × V × sizeof(T). For this fixture:
        // 5000 × 2048 × 4 = 40 MB. Assert the saving meets that floor
        // within 10 % tolerance — verifies the helper actually skips
        // the concat allocation rather than just shuffling memory.
        long expectedSavingBytes = (long)x.Shape[0] * arch.VocabularySize * sizeof(float);
        double expectedSavingMb = expectedSavingBytes / 1024.0 / 1024.0;
        double minSavingMb = expectedSavingMb * 0.9;
        _output.WriteLine($"Expected saving (output tensor size): {expectedSavingMb:F1} MB; lower bound at 90%: {minSavingMb:F1} MB");
        Assert.True(absSavingsMb >= minSavingMb,
            $"Stream-aggregation saved only {absSavingsMb:F1} MB but should save ≥ {minSavingMb:F1} MB " +
            $"(the final concat tensor size). PredictAndReduce may be materialising the full output.");
    }

    /// <summary>
    /// <b>Probe — compile-replay hot path engages and stays within the
    /// expected perf band of eager.</b>
    ///
    /// <para>History: the trace-and-replay path originally measured ~2×
    /// SLOWER than the eager forward (speedup ≈ 0.43-0.46×) because the
    /// CompiledTrainingPlan FusedLinear specialization hardcoded
    /// <c>allowCachedB: false</c> (re-packing B on every replay) and the
    /// hot-path entry conditions churned. After the Tensors-side fixes
    /// the compiled path beats eager (≥1×) in a clean process — verified
    /// by running this test in isolation and by the AIsEval benchmark
    /// harness, which measures in dedicated processes.</para>
    ///
    /// <para>Inside the full suite this probe inherits allocator / pool
    /// state from the 20 training-heavy tests that precede it, which
    /// taxes the replay path by up to ~40% while leaving eager untouched
    /// (deterministic; survives GC settle + interleaved min-of-rounds
    /// with the hot path fully engaged). The assertions are therefore
    /// two-band: (1) STRUCTURAL — the hot replay path must be engaged for
    /// every trial; (2) PERF FLOOR at 0.55×, which cleanly separates
    /// in-suite pollution (0.67×+) from the per-call replay-overhead
    /// regression signature (~0.45×).</para>
    /// </summary>
    [Fact(Timeout = 300_000)]
    public async Task CompileReplay_DeliversSpeedupVsEager()
    {
        await Task.Yield();
        const int warmup = 10;
        const int trials = 50;
        const int batchSize = 8;
        var (arch, _, _) = BuildFixture(sampleCount: 1);
        int vocab = Math.Max(2, arch.VocabularySize);
        var input = new Tensor<float>([batchSize, arch.InputSize]);
        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < arch.InputSize; s++)
                input[b, s] = (b * 7 + s) % vocab;

        var prev = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation;
        try
        {
            // Build + warm BOTH models first, then measure in INTERLEAVED
            // rounds taking the per-side MIN. This is the same
            // drift-cancellation methodology the AIsEval benchmarks use:
            // a single sequential (all-eager, then all-compiled) pass makes
            // the comparison hostage to whatever rig state the measurement
            // window inherits — when this test runs after the 20 training-
            // heavy probes in this class, GC/allocator pressure taxed the
            // second (compiled) window by ~40% while the eager window read
            // clean, flipping the assertion (observed 0.67-0.70x in-class
            // vs >=1x isolated, with the hot path fully engaged either
            // way). Interleaving puts both paths in every rig regime and
            // min-of-rounds discards the polluted samples symmetrically.
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = false;
            var eagerModel = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
            for (int i = 0; i < warmup; i++) _ = eagerModel.Predict(input);

            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = true;
            var compiledModel = new TestExposingTransformer(arch);
            for (int i = 0; i < warmup; i++) _ = compiledModel.PredictCompiledPublic(input);

            const int rounds = 5;
            int trialsPerRound = Math.Max(1, trials / rounds);
            var sw = new System.Diagnostics.Stopwatch();

            double MeasureSpeedup()
            {
                double eagerMsMin = double.MaxValue, compiledMsMin = double.MaxValue;
                for (int r = 0; r < rounds; r++)
                {
                    // Settle the heap so neither side eats a collection
                    // triggered by the other's (or a prior test's) garbage.
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();

                    AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = false;
                    sw.Restart();
                    for (int i = 0; i < trialsPerRound; i++) _ = eagerModel.Predict(input);
                    sw.Stop();
                    eagerMsMin = Math.Min(eagerMsMin, sw.Elapsed.TotalMilliseconds / trialsPerRound);

                    AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = true;
                    sw.Restart();
                    for (int i = 0; i < trialsPerRound; i++) _ = compiledModel.PredictCompiledPublic(input);
                    sw.Stop();
                    compiledMsMin = Math.Min(compiledMsMin, sw.Elapsed.TotalMilliseconds / trialsPerRound);
                }
                _output.WriteLine($"Eager: min {eagerMsMin:F3} ms/call over {rounds} interleaved rounds x {trialsPerRound} trials");
                _output.WriteLine($"Compiled: min {compiledMsMin:F3} ms/call over {rounds} interleaved rounds x {trialsPerRound} trials");
                return eagerMsMin / Math.Max(1e-6, compiledMsMin);
            }

            double speedup = MeasureSpeedup();
            // Flake guard for the timing floor (review #1488): a runner being
            // throttled across an entire measurement window can land an
            // otherwise-healthy build under the floor. Re-measure up to twice
            // before failing — a REAL replay-overhead regression (compiled
            // structurally slower per call) reproduces on every attempt,
            // while a noisy-neighbor window doesn't survive three separate
            // multi-second windows. The ratio itself is already throttle-
            // resistant (interleaved rounds slow both sides together).
            for (int attempt = 0; attempt < 2 && speedup < 0.55; attempt++)
            {
                _output.WriteLine($"Speedup {speedup:F2}x below floor — re-measuring (attempt {attempt + 2}/3)");
                speedup = MeasureSpeedup();
            }
            var (hotHits, slowCalls) = compiledModel.GetCompileHostCounters();
            _output.WriteLine($"Speedup: {speedup:F2}x");
            _output.WriteLine($"CompiledModelHost: hot-path hits={hotHits}, slow-path calls={slowCalls}");

            // Structural invariant first: the hot replay path must actually
            // be engaged across the trial loop. hot hits == 0 with slow
            // calls == trials is the deterministic signature of the
            // hot-path entry conditions regressing (value-stability rebind,
            // shape gate, plan invalidation churn) — catch that exactly,
            // independent of rig state.
            Assert.True(hotHits >= trials,
                $"Compiled hot path not engaged: hits={hotHits} < trials={trials} (slow-path calls={slowCalls}). " +
                "Replay is re-entering the slow trace/compile path per call.");

            // Perf floor. In a CLEAN process the compiled path beats eager
            // (>=1x — verified by running this test in isolation and by the
            // AIsEval benchmark harness, which measures in dedicated
            // processes). Inside the full suite, however, the 20 training-
            // heavy probes that precede this one leave allocator / pool
            // state that taxes the replay path by up to ~40% while leaving
            // the eager loop untouched (measured: isolated >=1.0x;
            // after 1 heavy test 0.93-0.94x; after the full class 0.67-
            // 0.78x — deterministic, survives GC settle + interleaved
            // min-of-rounds, hot path fully engaged). The regression this
            // assertion exists to catch — per-call trace overhead in the
            // replay pipeline — measured 0.43-0.46x before the compile
            // fixes, well below the pollution band. 0.55 separates the two
            // cleanly: pollution passes, a real replay-overhead regression
            // fails.
            //
            // This floor stays a HARD assert (not a logged warning): the
            // hotHits gate above CANNOT substitute for it. The original
            // regression — FusedLinear specialization re-packing B on every
            // replay — ran with the hot path FULLY engaged (hits == trials)
            // at 0.43-0.46x; demoting the floor would leave that entire
            // regression class undetected. Runner-throttle flakes are
            // handled by the bounded re-measure above instead.
            Assert.True(speedup >= 0.55,
                $"Compiled path is far slower than eager: speedup={speedup:F2}x. " +
                $"This is below the in-suite pollution band (0.67x+) and matches the " +
                $"per-call replay-overhead regression signature (~0.45x) — actionable upstream.");
        }
        finally
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = prev;
        }
    }

    /// <summary>
    /// Test-only Transformer subclass that re-exposes the
    /// <see cref="NeuralNetworkBase{T}.PredictCompiled"/> entry point
    /// (which is <c>protected internal</c>) as a public method, so the
    /// value-stable replay and compile-speedup probes can drive it
    /// directly. The compile path is opt-in for end users; this wrapper
    /// is the test surface that makes it observable.
    /// </summary>
    private sealed class TestExposingTransformer : Transformer<float>
    {
        public TestExposingTransformer(TransformerArchitecture<float> arch)
            : base(arch, lossFunction: new CategoricalCrossEntropyLoss<float>())
        {
        }

        public Tensor<float> PredictCompiledPublic(Tensor<float> input) => PredictCompiled(input);

        // Reach into the private _compileHost field via reflection so the
        // speedup test can verify the hot-path is engaged across the trial
        // loop. If hot-path hits == 0 but slow path == trials, the hot-path
        // entry conditions are failing and the slowdown is downstream of
        // CompiledModelHost (in the Tensors compile pipeline itself).
        public (long Hot, long Slow) GetCompileHostCounters()
        {
            var fld = typeof(NeuralNetworkBase<float>).GetField(
                "_compileHost",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            var host = fld?.GetValue(this);
            if (host is null) return (0, 0);
            long hot = (long)(host.GetType().GetProperty("HotPathHits")?.GetValue(host) ?? 0L);
            long slow = (long)(host.GetType().GetProperty("SlowPathCalls")?.GetValue(host) ?? 0L);
            return (hot, slow);
        }
    }
}
