using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Hard reproduction attempts for #1221 — the previous scaffold tests
/// (TransformerProductionScaleConvergenceIssue1221Tests) called
/// <c>model.Train</c> / <c>model.Predict</c> directly on a freshly-constructed
/// <see cref="Transformer{T}"/>, bypassing the real builder pipeline that
/// the user's repro uses. That is the wrong harness — the actual user
/// code path is:
///
/// <code>
/// var builder = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(model)
///     .ConfigureOptimizer(optimizer)
///     .ConfigureDataLoader(DataLoaders.FromTensors(features, labels));
/// var result = await builder.BuildAsync();
/// var output = result.Predict(input);   // <- AiModelResult.Predict, not Transformer.Predict
/// </code>
///
/// Critical differences this harness exercises that the prior tests missed:
///
///   1. <see cref="AiModelBuilder{T,TInput,TOutput}.BuildAsync"/> wraps the
///      training loop, applies preprocessing/normalization, optionally
///      builds a JIT-compiled Predict plan, and may swap the model for an
///      AutoML/distributed/quantized variant. Any of those wrappings
///      could break gradient-flow continuity between training and eval.
///
///   2. <see cref="AiModelResult{T,TInput,TOutput}.Predict"/> is the eval
///      entry-point users actually call. It dispatches through
///      <c>JitCompiledFunction</c> when configured, applies preprocessing
///      InverseTransform, and runs SafetyFilter wrapping. The prior
///      <c>Transformer.Predict</c> tests skipped all of this.
///
///   3. The user's #1221 repro uses 1 MB Shakespeare byte-LM data —
///      tokens follow a Zipf-shaped distribution, NOT the uniform
///      identity-mapping pattern that the prior tests used. Some
///      tape-chaining defects only surface when token frequency is
///      skewed (rare tokens accumulate gradient through fewer
///      forward-pass nodes than common tokens).
///
///   4. Training-mode vs eval-mode forward divergence — if
///      <c>SetTrainingMode(false)</c> + <c>result.Predict</c> reads from
///      different tensor references than <c>SetTrainingMode(true)</c> +
///      <c>model.Train</c>, the eval path can produce uniform output
///      while training updates real weights. The prior tests used
///      <c>model.Predict</c> in both modes on the SAME instance, so
///      they would not catch a builder-side reference swap.
///
/// <para>
/// <b>Test strategy:</b> If any of these tests fails, we have isolated
/// the user's bug to a specific code path. If all pass, the bug is
/// genuinely environment-side (OpenCL backend, Tensors version drift,
/// or some interaction with their Shakespeare-corpus byte loader) and
/// the user-side suspects in the issue comment are the right next
/// investigation step.
/// </para>
/// </summary>
public class TransformerProductionScaleHarnessIssue1221Tests
{
    private readonly ITestOutputHelper _output;

    public TransformerProductionScaleHarnessIssue1221Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static Transformer<float> BuildTransformer(
        int vocab, int modelDim, int feedForwardDim, int seqLen,
        int numEncoderLayers, int numHeads, double learningRate)
    {
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: numEncoderLayers,
            numDecoderLayers: 0,
            numHeads: numHeads,
            modelDimension: modelDim,
            feedForwardDimension: feedForwardDim,
            inputSize: seqLen,
            outputSize: vocab,
            maxSequenceLength: seqLen,
            vocabularySize: vocab);

        var optOptions = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = learningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, optOptions);

        return new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            optimizer: optimizer);
    }

    /// <summary>
    /// Build a Zipf-distributed training corpus matching the user's #1221
    /// production workload shape (V=256 byte-LM, 1 MB → ~1M tokens, but
    /// scaled to ~6k samples for test tractability). Token frequency
    /// follows the Zipf law: rank-1 token appears ~1/H_V of the time,
    /// rank-2 appears ~1/(2·H_V), etc. This matches real text far better
    /// than the uniform-identity pattern the prior tests used and
    /// exercises the rare-token gradient paths the user's repro hits
    /// but our prior synthetic tests skipped.
    /// </summary>
    private static (Tensor<float> features, Tensor<float> labels) BuildZipfCorpus(
        int vocab, int seqLen, int numSamples, int seed = 42)
    {
        var rng = new Random(seed);

        // Precompute Zipf CDF for inverse-transform sampling.
        var zipfWeights = new double[vocab];
        double harmonic = 0.0;
        for (int i = 0; i < vocab; i++)
        {
            zipfWeights[i] = 1.0 / (i + 1);
            harmonic += zipfWeights[i];
        }
        for (int i = 0; i < vocab; i++) zipfWeights[i] /= harmonic;
        var cdf = new double[vocab];
        cdf[0] = zipfWeights[0];
        for (int i = 1; i < vocab; i++) cdf[i] = cdf[i - 1] + zipfWeights[i];

        int SampleZipf()
        {
            double u = rng.NextDouble();
            for (int i = 0; i < vocab; i++)
                if (u <= cdf[i]) return i;
            return vocab - 1;
        }

        var features = new Tensor<float>([numSamples, seqLen]);
        var labels = new Tensor<float>([numSamples, vocab]);
        for (int n = 0; n < numSamples; n++)
        {
            // Each sequence is Zipf-sampled tokens — matches real text
            // statistics (common tokens dominate, rare tokens occasionally
            // appear). The label is the next token in a synthetic
            // identity-style continuation: target = features[n, seqLen-1]
            // (predict last input token). Trivially learnable in a
            // sense the model HAS to learn the embedding-attention path
            // to solve.
            int lastToken = 0;
            for (int s = 0; s < seqLen; s++)
            {
                int tok = SampleZipf();
                features[n, s] = tok;
                lastToken = tok;
            }
            labels[n, lastToken] = 1f;
        }
        return (features, labels);
    }

    private static double L2Distance(float[] a, float[] b)
    {
        double s = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) { double d = a[i] - b[i]; s += d * d; }
        return Math.Sqrt(s);
    }

    private static float[] ToArray(Tensor<float> t)
    {
        var a = new float[t.Length];
        for (int i = 0; i < t.Length; i++) a[i] = t[i];
        return a;
    }

    /// <summary>
    /// PROBE A: Full <see cref="AiModelBuilder{T,TInput,TOutput}.BuildAsync"/>
    /// pipeline on V=256 / d=64 / L=2 / Zipf-sampled corpus. This is the
    /// closest synthetic match for the user's #1221 1 MB Shakespeare repro
    /// short of pulling Shakespeare itself. Asserts the SAME post-training
    /// dispersion invariant via <see cref="AiModelResult{T,TInput,TOutput}.Predict"/>
    /// — NOT through direct <c>model.Predict</c> as the prior tests did.
    ///
    /// If this fails, the bug is real and lives in the builder/result
    /// wrapping path; we have a tight reproducer for #1221. If this
    /// passes, the bug is environment-side.
    /// </summary>
    /// <summary>
    /// CRITICAL DIAGNOSTIC: enumerate the MultiHeadAttentionLayer's
    /// auto-generated <c>GetTrainableParameters</c> output and verify it
    /// includes ALL FIVE parameter tensors (Q/K/V/O weights + outputBias).
    /// During #1221 investigation we observed the auto-generated method
    /// returning ONLY 2 of 5 parameters (just <c>_queryWeights</c> and
    /// <c>_outputBias</c>), causing K/V/O weights to be excluded from
    /// the optimizer's gradient flow on every Transformer/BERT/ViT model
    /// in the library.
    /// </summary>
    [Fact]
    public void MultiHeadAttention_GetTrainableParameters_ReturnsAllFiveTensors()
    {
        var mha = new MultiHeadAttentionLayer<float>(headCount: 4, headDimension: 16);
        // Trigger lazy weight allocation by running a forward pass.
        var input = new Tensor<float>([1, 8, 64]);
        for (int i = 0; i < input.Length; i++) input[i] = (float)((i % 7) * 0.1);
        mha.Forward(input);

        var trainable = mha.GetTrainableParameters();
        _output.WriteLine($"MHA GetTrainableParameters returned {trainable.Count} tensors:");
        for (int i = 0; i < trainable.Count; i++)
        {
            _output.WriteLine($"  [{i}] shape=[{string.Join(",", trainable[i].Shape)}] length={trainable[i].Length}");
        }

        Assert.Equal(5, trainable.Count);
        // All five must be non-empty after the forward pass materialized them.
        for (int i = 0; i < trainable.Count; i++)
        {
            Assert.True(trainable[i].Length > 0,
                $"Trainable parameter [{i}] has Length=0 — auto-gen GetTrainableParameters " +
                $"may be returning lazy placeholders that never get included in the optimizer's " +
                $"gradient flow. This is the root cause of #1221: K/V/O attention weights and " +
                $"any other [TrainableParameter] field with the same Role get silently skipped.");
        }
    }

    /// <summary>
    /// Serialize → Deserialize round-trip MUST preserve trained weights and
    /// produce bit-identical predictions. This is the foundational guarantee
    /// of any model save/load workflow and the path Clone()/DeepCopy() goes
    /// through. #1221 root cause was that lazy-layer SetParameters silently
    /// dropped trained weights when called on an unresolved layer post-deserialize.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task Transformer_SerializeDeserialize_PreservesTrainedWeights()
    {
        await Task.Yield();
        const int vocab = 64;
        const int seqLen = 8;

        var model = BuildTransformer(
            vocab: vocab, modelDim: 32, feedForwardDim: 64, seqLen: seqLen,
            numEncoderLayers: 1, numHeads: 2, learningRate: 0.001);

        // Train the model so weights have real, non-default values.
        model.SetTrainingMode(true);
        var rng = new Random(7);
        for (int i = 0; i < 30; i++)
        {
            var inp = new Tensor<float>([1, seqLen]);
            for (int s = 0; s < seqLen; s++) inp[0, s] = rng.Next(vocab);
            var tgt = new Tensor<float>([1, vocab]);
            tgt[0, rng.Next(vocab)] = 1f;
            model.Train(inp, tgt);
        }

        // Capture trained predictions on diverse inputs.
        model.SetTrainingMode(false);
        var probeInputs = new Tensor<float>[8];
        var trainedOutputs = new float[8][];
        for (int k = 0; k < 8; k++)
        {
            var inp = new Tensor<float>([1, seqLen]);
            for (int s = 0; s < seqLen; s++) inp[0, s] = (k * 7 + s) % vocab;
            probeInputs[k] = inp;
            trainedOutputs[k] = ToArray(model.Predict(inp));
        }

        // Serialize → deserialize round-trip via DeepCopy.
        var clone = model.Clone() as Transformer<float>;
        Assert.NotNull(clone);

        // Cloned model MUST produce IDENTICAL predictions on every input.
        // Pre-fix #1221: cloned predictions were uniform / random because
        // lazy layer SetParameters dropped most of the trained weights.
        for (int k = 0; k < 8; k++)
        {
            var clonedOutput = ToArray(clone.Predict(probeInputs[k]));
            double diff = L2Distance(trainedOutputs[k], clonedOutput);
            double mag = Math.Sqrt(trainedOutputs[k].Sum(x => (double)x * x));
            _output.WriteLine($"  input[{k}] ||trained - cloned|| = {diff:E3}, ||trained|| = {mag:E3}");
            // Allow tiny float drift (1e-5 relative or absolute) but reject
            // anything resembling the #1221 magnitude (uniform output =
            // distance similar to ||trained|| itself).
            double tolerance = Math.Max(1e-5, mag * 1e-5);
            Assert.True(diff <= tolerance,
                $"Cloned model predicts differently from trained model on input {k}: " +
                $"||Δ|| = {diff:E3}, tolerance = {tolerance:E3}, ||trained|| = {mag:E3}. " +
                $"Serialize/deserialize round-trip dropped trained weights — this is the " +
                $"#1221 root cause: lazy layer SetParameters silently skips when called " +
                $"on an unresolved layer post-deserialize.");
        }
    }

    /// <summary>Probe whether BuildAsync actually updates the configured model's weights.</summary>
    [Fact(Timeout = 60000)]
    public async Task Builder_BuildAsync_UpdatesEmbeddingWeights_InPlace()
    {
        const int vocab = 64;
        const int seqLen = 8;

        var model = BuildTransformer(
            vocab: vocab, modelDim: 32, feedForwardDim: 64, seqLen: seqLen,
            numEncoderLayers: 1, numHeads: 2, learningRate: 0.001);
        var embedding = model.Layers.OfType<EmbeddingLayer<float>>().First();

        // Materialize embedding via one Forward so we can snapshot pre-build weights.
        var warmup = new Tensor<float>([1, seqLen]);
        for (int i = 0; i < seqLen; i++) warmup[0, i] = i % vocab;
        model.Predict(warmup);
        var before = ToArray(embedding.GetTrainableParameters()[0]);

        var (features, labels) = BuildZipfCorpus(vocab, seqLen, 64);
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.001,
                MaxIterations = 5
            });

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(DataLoaders.FromTensors(features, labels))
            .BuildAsync();

        // BuildAsync trains a Clone of `model`, returns the clone in result.Model.
        // The original `model` reference is unchanged. Check the trained CLONE.
        var trainedTransformer = result.Model as Transformer<float>;
        Assert.NotNull(trainedTransformer);
        var trainedEmbedding = trainedTransformer.Layers.OfType<EmbeddingLayer<float>>().First();
        // Force resolution of the trained clone's embedding then read.
        trainedTransformer.Predict(warmup);
        var afterTrained = ToArray(trainedEmbedding.GetTrainableParameters()[0]);

        int movedTrained = 0;
        for (int i = 0; i < Math.Min(before.Length, afterTrained.Length); i++)
            if (Math.Abs(afterTrained[i] - before[i]) > 1e-6) movedTrained++;

        _output.WriteLine($"original model is same ref as result.Model: {ReferenceEquals(result.Model, model)}");
        _output.WriteLine($"Embedding entries moved (result.Model trained clone): {movedTrained}/{Math.Min(before.Length, afterTrained.Length)}");

        Assert.True(movedTrained > before.Length / 4,
            $"BuildAsync trained a clone of the model but did not update its weights. " +
            $"Only {movedTrained}/{before.Length} entries moved after MaxIterations=5. " +
            $"This is the #1221 root cause: optimizer iterations don't apply to the " +
            $"clone's parameters.");

        // Now probe whether the trained model produces distinct outputs
        // for distinct inputs through result.Predict (the user-facing path).
        var inp1 = new Tensor<float>([1, seqLen]);
        for (int s = 0; s < seqLen; s++) inp1[0, s] = 3;
        var inp2 = new Tensor<float>([1, seqLen]);
        for (int s = 0; s < seqLen; s++) inp2[0, s] = 17;

        var fromResult1 = ToArray(result.Predict(inp1));
        var fromResult2 = ToArray(result.Predict(inp2));
        var fromModel1 = ToArray(trainedTransformer.Predict(inp1));
        var fromModel2 = ToArray(trainedTransformer.Predict(inp2));
        double resultDiff = L2Distance(fromResult1, fromResult2);
        double modelDiff = L2Distance(fromModel1, fromModel2);
        _output.WriteLine($"||result.Predict(inp1) - result.Predict(inp2)|| = {resultDiff:E3}");
        _output.WriteLine($"||trainedTransformer.Predict(inp1) - trainedTransformer.Predict(inp2)|| = {modelDiff:E3}");

        Assert.True(modelDiff > 1e-6,
            $"trainedTransformer.Predict (DIRECT model call) produces identical output for distinct inputs: ||Δ|| = {modelDiff:E3}. " +
            $"Embedding has moved 2048/2048 but Forward path collapses input differentiation.");
        Assert.True(resultDiff > 1e-6,
            $"result.Predict (wrapped) produces identical output even though direct model call works. " +
            $"AiModelResult wrapping path is the bug: result={resultDiff:E3}, model={modelDiff:E3}.");
    }

    [Fact(Timeout = 180000)]
    public async Task Builder_BuildAsync_V256_ZipfCorpus_ProducesNonUniformLogits()
    {
        const int vocab = 256;
        const int seqLen = 32;
        const int numSamples = 256;  // small enough for ~60s test budget but >> 8 distinct classes

        var model = BuildTransformer(
            vocab: vocab, modelDim: 64, feedForwardDim: 256, seqLen: seqLen,
            numEncoderLayers: 2, numHeads: 4, learningRate: 0.0003);

        var (features, labels) = BuildZipfCorpus(vocab, seqLen, numSamples);
        var loader = DataLoaders.FromTensors(features, labels);

        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.0003
            });

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(loader);

        AiModelResult<float, Tensor<float>, Tensor<float>> result;
        try
        {
            result = await builder.BuildAsync();
        }
        catch (Exception ex)
        {
            // BuildAsync may throw on unsupported wiring (e.g., the
            // sequence-classification head may not match the loader's
            // labels shape end-to-end). Surface it in the test output
            // so we know whether the bug is the builder rejecting our
            // setup vs producing uniform output.
            _output.WriteLine($"BuildAsync threw: {ex.GetType().Name}: {ex.Message}");
            throw;
        }

        // Eval through result.Predict — the user-facing path that
        // dispatches through JitCompiledFunction / preprocessing /
        // safety-filter wrapping. The prior tests called
        // model.Predict directly and missed this.
        const int numEvalInputs = 16;
        var evalInputs = new Tensor<float>[numEvalInputs];
        for (int k = 0; k < numEvalInputs; k++)
        {
            // Each eval input is a constant-token sequence — like the
            // #1208 / #1221 repro inputs. Distinct token ids must
            // produce distinct outputs through the builder/result path.
            var inp = new Tensor<float>([1, seqLen]);
            for (int s = 0; s < seqLen; s++) inp[0, s] = k;
            evalInputs[k] = inp;
        }

        var logits = new float[numEvalInputs][];
        for (int k = 0; k < numEvalInputs; k++)
        {
            var pred = result.Predict(evalInputs[k]);
            logits[k] = ToArray(pred);
        }

        double maxPairwise = 0.0;
        for (int i = 0; i < numEvalInputs; i++)
            for (int j = i + 1; j < numEvalInputs; j++)
                maxPairwise = Math.Max(maxPairwise, L2Distance(logits[i], logits[j]));

        _output.WriteLine($"Builder.BuildAsync + result.Predict — V=256, Zipf corpus");
        _output.WriteLine($"  max pairwise L2 across {numEvalInputs} eval inputs: {maxPairwise:E3}");

        Assert.True(maxPairwise > 5e-4,
            $"Builder.BuildAsync + result.Predict at V=256 / d=64 / L=2 / Zipf " +
            $"corpus produces uniform output across {numEvalInputs} distinct " +
            $"eval inputs (issue #1221, full builder/result harness): " +
            $"max pairwise L2 = {maxPairwise:E3}. This is the user's exact " +
            $"reported failure mode — eval NLL = ln(V), PPL = |V|, top-1 = 0%. " +
            $"Pre-fix this is exactly 0; post-fix healthy runs see dispersion " +
            $"in the 0.1-1.0 range.");
    }

    /// <summary>
    /// PROBE B: Same as Probe A but with JIT compilation EXPLICITLY ENABLED
    /// via <see cref="AiModelBuilder{T,TInput,TOutput}.ConfigureJitCompilation"/>.
    /// The user's repro does not show this call but it can be implicitly
    /// engaged by other builder configurations; this probe forces the
    /// JIT-cached <c>Predict</c> plan path through
    /// <see cref="AiModelResult{T,TInput,TOutput}.Predict"/>'s
    /// JitCompiledFunction branch.
    ///
    /// The JIT plan is built from a pre-training Predict trace at
    /// <c>BuildAsync</c> time. If that plan captures stale (pre-training)
    /// tensor references, eval-time <c>result.Predict</c> would replay the
    /// untrained model regardless of how training updated weights — exactly
    /// the user-reported "training completes successfully but model emits
    /// uniform distribution" pattern. This probe binary-tests that
    /// hypothesis.
    /// </summary>
    [Fact(Timeout = 180000)]
    public async Task Builder_BuildAsync_WithJitEnabled_V256_ProducesNonUniformLogits()
    {
        const int vocab = 256;
        const int seqLen = 32;
        const int numSamples = 256;

        var model = BuildTransformer(
            vocab: vocab, modelDim: 64, feedForwardDim: 256, seqLen: seqLen,
            numEncoderLayers: 2, numHeads: 4, learningRate: 0.0003);

        var (features, labels) = BuildZipfCorpus(vocab, seqLen, numSamples);
        var loader = DataLoaders.FromTensors(features, labels);
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.0003
            });

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(loader)
            .ConfigureJitCompilation(JitCompilationConfig.Default);   // <-- forces JIT path

        AiModelResult<float, Tensor<float>, Tensor<float>> result;
        try { result = await builder.BuildAsync(); }
        catch (Exception ex)
        {
            _output.WriteLine($"BuildAsync (JIT on) threw: {ex.GetType().Name}: {ex.Message}");
            throw;
        }

        const int numEvalInputs = 16;
        var logits = new float[numEvalInputs][];
        for (int k = 0; k < numEvalInputs; k++)
        {
            var inp = new Tensor<float>([1, seqLen]);
            for (int s = 0; s < seqLen; s++) inp[0, s] = k;
            logits[k] = ToArray(result.Predict(inp));
        }

        double maxPairwise = 0.0;
        for (int i = 0; i < numEvalInputs; i++)
            for (int j = i + 1; j < numEvalInputs; j++)
                maxPairwise = Math.Max(maxPairwise, L2Distance(logits[i], logits[j]));

        _output.WriteLine($"Builder.BuildAsync (JIT enabled) + result.Predict — V=256");
        _output.WriteLine($"  max pairwise L2 across {numEvalInputs} eval inputs: {maxPairwise:E3}");

        Assert.True(maxPairwise > 5e-4,
            $"Builder.BuildAsync with JIT enabled produces uniform output at " +
            $"V=256 / d=64 / L=2 (issue #1221, JIT-cached Predict path): " +
            $"max pairwise L2 = {maxPairwise:E3}. If this fails while Probe A " +
            $"passes, the bug is in the JIT compilation plan capturing stale " +
            $"tensor references at build time and replaying the untrained " +
            $"model regardless of training updates.");
    }

    /// <summary>
    /// PROBE C: Diff between <c>result.Predict</c> (the wrapped path) and
    /// <c>model.Predict</c> (the direct path) on the SAME trained model
    /// with identical inputs. After <c>BuildAsync</c>, both paths must
    /// agree to within float-noise tolerance — they should be reading
    /// from the same trained weights.
    ///
    /// If <c>result.Predict</c> outputs uniform values while
    /// <c>model.Predict</c> outputs normal trained values, the bug is in
    /// the AiModelResult wrapping (preprocessing inverse-transform,
    /// JitCompiledFunction replay, or SafetyFilter overwriting output).
    /// If both produce uniform output, the bug is in training itself.
    /// If both produce normal output, the bug is environment-side.
    /// </summary>
    [Fact(Timeout = 180000)]
    public async Task Builder_ResultPredict_AgreesWith_ModelPredict_PostTraining()
    {
        const int vocab = 256;
        const int seqLen = 32;
        const int numSamples = 256;

        var model = BuildTransformer(
            vocab: vocab, modelDim: 64, feedForwardDim: 256, seqLen: seqLen,
            numEncoderLayers: 2, numHeads: 4, learningRate: 0.0003);

        var (features, labels) = BuildZipfCorpus(vocab, seqLen, numSamples);
        var loader = DataLoaders.FromTensors(features, labels);
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.0003
            });

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(loader);

        var result = await builder.BuildAsync();

        // Use a few diverse eval inputs — one constant-token sequence
        // and one Zipf-sampled to cover both the synthetic-identity and
        // real-distribution paths.
        var inp1 = new Tensor<float>([1, seqLen]);
        for (int s = 0; s < seqLen; s++) inp1[0, s] = 7;

        var inp2 = new Tensor<float>([1, seqLen]);
        var rng = new Random(123);
        for (int s = 0; s < seqLen; s++) inp2[0, s] = rng.Next(vocab);

        // model.Predict — direct call on the trained Transformer.
        // result.Predict — through the AiModelResult wrapper.
        // The trained model returned by BuildAsync should be the SAME
        // object as the original `model` we configured.
        var fromModel1 = ToArray(model.Predict(inp1));
        var fromResult1 = ToArray(result.Predict(inp1));
        var fromModel2 = ToArray(model.Predict(inp2));
        var fromResult2 = ToArray(result.Predict(inp2));

        double diff1 = L2Distance(fromModel1, fromResult1);
        double diff2 = L2Distance(fromModel2, fromResult2);
        double mag1 = Math.Sqrt(fromModel1.Sum(x => (double)x * x));
        double mag2 = Math.Sqrt(fromModel2.Sum(x => (double)x * x));

        _output.WriteLine($"||model.Predict(inp1) - result.Predict(inp1)|| = {diff1:E3}, ||model.Predict(inp1)|| = {mag1:E3}");
        _output.WriteLine($"||model.Predict(inp2) - result.Predict(inp2)|| = {diff2:E3}, ||model.Predict(inp2)|| = {mag2:E3}");

        // Allow 1% relative drift to absorb preprocessing inverse-
        // transform float noise — but if result.Predict is replaying
        // a stale plan or applying degenerate normalization, the
        // outputs will diverge by orders of magnitude.
        double tol1 = Math.Max(1e-4, mag1 * 0.01);
        double tol2 = Math.Max(1e-4, mag2 * 0.01);

        Assert.True(diff1 <= tol1,
            $"result.Predict diverges from model.Predict on constant input " +
            $"after BuildAsync (issue #1221, AiModelResult wrapping bug): " +
            $"||Δ|| = {diff1:E3}, tolerance = {tol1:E3}, ||model.Predict|| = " +
            $"{mag1:E3}. This isolates the bug to the AiModelResult " +
            $"path — preprocessing inverse-transform, JitCompiledFunction " +
            $"replay capturing stale weights, or SafetyFilter zeroing output.");

        Assert.True(diff2 <= tol2,
            $"result.Predict diverges from model.Predict on Zipf-sampled input " +
            $"after BuildAsync (issue #1221, AiModelResult wrapping bug): " +
            $"||Δ|| = {diff2:E3}, tolerance = {tol2:E3}.");
    }

    /// <summary>
    /// PROBE D: TrainingMode toggle — the eval-time forward pass must use
    /// the same trained weights as the training-time forward pass. Reads
    /// the embedding tensor directly via reflection, snapshots before and
    /// after a training step, runs the model in eval mode (SetTrainingMode
    /// false) and checks that <c>Predict</c> exposes those updated weights
    /// rather than reading from a stale buffer.
    ///
    /// The user-reported bug pattern (uniform output post-training) could
    /// happen if eval-mode forward reads from a cached pre-training tensor
    /// that the buffer-view + RestoreOriginalParameters dance restored
    /// to its pre-training state somewhere in the toggle path.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Transformer_TrainingMode_Toggle_PreservesUpdatedWeights()
    {
        await Task.Yield();
        const int vocab = 64;   // smaller for tractability
        const int seqLen = 16;

        var model = BuildTransformer(
            vocab: vocab, modelDim: 32, feedForwardDim: 64, seqLen: seqLen,
            numEncoderLayers: 1, numHeads: 2, learningRate: 0.001);

        var embedding = model.Layers.OfType<EmbeddingLayer<float>>().FirstOrDefault()
            ?? throw new InvalidOperationException("Transformer must have an EmbeddingLayer.");

        // Warm up: materialize embedding.
        model.SetTrainingMode(true);
        var warmInput = new Tensor<float>([1, seqLen]);
        for (int s = 0; s < seqLen; s++) warmInput[0, s] = s % vocab;
        var warmTarget = new Tensor<float>([1, vocab]); warmTarget[0, 0] = 1f;
        model.Train(warmInput, warmTarget);

        // Snapshot embedding params after warm-up.
        var beforeParams = embedding.GetTrainableParameters();
        var snapshot = ToArray(beforeParams[0]);

        // Run real training steps with RANDOM inputs.
        const int trainSteps = 50;
        var rngTrain = new Random(7);
        for (int i = 0; i < trainSteps; i++)
        {
            var inp = new Tensor<float>([1, seqLen]);
            for (int s = 0; s < seqLen; s++) inp[0, s] = rngTrain.Next(vocab);
            int targetClass = rngTrain.Next(vocab);
            var tgt = new Tensor<float>([1, vocab]); tgt[0, targetClass] = 1f;
            model.Train(inp, tgt);
        }

        // Toggle to eval mode and probe.
        model.SetTrainingMode(false);

        // Re-read embedding params — must reflect training updates.
        var afterParams = embedding.GetTrainableParameters();
        var afterSnap = ToArray(afterParams[0]);

        int movedEntries = 0;
        double maxDelta = 0.0;
        int n = Math.Min(snapshot.Length, afterSnap.Length);
        for (int i = 0; i < n; i++)
        {
            double delta = afterSnap[i] - snapshot[i];
            double abs = Math.Abs(delta);
            if (abs > maxDelta) maxDelta = abs;
            if (abs > 1e-5) movedEntries++;
        }

        _output.WriteLine($"Post-toggle embedding deltas: {movedEntries}/{n} moved > 1e-5; max |Δ| = {maxDelta:E3}");

        Assert.True(movedEntries > n / 4,
            $"After {trainSteps} training steps + SetTrainingMode(false), " +
            $"embedding tensor visible through GetTrainableParameters has " +
            $"NOT been updated (issue #1221 mode-toggle hypothesis): " +
            $"only {movedEntries}/{n} entries moved. The training-mode " +
            $"toggle may be restoring pre-training weights or reading " +
            $"from a stale buffer.");

        // Now run Predict in eval mode and check it produces non-uniform
        // output across distinct inputs — the eval-time forward must use
        // the trained weights we just verified are present.
        var logits = new float[vocab][];
        for (int k = 0; k < vocab; k++)
        {
            var inp = new Tensor<float>([1, seqLen]);
            for (int s = 0; s < seqLen; s++) inp[0, s] = k;
            logits[k] = ToArray(model.Predict(inp));
        }

        double maxPairwise = 0.0;
        for (int i = 0; i < vocab; i++)
            for (int j = i + 1; j < vocab; j++)
                maxPairwise = Math.Max(maxPairwise, L2Distance(logits[i], logits[j]));

        _output.WriteLine($"Eval-mode Predict — max pairwise L2 across {vocab} inputs: {maxPairwise:E3}");

        Assert.True(maxPairwise > 5e-4,
            $"Eval-mode Predict produces uniform output despite trained " +
            $"embedding being present (issue #1221 eval-vs-train forward " +
            $"divergence): max pairwise L2 = {maxPairwise:E3}. The training " +
            $"updates the embedding but the eval forward pass reads from " +
            $"a different tensor.");
    }
}
