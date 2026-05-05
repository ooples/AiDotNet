using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// End-to-end integration tests that catch the class of regressions which let
/// ooples/AiDotNet#1264 slip through. These tests exercise <see cref="Transformer{T}"/>
/// at three vocab sizes (V=4 / 16 / 256) with deterministic single-example
/// memorization and batched training, and assert <i>concrete numerical outcomes</i>
/// — not just "loss decreased somewhat". Any future regression that breaks the
/// per-sample or per-batch training path on Transformer (default optimizer
/// regression, gradient sign, optimizer step semantics, learning-rate decay,
/// loss reduction direction) flips at least one of these assertions.
/// </summary>
/// <remarks>
/// <para>
/// Test gap closed by this file: the previous integration coverage only required
/// loss to monotonically <i>decrease</i>, which a vanilla-SGD-default Transformer
/// satisfies trivially without ever actually learning. The strong-form assertion
/// here is "after N steps overfitting on a single fixed example, P(target) > 0.5"
/// — which is the simplest possible signal that the model is doing more than
/// taking randomly-sized steps in roughly the right direction.
/// </para>
/// </remarks>
public class TransformerEndToEndIntegrationTests
{
    private readonly ITestOutputHelper _output;

    public TransformerEndToEndIntegrationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Default optimizer must be Adam, not vanilla GradientDescent. Vaswani 2017
    /// and every modern Transformer paper (BERT/GPT/T5/ViT) use Adam or AdamW.
    /// Vanilla SGD on Transformers does not converge in practical step budgets
    /// because the gradient surface across attention's softmax + LayerNorm has
    /// very different scales across parameters.
    /// </summary>
    [Fact]
    public void Constructor_DefaultOptimizer_IsAdamNotGradientDescent()
    {
        var arch = MakeArch(vocab: 4, ctxLen: 4, dModel: 8, dFf: 16, layers: 1, heads: 2);
        var transformer = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Reflective check on the private _optimizer field. Direct property
        // access is intentionally not exposed (consumers should configure
        // optimizer at construction time), but the field is observable via
        // reflection for this regression check.
        var field = typeof(Transformer<float>).GetField("_optimizer",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        Assert.NotNull(field);
        var actualOptimizer = field!.GetValue(transformer);
        Assert.NotNull(actualOptimizer);

        var actualType = actualOptimizer!.GetType();
        _output.WriteLine($"Transformer default optimizer type: {actualType.Name}");

        Assert.True(
            actualType.Name.Contains("Adam"),
            $"Transformer default optimizer must be an Adam family member; got {actualType.Name}. "
            + "Reverting this default to GradientDescent (or any non-adaptive optimizer) silently breaks "
            + "byte-LM training — see ooples/AiDotNet#1264 for the full reproducer.");
    }

    /// <summary>
    /// V=4 single-example memorization. Cumulative gradient signal is high
    /// enough that any reasonable optimizer + LR combination memorizes one
    /// example in a few hundred steps. If this test fails, the training
    /// pipeline is fundamentally broken (gradient sign, parameter update,
    /// loss direction).
    /// </summary>
    [Fact]
    public void Train_SingleSample_V4_MemorisesAfter5000Steps()
    {
        // 5000 steps is the realistic budget for single-sample SGD/Adam at
        // default LR=1e-3 to memorize one V=4 example. PyTorch's torch.optim.Adam
        // at the same LR on the same toy task takes roughly the same number
        // of steps to cross P>0.80. If this fires, the training-pipeline
        // gradient direction is wrong, not the rate.
        const int vocab = 4;
        const int ctxLen = 8;
        const int targetClass = 1;
        const int trainSteps = 5000;

        var arch = MakeArch(vocab: vocab, ctxLen: ctxLen, dModel: 16, dFf: 32, layers: 1, heads: 2);
        var transformer = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var input = new Tensor<float>([1, ctxLen]);
        for (int s = 0; s < ctxLen; s++) input[0, s] = (float)(s % vocab);
        var target = new Tensor<float>([1, vocab]);
        target[0, targetClass] = 1f;

        transformer.SetTrainingMode(true);
        for (int step = 0; step < trainSteps; step++) transformer.Train(input, target);

        transformer.SetTrainingMode(false);
        float pTarget = SoftmaxAndPickClass(transformer.Predict(input), vocab, targetClass);
        _output.WriteLine($"V={vocab} after {trainSteps} steps: P(target={targetClass})={pTarget:F4}");

        Assert.True(pTarget > 0.80f,
            $"V={vocab} single-example memorization should exceed P>0.80 after {trainSteps} steps "
            + $"of per-sample Adam at default LR=1e-3; got {pTarget:F4}. "
            + "If this fails, training pipeline is fundamentally broken (gradient sign, optimizer step semantics).");
    }

    /// <summary>
    /// V=16 single-example memorization. This is the bar the existing
    /// TransformerTrainConvergenceTests sets, but verified more strictly
    /// (probability bound rather than loss-decrease).
    /// </summary>
    [Fact]
    public void Train_SingleSample_V16_MemorisesAfter1000Steps()
    {
        const int vocab = 16;
        const int ctxLen = 4;
        const int targetClass = 7;
        const int trainSteps = 1000;

        var arch = MakeArch(vocab: vocab, ctxLen: ctxLen, dModel: 16, dFf: 32, layers: 2, heads: 2);
        var transformer = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var input = new Tensor<float>([1, ctxLen]);
        for (int s = 0; s < ctxLen; s++) input[0, s] = (float)(s % vocab);
        var target = new Tensor<float>([1, vocab]);
        target[0, targetClass] = 1f;

        transformer.SetTrainingMode(true);
        for (int step = 0; step < trainSteps; step++) transformer.Train(input, target);

        transformer.SetTrainingMode(false);
        float pTarget = SoftmaxAndPickClass(transformer.Predict(input), vocab, targetClass);
        _output.WriteLine($"V={vocab} after {trainSteps} steps: P(target={targetClass})={pTarget:F4}");

        Assert.True(pTarget > 0.50f,
            $"V={vocab} single-example memorization should exceed P>0.50 after {trainSteps} steps; "
            + $"got {pTarget:F4}. Random would be {1.0/vocab:F4}.");
    }

    /// <summary>
    /// V=256 batched training (B=32). Batched gradients are the practical
    /// path for high-V tasks (byte-LM, token classification). After 100
    /// batch updates on a 32-example memorization set, the model should
    /// classify the training set with >50% top-1 accuracy.
    /// </summary>
    [Fact]
    public void TrainBatched_V256_LearnsBatchAfter100Steps()
    {
        const int vocab = 256;
        const int ctxLen = 8;
        const int batchSize = 32;
        const int trainSteps = 100;

        var arch = MakeArch(vocab: vocab, ctxLen: ctxLen, dModel: 32, dFf: 64, layers: 1, heads: 2);
        var transformer = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Build a batch of 32 input/target pairs with deterministic distinct
        // patterns (input[i] uses tokens shifted by i; target[i] is class i).
        var inputs = new Tensor<float>[batchSize];
        var targets = new Tensor<float>[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            inputs[b] = new Tensor<float>([1, ctxLen]);
            for (int s = 0; s < ctxLen; s++) inputs[b][0, s] = (float)((s + b) % vocab);
            targets[b] = new Tensor<float>([1, vocab]);
            targets[b][0, b] = 1f;
        }

        transformer.SetTrainingMode(true);
        for (int step = 0; step < trainSteps; step++) transformer.TrainBatched(inputs, targets);

        transformer.SetTrainingMode(false);
        int correct = 0;
        for (int b = 0; b < batchSize; b++)
        {
            int argmax = SoftmaxArgmax(transformer.Predict(inputs[b]), vocab);
            if (argmax == b) correct++;
        }
        double topAcc = (double)correct / batchSize;
        _output.WriteLine($"V={vocab} batched (B={batchSize}) after {trainSteps} steps: top-1 acc on training set = {topAcc:P2}");

        Assert.True(topAcc > 0.50,
            $"V={vocab} TrainBatched should achieve >50% top-1 on the {batchSize}-example training set "
            + $"after {trainSteps} batch updates; got {topAcc:P2}. "
            + "This catches regressions in the batch-stacking path or in the optimizer's per-step update magnitude.");
    }

    /// <summary>
    /// Loss must DECREASE — not just "spread is non-zero". Concrete numerical
    /// bound: after 200 steps on a single example, loss must be below 50% of
    /// the initial loss. A Transformer that's actually training cuts loss in
    /// half on a memorization task within a few hundred steps; one that isn't
    /// (e.g. wrong gradient sign, broken backward pass) leaves loss flat or
    /// rising.
    /// </summary>
    [Fact]
    public void Train_LossDecreasesByAtLeastHalfOnMemorizationTask()
    {
        const int vocab = 8;
        const int ctxLen = 4;
        const int trainSteps = 200;

        var arch = MakeArch(vocab: vocab, ctxLen: ctxLen, dModel: 16, dFf: 32, layers: 1, heads: 2);
        var transformer = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var input = new Tensor<float>([1, ctxLen]);
        for (int s = 0; s < ctxLen; s++) input[0, s] = (float)(s % vocab);
        var target = new Tensor<float>([1, vocab]);
        target[0, 0] = 1f;

        transformer.SetTrainingMode(true);
        // Capture initial loss (one forward pass with grad on, before any update).
        transformer.Train(input, target);
        float initialLoss = transformer.GetLastLoss();

        for (int step = 1; step < trainSteps; step++) transformer.Train(input, target);
        float finalLoss = transformer.GetLastLoss();
        _output.WriteLine($"loss before training: {initialLoss:F4} → after {trainSteps} steps: {finalLoss:F4} (ratio: {finalLoss / initialLoss:F4})");

        Assert.True(finalLoss < 0.5f * initialLoss,
            $"Loss should drop to below 50% of initial after {trainSteps} memorization steps; "
            + $"got {finalLoss:F4} (initial {initialLoss:F4}, ratio {finalLoss / initialLoss:F4}). "
            + "If this fires, gradient sign / optimizer step / loss direction is wrong.");
    }

    /// <summary>
    /// Explicit Adam optimizer at PyTorch default LR=1e-3 must produce identical
    /// behavior to the new default-Adam path. This is the regression test for
    /// the constructor logic that selects the default — if someone breaks the
    /// "if (optimizer is null) { adam } else { user-supplied }" branching, this
    /// fires.
    /// </summary>
    [Fact]
    public void ExplicitAdamMatchesDefaultBehavior()
    {
        const int vocab = 4;
        const int ctxLen = 4;
        const int trainSteps = 200;

        var arch = MakeArch(vocab: vocab, ctxLen: ctxLen, dModel: 8, dFf: 16, layers: 1, heads: 2);

        // Default-construction path: Adam with the Vaswani 2017 hyperparameters
        // the ctor constructs internally (β₁=0.9, β₂=0.98, ε=1e-9, lr=1e-3).
        var defaultTransformer = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Explicit-Adam path. Mirror the ctor's exact hyperparameters so this
        // test reflects "user constructs the same Adam by hand vs. user lets
        // the ctor build it" — NOT "ctor's Vaswani Adam vs library-default
        // Adam." The latter is a different (and noisier) comparison that
        // would conflate optimizer-config drift with anything the ctor does.
        var adamOpts = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-3,
            Beta2 = 0.98,
            Epsilon = 1e-9,
        };
        var explicitTransformer = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            optimizer: new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, adamOpts));

        // DETERMINISM: the two transformers are constructed independently so
        // their initial weights differ (Xavier/He init draws from the static
        // Random). Without a seed, the convergence-after-N-steps comparison
        // can drift across runs. Copy the default model's parameter vector
        // into the explicit model before training so both start from the
        // SAME weights — this makes the test robust to any global RNG state
        // and isolates the comparison to "do the two construction paths
        // wire up identical optimizers."
        var sharedInitialParams = defaultTransformer.GetParameters();
        explicitTransformer.UpdateParameters(sharedInitialParams);

        var input = new Tensor<float>([1, ctxLen]);
        for (int s = 0; s < ctxLen; s++) input[0, s] = (float)(s % vocab);
        var target = new Tensor<float>([1, vocab]);
        target[0, 1] = 1f;

        defaultTransformer.SetTrainingMode(true);
        explicitTransformer.SetTrainingMode(true);
        for (int step = 0; step < trainSteps; step++)
        {
            defaultTransformer.Train(input, target);
            explicitTransformer.Train(input, target);
        }

        defaultTransformer.SetTrainingMode(false);
        explicitTransformer.SetTrainingMode(false);

        float pDefault = SoftmaxAndPickClass(defaultTransformer.Predict(input), vocab, targetClass: 1);
        float pExplicit = SoftmaxAndPickClass(explicitTransformer.Predict(input), vocab, targetClass: 1);
        _output.WriteLine($"default-Adam P(target)={pDefault:F4}  explicit-Adam P(target)={pExplicit:F4}");

        // Same initial weights, same optimizer, same training loop — both
        // should converge to the SAME answer (modulo floating-point summation
        // order across the two parallel Train() calls). A 5% absolute spread
        // is conservative; bit-identical floats would require deterministic
        // SIMD reductions which we don't guarantee yet.
        Assert.True(System.Math.Abs(pDefault - pExplicit) < 0.05f,
            $"Default-Adam and explicit-Adam should produce essentially identical "
            + $"convergence after weight cloning; got default={pDefault:F4}, "
            + $"explicit={pExplicit:F4}. A spread > 5% indicates the two paths "
            + "wire up different optimizers — i.e., the ctor's default-Adam config "
            + "drifted from the explicit AdamOptimizerOptions used here.");
    }

    // ---- helpers ----

    private static TransformerArchitecture<float> MakeArch(int vocab, int ctxLen, int dModel, int dFf, int layers, int heads)
        => new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: layers,
            numDecoderLayers: 0,
            numHeads: heads,
            modelDimension: dModel,
            feedForwardDimension: dFf,
            inputSize: ctxLen,
            outputSize: vocab,
            maxSequenceLength: ctxLen,
            vocabularySize: vocab);

    private static float SoftmaxAndPickClass(Tensor<float> pred, int vocab, int targetClass)
    {
        // If pred is already a normalized probability distribution (every entry
        // in [0,1] and the row sums to ~1), don't re-apply softmax — that would
        // re-normalize and lower confident predictions, masking real model
        // behavior on a network whose final layer is softmax. Otherwise treat
        // pred as logits and apply numerically-stable softmax.
        // Use System.Math.Exp (double precision) cast to float instead of
        // MathF.Exp because MathF was added in .NET 5 and this test compiles
        // against net471 too.
        float rowSum = 0f;
        bool looksNormalized = true;
        for (int v = 0; v < vocab; v++)
        {
            float pv = pred[0, v];
            if (pv < 0f || pv > 1f) looksNormalized = false;
            rowSum += pv;
        }
        if (looksNormalized && Math.Abs(rowSum - 1f) <= 1e-3f)
        {
            return pred[0, targetClass];
        }

        float max = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++) if (pred[0, v] > max) max = pred[0, v];
        float sum = 0f;
        var p = new float[vocab];
        for (int v = 0; v < vocab; v++)
        {
            p[v] = (float)Math.Exp(pred[0, v] - max);
            sum += p[v];
        }
        return p[targetClass] / sum;
    }

    private static int SoftmaxArgmax(Tensor<float> pred, int vocab)
    {
        int best = 0;
        float bestV = pred[0, 0];
        for (int v = 1; v < vocab; v++) if (pred[0, v] > bestV) { bestV = pred[0, v]; best = v; }
        return best;
    }
}

// =====================================================
// FACADE / DIRECT-MODEL PARITY INVARIANT (closes #1267 gap)
// =====================================================

public class AiModelBuilderFacadePredictParityTests
{
    private readonly Xunit.Abstractions.ITestOutputHelper _output;
    public AiModelBuilderFacadePredictParityTests(Xunit.Abstractions.ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// After training a Transformer through AiModelBuilder.BuildAsync,
    /// the facade-wrapping AiModelResult.Predict MUST produce identical
    /// (or near-identical, allowing for FP rounding) output to the
    /// underlying Transformer's own Predict. Issue #1267: the facade
    /// returned uniform-zero on byte-LM inference even though the
    /// underlying model produced trained logits.
    /// </summary>
    [Xunit.Fact]
    public void Facade_Predict_MatchesDirectModelPredict_AfterBuildAsync()
    {
        const int vocab = 4;
        const int ctxLen = 4;
        const int batchSize = 8;

        var arch = new AiDotNet.NeuralNetworks.TransformerArchitecture<float>(
            inputType: AiDotNet.Enums.InputType.TwoDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1, numDecoderLayers: 0, numHeads: 2,
            modelDimension: 16, feedForwardDimension: 32,
            inputSize: ctxLen, outputSize: vocab,
            maxSequenceLength: ctxLen, vocabularySize: vocab);
        var model = new AiDotNet.NeuralNetworks.Transformer<float>(
            arch,
            lossFunction: new AiDotNet.LossFunctions.CategoricalCrossEntropyLoss<float>());

        // Build a tiny training set.
        var features = new Tensor<float>([batchSize, ctxLen]);
        var labels = new Tensor<float>([batchSize, vocab]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < ctxLen; s++) features[b, s] = (float)((b + s) % vocab);
            labels[b, b % vocab] = 1.0f;
        }

        var loader = AiDotNet.Data.Loaders.DataLoaders.FromTensors<float>(features, labels);
        var optimizer = new AiDotNet.Optimizers.AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AiDotNet.Models.Options.AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 1e-3,
                MaxIterations = 5,
                UseAdaptiveLearningRate = false,
            });

        var builder = new AiDotNet.AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(loader);
        var result = builder.BuildAsync().GetAwaiter().GetResult();

        // Pick one sample for parity check.
        var probe = new Tensor<float>([1, ctxLen]);
        for (int s = 0; s < ctxLen; s++) probe[0, s] = features[0, s];

        // Direct model prediction (post-training).
        model.SetTrainingMode(false);
        var directPred = model.Predict(probe);

        // Facade prediction.
        var facadePred = result.Predict(probe);

        // Compute max abs diff and L2 of each.
        double maxDiff = 0, l2Direct = 0, l2Facade = 0;
        int n = directPred.Length;
        Xunit.Assert.Equal(n, facadePred.Length);
        for (int i = 0; i < n; i++)
        {
            double d = Math.Abs(directPred[i] - facadePred[i]);
            if (d > maxDiff) maxDiff = d;
            l2Direct += directPred[i] * directPred[i];
            l2Facade += facadePred[i] * facadePred[i];
        }
        l2Direct = Math.Sqrt(l2Direct);
        l2Facade = Math.Sqrt(l2Facade);

        _output.WriteLine($"L2 direct={l2Direct:F6} facade={l2Facade:F6} maxDiff={maxDiff:F6}");

        // Bound: max-abs-diff between facade and direct must be tiny
        // (numerics only). If facade returns uniform-zero while direct
        // returns trained logits, maxDiff = max(direct logit) which is
        // typically O(1) — much larger than this 1e-3 bound.
        Xunit.Assert.True(maxDiff < 1e-3,
            $"AiModelBuilder facade Predict diverged from direct Model.Predict: "
            + $"maxDiff={maxDiff:F6} (bound 1e-3). Direct L2={l2Direct:F4}, facade L2={l2Facade:F4}. "
            + "Catches #1267-class bugs where AiModelResult wraps the model in a way that "
            + "loses post-training state (JIT capture timing, stale preprocessing, etc.).");

        // Also a sanity check: direct prediction must NOT be all-zero.
        // If both directs were zero, maxDiff would still be 0 and the
        // assertion above would pass — but the model would be untrained.
        Xunit.Assert.True(l2Direct > 1e-6,
            $"Direct Model.Predict returned all-zero (L2={l2Direct:F6}). Training itself is broken.");
    }
}
