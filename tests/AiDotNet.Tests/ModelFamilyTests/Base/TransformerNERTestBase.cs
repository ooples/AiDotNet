using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.NER.Options;
using AiDotNet.NER.TransformerBased;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using System.Reflection;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for transformer-based NER models (BERT-NER, RoBERTa-NER, etc.).
/// Inherits NER invariants and adds transformer-specific: contextual sensitivity
/// and attention-based output variation.
/// </summary>
public abstract class TransformerNERTestBase<T> : NERModelTestBase<T>
{
    /// <summary>
    /// Explicitly opts a generated smoke fixture into a positive first warmup update. This retains
    /// its established learning-rate trajectory without changing the production zero-start contract.
    /// </summary>
    protected static TOptions WithPositiveSmokeWarmup<TOptions>(TOptions options)
        where TOptions : TransformerNEROptions
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (options.WarmupSteps > 0 && options.WarmupInitialLearningRate == 0.0)
        {
            options.WarmupInitialLearningRate = options.LearningRate / options.WarmupSteps;
        }
        return options;
    }

    /// <inheritdoc />
    protected override void PrepareForGradientFlowInvariant(
        INeuralNetworkModel<T> network, Tensor<T> input, Tensor<T> target)
    {
        base.PrepareForGradientFlowInvariant(network, input, target);
        if (network is not TransformerNERBase<T>
            || network.GetOptions() is not TransformerNEROptions options
            || options.WarmupSteps <= 0 || options.WarmupInitialLearningRate != 0.0)
        {
            return;
        }

        // Read the actual model-owned optimizer: its constructor may have received a custom
        // optimizer, and a clone may already have advanced its scheduler. Options alone cannot
        // establish that this particular next step is intentionally zero. Never mutate that state.
        FieldInfo? optimizerField = typeof(TransformerNERBase<T>).GetField("_optimizer",
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(optimizerField);
        if (optimizerField.GetValue(network) is not GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> optimizer
            || optimizer.SchedulerStepMode != SchedulerStepMode.StepPerBatch
            || optimizer.CurrentStep != 0 || optimizer.GetCurrentLearningRate() != 0.0
            || optimizer.LearningRateScheduler is not LinearWarmupScheduler scheduler
            || scheduler.CurrentStep != 0 || scheduler.CurrentLearningRate != 0.0)
        {
            return;
        }
        double nextRate = scheduler.GetLearningRateAtStep(1);
        if (!(nextRate > 0.0) || double.IsInfinity(nextRate)) return;

        var initialHashes = ComputeChunkHashes(network);
        Assert.NotEmpty(initialHashes);
        network.Train(input, target);
        Assert.Equal(initialHashes, ComputeChunkHashes(network));
        Assert.Equal(1, optimizer.CurrentStep);
        Assert.Equal(1, scheduler.CurrentStep);
        Assert.Equal(nextRate, scheduler.CurrentLearningRate);
        Assert.Equal(nextRate, optimizer.GetCurrentLearningRate());
        foreach (var chunk in EnumerateParameterChunks(network))
        {
            for (int i = 0; i < chunk.Length; i++)
            {
                double value = ConvertToDouble(chunk[i]);
                if (double.IsNaN(value) || double.IsInfinity(value))
                {
                    Assert.Fail("A parameter became non-finite during the intentional zero-rate warmup step.");
                }
            }
        }
    }

    [Fact(Timeout = 120000)]
    public virtual async Task ContextualSensitivity_DifferentContext_DifferentLabels()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        // Two CONTENT-distinct contexts. The previous probe used two SPATIALLY
        // CONSTANT inputs (0.3 vs 0.7) that differ only by a global scale, but a
        // BERT-class NER encoder is LayerNorm-first and therefore invariant to a
        // uniform scale/shift of its input — it correctly maps both constants to
        // the same output, so the test was a false positive for "attention is
        // broken". Use two inputs that differ in their PER-POSITION pattern (which
        // survives LayerNorm) so the test exercises genuine contextual sensitivity
        // and still fails loudly if the encoder truly ignores its input.
        //
        // The pattern is driven from BOTH the sequence axis (token position) and the
        // feature axis. A feature-only pattern (i % lastDim) gives every token the
        // SAME row vector, so a model that ignores cross-token attention — reacting
        // only to a single repeated embedding — could still pass. Varying per token
        // index makes each position distinct, so the probe genuinely requires the
        // encoder to attend across the sequence.
        int lastDim = InputShape[InputShape.Length - 1];
        int seqDim = InputShape.Length > 1 ? InputShape[InputShape.Length - 2] : 1;
        var input1 = new Tensor<T>(InputShape);
        var input2 = new Tensor<T>(InputShape);
        for (int i = 0; i < input1.Length; i++)
        {
            int featureIndex = i % lastDim;
            int tokenIndex = (i / lastDim) % seqDim;
            double featurePhase = featureIndex / (double)lastDim;
            double tokenPhase = tokenIndex / (double)System.Math.Max(1, seqDim - 1);
            input1[i] = NumOps.FromDouble(0.2 + 0.3 * tokenPhase + 0.3 * featurePhase);
            input2[i] = NumOps.FromDouble(0.8 - 0.3 * tokenPhase - 0.3 * featurePhase);
        }

        var labels1 = network.Predict(input1);
        var labels2 = network.Predict(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(labels1.Length, labels2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(ConvertToDouble(labels1[i]) - ConvertToDouble(labels2[i])) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Transformer NER produces identical labels for different contexts — attention may be broken.");
    }

    [Fact(Timeout = 120000)]
    public async Task Output_ShouldBeFiniteSequence()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            double v = ConvertToDouble(output[i]);
            Assert.False(double.IsNaN(v), $"Transformer NER output[{i}] is NaN.");
            Assert.False(double.IsInfinity(v), $"Transformer NER output[{i}] is Infinity.");
        }
    }
}

/// <summary>Double-precision default for <see cref="TransformerNERTestBase{T}"/>.</summary>
public abstract class TransformerNERTestBase : TransformerNERTestBase<double> { }
