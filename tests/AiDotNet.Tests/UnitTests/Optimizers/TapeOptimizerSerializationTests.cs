using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

public class TapeOptimizerSerializationTests
{
    [Theory]
    [InlineData(39_999, 0.0001)]
    [InlineData(40_000, 0.00005)]
    [InlineData(80_000, 0.000025)]
    public void AdamWSerializeDeserialize_RestoresStepSchedulerAtDecayBoundaries(
        int step,
        double expectedLearningRate)
    {
        var scheduler = new StepLRScheduler(0.0001, 40_000, 0.5);
        var schedulerState = scheduler.GetState();
        schedulerState["current_step"] = step;
        schedulerState["current_lr"] = scheduler.GetLearningRateAtStep(step);
        scheduler.LoadState(schedulerState);

        var source = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            Common(new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                WeightDecay = 0.0,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = SchedulerStepMode.StepPerBatch
            }));

        var restored = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            Common(new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                WeightDecay = 0.0,
                // Deliberately wrong recipe and cadence: deserialization must use the checkpoint,
                // not retain same-typed constructor state from the restore target.
                LearningRateScheduler = new StepLRScheduler(0.01, 2, 0.9),
                SchedulerStepMode = SchedulerStepMode.StepPerEpoch
            }));

        restored.Deserialize(source.Serialize());

        var restoredScheduler = Assert.IsType<StepLRScheduler>(restored.LearningRateScheduler);
        Assert.Equal(step, restoredScheduler.CurrentStep);
        Assert.Equal(40_000, restoredScheduler.StepSize);
        Assert.Equal(0.5, restoredScheduler.Gamma, 12);
        Assert.Equal(expectedLearningRate, restoredScheduler.CurrentLearningRate, 12);
        Assert.Equal(expectedLearningRate, restoredScheduler.GetLearningRateAtStep(step), 12);
        Assert.Equal(SchedulerStepMode.StepPerBatch, restored.SchedulerStepMode);
        Assert.Equal(scheduler.GetLearningRateAtStep(step + 1), restored.StepScheduler(), 12);
    }

    [Fact]
    public void AdamWSerializeDeserialize_ReconstructsOneCycleSchedulerOnFreshOptimizer()
    {
        var scheduler = new OneCycleLRScheduler(
            maxLearningRate: 0.01,
            totalSteps: 20,
            pctStart: 0.25,
            divFactor: 10.0,
            finalDivFactor: 100.0,
            annealStrategy: OneCycleLRScheduler.AnnealingStrategy.Linear);
        for (int i = 0; i < 7; i++)
            scheduler.Step();

        var source = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            Common(new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                WeightDecay = 0.0,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = SchedulerStepMode.StepPerBatch
            }));
        var restored = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            Common(new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                WeightDecay = 0.0
            }));

        restored.Deserialize(source.Serialize());

        var restoredScheduler = Assert.IsType<OneCycleLRScheduler>(restored.LearningRateScheduler);
        Assert.Equal(0.01, restoredScheduler.MaxLearningRate, 12);
        Assert.Equal(20, restoredScheduler.TotalSteps);
        Assert.Equal(0.25, restoredScheduler.PctStart, 12);
        Assert.Equal(scheduler.CurrentStep, restoredScheduler.CurrentStep);
        Assert.Equal(scheduler.CurrentLearningRate, restoredScheduler.CurrentLearningRate, 12);
        Assert.Equal(
            scheduler.GetLearningRateAtStep(scheduler.CurrentStep + 1),
            restored.StepScheduler(),
            12);
    }

    public static IEnumerable<object[]> CheckpointSupportedBuiltInSchedulers()
    {
        yield return ["Constant", (Func<ILearningRateScheduler>)(() => new ConstantLRScheduler(0.01))];
        yield return ["Step", (Func<ILearningRateScheduler>)(() => new StepLRScheduler(0.01, 3, 0.5))];
        yield return ["Exponential", (Func<ILearningRateScheduler>)(() => new ExponentialLRScheduler(0.01, 0.9))];
        yield return ["Cosine", (Func<ILearningRateScheduler>)(() => new CosineAnnealingLRScheduler(0.01, 12, 0.0001))];
        yield return ["WarmRestarts", (Func<ILearningRateScheduler>)(() => new CosineAnnealingWarmRestartsScheduler(0.01, 4, 2, 0.0001))];
        yield return ["Cyclic", (Func<ILearningRateScheduler>)(() => new CyclicLRScheduler(0.001, 0.01, 2, 3, CyclicLRScheduler.CyclicMode.Triangular2, 0.9))];
        yield return ["LinearWarmup", (Func<ILearningRateScheduler>)(() => new LinearWarmupScheduler(0.01, 2, 10, 0.001, LinearWarmupScheduler.DecayMode.Cosine, 0.0001))];
        yield return ["MultiStep", (Func<ILearningRateScheduler>)(() => new MultiStepLRScheduler(0.01, [2, 5], 0.5, 0.0001))];
        yield return ["Noam", (Func<ILearningRateScheduler>)(() => new NoamSchedule(16, 4, 0.75))];
        yield return ["OneCycle", (Func<ILearningRateScheduler>)(() => new OneCycleLRScheduler(0.01, 12, 0.25, 10.0, 100.0, OneCycleLRScheduler.AnnealingStrategy.Linear))];
        yield return ["Polynomial", (Func<ILearningRateScheduler>)(() => new PolynomialLRScheduler(0.01, 12, 2.0, 0.0001))];
        yield return ["Plateau", (Func<ILearningRateScheduler>)(() => new ReduceOnPlateauScheduler(0.01, 0.5, 2, 0.001, ReduceOnPlateauScheduler.ThresholdMode.Absolute, 1, ReduceOnPlateauScheduler.Mode.Min, 0.0001))];
        yield return ["AdaptiveFitness", (Func<ILearningRateScheduler>)(() => new AdaptiveFitnessScheduler(0.01, 0.8, 0.0001, 0.1, higherIsBetter: true))];
        yield return ["Sequential", (Func<ILearningRateScheduler>)(() => new SequentialLRScheduler(
            [new LinearWarmupScheduler(0.01, 1), new ExponentialLRScheduler(0.01, 0.9)],
            [2]))];
    }

    [Theory]
    [MemberData(nameof(CheckpointSupportedBuiltInSchedulers))]
    public void AdamWSerializeDeserialize_ReconstructsEveryCheckpointSupportedBuiltIn(
        string schedulerName,
        Func<ILearningRateScheduler> schedulerFactory)
    {
        var scheduler = schedulerFactory();
        if (scheduler is AdaptiveFitnessScheduler or ReduceOnPlateauScheduler)
            scheduler.Step(1.0);
        else
            scheduler.Step();

        var source = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            Common(new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                WeightDecay = 0.0,
                LearningRateScheduler = scheduler
            }));
        var restored = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            Common(new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                WeightDecay = 0.0
            }));

        restored.Deserialize(source.Serialize());

        Assert.NotNull(restored.LearningRateScheduler);
        Assert.True(
            scheduler.GetType() == restored.LearningRateScheduler.GetType(),
            $"{schedulerName} restored as {restored.LearningRateScheduler.GetType().Name}.");
        Assert.Equal(scheduler.CurrentStep, restored.LearningRateScheduler.CurrentStep);
        Assert.Equal(
            scheduler.CurrentLearningRate,
            restored.LearningRateScheduler.CurrentLearningRate,
            12);
    }

    [Fact]
    public void AdamWSerialize_RejectsLambdaSchedulerWithoutPortableRecipe()
    {
        var source = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            Common(new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                WeightDecay = 0.0,
                LearningRateScheduler = new LambdaLRScheduler(0.01, step => 1.0 / (step + 1)),
            }));

        var exception = Assert.Throws<NotSupportedException>(() => source.Serialize());
        Assert.Contains("delegates do not have a serializable reconstruction recipe", exception.Message);
    }

    public static IEnumerable<object[]> StatefulTapeOptimizers()
    {
#pragma warning disable CS8625
        yield return new object[] { "Adam", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { UseAMSGrad = false }))) };
        yield return new object[] { "AdamAMSGrad", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { UseAMSGrad = true }))) };
        yield return new object[] { "AdamW", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>> { WeightDecay = 0.0 }))) };
        yield return new object[] { "Adam8Bit", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new Adam8BitOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new Adam8BitOptimizerOptions<double, Tensor<double>, Tensor<double>> { BlockSize = 2, CompressBothMoments = true, QuantizationPercentile = 100.0, UseStochasticRounding = false, UseBFloat16MomentStorage = false }))) };
        yield return new object[] { "AMSGrad", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new AMSGradOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new AMSGradOptimizerOptions<double, Tensor<double>, Tensor<double>>()))) };
        yield return new object[] { "AdaMax", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new AdaMaxOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new AdaMaxOptimizerOptions<double, Tensor<double>, Tensor<double>>()))) };
        yield return new object[] { "AdaDelta", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new AdaDeltaOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new AdaDeltaOptimizerOptions<double, Tensor<double>, Tensor<double>>()))) };
        yield return new object[] { "Adagrad", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new AdagradOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new AdagradOptimizerOptions<double, Tensor<double>, Tensor<double>>()))) };
        yield return new object[] { "ASGD", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new ASGDOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new ASGDOptimizerOptions<double, Tensor<double>, Tensor<double>> { T0 = 0 }))) };
        yield return new object[] { "RAdam", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new RAdamOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new RAdamOptimizerOptions<double, Tensor<double>, Tensor<double>>()))) };
        yield return new object[] { "Rprop", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new RpropOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new RpropOptimizerOptions<double, Tensor<double>, Tensor<double>>()))) };
        yield return new object[] { "FTRL", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new FTRLOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new FTRLOptimizerOptions<double, Tensor<double>, Tensor<double>> { Alpha = 0.01, Lambda1 = 0.0, Lambda2 = 0.0 }))) };
        yield return new object[] { "LAMB", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new LAMBOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new LAMBOptimizerOptions<double, Tensor<double>, Tensor<double>> { WeightDecay = 0.0 }))) };
        yield return new object[] { "LARS", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new LARSOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new LARSOptimizerOptions<double, Tensor<double>, Tensor<double>> { WeightDecay = 0.0 }))) };
        yield return new object[] { "Lion", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new LionOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new LionOptimizerOptions<double, Tensor<double>, Tensor<double>> { WeightDecay = 0.0 }))) };
        yield return new object[] { "Momentum", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new MomentumOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new MomentumOptimizerOptions<double, Tensor<double>, Tensor<double>>()))) };
        yield return new object[] { "Nadam", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new NadamOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new NadamOptimizerOptions<double, Tensor<double>, Tensor<double>>()))) };
        yield return new object[] { "Nesterov", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new NesterovAcceleratedGradientOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new NesterovAcceleratedGradientOptimizerOptions<double, Tensor<double>, Tensor<double>>()))) };
        yield return new object[] { "RMSProp", (Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)(() => new RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>>(null, Common(new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>()))) };
#pragma warning restore CS8625
    }

    [Theory]
    [MemberData(nameof(StatefulTapeOptimizers))]
    public void SerializeDeserialize_RestoresTapeStateForFreshParameterReferences(
        string optimizerName,
        Func<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>> optimizerFactory)
    {
        var uninterrupted = optimizerFactory();
        var checkpointSource = optimizerFactory();
        var restored = optimizerFactory();

        var uninterruptedParameters = CreateParameters();
        var checkpointParameters = CreateParameters();

        Step(uninterrupted, uninterruptedParameters, CreateFirstGradients(uninterruptedParameters));
        Step(checkpointSource, checkpointParameters, CreateFirstGradients(checkpointParameters));

        byte[] optimizerState = checkpointSource.Serialize();
        restored.Deserialize(optimizerState);

        var restoredParameters = CloneParameters(checkpointParameters);

        Step(uninterrupted, uninterruptedParameters, CreateSecondGradients(uninterruptedParameters));
        Step(restored, restoredParameters, CreateSecondGradients(restoredParameters));

        AssertParametersEqual(optimizerName, uninterruptedParameters, restoredParameters);

        AssertAsgdAverageRestored(optimizerName, uninterrupted, uninterruptedParameters, restored, restoredParameters);
    }

    /// <summary>
    /// ASGD's averaged iterate is state the updated parameters cannot reveal, so it needs its own check.
    /// </summary>
    /// <remarks>
    /// The parameter comparison above passes whether or not <c>_tapeAx</c> survived the round trip: the
    /// running average is written but never read back into the parameters, so a restore that dropped it
    /// would look identical. It is the value <c>GetAveragedParameters</c> hands back as the solution,
    /// which is the whole point of Polyak-Ruppert averaging, so a silent reset would surface only as a
    /// worse final model.
    /// </remarks>
    private static void AssertAsgdAverageRestored(
        string optimizerName,
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> uninterrupted,
        Tensor<double>[] uninterruptedParameters,
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> restored,
        Tensor<double>[] restoredParameters)
    {
        if (uninterrupted is not ASGDOptimizer<double, Tensor<double>, Tensor<double>> uninterruptedAsgd
            || restored is not ASGDOptimizer<double, Tensor<double>, Tensor<double>> restoredAsgd)
        {
            return;
        }

        for (int p = 0; p < uninterruptedParameters.Length; p++)
        {
            var expected = uninterruptedAsgd.GetTapeAveragedParameterForTests(uninterruptedParameters[p]);
            var actual = restoredAsgd.GetTapeAveragedParameterForTests(restoredParameters[p]);

            Assert.NotNull(expected);
            Assert.NotNull(actual);
            for (int i = 0; i < expected!.Length; i++)
            {
                Assert.True(Math.Abs(expected[i] - actual![i]) < 1e-12,
                    $"{optimizerName} restored a different averaged iterate at parameter {p} index {i}: " +
                    $"expected {expected[i]:R}, actual {actual[i]:R}.");
            }
        }
    }

    [Fact]
    public void Adam8BitDeserialize_LegacyPayloadWithoutTapeHeader_ColdStartsTapeState()
    {
        var optimizer = CreateAdam8BitOptimizer();
        byte[] legacyPayload = StripAdam8BitTapePayload(optimizer.Serialize());
        var restored = CreateAdam8BitOptimizer();

        var exception = Record.Exception(() => restored.Deserialize(legacyPayload));

        Assert.Null(exception);
        var parameters = CreateParameters();
        var before = CloneParameters(parameters);
        Step(restored, parameters, CreateFirstGradients(parameters));

        // The cold-started tape state must drive a real parameter update — otherwise this test would
        // pass even if Deserialize left the optimizer in a state where Step is a silent no-op.
        bool anyChanged = false;
        for (int p = 0; p < parameters.Length && !anyChanged; p++)
        {
            var after = parameters[p].AsSpan();
            var prior = before[p].AsSpan();
            for (int i = 0; i < after.Length; i++)
            {
                if (after[i] != prior[i]) { anyChanged = true; break; }
            }
        }
        Assert.True(anyChanged, "Cold-started tape state must update parameters on Step, not no-op.");
    }

    [Fact]
    public void Adam8BitDeserialize_TruncatedTapeStatePayload_ThrowsInvalidOperationException()
    {
        var optimizer = CreateAdam8BitOptimizer();
        byte[] truncatedPayload = optimizer.Serialize();
        Array.Resize(ref truncatedPayload, truncatedPayload.Length - 1);
        var restored = CreateAdam8BitOptimizer();

        var exception = Assert.Throws<InvalidOperationException>(() => restored.Deserialize(truncatedPayload));

        Assert.Contains("truncated tape-state payload after the tape-step header", exception.Message);
    }

    private static TOptions Common<TOptions>(TOptions options)
        where TOptions : GradientBasedOptimizerOptions<double, Tensor<double>, Tensor<double>>
    {
        options.InitialLearningRate = 0.01;
        options.EnableGradientClipping = false;
        options.MaxIterations = 2;
        return options;
    }

    private static Adam8BitOptimizer<double, Tensor<double>, Tensor<double>> CreateAdam8BitOptimizer()
    {
        return new Adam8BitOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            Common(new Adam8BitOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                BlockSize = 2,
                CompressBothMoments = true,
                QuantizationPercentile = 100.0,
                UseStochasticRounding = false,
                UseBFloat16MomentStorage = false
            }));
    }

    private static byte[] StripAdam8BitTapePayload(byte[] serialized)
    {
        using var stream = new MemoryStream(serialized);
        using var reader = new BinaryReader(stream);

        int baseDataLength = reader.ReadInt32();
        Assert.True(baseDataLength >= 0);
        Assert.True(stream.Position + baseDataLength <= serialized.Length);
        stream.Position += baseDataLength;

        _ = reader.ReadString();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        int numBlocks = reader.ReadInt32();

        bool compressBothMoments = reader.ReadBoolean();
        bool hasMState = reader.ReadBoolean();
        if (hasMState)
        {
            int mLength = reader.ReadInt32();
            stream.Position += compressBothMoments ? mLength : sizeof(double) * mLength;
            if (compressBothMoments)
            {
                stream.Position += sizeof(double) * numBlocks;
            }
        }

        bool hasVState = reader.ReadBoolean();
        if (hasVState)
        {
            int vLength = reader.ReadInt32();
            stream.Position += vLength + sizeof(double) * numBlocks;
        }

        int tapeOffset = checked((int)stream.Position);
        var legacyPayload = new byte[tapeOffset];
        Array.Copy(serialized, legacyPayload, tapeOffset);
        return legacyPayload;
    }

    private static Tensor<double>[] CreateParameters()
    {
        return new[]
        {
            Tensor(new[] { 2, 2 }, 0.25, -0.5, 0.75, -1.0),
            Tensor(new[] { 3 }, 1.25, -1.5, 0.5)
        };
    }

    private static Dictionary<Tensor<double>, Tensor<double>> CreateFirstGradients(Tensor<double>[] parameters)
    {
        return new Dictionary<Tensor<double>, Tensor<double>>
        {
            [parameters[0]] = Tensor(new[] { 2, 2 }, 0.10, -0.20, 0.05, 0.30),
            [parameters[1]] = Tensor(new[] { 3 }, -0.15, 0.25, -0.05)
        };
    }

    private static Dictionary<Tensor<double>, Tensor<double>> CreateSecondGradients(Tensor<double>[] parameters)
    {
        return new Dictionary<Tensor<double>, Tensor<double>>
        {
            [parameters[0]] = Tensor(new[] { 2, 2 }, -0.07, 0.11, -0.13, 0.17),
            [parameters[1]] = Tensor(new[] { 3 }, 0.19, -0.23, 0.29)
        };
    }

    private static void Step(
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer,
        Tensor<double>[] parameters,
        Dictionary<Tensor<double>, Tensor<double>> gradients)
    {
        optimizer.Step(new TapeStepContext<double>(parameters, gradients, 0.0));
    }

    private static Tensor<double>[] CloneParameters(Tensor<double>[] parameters)
    {
        return parameters.Select(CloneTensor).ToArray();
    }

    private static Tensor<double> CloneTensor(Tensor<double> source)
    {
        var clone = new Tensor<double>(source._shape);
        source.AsSpan().CopyTo(clone.AsWritableSpan());
        return clone;
    }

    private static Tensor<double> Tensor(int[] shape, params double[] values)
    {
        var tensor = new Tensor<double>(shape);
        Assert.Equal(tensor.Length, values.Length);
        values.CopyTo(tensor.AsWritableSpan());
        return tensor;
    }

    private static void AssertParametersEqual(string optimizerName, Tensor<double>[] expected, Tensor<double>[] actual)
    {
        for (int parameterIndex = 0; parameterIndex < expected.Length; parameterIndex++)
        {
            var expectedSpan = expected[parameterIndex].AsSpan();
            var actualSpan = actual[parameterIndex].AsSpan();
            Assert.Equal(expectedSpan.Length, actualSpan.Length);
            for (int elementIndex = 0; elementIndex < expectedSpan.Length; elementIndex++)
            {
                double difference = Math.Abs(expectedSpan[elementIndex] - actualSpan[elementIndex]);
                Assert.True(
                    difference <= 1e-9,
                    $"{optimizerName} parameter {parameterIndex} element {elementIndex} differed by {difference}. " +
                    $"Expected {expectedSpan[elementIndex]}, actual {actualSpan[elementIndex]}.");
            }
        }
    }
}
