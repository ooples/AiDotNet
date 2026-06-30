using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

public class TapeOptimizerSerializationTests
{
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
        Step(restored, parameters, CreateFirstGradients(parameters));
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
