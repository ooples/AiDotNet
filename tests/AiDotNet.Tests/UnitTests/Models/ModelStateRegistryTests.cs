using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

public sealed class ModelStateRegistryTests
{
    [Fact]
    public void AssignableChild_RestoresModelWithOneRequiredVectorConstructor()
    {
        var coefficients = new Vector<double>(new[] { 1.25, -2.5, 3.75 });
        using var source = new VectorModel<double>(coefficients);
        var writingRegistry = new ModelStateRegistry<double>();
        writingRegistry.DeclareChild<IModelSerializer>("child", () => source, _ => { });
        using var payload = Write(writingRegistry);

        IModelSerializer? destination = null;
        var readingRegistry = new ModelStateRegistry<double>();
        readingRegistry.DeclareChild<IModelSerializer>(
            "child",
            () => destination,
            value => destination = value);
        using var reader = new BinaryReader(
            payload, System.Text.Encoding.UTF8, leaveOpen: true);
        readingRegistry.ReadAll(reader);

        using var restored = Assert.IsType<VectorModel<double>>(destination);
        Assert.Equal(payload.Length, payload.Position);
        Assert.Equal(coefficients.Length, restored.Coefficients.Length);
        for (int i = 0; i < coefficients.Length; i++)
            Assert.Equal(coefficients[i], restored.Coefficients[i]);
    }

    [Fact]
    public void RandomState_RestoresSeededContinuationIntoIndependentInstance()
    {
        var source = RandomHelper.CreateSeededRandom(173);
        for (int i = 0; i < 257; i++)
            _ = source.Next();

        var destination = RandomHelper.CreateSeededRandom(173);
        RoundTripRandom(source, destination);

        Assert.NotSame(source, destination);
        for (int i = 0; i < 128; i++)
        {
            Assert.Equal(source.Next(), destination.Next());
            Assert.Equal(source.NextDouble(), destination.NextDouble());
        }
    }

    [Fact]
    public void RandomState_RestoresSecureRuntimeImplementation()
    {
        var source = RandomHelper.CreateSecureRandom();
        for (int i = 0; i < 257; i++)
            _ = source.Next();

        var destination = RandomHelper.CreateSecureRandom();
        RoundTripRandom(source, destination);

        Assert.NotSame(source, destination);
        for (int i = 0; i < 128; i++)
            Assert.Equal(source.Next(), destination.Next());
    }

    [Fact]
    public void RandomState_MaterializesSeededContinuationForAssignableNullDestination()
    {
        var source = RandomHelper.CreateSeededRandom(347);
        for (int i = 0; i < 257; i++)
            _ = source.Next();

        Random? destination = RoundTripRandomIntoNullDestination(source);

        Assert.NotNull(destination);
        Assert.NotSame(source, destination);
        for (int i = 0; i < 128; i++)
            Assert.Equal(source.Next(), destination.Next());
    }

    [Fact]
    public void RandomState_MaterializesSecureContinuationForAssignableNullDestination()
    {
        var source = RandomHelper.CreateSecureRandom();
        for (int i = 0; i < 257; i++)
            _ = source.Next();

        Random? destination = RoundTripRandomIntoNullDestination(source);

        Assert.NotNull(destination);
        Assert.NotSame(source, destination);
        for (int i = 0; i < 128; i++)
            Assert.Equal(source.Next(), destination.Next());
    }

    [Fact]
    public void RandomState_NameFramingRemainsCompatibleWhenEitherSideLacksTheEntry()
    {
        var source = RandomHelper.CreateSeededRandom(41);
        var writingRegistry = new ModelStateRegistry<double>();
        writingRegistry.DeclareRandom("rng", () => source);

        using var newPayload = Write(writingRegistry);
        var legacyReader = new ModelStateRegistry<double>();
        using (var reader = new BinaryReader(
                   newPayload, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            legacyReader.ReadAll(reader);
            Assert.Equal(newPayload.Length, newPayload.Position);
        }

        var oldWriter = new ModelStateRegistry<double>();
        using var oldPayload = Write(oldWriter);
        var destination = RandomHelper.CreateSeededRandom(73);
        var untouched = RandomHelper.CreateSeededRandom(73);
        var currentReader = new ModelStateRegistry<double>();
        currentReader.DeclareRandom("rng", () => destination);
        using (var reader = new BinaryReader(
                   oldPayload, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            currentReader.ReadAll(reader);
            Assert.Equal(oldPayload.Length, oldPayload.Position);
        }

        for (int i = 0; i < 32; i++)
            Assert.Equal(untouched.Next(), destination.Next());
    }

    [Fact]
    public void Options_WithAForwardingSetter_StageLeavesTheLiveOptionsUntouchedAndReportsAFallibleCommit()
    {
        // The dry run writes to a shallow copy, which shares Inner with the live options, so the
        // forwarding setter reaches the live object through it. Staging must undo that and say the
        // commit is no longer side-effect free.
        var restored = RoundTripOptions(
            new ForwardingOptions { Label = "checkpoint" }, new ForwardingOptions { Label = "live" },
            out var commit, out bool hasFallibleCommit);

        Assert.Equal("live", restored.Label);
        Assert.True(hasFallibleCommit);
        commit();
        Assert.Equal("checkpoint", restored.Label);
    }

    [Fact]
    public void Options_WithPlainSetters_StageIsSideEffectFreeAndNotFallible()
    {
        // Control: detection must not mark every options restore fallible, or every restore would
        // pay for a rollback snapshot.
        var restored = RoundTripOptions(
            new PlainOptions { Label = "checkpoint" }, new PlainOptions { Label = "live" },
            out var commit, out bool hasFallibleCommit);

        Assert.Equal("live", restored.Label);
        Assert.False(hasFallibleCommit);
        commit();
        Assert.Equal("checkpoint", restored.Label);
    }

    private static TOptions RoundTripOptions<TOptions>(
        TOptions source, TOptions destination, out Action commit, out bool hasFallibleCommit)
        where TOptions : class
    {
        var writingRegistry = new ModelStateRegistry<double>();
        writingRegistry.DeclareOptions("options", () => source);
        using var payload = Write(writingRegistry);

        var readingRegistry = new ModelStateRegistry<double>();
        readingRegistry.DeclareOptions("options", () => destination);
        using var reader = new BinaryReader(
            payload, System.Text.Encoding.UTF8, leaveOpen: true);
        commit = readingRegistry.StageAll(reader, restoreAfterParameters: null, out hasFallibleCommit);
        Assert.Equal(payload.Length, payload.Position);
        return destination;
    }

    public sealed class ForwardingOptions
    {
        public LabelHolder Inner { get; set; } = new LabelHolder();
        public string Label { get => Inner.Value; set => Inner.Value = value; }
    }

    public sealed class LabelHolder
    {
        public string Value { get; set; } = string.Empty;
    }

    public sealed class PlainOptions
    {
        public string Label { get; set; } = string.Empty;
    }
    private static void RoundTripRandom(Random source, Random destination)
    {
        var writingRegistry = new ModelStateRegistry<double>();
        writingRegistry.DeclareRandom("rng", () => source);
        using var payload = Write(writingRegistry);

        var readingRegistry = new ModelStateRegistry<double>();
        readingRegistry.DeclareRandom("rng", () => destination);
        using var reader = new BinaryReader(
            payload, System.Text.Encoding.UTF8, leaveOpen: true);
        readingRegistry.ReadAll(reader);
        Assert.Equal(payload.Length, payload.Position);
    }

    private static Random? RoundTripRandomIntoNullDestination(Random source)
    {
        var writingRegistry = new ModelStateRegistry<double>();
        writingRegistry.DeclareRandom("rng", () => source);
        using var payload = Write(writingRegistry);

        Random? destination = null;
        var readingRegistry = new ModelStateRegistry<double>();
        readingRegistry.DeclareRandom("rng", () => destination, value => destination = value);
        using var reader = new BinaryReader(
            payload, System.Text.Encoding.UTF8, leaveOpen: true);
        readingRegistry.ReadAll(reader);
        Assert.Equal(payload.Length, payload.Position);
        return destination;
    }

    private static MemoryStream Write(ModelStateRegistry<double> registry)
    {
        var stream = new MemoryStream();
        using (var writer = new BinaryWriter(
                   stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            registry.WriteAll(writer);
            writer.Flush();
        }
        stream.Position = 0;
        return stream;
    }
}
