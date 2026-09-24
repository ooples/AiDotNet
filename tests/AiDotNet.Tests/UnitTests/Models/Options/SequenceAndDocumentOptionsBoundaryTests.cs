using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

public sealed class SequenceAndDocumentOptionsBoundaryTests
{
    [Theory]
    [InlineData(typeof(JambaOptions))]
    [InlineData(typeof(Mamba2Options))]
    [InlineData(typeof(XLSTMOptions))]
    public void SequenceValidationRemainsAnInternalConstructorBoundary(Type optionsType)
    {
        Assert.Null(optionsType.GetMethod("Validate", BindingFlags.Public | BindingFlags.Instance));
        MethodInfo method = optionsType.GetMethod("Validate", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("The constructor validator must remain available internally.");
        Assert.True(method.IsAssembly);
    }

    [Fact]
    public void ConcreteDocumentOptionsExposeTheirCopyConstructor()
        => Assert.NotNull(CopyConstructor());

    [Fact]
    public void DocumentCopyPreservesEveryDocumentPropertyAndTheInheritedConfiguration()
    {
        var source = new DocumentNeuralNetworkOptions
        {
            ImageSize = 101, ImageWidth = 102, ImageHeight = 103, PatchSize = 104,
            MaxSequenceLength = 105, VocabSize = 106, HiddenDim = 107, NumHeads = 108,
            NumLayers = 109, NumEncoderLayers = 110, NumDecoderLayers = 111,
            VisionDim = 112, VisionLayers = 113, BackboneChannels = 114, NumClasses = 115,
            EncoderLayerCount = 116,
        };
        var copied = Assert.IsType<DocumentNeuralNetworkOptions>(CopyConstructor().Invoke(new object[] { source }));
        Assert.NotSame(source, copied);
        foreach (var property in typeof(DocumentNeuralNetworkOptions).GetProperties(BindingFlags.Public | BindingFlags.Instance)
                     .Where(property => property.DeclaringType == typeof(DocumentNeuralNetworkOptions)))
            Assert.Equal(property.GetValue(source), property.GetValue(copied));
        Assert.Equal(source.EncoderLayerCount, copied.EncoderLayerCount);
        copied.ImageSize = 201;
        copied.EncoderLayerCount = 202;
        Assert.Equal(101, source.ImageSize);
        Assert.Equal(116, source.EncoderLayerCount);
    }

    [Fact]
    public void DocumentCopyRejectsNullAtTheExistingBaseGuard()
    {
        var exception = Assert.Throws<TargetInvocationException>(() => CopyConstructor().Invoke(new object?[] { null }));
        var guard = Assert.IsType<ArgumentNullException>(exception.InnerException);
        Assert.Equal("other", guard.ParamName);
    }

    private static ConstructorInfo CopyConstructor()
    {
        var constructor = typeof(DocumentNeuralNetworkOptions).GetConstructor(new[] { typeof(DocumentNeuralNetworkOptions) });
        Assert.NotNull(constructor);
        return constructor;
    }
}
