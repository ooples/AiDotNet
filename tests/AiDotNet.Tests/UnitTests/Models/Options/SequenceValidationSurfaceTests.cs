using System;
using System.Reflection;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

/// <summary>Options stay public while constructor-owned validation stays inside the assembly.</summary>
public class SequenceValidationSurfaceTests
{
    [Theory]
    [InlineData(typeof(FinchOptions))]
    [InlineData(typeof(GLAOptions))]
    [InlineData(typeof(GatedDeltaNetOptions))]
    [InlineData(typeof(GriffinOptions))]
    [InlineData(typeof(HawkOptions))]
    [InlineData(typeof(RecurrentGemmaOptions))]
    public void ConstructorValidation_IsInternalWithoutChangingTheOptionsContract(Type optionsType)
    {
        Assert.True(optionsType.IsPublic);
        var constructor = optionsType.GetConstructor(Type.EmptyTypes)
            ?? throw new InvalidOperationException($"{optionsType.Name} has no public default constructor.");
        var options = Assert.IsAssignableFrom<SequenceModelOptions>(constructor.Invoke(Array.Empty<object>()));
        var validate = optionsType.GetMethod(nameof(GLAOptions.Validate),
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly,
            binder: null, types: Type.EmptyTypes, modifiers: null)
            ?? throw new InvalidOperationException($"{optionsType.Name} has no declared validation method.");

        Assert.True(validate.IsAssembly, $"{optionsType.Name}.Validate must be internal, not public or protected.");
        Assert.Null(optionsType.GetMethod(nameof(GLAOptions.Validate), BindingFlags.Instance | BindingFlags.Public));

        // A visibility change cannot silently remove validation or invalidate shipped defaults.
        validate.Invoke(options, Array.Empty<object>());
        options.VocabSize = 0;
        var failure = Assert.Throws<TargetInvocationException>(() => validate.Invoke(options, Array.Empty<object>()));
        var argument = Assert.IsType<ArgumentException>(failure.InnerException);
        Assert.Equal("options", argument.ParamName);
        Assert.Contains($"{optionsType.Name}.{nameof(SequenceModelOptions.VocabSize)}", argument.Message);
    }
}
