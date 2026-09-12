using System;
using System.Reflection;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

public class ModelHyperparameterSurfaceTests
{
    [Fact]
    public void SharedFamilyOptions_DoNotExposeAnUnusedGradientAlias()
    {
        // This is the removed API member's identity, not a policy-dispatch string.
        Assert.Null(typeof(ModelHyperparameterOptions).GetProperty("MaxGradNorm",
            BindingFlags.Instance | BindingFlags.Public));
    }

    [Theory]
    [InlineData(typeof(FinchOptions))]
    [InlineData(typeof(GriffinOptions))]
    [InlineData(typeof(HawkOptions))]
    [InlineData(typeof(RecurrentGemmaOptions))]
    public void ExistingOptimizerThresholdAndEnableSwitch_KeepTheirNamesDefaultsAndCopies(Type optionsType)
    {
        var options = Activator.CreateInstance(optionsType)
            ?? throw new InvalidOperationException($"Could not create {optionsType.Name}.");
        var threshold = optionsType.GetProperty(nameof(GriffinOptions.MaxGradientNorm))
            ?? throw new InvalidOperationException($"{optionsType.Name} lost its optimizer threshold.");
        var enabled = optionsType.GetProperty(nameof(GriffinOptions.EnableGradientClipping))
            ?? throw new InvalidOperationException($"{optionsType.Name} lost its optimizer clipping switch.");
        Assert.Equal(typeof(double), threshold.PropertyType);
        Assert.Equal(typeof(bool), enabled.PropertyType);
        Assert.Equal(1.0, threshold.GetValue(options));
        Assert.Equal(true, enabled.GetValue(options));

        threshold.SetValue(options, 2.5);
        enabled.SetValue(options, false);
        var copyConstructor = optionsType.GetConstructor(new[] { optionsType })
            ?? throw new InvalidOperationException($"{optionsType.Name} lost its copy constructor.");
        var copy = copyConstructor.Invoke(new[] { options });
        Assert.NotSame(options, copy);
        Assert.Equal(2.5, threshold.GetValue(copy));
        Assert.Equal(false, enabled.GetValue(copy));
        threshold.SetValue(options, 7.0);
        enabled.SetValue(options, true);
        Assert.Equal(2.5, threshold.GetValue(copy));
        Assert.Equal(false, enabled.GetValue(copy));
    }
}
