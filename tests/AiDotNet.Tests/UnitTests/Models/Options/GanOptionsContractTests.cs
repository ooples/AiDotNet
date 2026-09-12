using System;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

/// <summary>Validates the shared GAN options contract without inventing model defaults or constructing a model.</summary>
public sealed class GanOptionsContractTests
{
    public enum Channel { Generator, Discriminator }
    public enum ExistingField { LatentSize, ImageChannels, InitialLearningRate }

    // No production options class currently derives from GanOptions. This probe exercises
    // its protected validation contract, not the separate BigGAN/SAGAN constructor APIs.
    private sealed class ProbeGanOptions : GanOptions
    {
        internal ProbeGanOptions()
        {
            LatentSize = 1;
            GeneratorChannels = 1;
            DiscriminatorChannels = 1;
            InitialLearningRate = 0.001;
        }

        internal void Validate() => ValidateCore();
    }

    [Theory]
    [InlineData(Channel.Generator, 0)]
    [InlineData(Channel.Generator, -1)]
    [InlineData(Channel.Generator, int.MinValue)]
    [InlineData(Channel.Discriminator, 0)]
    [InlineData(Channel.Discriminator, -1)]
    [InlineData(Channel.Discriminator, int.MinValue)]
    public void RequiredChannelsRejectNonpositiveValuesAtTheSharedBoundary(Channel channel, int value)
    {
        var options = new ProbeGanOptions();
        options.Validate();
        string property;
        switch (channel)
        {
            case Channel.Generator:
                options.GeneratorChannels = value;
                property = nameof(GanOptions.GeneratorChannels);
                break;
            case Channel.Discriminator:
                options.DiscriminatorChannels = value;
                property = nameof(GanOptions.DiscriminatorChannels);
                break;
            default: throw new ArgumentOutOfRangeException(nameof(channel));
        }
        var exception = Assert.Throws<ArgumentException>(options.Validate);
        Assert.Equal("options", exception.ParamName);
        Assert.Contains(property, exception.Message);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(64, 96)]
    [InlineData(int.MaxValue, int.MaxValue)]
    public void PositiveChannelCountsRemainValidWithoutRequiringCriticIterations(int generator, int discriminator)
    {
        var options = new ProbeGanOptions { GeneratorChannels = generator, DiscriminatorChannels = discriminator };
        Assert.Equal(0, options.CriticIterations);
        Assert.Equal(3, options.ImageChannels);
        options.Validate();
    }

    [Theory]
    [InlineData(ExistingField.LatentSize)]
    [InlineData(ExistingField.ImageChannels)]
    [InlineData(ExistingField.InitialLearningRate)]
    public void ExistingRequiredFieldsStillRejectMissingValues(ExistingField field)
    {
        var options = new ProbeGanOptions();
        options.Validate();
        string property;
        switch (field)
        {
            case ExistingField.LatentSize: options.LatentSize = 0; property = nameof(GanOptions.LatentSize); break;
            case ExistingField.ImageChannels: options.ImageChannels = 0; property = nameof(GanOptions.ImageChannels); break;
            case ExistingField.InitialLearningRate: options.InitialLearningRate = 0; property = nameof(GanOptions.InitialLearningRate); break;
            default: throw new ArgumentOutOfRangeException(nameof(field));
        }
        var exception = Assert.Throws<ArgumentException>(options.Validate);
        Assert.Equal("options", exception.ParamName);
        Assert.Contains(property, exception.Message);
    }
}
