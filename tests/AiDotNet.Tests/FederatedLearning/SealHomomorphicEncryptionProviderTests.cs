using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class SealHomomorphicEncryptionProviderTests
{
    [Fact]
    public void AggregateEncryptedWeightedAverage_ValidatesInputs()
    {
        var provider = new SealHomomorphicEncryptionProvider<double>();
        var baseline = new Vector<double>(new[] { 1.0 });
        var options = new HomomorphicEncryptionOptions();

        Assert.Throws<ArgumentException>(() => provider.AggregateEncryptedWeightedAverage(null!, new Dictionary<int, double>(), baseline, new[] { 0 }, options));
        Assert.Throws<ArgumentException>(() => provider.AggregateEncryptedWeightedAverage(new Dictionary<int, Vector<double>>(), new Dictionary<int, double>(), baseline, new[] { 0 }, options));
        Assert.Throws<ArgumentException>(() => provider.AggregateEncryptedWeightedAverage(new Dictionary<int, Vector<double>> { [1] = baseline }, null!, baseline, new[] { 0 }, options));
        Assert.Throws<ArgumentException>(() => provider.AggregateEncryptedWeightedAverage(new Dictionary<int, Vector<double>> { [1] = baseline }, new Dictionary<int, double>(), baseline, new[] { 0 }, options));
        Assert.Throws<ArgumentNullException>(() => provider.AggregateEncryptedWeightedAverage(new Dictionary<int, Vector<double>> { [1] = baseline }, new Dictionary<int, double> { [1] = 1.0 }, null!, new[] { 0 }, options));
        Assert.Throws<ArgumentNullException>(() => provider.AggregateEncryptedWeightedAverage(new Dictionary<int, Vector<double>> { [1] = baseline }, new Dictionary<int, double> { [1] = 1.0 }, baseline, new[] { 0 }, null!));
    }

    [Fact]
    public void AggregateEncryptedWeightedAverage_ReturnsBaseline_WhenNoValidEncryptedIndices()
    {
        var provider = new SealHomomorphicEncryptionProvider<double>();
        var baseline = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var options = new HomomorphicEncryptionOptions();

        var result = provider.AggregateEncryptedWeightedAverage(
            clientParameters: new Dictionary<int, Vector<double>> { [1] = baseline },
            clientWeights: new Dictionary<int, double> { [1] = 1.0 },
            globalBaseline: baseline,
            encryptedIndices: new[] { -1, 99 },
            options: options);

        Assert.False(ReferenceEquals(baseline, result));
        Assert.Equal(1.0, result[0], precision: 12);
        Assert.Equal(2.0, result[1], precision: 12);
        Assert.Equal(3.0, result[2], precision: 12);
    }

    [Fact]
    public void AggregateEncryptedWeightedAverage_ThrowsWhenWeightsMissingOrInvalid()
    {
        var provider = new SealHomomorphicEncryptionProvider<double>();
        var baseline = new Vector<double>(new[] { 1.0 });

        Assert.Throws<ArgumentException>(() => provider.AggregateEncryptedWeightedAverage(
            clientParameters: new Dictionary<int, Vector<double>> { [5] = baseline },
            clientWeights: new Dictionary<int, double> { [6] = 1.0 },
            globalBaseline: baseline,
            encryptedIndices: new[] { 0 },
            options: new HomomorphicEncryptionOptions { Scheme = (HomomorphicEncryptionScheme)999 }));

        Assert.Throws<ArgumentException>(() => provider.AggregateEncryptedWeightedAverage(
            clientParameters: new Dictionary<int, Vector<double>> { [5] = baseline },
            clientWeights: new Dictionary<int, double> { [5] = 0.0 },
            globalBaseline: baseline,
            encryptedIndices: new[] { 0 },
            options: new HomomorphicEncryptionOptions { Scheme = (HomomorphicEncryptionScheme)999 }));
    }

    [Fact]
    public void AggregateEncryptedWeightedAverage_ThrowsForUnknownScheme()
    {
        var provider = new SealHomomorphicEncryptionProvider<double>();
        var baseline = new Vector<double>(new[] { 1.0 });

        Assert.Throws<InvalidOperationException>(() => provider.AggregateEncryptedWeightedAverage(
            clientParameters: new Dictionary<int, Vector<double>> { [1] = baseline },
            clientWeights: new Dictionary<int, double> { [1] = 1.0 },
            globalBaseline: baseline,
            encryptedIndices: new[] { 0 },
            options: new HomomorphicEncryptionOptions { Scheme = (HomomorphicEncryptionScheme)999 }));
    }

    [Fact]
    public void GetProviderName_IsStable()
    {
        var provider = new SealHomomorphicEncryptionProvider<double>();
        Assert.Equal("SEAL", provider.GetProviderName());
    }
}
