using FrameworkAttention = AiDotNet.NeuralNetworks.Attention.FlashAttention<float>;
using FrameworkConfig = AiDotNet.NeuralNetworks.Attention.FlashAttentionConfig;
using TensorAttention = AiDotNet.Tensors.Engines.Autodiff.FusedAttention<float>;
using TensorConfig = AiDotNet.Tensors.Engines.Autodiff.FlashAttentionConfig;
using System.IO;
using System.Security.Cryptography;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

public sealed class RectangularAttentionReviewTests
{
    public RectangularAttentionReviewTests(ITestOutputHelper output)
    {
        TestModuleInitializer.EnsureInitialized();
        string loadedPath = typeof(TensorAttention).Assembly.Location;
        using var stream = File.OpenRead(loadedPath);
        using var sha = SHA256.Create();
        string loadedHash = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty);
        output.WriteLine("Loaded Tensors assembly: " + loadedPath);
        output.WriteLine("Loaded Tensors SHA-256: " + loadedHash);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void NoncausalMoreQueriesThanKeys_MatchesManualSoftmax(bool framework, bool fourDimensions)
    {
        var query = new Tensor<float>(fourDimensions ? new[] { 1, 1, 3, 2 } : new[] { 1, 3, 2 },
            new Vector<float>(new[] { 1f, 0f, 0f, 1f, 1f, 1f }));
        var key = new Tensor<float>(fourDimensions ? new[] { 1, 1, 2, 2 } : new[] { 1, 2, 2 },
            new Vector<float>(new[] { 1f, 0f, 0f, 1f }));
        var value = new Tensor<float>(key.Shape.ToArray(), new Vector<float>(new[] { 2f, 3f, 5f, 7f }));
        var output = framework
            ? FrameworkAttention.Forward(query, key, value, new FrameworkConfig { UseCausalMask = false }).Output
            : TensorAttention.Forward(query, key, value, new TensorConfig { IsCausal = false }).Output;
        Assert.Equal(query.Shape.ToArray(), output.Shape.ToArray());
        for (int row = 0; row < 3; row++)
        {
            double first = Math.Exp(query[row * 2] / Math.Sqrt(2));
            double second = Math.Exp(query[row * 2 + 1] / Math.Sqrt(2));
            for (int column = 0; column < 2; column++)
            {
                double expected = (first * value[column] + second * value[2 + column]) / (first + second);
                Assert.InRange(Math.Abs(output[row * 2 + column] - expected), 0, 2e-6);
            }
        }
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(2)]
    [InlineData(int.MaxValue)]
    public void CausalQueryWindow_RejectsNegativeOutOfRangeAndOverflowOffsets(int queryOffset)
    {
        var query = new Tensor<float>(new[] { 1, 2, 2 });
        var key = new Tensor<float>(new[] { 1, 3, 2 });
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => FrameworkAttention.Forward(query, key, key,
            new FrameworkConfig { UseCausalMask = true }, queryOffset));
        Assert.Equal("queryOffset", error.ParamName);
    }

    [Fact]
    public void CausalQueryWindow_AtExactEndRemainsValid()
    {
        var query = new Tensor<float>(new[] { 1, 2, 2 });
        var key = new Tensor<float>(new[] { 1, 3, 2 });
        var value = new Tensor<float>(new[] { 1, 3, 2 }, new Vector<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }));
        var output = FrameworkAttention.Forward(query, key, value, new FrameworkConfig { UseCausalMask = true }, 1).Output;
        Assert.Equal(new[] { 2f, 3f, 3f, 4f }, output.ToArray());
    }

    [Theory]
    [InlineData(2)]
    [InlineData(int.MaxValue)]
    public void NoncausalExplicitOffset_PreservesWindowBounds(int queryOffset)
    {
        var query = new Tensor<float>(new[] { 1, 2, 2 });
        var key = new Tensor<float>(new[] { 1, 3, 2 });
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => FrameworkAttention.Forward(query, key, key,
            new FrameworkConfig { UseCausalMask = false }, queryOffset));
        Assert.Equal("queryOffset", error.ParamName);
    }
}
