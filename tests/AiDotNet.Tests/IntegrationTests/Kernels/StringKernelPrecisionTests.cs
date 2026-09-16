using System.Linq;
using AiDotNet.Kernels;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Kernels;

/// <summary>
/// Regression tests for integer-overflow in the <see cref="StringKernel{T}"/> count products.
/// A single k-mer / word occurring more than 46,341 times made <c>count * count</c> overflow
/// int32 before being accumulated into a double, turning the norms negative and the kernel NaN.
/// </summary>
public class StringKernelPrecisionTests
{
    private const int RepeatCount = 50_000; // 50,000^2 = 2.5e9 > int.MaxValue

    [Fact(Timeout = 120000)]
    public async Task Spectrum_HighlyRepeatedKmer_DoesNotOverflow()
    {
        var kernel = new StringKernel<double>(StringKernel<double>.KernelType.Spectrum, substringLength: 1);
        string s1 = new string('A', RepeatCount);
        string s2 = new string('A', RepeatCount + 10);

        double similarity = kernel.Calculate(s1, s2);

        Assert.False(double.IsNaN(similarity), "Spectrum kernel returned NaN (int32 overflow in count products).");
        Assert.Equal(1.0, similarity, 12);
    }

    [Fact(Timeout = 120000)]
    public async Task BagOfWords_HighlyRepeatedWord_DoesNotOverflow()
    {
        var kernel = new StringKernel<double>(StringKernel<double>.KernelType.BagOfWords);
        string s1 = string.Join(" ", Enumerable.Repeat("the", RepeatCount));
        string s2 = string.Join(" ", Enumerable.Repeat("the", RepeatCount + 10));

        double similarity = kernel.Calculate(s1, s2);

        Assert.False(double.IsNaN(similarity), "Bag-of-words kernel returned NaN (int32 overflow in count products).");
        Assert.Equal(1.0, similarity, 12);
    }
}
