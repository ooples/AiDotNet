using System;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Generators;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// The scaffold generator drops the <c>outputSize</c> argument from fixtures of models sized from their options
/// (#2147). It used to drop only the first number of the value, so <c>outputSize: 256 * 256 * 3</c> left
/// <c> * 256 * 3</c> attached to the argument before it: OpenSora's fixture declared <c>inputDepth: 3 * 256 * 3</c>,
/// 2304 input channels, and every one of its training tests threw on the nightly lane (#2365).
/// </summary>
public sealed class OmitStatedOutputSizeTests
{
    private static string Omit(string expression)
    {
        var omit = typeof(TestScaffoldGenerator).GetMethod("OmitStatedOutputSize", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("TestScaffoldGenerator.OmitStatedOutputSize was not found.");
        return omit.Invoke(null, new object[] { expression }) as string
            ?? throw new InvalidOperationException("OmitStatedOutputSize did not return a string.");
    }

    [Theory(Timeout = 60000)]
    [InlineData("new A(inputDepth: 3, outputSize: 256 * 256 * 3), options)", "new A(inputDepth: 3), options)")]
    [InlineData("new A(inputSize: 16, outputSize: 4)", "new A(inputSize: 16)")]
    [InlineData("new A(outputSize: 4, inputSize: 16)", "new A(inputSize: 16)")]
    [InlineData("new A(inputSize: 16, outputSize: 4, layers: null)", "new A(inputSize: 16, layers: null)")]
    [InlineData("new A(x: 1, outputSize: 8 / 2 )", "new A(x: 1 )")]
    [InlineData("new A(inputSize: 16)", "new A(inputSize: 16)")]
    public async Task RemovesTheWholeArgument(string expression, string expected)
    {
        await Task.Yield();
        Assert.Equal(expected, Omit(expression));
    }

    /// <summary>
    /// A value the pattern cannot delimit safely (a call or a parenthesized expression) is left whole rather than cut,
    /// so the worst case is a stated size the model refuses loudly, never a silently corrupted neighbour.
    /// </summary>
    [Theory(Timeout = 60000)]
    [InlineData("new A(inputDepth: 3, outputSize: Size(4, 4))")]
    [InlineData("new A(inputDepth: 3, outputSize: (256 * 3))")]
    public async Task LeavesAValueItCannotDelimitUntouched(string expression)
    {
        await Task.Yield();
        Assert.Equal(expression, Omit(expression));
    }
}
