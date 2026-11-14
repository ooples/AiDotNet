using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// ReverseUpdate must undo exactly the step UpdateParameters took. The Nesterov step subtracts
/// lr * (g + momentum * v); adding back only the velocity v restored the wrong point.
/// </summary>
public class NesterovReverseUpdateTests
{
    private static NesterovAcceleratedGradientOptimizer<double, Matrix<double>, Vector<double>> Create(double learningRate, double momentum)
        => new(null, new NesterovAcceleratedGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = learningRate,
            InitialMomentum = momentum
        });

    [Fact]
    public void ReverseUpdate_ZeroMomentum_RestoresTheParameter()
    {
        // lr 0.1, momentum 0, parameter 1, gradient 2: the step is 0.2, so 1 -> 0.8 -> 1 (adding the
        // velocity 2 instead gave 2.8).
        var optimizer = Create(0.1, 0.0);
        var start = new Vector<double>(new[] { 1.0 });
        var gradient = new Vector<double>(new[] { 2.0 });

        var updated = optimizer.UpdateParameters(start, gradient);
        var restored = optimizer.ReverseUpdate(updated, gradient);

        Assert.NotEqual(start[0], updated[0]);
        Assert.Equal(1.0, restored[0], 12);
    }

    [Fact]
    public void ReverseUpdate_WithMomentum_UndoesTheLatestStepExactly()
    {
        var optimizer = Create(0.05, 0.9);
        var start = new Vector<double>(new[] { 1.0, -2.0, 0.5 });

        var first = optimizer.UpdateParameters(start, new Vector<double>(new[] { 0.3, -0.4, 1.0 }));
        var gradient = new Vector<double>(new[] { -0.2, 0.7, 0.1 });
        var second = optimizer.UpdateParameters(first, gradient);
        var restored = optimizer.ReverseUpdate(second, gradient);

        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(first[i], restored[i], 12);
        }
    }

    [Fact]
    public void ReverseUpdate_BeforeAnyUpdate_Throws()
    {
        var optimizer = Create(0.1, 0.9);
        var parameters = new Vector<double>(new[] { 1.0 });

        Assert.Throws<InvalidOperationException>(() => optimizer.ReverseUpdate(parameters, parameters));
    }
}
