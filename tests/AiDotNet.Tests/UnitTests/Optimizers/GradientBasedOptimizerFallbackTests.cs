using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Moq;
using Xunit;

namespace AiDotNetTests.UnitTests.Optimizers;

/// <summary>
/// Regression tests for #1837 — <see cref="GradientBasedOptimizerBase{T, TInput, TOutput}"/>'s
/// 4-argument <c>CalculateGradient(solution, xTrain, yTrain, batchIndices)</c> virtual fallback
/// historically shipped an algebraically wrong gradient formula (<c>gradient[j] += loss × input[j]</c>,
/// where <c>loss</c> is a scalar MSE — not even the linear-regression formula it appeared to be).
/// It was unreachable in the shipped model set because every model that extends
/// <c>NeuralNetworkBase&lt;T&gt;</c> also implements <c>IGradientComputable&lt;T, TInput, TOutput&gt;</c>,
/// but the code was a latent landmine — any subclass that overrode the virtual "to extend the base,"
/// or any new model class that skipped <c>IGradientComputable</c>, silently trained with algebraic
/// garbage.
///
/// The fix (#1837) deletes the wrong-formula body and routes callers to
/// <c>InterfaceGuard.GradientComputable(solution)</c>, which throws a clear
/// <see cref="InvalidOperationException"/> when the model doesn't implement
/// <see cref="IGradientComputable{T, TInput, TOutput}"/>. This matches PyTorch, Keras, and JAX's
/// fail-fast convention for the same case.
///
/// These tests pin that contract so a future refactor can't accidentally reintroduce the wrong
/// formula.
/// </summary>
public class GradientBasedOptimizerFallbackTests
{
    // Minimal concrete subclass of AdamOptimizer that exposes the protected 4-argument
    // CalculateGradient. Tests can then invoke the virtual directly with a mock model that
    // deliberately lacks IGradientComputable.
    private sealed class TestOptimizer<T, TInput, TOutput> : AdamOptimizer<T, TInput, TOutput>
    {
        public TestOptimizer() : base(model: null, options: new AdamOptimizerOptions<T, TInput, TOutput>()) { }

        public Vector<T> InvokeCalculateGradient(
            IFullModel<T, TInput, TOutput> solution, TInput x, TOutput y, int[] batchIndices)
            => CalculateGradient(solution, x, y, batchIndices);
    }

    [Fact]
    public void CalculateGradient_ModelWithoutIGradientComputable_ThrowsWithClearMessage()
    {
        var optimizer = new TestOptimizer<double, Matrix<double>, Vector<double>>();

        // Deliberately mock a model that does NOT implement IGradientComputable.
        // The pre-fix fallback would silently compute a bogus gradient here.
        var mockModel = new Mock<IFullModel<double, Matrix<double>, Vector<double>>>();
        var x = new Matrix<double>(2, 3);
        var y = new Vector<double>(2);
        var batchIndices = new[] { 0, 1 };

        var ex = Assert.Throws<InvalidOperationException>(
            () => optimizer.InvokeCalculateGradient(mockModel.Object, x, y, batchIndices));

        // Message must name the missing interface so users can trace the fix — either implement
        // IGradientComputable on a custom model, or inherit from NeuralNetworkBase<T> which
        // implements it.
        Assert.Contains("IGradientComputable", ex.Message);
        Assert.Contains("gradient", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}
