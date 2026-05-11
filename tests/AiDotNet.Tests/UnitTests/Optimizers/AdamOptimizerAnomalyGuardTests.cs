using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Focused coverage for the PyTorch GradScaler-style anomaly guard added to
/// <see cref="AdamOptimizer{T,TInput,TOutput}.Step"/>. The guard scans every
/// gradient tensor for NaN/Inf and aborts the entire step (parameters, m/v
/// moment accumulators, AND step index unchanged) so a single poisoned
/// gradient cannot permanently corrupt the optimizer state.
/// <para>
/// Verifies the two pieces of the guard in isolation:
/// <list type="bullet">
/// <item><c>AnyGradientIsAnomalous</c> correctly detects NaN, Inf, and clean
///   gradient sets across the supported numeric types.</item>
/// <item><c>ShouldRunAnomalyGuard</c> honors every value of
///   <see cref="AdamAnomalyGuardMode"/>.</item>
/// </list>
/// End-to-end semantics (a poisoned step is a true no-op including m/v and
/// step-index) are covered by the existing HopeNetwork model-family tests —
/// before this guard, <c>ForwardPass_ShouldBeFinite_AfterTraining</c> /
/// <c>DifferentInputs_AfterTraining</c> / <c>Clone_AfterTraining</c>
/// triple-failed at step ~10 when a NaN gradient poisoned <c>m</c> and
/// every subsequent step produced NaN weights; with the guard they now pass.
/// </para>
/// </summary>
public class AdamOptimizerAnomalyGuardTests
{
    [Fact]
    public void AnyGradientIsAnomalous_FiniteGradients_ReturnsFalse()
    {
        var opt = BuildOptimizer(AdamAnomalyGuardMode.Always);
        var context = BuildTapeStepContextWithGradients(new[] { 0.1, -0.2, 0.3, 1e-6, -1e-6 });
        Assert.False(InvokeAnyGradientIsAnomalous(opt, context));
    }

    [Fact]
    public void AnyGradientIsAnomalous_OneNaNAmongFinites_ReturnsTrue()
    {
        var opt = BuildOptimizer(AdamAnomalyGuardMode.Always);
        var context = BuildTapeStepContextWithGradients(new[] { 0.1, -0.2, double.NaN, 0.3, 1e-6 });
        Assert.True(InvokeAnyGradientIsAnomalous(opt, context));
    }

    [Fact]
    public void AnyGradientIsAnomalous_PositiveInfinity_ReturnsTrue()
    {
        var opt = BuildOptimizer(AdamAnomalyGuardMode.Always);
        var context = BuildTapeStepContextWithGradients(new[] { 0.1, double.PositiveInfinity, 0.3 });
        Assert.True(InvokeAnyGradientIsAnomalous(opt, context));
    }

    [Fact]
    public void AnyGradientIsAnomalous_NegativeInfinity_ReturnsTrue()
    {
        var opt = BuildOptimizer(AdamAnomalyGuardMode.Always);
        var context = BuildTapeStepContextWithGradients(new[] { double.NegativeInfinity, 0.1, 0.3 });
        Assert.True(InvokeAnyGradientIsAnomalous(opt, context));
    }

    [Theory]
    [InlineData(AdamAnomalyGuardMode.Auto, true)]
    [InlineData(AdamAnomalyGuardMode.Always, true)]
    [InlineData(AdamAnomalyGuardMode.Never, false)]
    public void ShouldRunAnomalyGuard_HonorsMode(AdamAnomalyGuardMode mode, bool expected)
    {
        var opt = BuildOptimizer(mode);
        Assert.Equal(expected, InvokeShouldRunAnomalyGuard(opt));
    }

    private static AdamOptimizer<double, Tensor<double>, Tensor<double>> BuildOptimizer(AdamAnomalyGuardMode mode)
    {
        return new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            model: null,
            options: new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.01,
                EnableGradientClipping = false,
                AnomalyGuardMode = mode,
            });
    }

    /// <summary>
    /// Builds a minimal <c>TapeStepContext&lt;double&gt;</c> whose
    /// <c>Gradients</c> dictionary has a single (param → gradient) pair. The
    /// gradient tensor's values come from <paramref name="gradValues"/> — pass
    /// any combination of finite, NaN, or Inf to drive the AnyGradientIsAnomalous
    /// scan. The other context fields (input, expected, loss, forward/loss
    /// closures, paramBuffer) are reflection-stubbed because the anomaly-guard
    /// path returns before touching them.
    /// </summary>
    private static object BuildTapeStepContextWithGradients(double[] gradValues)
    {
        var param = new Tensor<double>(new[] { gradValues.Length });
        for (int i = 0; i < gradValues.Length; i++) param[i] = 0.0;
        var grad = new Tensor<double>(new[] { gradValues.Length });
        for (int i = 0; i < gradValues.Length; i++) grad[i] = gradValues[i];

        var ctxType = typeof(AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<double>);
        // Deterministic ctor selection: pick the public ctor with the most
        // parameters. Indexing [0] on GetConstructors() depended on ordering
        // (not guaranteed by reflection) and would silently bind to the wrong
        // overload if a new ctor were added later. Selecting the widest
        // signature matches the construction site in NeuralNetworkBase
        // which passes every available context field.
        var ctor = ctxType.GetConstructors()
            .OrderByDescending(c => c.GetParameters().Length)
            .First();
        var parameters = ctor.GetParameters();
        var args = new object?[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            var p = parameters[i];
            // Match by name where possible — Gradients dict is the only field
            // the anomaly guard touches; everything else can be a zero/null stub.
            args[i] = p.Name switch
            {
                "trainableParameters" or "parameters" => new System.Collections.Generic.List<Tensor<double>> { param },
                "gradients" => new System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> { [param] = grad },
                "lossValue" or "loss" => 0.0,
                _ => p.ParameterType.IsValueType ? Activator.CreateInstance(p.ParameterType) : null,
            };
        }
        return ctor.Invoke(args)!;
    }

    private static bool InvokeAnyGradientIsAnomalous(
        AdamOptimizer<double, Tensor<double>, Tensor<double>> opt, object context)
    {
        var method = typeof(AdamOptimizer<double, Tensor<double>, Tensor<double>>)
            .GetMethod("AnyGradientIsAnomalous", BindingFlags.NonPublic | BindingFlags.Instance)!;
        return (bool)method.Invoke(opt, new[] { context })!;
    }

    private static bool InvokeShouldRunAnomalyGuard(
        AdamOptimizer<double, Tensor<double>, Tensor<double>> opt)
    {
        var method = typeof(AdamOptimizer<double, Tensor<double>, Tensor<double>>)
            .GetMethod("ShouldRunAnomalyGuard", BindingFlags.NonPublic | BindingFlags.Instance)!;
        return (bool)method.Invoke(opt, null)!;
    }
}
