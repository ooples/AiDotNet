using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// Regression tests for #1833 — <see cref="GaussianSplatting{T}"/> used to silently ignore
/// every <see cref="OptimizationAlgorithmOptions{T, TInput, TOutput}"/> knob the caller
/// configured via <c>PredictionModelBuilder.ConfigureOptimizer</c>. The facade's Adam step
/// walked the standard <c>NeuralNetworkBase.Layers</c> path — GS has zero standard Layers
/// (parameters live in <c>_gaussians</c>) — so the step ran but updated nothing. Meanwhile
/// GS's own internal training loop used whatever LR schedule the ctor's
/// <see cref="GaussianSplattingOptions"/> had baked in, regardless of what the caller
/// asked for.
///
/// The fix threads the configured optimizer options into GS via
/// <see cref="IHyperparameterAware{T, TInput, TOutput}"/> — the builder calls
/// <c>ApplyOptimizerHyperparameters</c> once, immediately after <c>SetModel</c>, and GS
/// derives its full per-attribute LR schedule from the caller's base LR anchored on the
/// 3DGS paper's canonical ratios (position:color:opacity:scale:rotation ≈
/// 1 : 15.6 : 312 : 31 : 6).
///
/// These tests pin the wiring contract at the model layer — they exercise the interface
/// method directly rather than the full facade round-trip so the assertion targets the
/// piece of the fix that #1833 owns.
/// </summary>
public class GaussianSplattingHyperparameterAwareTests
{
    private static GaussianSplatting<double> BuildGS()
    {
        var options = new GaussianSplattingOptions
        {
            EnableDensification = false,
            EnableSpatialIndex = false,
            MaxGaussians = 8,
            ShDegree = 0,
        };
        // Minimal 4-point seed so the ctor accepts the model.
        var points = new Matrix<double>(4, 3);
        var colors = new Matrix<double>(4, 3);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                points[i, j] = 0.1 * (i + j);
                colors[i, j] = 0.5;
            }
        }
        return new GaussianSplatting<double>(options, points, colors);
    }

    [Fact]
    public void ImplementsIHyperparameterAware()
    {
        // If this fails, the facade routing in AiModelBuilder.BuildPipeline.cs stops
        // finding the model — the interface IS the contract.
        var gs = BuildGS();
        Assert.IsAssignableFrom<IHyperparameterAware<double, Tensor<double>, Tensor<double>>>(gs);
    }

    [Fact]
    public void ApplyOptimizerHyperparameters_ExplicitNonDefaultLR_DerivesPaperSchedule()
    {
        var gs = BuildGS();
        // Snapshot the pre-call schedule so we can assert it moved.
        double preColor = gs.ColorLearningRate;
        double preOpacity = gs.OpacityLearningRate;

        // Base LR = 1.6e-4 is the paper canonical. Any value != the base-class default
        // (0.01) triggers full re-derivation.
        var options = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 1.6e-4,
        };

        ((IHyperparameterAware<double, Tensor<double>, Tensor<double>>)gs)
            .ApplyOptimizerHyperparameters(options);

        // Position = caller LR (identity).
        Assert.Equal(1.6e-4, gs.PositionLearningRate, precision: 12);
        // Paper ratios anchor at base=1.6e-4 → these are the direct paper values.
        Assert.Equal(2.5e-3, gs.ColorLearningRate,    precision: 8);
        Assert.Equal(5.0e-2, gs.OpacityLearningRate,  precision: 8);
        Assert.Equal(5.0e-3, gs.ScaleLearningRate,    precision: 8);
        Assert.Equal(1.0e-3, gs.RotationLearningRate, precision: 8);

        // Sanity: something actually moved (avoids the test passing on a no-op path).
        Assert.NotEqual(preColor, gs.ColorLearningRate);
        Assert.NotEqual(preOpacity, gs.OpacityLearningRate);
    }

    [Fact]
    public void ApplyOptimizerHyperparameters_ScalesLinearlyFromBase()
    {
        var gs = BuildGS();

        // 10× the paper base → every attribute LR is 10× the paper's per-attribute default.
        var options = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 1.6e-3,
        };
        ((IHyperparameterAware<double, Tensor<double>, Tensor<double>>)gs)
            .ApplyOptimizerHyperparameters(options);

        Assert.Equal(1.6e-3, gs.PositionLearningRate, precision: 12);
        Assert.Equal(2.5e-2, gs.ColorLearningRate,    precision: 8);
        Assert.Equal(5.0e-1, gs.OpacityLearningRate,  precision: 8);
        Assert.Equal(5.0e-2, gs.ScaleLearningRate,    precision: 8);
        Assert.Equal(1.0e-2, gs.RotationLearningRate, precision: 8);
    }

    [Fact]
    public void ApplyOptimizerHyperparameters_BaseClassDefaultLR_LeavesGSDefaultsUntouched()
    {
        var gs = BuildGS();

        // Snapshot GS's constructor-set (paper-quality) defaults.
        double origPos = gs.PositionLearningRate;
        double origColor = gs.ColorLearningRate;
        double origOpacity = gs.OpacityLearningRate;
        double origScale = gs.ScaleLearningRate;
        double origRotation = gs.RotationLearningRate;

        // Caller didn't touch InitialLearningRate → OptimizationAlgorithmOptions default = 0.01.
        // For GS, 0.01 is 62× the paper's position LR — treating that as "user asked for it"
        // would blow training. GS explicitly no-ops on the base-class default.
        var options = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        Assert.Equal(0.01, options.InitialLearningRate);

        ((IHyperparameterAware<double, Tensor<double>, Tensor<double>>)gs)
            .ApplyOptimizerHyperparameters(options);

        Assert.Equal(origPos, gs.PositionLearningRate);
        Assert.Equal(origColor, gs.ColorLearningRate);
        Assert.Equal(origOpacity, gs.OpacityLearningRate);
        Assert.Equal(origScale, gs.ScaleLearningRate);
        Assert.Equal(origRotation, gs.RotationLearningRate);
    }

    [Fact]
    public void ApplyOptimizerHyperparameters_NullOptions_ThrowsArgumentNullException()
    {
        var gs = BuildGS();
        Assert.Throws<ArgumentNullException>(() =>
            ((IHyperparameterAware<double, Tensor<double>, Tensor<double>>)gs)
                .ApplyOptimizerHyperparameters(null!));
    }
}
