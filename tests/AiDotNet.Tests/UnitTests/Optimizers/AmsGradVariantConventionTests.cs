using System;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// The AMSGrad switch on Adam and AdamW must run PyTorch's AMSGrad - the running max of the RAW second moment,
/// bias-corrected at use - on every eager path, because their fused configs dispatch to the Tensors AMSGrad
/// kernel that does exactly that. The eager paths used to keep max(vMax, v/(1-b2^t)) instead, so the same model
/// trained differently fused and eager; AdamW's matrix path ignored UseAMSGrad altogether; and fused AdamW + AMSGrad
/// ran L2 instead of decoupled decay.
/// </summary>
public class AmsGradVariantConventionTests
{
    private const double Lr = 0.01, B1 = 0.9, B2 = 0.999, Eps = 1e-8, Wd = 0.1, Tol = 1e-12;

    // A large first gradient then small ones: v decays below the running max, the regime where the two
    // max conventions disagree.
    private static readonly double[][] Grads =
    {
        new[] { 0.8, -1.2, 0.3 },
        new[] { 0.01, 0.02, -0.01 },
        new[] { -0.02, 0.01, 0.03 },
        new[] { 0.015, -0.005, 0.02 },
    };
    private static readonly double[] Init = { 0.5, -1.0, 2.0 };

    /// <summary>PyTorch Adam/AdamW(amsgrad=True); <paramref name="decoupledWd"/> &gt; 0 selects AdamW decay.</summary>
    private static double[][] Reference(double decoupledWd)
    {
        var p = (double[])Init.Clone();
        double[] m = new double[3], v = new double[3], vMax = new double[3];
        var trajectory = new double[Grads.Length][];
        for (int t = 1; t <= Grads.Length; t++)
        {
            for (int i = 0; i < 3; i++)
            {
                double g = Grads[t - 1][i];
                m[i] = B1 * m[i] + (1 - B1) * g;
                v[i] = B2 * v[i] + (1 - B2) * g * g;
                vMax[i] = Math.Max(vMax[i], v[i]);
                double adam = (m[i] / (1 - Math.Pow(B1, t))) / (Math.Sqrt(vMax[i] / (1 - Math.Pow(B2, t))) + Eps);
                p[i] = p[i] - Lr * adam - Lr * decoupledWd * p[i];
            }
            trajectory[t - 1] = (double[])p.Clone();
        }
        return trajectory;
    }

    [Fact]
    public void Adam_UseAMSGrad_vector_path_matches_PyTorch()
    {
        var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null!,
            new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            { InitialLearningRate = Lr, Beta1 = B1, Beta2 = B2, Epsilon = Eps, UseAMSGrad = true });
        var expected = Reference(0);
        var p = new Vector<double>(Init);
        for (int t = 0; t < Grads.Length; t++)
        {
            p = optimizer.UpdateParameters(p, new Vector<double>(Grads[t]));
            for (int i = 0; i < 3; i++) Assert.Equal(expected[t][i], p[i], Tol);
        }
    }

    [Fact]
    public void AdamW_UseAMSGrad_vector_path_matches_PyTorch()
    {
        var optimizer = new AdamWOptimizer<double, Matrix<double>, Vector<double>>(null!,
            new AdamWOptimizerOptions<double, Matrix<double>, Vector<double>>
            { InitialLearningRate = Lr, Beta1 = B1, Beta2 = B2, Epsilon = Eps, WeightDecay = Wd, UseAMSGrad = true });
        var expected = Reference(Wd);
        var p = new Vector<double>(Init);
        for (int t = 0; t < Grads.Length; t++)
        {
            p = optimizer.UpdateParameters(p, new Vector<double>(Grads[t]));
            for (int i = 0; i < 3; i++) Assert.Equal(expected[t][i], p[i], Tol);
        }
    }

    [Fact]
    public void AdamW_UseAMSGrad_matrix_path_applies_AMSGrad()
    {
        var optimizer = new AdamWOptimizer<double, Matrix<double>, Vector<double>>(null!,
            new AdamWOptimizerOptions<double, Matrix<double>, Vector<double>>
            { InitialLearningRate = Lr, Beta1 = B1, Beta2 = B2, Epsilon = Eps, WeightDecay = Wd, UseAMSGrad = true });
        var expected = Reference(Wd);
        var p = new Matrix<double>(1, 3);
        for (int i = 0; i < 3; i++) p[0, i] = Init[i];
        for (int t = 0; t < Grads.Length; t++)
        {
            var g = new Matrix<double>(1, 3);
            for (int i = 0; i < 3; i++) g[0, i] = Grads[t][i];
            p = optimizer.UpdateParameters(p, g);
            for (int i = 0; i < 3; i++) Assert.Equal(expected[t][i], p[0, i], Tol);
        }
    }

    [Fact]
    public void Fused_AdamW_UseAMSGrad_requests_decoupled_decay_and_plain_AdamW_does_not()
    {
        var amsgrad = new AdamWOptimizer<float, Matrix<float>, Vector<float>>(null!,
            new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>> { WeightDecay = 0.01, UseAMSGrad = true });
        Assert.True(((IFusedOptimizerSpec)amsgrad).TryGetFusedOptimizerConfig(out var cfg));
        Assert.Equal(AiDotNet.Tensors.Engines.Compilation.OptimizerType.AMSGrad, cfg.Type);
        Assert.NotNull(cfg.Extras);
        Assert.True(cfg.Extras!.DecoupledWeightDecay);

        var plain = new AdamWOptimizer<float, Matrix<float>, Vector<float>>(null!,
            new AdamWOptimizerOptions<float, Matrix<float>, Vector<float>> { WeightDecay = 0.01 });
        Assert.True(((IFusedOptimizerSpec)plain).TryGetFusedOptimizerConfig(out var plainCfg));
        Assert.Equal(AiDotNet.Tensors.Engines.Compilation.OptimizerType.AdamW, plainCfg.Type);
        Assert.Null(plainCfg.Extras);
    }
}
