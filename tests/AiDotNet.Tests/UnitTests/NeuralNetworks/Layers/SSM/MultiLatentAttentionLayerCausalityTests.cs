using System;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Pins the attention semantics of <see cref="MultiLatentAttentionLayer{T}"/> across engine attention kernels: strict
/// causality (position i reads only positions j &lt;= i), agreement between the fused host float32 kernel and the
/// decomposed chain every other element type runs, and a gradient that reaches the input.
/// </summary>
public class MultiLatentAttentionLayerCausalityTests
{
    private const int Batch = 2;
    private const int Seq = 6;
    private const int Dim = 8;
    private const int Heads = 2;
    private const int Latent = 4;

    private static Tensor<double> MakeInput(int seq, int seed)
    {
        var rng = new Random(seed);
        var x = new Tensor<double>(new[] { Batch, seq, Dim });
        for (int i = 0; i < x.Length; i++) x.SetFlat(i, rng.NextDouble() * 2.0 - 1.0);
        return x;
    }

    private static Tensor<float> ToFloat(Tensor<double> x)
    {
        var f = new Tensor<float>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++) f.SetFlat(i, (float)x.GetFlat(i));
        return f;
    }

    /// <summary>A double layer and a float layer holding the same (float-representable) parameters.</summary>
    private static (MultiLatentAttentionLayer<double> D, MultiLatentAttentionLayer<float> F) MakePair()
    {
        var d = new MultiLatentAttentionLayer<double>(Seq, Dim, Heads, Latent);
        var f = new MultiLatentAttentionLayer<float>(Seq, Dim, Heads, Latent);
        var rng = new Random(4242);
        var pf = f.GetParameters();
        var valuesF = new float[pf.Length];
        var valuesD = new double[pf.Length];
        for (int i = 0; i < valuesF.Length; i++)
        {
            valuesF[i] = (float)(rng.NextDouble() * 1.2 - 0.6);
            valuesD[i] = valuesF[i];
        }
        f.SetParameters(new Vector<float>(valuesF));
        d.SetParameters(new Vector<double>(valuesD));
        return (d, f);
    }

    [Fact]
    public void Forward_IsStrictlyCausal_InBothAttentionKernels()
    {
        var (d, f) = MakePair();
        var x = MakeInput(Seq, 11);
        var baseD = d.Forward(x);
        var baseF = f.Forward(ToFloat(x));

        const int changed = 3;
        var x2 = MakeInput(Seq, 11);
        for (int b = 0; b < Batch; b++)
            for (int c = 0; c < Dim; c++)
                x2.SetFlat((b * Seq + changed) * Dim + c, x2.GetFlat((b * Seq + changed) * Dim + c) + 0.75);

        var movedD = d.Forward(x2);
        var movedF = f.Forward(ToFloat(x2));

        double laterDeltaD = 0.0, laterDeltaF = 0.0;
        for (int b = 0; b < Batch; b++)
        {
            for (int s = 0; s < Seq; s++)
            {
                for (int c = 0; c < Dim; c++)
                {
                    int i = (b * Seq + s) * Dim + c;
                    double dd = Math.Abs(baseD.GetFlat(i) - movedD.GetFlat(i));
                    double df = Math.Abs(baseF.GetFlat(i) - movedF.GetFlat(i));
                    if (s < changed)
                    {
                        Assert.True(dd <= 1e-12, $"double: position {s} read future position {changed} (delta {dd})");
                        Assert.True(df <= 1e-6, $"float: position {s} read future position {changed} (delta {df})");
                    }
                    else
                    {
                        laterDeltaD = Math.Max(laterDeltaD, dd);
                        laterDeltaF = Math.Max(laterDeltaF, df);
                    }
                }
            }
        }

        // The perturbation has to matter somewhere, or the assertions above prove nothing.
        Assert.True(laterDeltaD > 1e-4, $"double: perturbing position {changed} changed nothing at or after it");
        Assert.True(laterDeltaF > 1e-4, $"float: perturbing position {changed} changed nothing at or after it");
    }

    [Fact]
    public void Forward_PrefixOfSequence_MatchesFullSequencePrefix()
    {
        var (d, f) = MakePair();
        var full = MakeInput(Seq, 23);
        const int prefix = 4;
        var head = new Tensor<double>(new[] { Batch, prefix, Dim });
        for (int b = 0; b < Batch; b++)
            for (int s = 0; s < prefix; s++)
                for (int c = 0; c < Dim; c++)
                    head.SetFlat((b * prefix + s) * Dim + c, full.GetFlat((b * Seq + s) * Dim + c));

        var fullD = d.Forward(full);
        var headD = d.Forward(head);
        var fullF = f.Forward(ToFloat(full));
        var headF = f.Forward(ToFloat(head));

        for (int b = 0; b < Batch; b++)
        {
            for (int s = 0; s < prefix; s++)
            {
                for (int c = 0; c < Dim; c++)
                {
                    int iFull = (b * Seq + s) * Dim + c;
                    int iHead = (b * prefix + s) * Dim + c;
                    Assert.True(Math.Abs(fullD.GetFlat(iFull) - headD.GetFlat(iHead)) <= 1e-12,
                        $"double: prefix output differs at b={b}, s={s}, c={c}");
                    Assert.True(Math.Abs(fullF.GetFlat(iFull) - headF.GetFlat(iHead)) <= 1e-5,
                        $"float: prefix output differs at b={b}, s={s}, c={c}");
                }
            }
        }
    }

    [Fact]
    public void Forward_FusedFloatKernel_AgreesWithDecomposedDoubleChain()
    {
        var (d, f) = MakePair();
        var x = MakeInput(Seq, 37);
        var yd = d.Forward(x);
        var yf = f.Forward(ToFloat(x));
        Assert.Equal(yd.Shape.ToArray(), yf.Shape.ToArray());
        for (int i = 0; i < yd.Length; i++)
        {
            Assert.True(Math.Abs(yd.GetFlat(i) - yf.GetFlat(i)) <= 1e-4 * Math.Max(1.0, Math.Abs(yd.GetFlat(i))),
                $"float and double attention disagree at {i}: double={yd.GetFlat(i)}, float={yf.GetFlat(i)}");
        }
    }

    [Fact]
    public void Backward_InputGradient_MatchesFiniteDifferences_AndFloatMatchesDouble()
    {
        var (d, f) = MakePair();
        var x = MakeInput(Seq, 51);
        var xf = ToFloat(x);
        var engine = AiDotNetEngine.Current;

        Tensor<double> gradD;
        using (var tape = new GradientTape<double>())
        {
            var y = d.Forward(x);
            var loss = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(y, y), null), 0.5);
            gradD = tape.ComputeGradients(loss, sources: new[] { x })[x];
        }

        Tensor<float> gradF;
        using (var tape = new GradientTape<float>())
        {
            var y = f.Forward(xf);
            var loss = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(y, y), null), 0.5f);
            gradF = tape.ComputeGradients(loss, sources: new[] { xf })[xf];
        }

        double Loss()
        {
            var y = d.Forward(x);
            double sum = 0.0;
            for (int i = 0; i < y.Length; i++) sum += 0.5 * y.GetFlat(i) * y.GetFlat(i);
            return sum;
        }

        const double eps = 1e-6;
        double gradScale = 0.0;
        for (int i = 0; i < x.Length; i++)
        {
            double orig = x.GetFlat(i);
            x.SetFlat(i, orig + eps);
            double lp = Loss();
            x.SetFlat(i, orig - eps);
            double lm = Loss();
            x.SetFlat(i, orig);
            double numeric = (lp - lm) / (2.0 * eps);
            gradScale = Math.Max(gradScale, Math.Abs(numeric));
            Assert.True(Math.Abs(gradD.GetFlat(i) - numeric) <= 1e-6 * Math.Max(1.0, Math.Abs(numeric)),
                $"dL/dx mismatch at {i}: analytic={gradD.GetFlat(i)}, numeric={numeric}");
            Assert.True(Math.Abs(gradF.GetFlat(i) - numeric) <= 2e-3 * Math.Max(1.0, Math.Abs(numeric)),
                $"float dL/dx mismatch at {i}: analytic={gradF.GetFlat(i)}, numeric={numeric}");
        }

        Assert.True(gradScale > 1e-6, "the loss does not depend on the input: the gradient check is vacuous");
    }
}
