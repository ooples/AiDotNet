using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// <see cref="GatedDeltaNetLayer{T}"/> against an independent float64 implementation of Gated DeltaNet (Yang, Kautz and
/// Hatamizadeh, ICLR 2025): per-head L2-normalized queries and keys (queries then scaled by 1/sqrt(headDim)), the
/// transition S_t = a_t S_{t-1} (I - b_t k_t k_tᵀ) + b_t v_t k_tᵀ formed as a matrix, o_t = S_t q_t, and the per-head
/// output RMSNorm before the swish gate.
/// </summary>
public class GatedDeltaNetPaperRuleTests
{
    private const int Batch = 2, SeqLen = 7, ModelDim = 12, Heads = 3, Kernel = 4, Hd = ModelDim / Heads;

    [Fact]
    public void Forward_matches_the_paper_recurrence_in_float64()
    {
        var layer = new GatedDeltaNetLayer<double>(SeqLen, ModelDim, Heads, Kernel);
        var rng = new Random(7);
        var p = new double[layer.ParameterCount];
        for (int i = 0; i < p.Length; i++) p[i] = rng.NextDouble() * 1.2 - 0.6;
        layer.SetParameters(new Vector<double>(p));
        var x = new double[Batch * SeqLen * ModelDim];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextDouble() * 4 - 2;

        var got = layer.Forward(new Tensor<double>(new[] { Batch, SeqLen, ModelDim }, new Vector<double>(x))).ToArray();
        var expected = Reference(p, x);

        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(expected[i] - got[i]) <= 1e-10 * (1 + Math.Abs(expected[i])),
                $"output[{i}]: reference {expected[i]:R}, layer {got[i]:R}");
    }

    [Fact]
    public void Large_inputs_keep_the_state_bounded()
    {
        // The unnormalized rule's transition has the eigenvalue a - b||k||^2 along k, so large keys made it expand and
        // the state overflow; with unit keys and b in (0, 1) the transition is a contraction and the output stays finite.
        var layer = new GatedDeltaNetLayer<float>(512, ModelDim, Heads, Kernel);
        var rng = new Random(3);
        var p = new float[layer.ParameterCount];
        for (int i = 0; i < p.Length; i++) p[i] = (float)(rng.NextDouble() * 4 - 2);
        layer.SetParameters(new Vector<float>(p));
        var x = new float[512 * ModelDim];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2000 - 1000);
        var y = layer.Forward(new Tensor<float>(new[] { 1, 512, ModelDim }, new Vector<float>(x))).ToArray();
        Assert.All(y, v => Assert.True(float.IsFinite(v), $"non-finite output {v}"));
    }

    private static double[] Reference(double[] p, double[] x)
    {
        int o = 0;
        double[] Take(int n) { var a = new double[n]; Array.Copy(p, o, a, 0, n); o += n; return a; }
        // Declaration order of the layer's trainable parameters.
        var convW = Take(ModelDim * Kernel); var convB = Take(ModelDim);
        var wq = Take(ModelDim * ModelDim); var wk = Take(ModelDim * ModelDim); var wv = Take(ModelDim * ModelDim);
        var wb = Take(ModelDim * Heads); var bb = Take(Heads);
        var wa = Take(ModelDim * Heads); var ba = Take(Heads);
        var gn = Take(Hd);
        var wg = Take(ModelDim * ModelDim); var bg = Take(ModelDim);
        var wo = Take(ModelDim * ModelDim); var bo = Take(ModelDim);
        Assert.Equal(p.Length, o);

        static double Sig(double z) => 1 / (1 + Math.Exp(-z));
        double[] MatVec(double[] row, double[] w, int cols, double[]? bias)
        {
            var r = new double[cols];
            for (int c = 0; c < cols; c++)
            {
                double s = bias?[c] ?? 0;
                for (int i = 0; i < ModelDim; i++) s += row[i] * w[i * cols + c];
                r[c] = s;
            }
            return r;
        }

        var y = new double[x.Length];
        for (int b = 0; b < Batch; b++)
        {
            var S = new double[Heads, Hd, Hd];
            for (int t = 0; t < SeqLen; t++)
            {
                var u = new double[ModelDim];
                for (int c = 0; c < ModelDim; c++)
                {
                    double s = convB[c];
                    for (int m = 0; m < Kernel && m <= t; m++) s += convW[c * Kernel + m] * x[(b * SeqLen + t - m) * ModelDim + c];
                    u[c] = s * Sig(s);
                }
                var q = MatVec(u, wq, ModelDim, null); var k = MatVec(u, wk, ModelDim, null); var v = MatVec(u, wv, ModelDim, null);
                var beta = MatVec(u, wb, Heads, bb); var alpha = MatVec(u, wa, Heads, ba);
                var gate = MatVec(u, wg, ModelDim, bg);
                var head = new double[ModelDim];
                for (int h = 0; h < Heads; h++)
                {
                    double nq = 0, nk = 0;
                    for (int i = 0; i < Hd; i++) { nq += q[h * Hd + i] * q[h * Hd + i]; nk += k[h * Hd + i] * k[h * Hd + i]; }
                    var qh = new double[Hd]; var kh = new double[Hd];
                    for (int i = 0; i < Hd; i++)
                    {
                        qh[i] = q[h * Hd + i] / Math.Sqrt(nq + 1e-6) / Math.Sqrt(Hd);
                        kh[i] = k[h * Hd + i] / Math.Sqrt(nk + 1e-6);
                    }
                    double a = Sig(alpha[h]), be = Sig(beta[h]);
                    // Transition matrix M = a (I - b k kᵀ); S_t = S_{t-1} M + b v kᵀ (S is [dv, dk]).
                    var M = new double[Hd, Hd];
                    for (int i = 0; i < Hd; i++)
                        for (int j = 0; j < Hd; j++) M[i, j] = a * ((i == j ? 1 : 0) - be * kh[i] * kh[j]);
                    var next = new double[Hd, Hd];
                    for (int r = 0; r < Hd; r++)
                        for (int j = 0; j < Hd; j++)
                        {
                            double s = be * v[h * Hd + r] * kh[j];
                            for (int i = 0; i < Hd; i++) s += S[h, r, i] * M[i, j];
                            next[r, j] = s;
                        }
                    var oh = new double[Hd];
                    for (int r = 0; r < Hd; r++)
                        for (int j = 0; j < Hd; j++) { S[h, r, j] = next[r, j]; oh[r] += next[r, j] * qh[j]; }
                    double ms = 0;
                    for (int r = 0; r < Hd; r++) ms += oh[r] * oh[r];
                    double inv = 1 / Math.Sqrt(ms / Hd + 1e-5);
                    for (int r = 0; r < Hd; r++) head[h * Hd + r] = oh[r] * inv * gn[r];
                }
                var gated = new double[ModelDim];
                for (int c = 0; c < ModelDim; c++) gated[c] = head[c] * gate[c] * Sig(gate[c]);
                var outRow = MatVec(gated, wo, ModelDim, bo);
                Array.Copy(outRow, 0, y, (b * SeqLen + t) * ModelDim, ModelDim);
            }
        }
        return y;
    }
}
