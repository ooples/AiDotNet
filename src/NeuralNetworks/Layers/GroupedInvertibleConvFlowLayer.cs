using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Glow-TTS's invertible 1×1 convolution with channel grouping: the channels are split into groups of
/// <c>nSplit</c> and the same <c>nSplit × nSplit</c> matrix mixes each group.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Kim et al. 2020, §3.3, Fig. 8c: a full invertible 1×1 convolution (Glow) over 160 channels costs a 160 × 160
/// determinant; Glow-TTS instead mixes groups of 4 channels with one shared 4 × 4 matrix, taking two channels from each
/// half so the following coupling layer sees a mix (reference implementation <c>modules.InvConvNear</c>, <c>n_split=4</c>).
/// The matrix starts as a random rotation (QR of a Gaussian matrix, with a positive determinant). The log-determinant is
/// <c>(C / nSplit) · T · log|det W|</c>.
/// </para>
/// <para><b>For Beginners:</b> Shuffles information between channels in a way that can be undone exactly.</para>
/// </remarks>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 8, 6", TestConstructorArgs = "8, 4")]
[ElementWiseShape(Note = "Mixes channels within groups; the shape is carried through.")]
[AutoParameters]
public partial class GroupedInvertibleConvFlowLayer<T> : LayerBase<T>, IInvertibleFlowStep<T>
{
    private readonly int _channels;
    private readonly int _split;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _weight;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="channels">Channels of the sequence; a multiple of <paramref name="split"/>.</param>
    /// <param name="split">Channels per group (even; 4 in Glow-TTS).</param>
    public GroupedInvertibleConvFlowLayer([LayerState] int channels, [LayerState] int split = 4)
        : base(new[] { channels }, new[] { channels })
    {
        if (split <= 0 || split % 2 != 0) throw new ArgumentOutOfRangeException(nameof(split), "The group size must be even.");
        if (channels <= 0 || channels % split != 0)
            throw new ArgumentException($"Channels ({channels}) must be a multiple of the group size ({split}).", nameof(channels));
        _channels = channels;
        _split = split;
        _weight = RandomRotation(split);
        RegisterTrainableParameter(_weight, PersistentTensorRole.Weights);
    }

    private Tensor<T> RandomRotation(int n)
    {
        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var a = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                a[i, j] = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            }
        // Gram-Schmidt on the columns gives Q of A = QR.
        var q = new double[n, n];
        for (int j = 0; j < n; j++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++) v[i] = a[i, j];
            for (int k = 0; k < j; k++)
            {
                double dot = 0;
                for (int i = 0; i < n; i++) dot += q[i, k] * a[i, j];
                for (int i = 0; i < n; i++) v[i] -= dot * q[i, k];
            }
            double norm = Math.Sqrt(v.Sum(x => x * x));
            for (int i = 0; i < n; i++) q[i, j] = v[i] / norm;
        }
        if (Determinant(q) < 0)
            for (int i = 0; i < n; i++) q[i, 0] = -q[i, 0];
        var w = new Tensor<T>(new[] { n, n });
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) w[i, j] = NumOps.FromDouble(q[i, j]);
        return w;
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => Transform(input, false).Output;

    /// <inheritdoc />
    public (Tensor<T> Output, Tensor<T>? LogDeterminant) Transform(Tensor<T> input, bool reverse)
    {
        if (input.Rank != 3 || input.Shape[1] != _channels)
            throw new ArgumentException($"Expected [batch, {_channels}, time], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], time = input.Shape[2], n = _split, groups = _channels / n;
        var w = Read(_weight, n);

        // [B, C, T] -> [B, 2, C/n, n/2, T] -> [B, 2, n/2, C/n, T] = [B, n, C/n, T] -> [n, B * C/n * T].
        var grouped = Engine.TensorPermute(Engine.Reshape(input, new[] { batch, 2, groups, n / 2, time }), new[] { 0, 1, 3, 2, 4 }).Contiguous();
        var columns = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(grouped, new[] { batch, n, groups * time }), new[] { 1, 0, 2 }).Contiguous(),
            new[] { n, batch * groups * time });

        Tensor<T> mixing = _weight;
        if (reverse)
        {
            var inverse = Inverse(w);
            mixing = new Tensor<T>(new[] { n, n });
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) mixing[i, j] = NumOps.FromDouble(inverse[i, j]);
        }
        var mixed = Engine.TensorMatMul(mixing, columns);                                   // [n, B * C/n * T]

        var back = Engine.TensorPermute(Engine.Reshape(mixed, new[] { n, batch, groups * time }), new[] { 1, 0, 2 }).Contiguous();
        var ungrouped = Engine.TensorPermute(Engine.Reshape(back, new[] { batch, 2, n / 2, groups, time }), new[] { 0, 1, 3, 2, 4 }).Contiguous();
        var output = Engine.Reshape(ungrouped, new[] { batch, _channels, time });
        if (reverse)
            return (output, null);

        // log|det W| with its exact gradient W^{-T}: value (C/n)·B·T·log|det W| from the host, gradient through the
        // surrogate Σ W ⊙ stop(W^{-T}), whose derivative is W^{-T}.
        double factor = groups * (double)batch * time;
        var inverseT = Inverse(w);
        var gradient = new Tensor<T>(new[] { n, n });
        double surrogateValue = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                gradient[i, j] = NumOps.FromDouble(inverseT[j, i]);
                surrogateValue += w[i, j] * inverseT[j, i];
            }
        var surrogate = Engine.ReduceSum(Engine.TensorMultiply(_weight, gradient), new[] { 0, 1 }, keepDims: false);
        double logAbsDet = Math.Log(Math.Abs(Determinant(w)));
        var logDet = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(surrogate, NumOps.FromDouble(logAbsDet - surrogateValue)), NumOps.FromDouble(factor));
        return (output, logDet);
    }

    private double[,] Read(Tensor<T> t, int n)
    {
        var m = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) m[i, j] = NumOps.ToDouble(t[i, j]);
        return m;
    }

    private static double Determinant(double[,] m)
    {
        int n = m.GetLength(0);
        var a = (double[,])m.Clone();
        double det = 1;
        for (int c = 0; c < n; c++)
        {
            int pivot = c;
            for (int r = c + 1; r < n; r++) if (Math.Abs(a[r, c]) > Math.Abs(a[pivot, c])) pivot = r;
            if (a[pivot, c] == 0) return 0;
            if (pivot != c)
            {
                for (int k = 0; k < n; k++) (a[c, k], a[pivot, k]) = (a[pivot, k], a[c, k]);
                det = -det;
            }
            det *= a[c, c];
            for (int r = c + 1; r < n; r++)
            {
                double f = a[r, c] / a[c, c];
                for (int k = c; k < n; k++) a[r, k] -= f * a[c, k];
            }
        }
        return det;
    }

    private static double[,] Inverse(double[,] m)
    {
        int n = m.GetLength(0);
        var a = new double[n, 2 * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) a[i, j] = m[i, j];
            a[i, n + i] = 1;
        }
        for (int c = 0; c < n; c++)
        {
            int pivot = c;
            for (int r = c + 1; r < n; r++) if (Math.Abs(a[r, c]) > Math.Abs(a[pivot, c])) pivot = r;
            for (int k = 0; k < 2 * n; k++) (a[c, k], a[pivot, k]) = (a[pivot, k], a[c, k]);
            double d = a[c, c];
            if (d == 0) throw new InvalidOperationException("The mixing matrix is singular.");
            for (int k = 0; k < 2 * n; k++) a[c, k] /= d;
            for (int r = 0; r < n; r++)
            {
                if (r == c) continue;
                double f = a[r, c];
                for (int k = 0; k < 2 * n; k++) a[r, k] -= f * a[c, k];
            }
        }
        var inv = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) inv[i, j] = a[i, n + j];
        return inv;
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != _weight.Length) return;
        for (int i = 0; i < _weight.Length; i++)
            _weight[i] = NumOps.Subtract(_weight[i], NumOps.Multiply(learningRate, gradients[i]));
        Engine.InvalidatePersistentTensor(_weight);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Channels"] = _channels.ToString(inv);
        metadata["Split"] = _split.ToString(inv);
        return metadata;
    }
}
