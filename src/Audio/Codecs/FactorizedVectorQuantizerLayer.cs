using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Codecs;

/// <summary>A residual vector quantizer as the neural audio codecs use it: <c>[1, dim, T]</c> in, the sum of the code
/// vectors of the active codebooks out, with codes per codebook.</summary>
public interface IResidualVectorQuantizer<T>
{
    /// <summary>The number of codebooks the forward uses (all when 0).</summary>
    int ActiveQuantizers { get; set; }

    /// <summary>The codes <c>[n_q, T]</c> of a latent <c>[1, dim, T]</c>.</summary>
    int[,] Encode(Tensor<T> latent, int? quantizers = null);

    /// <summary>The quantized latent <c>[1, dim, T]</c> of codes <c>[n_q, T]</c>.</summary>
    Tensor<T> Decode(int[,] codes);

    /// <summary>The quantized latent of the last forward, straight-through to its input.</summary>
    Tensor<T> Forward(Tensor<T> latent);

    /// <summary>The commitment loss of the last training forward (a tape-connected scalar), or null.</summary>
    Tensor<T>? CommitmentLoss { get; }
}

/// <summary>
/// DAC's residual vector quantizer with factorized, L2-normalized codes (Kumar et al. 2023, §3.2 and App. A; reference
/// <c>dac/nn/quantize.py</c>): each codebook projects its residual to a low-dimensional space (W_in, a weight-normalized
/// 1×1 convolution), looks up the nearest code by cosine similarity (both sides L2-normalized), and projects the code
/// back (W_out); the codebooks are learned by gradient with the VQ-VAE codebook and commitment losses and the
/// straight-through estimator — no EMA, k-means initialization or restarts.
/// </summary>
/// <remarks>
/// <para>App. A states both losses on the L2-normalized vectors:
/// <c>L_VQ = ‖sg[ℓ2(z_proj)] − ℓ2(e_k)‖² + β‖ℓ2(z_proj) − sg[ℓ2(e_k)]‖²</c>; the reference code takes them on the raw
/// projections and codes (<see cref="LossesOnRawVectors"/>). Each is the mean squared error over the projected dimensions
/// and frames, summed over the codebooks used.</para>
/// </remarks>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 4, 6", TestConstructorArgs = "4, 2, 8, 2")]
[ElementWiseShape(Note = "Quantizes each time step's vector; the shape is carried through.")]
public sealed partial class FactorizedVectorQuantizerLayer<T> : LayerBase<T>, IResidualVectorQuantizer<T>
{
    private readonly int _dim, _quantizers, _bins, _codeDim;
    private readonly bool _rawLosses;
    private readonly NormedConv1DLayer<T>[] _inProjection, _outProjection;
    private readonly Tensor<T>[] _codebooks;

    /// <summary>Creates the quantizer.</summary>
    /// <param name="dim">The latent dimension D.</param>
    /// <param name="quantizers">The codebooks N_q (9).</param>
    /// <param name="bins">The codes per codebook (1024).</param>
    /// <param name="codeDim">The factorized code dimension M (8).</param>
    /// <param name="lossesOnRawVectors">Whether the codebook and commitment losses compare the raw projection and code (the
    /// reference code) rather than their L2-normalized forms (App. A; false).</param>
    public FactorizedVectorQuantizerLayer([LayerState] int dim, [LayerState] int quantizers, [LayerState] int bins, [LayerState] int codeDim,
        [LayerState] bool lossesOnRawVectors = false)
        : base(new[] { dim }, new[] { dim })
    {
        if (dim <= 0 || quantizers <= 0 || bins <= 0 || codeDim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        _dim = dim;
        _quantizers = quantizers;
        _bins = bins;
        _codeDim = codeDim;
        _rawLosses = lossesOnRawVectors;
        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        _inProjection = new NormedConv1DLayer<T>[quantizers];
        _outProjection = new NormedConv1DLayer<T>[quantizers];
        _codebooks = new Tensor<T>[quantizers];
        for (int q = 0; q < quantizers; q++)
        {
            _inProjection[q] = new NormedConv1DLayer<T>(dim, codeDim, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight);
            _outProjection[q] = new NormedConv1DLayer<T>(codeDim, dim, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight);
            // The reference's init_weights zeroes every Conv1d bias, the projections' included.
            _inProjection[q].ZeroBias();
            _outProjection[q].ZeroBias();
            RegisterSubLayer(_inProjection[q]);
            RegisterSubLayer(_outProjection[q]);
            // nn.Embedding: N(0, 1).
            _codebooks[q] = new Tensor<T>(new[] { bins, codeDim });
            for (int i = 0; i < _codebooks[q].Length; i++)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                _codebooks[q][i] = NumOps.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
            }
            RegisterTrainableParameter(_codebooks[q], PersistentTensorRole.Weights);
        }
    }

    public override bool SupportsTraining => true;

    /// <summary>Whether the losses compare raw vectors (the reference code) rather than L2-normalized ones.</summary>
    public bool LossesOnRawVectors => _rawLosses;

    /// <inheritdoc />
    public int ActiveQuantizers { get; set; }

    /// <summary>The codebook loss of the last training forward, summed over the codebooks used.</summary>
    public Tensor<T>? CodebookLoss { get; private set; }

    /// <inheritdoc />
    public Tensor<T>? CommitmentLoss { get; private set; }

    /// <summary>The codes <c>[n_q, T]</c> of the last forward.</summary>
    public int[,]? LastCodes { get; private set; }

    /// <summary>Codebook <paramref name="q"/> <c>[bins, codeDim]</c>.</summary>
    public Tensor<T> Codebook(int q) => _codebooks[q];

    internal NormedConv1DLayer<T> InProjection(int q) => _inProjection[q];

    internal NormedConv1DLayer<T> OutProjection(int q) => _outProjection[q];

    // Rows [T, M] of [1, M, T], L2-normalized (F.normalize, eps 1e-12).
    private Tensor<T> Normalize(Tensor<T> rows)
    {
        var norms = Engine.TensorPow(Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(rows, rows), new[] { 1 }, keepDims: true),
            NumOps.FromDouble(1e-24)), NumOps.FromDouble(0.5));
        return Engine.TensorDivide(rows, Engine.TensorBroadcastTo(norms, rows._shape));
    }

    private Tensor<T> Rows(Tensor<T> x) => Engine.TensorTranspose(Engine.Reshape(x, new[] { x.Shape[1], x.Shape[2] }));

    // The nearest codes by cosine similarity: argmax ℓ2(z)·ℓ2(e).
    private int[] Nearest(Tensor<T> projected, int q)
    {
        using var _ = new NoGradScope<T>();
        var similarity = Engine.TensorMatMul(Normalize(Rows(projected)), Engine.TensorTranspose(Normalize(_codebooks[q])));   // [T, bins]
        var arg = Engine.TensorArgMax(similarity, 1);
        var codes = new int[projected.Shape[2]];
        for (int i = 0; i < codes.Length; i++) codes[i] = arg[i];
        return codes;
    }

    // The code vectors [1, M, T] of codes: rows of the codebook (on the tape, so the codebook loss trains it).
    private Tensor<T> Lookup(int[] codes, int q)
    {
        var index = new Tensor<int>(new[] { codes.Length });
        for (int i = 0; i < codes.Length; i++) index[i] = codes[i];
        var rows = Engine.TensorIndexSelect(_codebooks[q], index, 0);                                         // [T, M]
        return Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, _codeDim, codes.Length });
    }

    private Tensor<T> Mse(Tensor<T> a, Tensor<T> b)
    {
        var d = Engine.TensorSubtract(a, b);
        return Engine.ReduceMean(Engine.TensorMultiply(d, d), Enumerable.Range(0, d.Rank).ToArray(), keepDims: false);
    }

    private static Tensor<T> Detach(Tensor<T> x) => new(x._shape, x.ToVector());

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[1] != _dim)
            throw new ArgumentException($"Expected [1, {_dim}, T], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int nq = ActiveQuantizers > 0 ? Math.Min(ActiveQuantizers, _quantizers) : _quantizers, t = input.Shape[2];
        var residual = input;
        Tensor<T>? output = null, commitment = null, codebook = null;
        var allCodes = new int[nq, t];
        for (int q = 0; q < nq; q++)
        {
            var projected = _inProjection[q].Forward(residual);                                              // z_e [1, M, T]
            var codes = Nearest(projected, q);
            var code = Lookup(codes, q);                                                                     // e_k [1, M, T]
            if (IsTrainingMode)
            {
                Tensor<T> z = projected, e = code;
                if (!_rawLosses)
                {
                    z = Engine.Reshape(Engine.TensorTranspose(Normalize(Rows(projected))), projected._shape);
                    e = Engine.Reshape(Engine.TensorTranspose(Normalize(Rows(code))), code._shape);
                }
                var commit = Mse(z, Detach(e));
                var book = Mse(e, Detach(z));
                commitment = commitment is null ? commit : Engine.TensorAdd(commitment, commit);
                codebook = codebook is null ? book : Engine.TensorAdd(codebook, book);
            }
            // Straight-through: z_e + stop(e_k − z_e), then W_out.
            var straight = Engine.TensorAdd(projected, Engine.TensorSubtract(Detach(code), Detach(projected)));
            var quantized = _outProjection[q].Forward(straight);
            residual = Engine.TensorSubtract(residual, quantized);
            output = output is null ? quantized : Engine.TensorAdd(output, quantized);
            for (int i = 0; i < t; i++) allCodes[q, i] = codes[i];
        }
        LastCodes = allCodes;
        CommitmentLoss = commitment;
        CodebookLoss = codebook;
        return output!;
    }

    /// <inheritdoc />
    public int[,] Encode(Tensor<T> latent, int? quantizers = null)
    {
        using var _ = new NoGradScope<T>();
        int nq = Math.Min(quantizers ?? _quantizers, _quantizers), t = latent.Shape[2];
        var codes = new int[nq, t];
        var residual = latent;
        for (int q = 0; q < nq; q++)
        {
            var c = Nearest(_inProjection[q].Forward(residual), q);
            for (int i = 0; i < t; i++) codes[q, i] = c[i];
            residual = Engine.TensorSubtract(residual, _outProjection[q].Forward(Lookup(c, q)));
        }
        return codes;
    }

    /// <inheritdoc />
    public Tensor<T> Decode(int[,] codes)
    {
        using var _ = new NoGradScope<T>();
        int nq = codes.GetLength(0), t = codes.GetLength(1);
        Tensor<T>? output = null;
        for (int q = 0; q < nq; q++)
        {
            var row = new int[t];
            for (int i = 0; i < t; i++) row[i] = codes[q, i];
            var y = _outProjection[q].Forward(Lookup(row, q));
            output = output is null ? y : Engine.TensorAdd(output, y);
        }
        return output ?? new Tensor<T>(new[] { 1, _dim, t });
    }

    Tensor<T> IResidualVectorQuantizer<T>.Forward(Tensor<T> latent) => Forward(latent);

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Dim"] = _dim.ToString(inv);
        metadata["Quantizers"] = _quantizers.ToString(inv);
        metadata["Bins"] = _bins.ToString(inv);
        metadata["CodeDim"] = _codeDim.ToString(inv);
        metadata["LossesOnRawVectors"] = _rawLosses.ToString(inv);
        return metadata;
    }
}
