using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Codecs;

/// <summary>
/// Residual vector quantization (SoundStream, Zeghidour et al. 2021, Algorithm 1; EnCodec's reference
/// <c>quantization/core_vq.py</c>): each of n_q Euclidean codebooks quantizes the residual the previous ones left; the
/// codebooks are updated by exponential moving averages of their assigned vectors (decay 0.99, Laplace smoothing 1e-5),
/// initialized by k-means on the first training batch, and a code whose usage falls below the dead-code threshold is
/// replaced by a random vector of the batch. Gradients pass straight through the quantization; the commitment loss is
/// EnCodec's Eq. 3, the sum over the codebooks used of the squared distance between each residual and its (stopped)
/// code, or the reference code's mean over codebooks of the mean squared error.
/// </summary>
/// <remarks>
/// As the reference (and its issue #25, kept "for reproducibility"), each layer's residual is reduced by the
/// straight-through output, so gradient flows through the residual chain. The input and output are <c>[1, dim, T]</c>.
/// </remarks>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = false, TestInputShape = "1, 4, 6", TestConstructorArgs = "4, 2, 8")]
[ElementWiseShape(Note = "Quantizes each time step's vector; the shape is carried through.")]
[AutoParameters]
public sealed partial class ResidualVectorQuantizerLayer<T> : LayerBase<T>, IResidualVectorQuantizer<T>
{
    private readonly int _dim, _quantizers, _bins, _kmeansIterations, _deadCodeThreshold;
    private readonly double _decay, _epsilon;
    private readonly bool _commitmentAsMean;
    private readonly Tensor<T>[] _embed, _embedAverage, _clusterSize;
    private readonly Tensor<T>[] _initialized;
    private Random _random;

    /// <summary>Creates the quantizer.</summary>
    /// <param name="dim">The vectors' dimension.</param>
    /// <param name="quantizers">The residual codebooks n_q.</param>
    /// <param name="bins">The codes per codebook.</param>
    /// <param name="decay">The EMA decay (0.99).</param>
    /// <param name="kmeansIterations">k-means iterations of the first-batch initialization (50; 0 to start from a
    /// Kaiming-uniform codebook).</param>
    /// <param name="deadCodeThreshold">The EMA usage below which a code is replaced (2; 0 never replaces).</param>
    /// <param name="commitmentAsMean">Whether the commitment loss is the mean over codebooks of the MSE (the reference
    /// code) rather than the sum over codebooks of squared norms (EnCodec Eq. 3; false).</param>
    public ResidualVectorQuantizerLayer([LayerState] int dim, [LayerState] int quantizers, [LayerState] int bins,
        [LayerState] double decay = 0.99, [LayerState] int kmeansIterations = 50, [LayerState] int deadCodeThreshold = 2,
        [LayerState] bool commitmentAsMean = false)
        : base(new[] { dim }, new[] { dim })
    {
        if (dim <= 0 || quantizers <= 0 || bins <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        _dim = dim;
        _quantizers = quantizers;
        _bins = bins;
        _decay = decay;
        _epsilon = 1e-5;
        _kmeansIterations = kmeansIterations;
        _deadCodeThreshold = deadCodeThreshold;
        _commitmentAsMean = commitmentAsMean;
        _random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        _embed = new Tensor<T>[quantizers];
        _embedAverage = new Tensor<T>[quantizers];
        _clusterSize = new Tensor<T>[quantizers];
        _initialized = new Tensor<T>[quantizers];
        // nn.init.kaiming_uniform_ on [bins, dim]: U(±√(6 / dim)) (a = 0, fan_in = dim).
        double bound = Math.Sqrt(6.0 / dim);
        for (int q = 0; q < quantizers; q++)
        {
            _embed[q] = new Tensor<T>(new[] { bins, dim });
            if (kmeansIterations == 0)
                for (int i = 0; i < _embed[q].Length; i++) _embed[q][i] = NumOps.FromDouble((2 * _random.NextDouble() - 1) * bound);
            _embedAverage[q] = new Tensor<T>(_embed[q]._shape, _embed[q].ToVector());
            _clusterSize[q] = new Tensor<T>(new[] { bins });
            // A buffer, as the reference's "inited": a restored model must not re-run k-means over trained codebooks.
            _initialized[q] = new Tensor<T>(new[] { 1 });
            _initialized[q][0] = kmeansIterations == 0 ? NumOps.One : NumOps.Zero;
            // The codebooks are fitted state - EMA and k-means, not the optimizer - restored with the model.
            RegisterBuffer(_embed[q], $"layers.{q}._codebook.embed", stateRole: AiDotNet.Models.Parameters.ParameterSlotRole.LearnedState);
            RegisterBuffer(_embedAverage[q], $"layers.{q}._codebook.embed_avg", stateRole: AiDotNet.Models.Parameters.ParameterSlotRole.LearnedState);
            RegisterBuffer(_clusterSize[q], $"layers.{q}._codebook.cluster_size", stateRole: AiDotNet.Models.Parameters.ParameterSlotRole.LearnedState);
            RegisterBuffer(_initialized[q], $"layers.{q}._codebook.inited");
        }
    }

    public override bool SupportsTraining => true;

    /// <summary>The number of residual codebooks.</summary>
    public int Quantizers => _quantizers;

    /// <summary>The codes per codebook.</summary>
    public int Bins => _bins;

    /// <summary>The commitment loss of the last training forward over the codebooks used (a tape-connected scalar).</summary>
    public Tensor<T>? CommitmentLoss { get; private set; }

    /// <summary>The codes <c>[n_q, T]</c> of the last forward.</summary>
    public int[,]? LastCodes { get; private set; }

    /// <summary>The first codebook's quantized output <c>[1, dim, T]</c> of the last forward, straight-through to its input
    /// (SpeechTokenizer distills it toward a teacher).</summary>
    public Tensor<T>? FirstQuantized { get; private set; }

    /// <summary>The number of codebooks the forward uses (all by default; fewer for a lower bandwidth).</summary>
    public int ActiveQuantizers { get; set; }

    /// <summary>The codebook <c>[bins, dim]</c> of quantizer <paramref name="q"/>.</summary>
    public Tensor<T> Codebook(int q) => _embed[q];

    /// <summary>Loads codebook <paramref name="q"/>'s state: the codes <c>[bins, dim]</c>, their EMA sums, the EMA usage
    /// <c>[bins]</c>, and whether k-means initialization already ran (reference <c>embed</c>, <c>embed_avg</c>,
    /// <c>cluster_size</c>, <c>inited</c>).</summary>
    internal void LoadCodebook(int q, double[] embed, double[] embedAverage, double[] clusterSize, bool initialized)
    {
        if (embed.Length != _bins * _dim || embedAverage.Length != _bins * _dim || clusterSize.Length != _bins)
            throw new ArgumentException($"Expected codebooks [{_bins}, {_dim}] and usage [{_bins}].");
        for (int i = 0; i < embed.Length; i++)
        {
            _embed[q][i] = NumOps.FromDouble(embed[i]);
            _embedAverage[q][i] = NumOps.FromDouble(embedAverage[i]);
        }
        for (int k = 0; k < _bins; k++) _clusterSize[q][k] = NumOps.FromDouble(clusterSize[k]);
        _initialized[q][0] = initialized ? NumOps.One : NumOps.Zero;
        Invalidate(q);
    }

    // The rows [T, dim] of [1, dim, T].
    private Tensor<T> Rows(Tensor<T> x) => Engine.TensorTranspose(Engine.Reshape(x, new[] { _dim, x.Shape[2] }));

    // The nearest code of each row [T, dim] under a codebook [bins, dim]: argmin ||e||^2 - 2 x.e (||x||^2 is constant per row).
    private int[] Nearest(Tensor<T> rows, Tensor<T> codebook)
    {
        var cross = Engine.TensorMatMul(rows, Engine.TensorTranspose(codebook));                                  // [T, bins]
        var norms = Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(codebook, codebook), new[] { 1 }, keepDims: false),
            new[] { 1, codebook.Shape[0] });
        var distance = Engine.TensorAdd(Engine.TensorMultiplyScalar(cross, NumOps.FromDouble(-2.0)), Engine.TensorBroadcastTo(norms, cross._shape));
        var arg = Engine.TensorArgMin(distance, 1);
        var codes = new int[rows.Shape[0]];
        for (int i = 0; i < codes.Length; i++) codes[i] = arg[i];
        return codes;
    }

    private Tensor<T> OneHot(int[] codes)
    {
        var indices = new Tensor<int>(new[] { codes.Length });
        for (int i = 0; i < codes.Length; i++) indices[i] = codes[i];
        return Engine.TensorOneHot<T>(indices, _bins);                                                             // [T, bins]
    }

    // The code vectors [1, dim, T] of codes under a codebook (a constant, not on the tape).
    private Tensor<T> Lookup(int[] codes, Tensor<T> codebook)
        => Engine.Reshape(Engine.TensorTranspose(Engine.TensorMatMul(OneHot(codes), codebook)), new[] { 1, _dim, codes.Length });

    private static void Store(Tensor<T> source, Tensor<T> buffer) => source.Data.Span.CopyTo(buffer.AsWritableSpan());

    // Row indices for sample_vectors: a random permutation's first `count` rows, or rows drawn with replacement when the
    // batch has fewer.
    private int[] SampleRows(int n, int count)
    {
        var rows = new int[count];
        if (n >= count)
        {
            var p = new int[n];
            for (int i = 0; i < n; i++) p[i] = i;
            for (int i = n - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (p[i], p[j]) = (p[j], p[i]);
            }
            Array.Copy(p, rows, count);
        }
        else
        {
            for (int k = 0; k < count; k++) rows[k] = _random.Next(n);
        }
        return rows;
    }

    private Tensor<T> Gather(Tensor<T> rows, int[] indices)
    {
        var picked = new Tensor<T>(new[] { indices.Length, _dim });
        for (int k = 0; k < indices.Length; k++)
            for (int j = 0; j < _dim; j++) picked[k, j] = rows[indices[k], j];
        return picked;
    }

    // k-means of the first training batch (reference kmeans: sampled initial means; an empty cluster keeps its mean).
    private void KMeansInit(Tensor<T> rows, int q)
    {
        var means = Gather(rows, SampleRows(rows.Shape[0], _bins));
        var counts = new Tensor<T>(new[] { _bins });
        for (int it = 0; it < _kmeansIterations; it++)
        {
            var onehot = OneHot(Nearest(rows, means));
            counts = Engine.ReduceSum(onehot, new[] { 0 }, keepDims: false);
            var sums = Engine.TensorMatMul(Engine.TensorTranspose(onehot), rows);                                      // [bins, dim]
            var next = new Tensor<T>(means._shape);
            for (int k = 0; k < _bins; k++)
            {
                double c = NumOps.ToDouble(counts[k]);
                for (int j = 0; j < _dim; j++) next[k, j] = c > 0 ? NumOps.FromDouble(NumOps.ToDouble(sums[k, j]) / c) : means[k, j];
            }
            means = next;
        }
        Store(means, _embed[q]);
        Store(means, _embedAverage[q]);
        Store(counts, _clusterSize[q]);
        _initialized[q][0] = NumOps.One;
        Invalidate(q);
    }

    private void Invalidate(int q)
    {
        Engine.InvalidatePersistentTensor(_embed[q]);
        Engine.InvalidatePersistentTensor(_embedAverage[q]);
        Engine.InvalidatePersistentTensor(_clusterSize[q]);
        Engine.InvalidatePersistentTensor(_initialized[q]);
    }

    // Training-time codebook maintenance (reference EuclideanCodebook.forward): codes whose EMA usage is below the threshold
    // take a sampled batch vector, then cluster_size <- g cluster_size + (1 - g) counts, embed_avg <- g embed_avg + (1 - g) sum x,
    // embed = embed_avg / Laplace-smoothed cluster_size.
    private void UpdateCodebook(Tensor<T> rows, int[] codes, int q)
    {
        if (_deadCodeThreshold > 0)
        {
            var replacements = Gather(rows, SampleRows(rows.Shape[0], _bins));
            for (int k = 0; k < _bins; k++)
                if (NumOps.ToDouble(_clusterSize[q][k]) < _deadCodeThreshold)
                    for (int j = 0; j < _dim; j++) _embed[q][k, j] = replacements[k, j];
        }
        var onehot = OneHot(codes);
        var counts = Engine.ReduceSum(onehot, new[] { 0 }, keepDims: false);                                           // [bins]
        var sums = Engine.TensorMatMul(Engine.TensorTranspose(onehot), rows);                                          // [bins, dim]
        var keep = NumOps.FromDouble(_decay);
        var take = NumOps.FromDouble(1 - _decay);
        var size = Engine.TensorAdd(Engine.TensorMultiplyScalar(_clusterSize[q], keep), Engine.TensorMultiplyScalar(counts, take));
        var average = Engine.TensorAdd(Engine.TensorMultiplyScalar(_embedAverage[q], keep), Engine.TensorMultiplyScalar(sums, take));
        double total = 0;
        for (int k = 0; k < _bins; k++) total += NumOps.ToDouble(size[k]);
        var smoothed = Engine.TensorMultiplyScalar(Engine.TensorAddScalar(size, NumOps.FromDouble(_epsilon)),
            NumOps.FromDouble(total / (total + _bins * _epsilon)));
        var embed = Engine.TensorDivide(average, Engine.TensorBroadcastTo(Engine.Reshape(smoothed, new[] { _bins, 1 }), average._shape));
        Store(size, _clusterSize[q]);
        Store(average, _embedAverage[q]);
        Store(embed, _embed[q]);
        Invalidate(q);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[1] != _dim)
            throw new ArgumentException($"Expected [1, {_dim}, T], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int nq = ActiveQuantizers > 0 ? Math.Min(ActiveQuantizers, _quantizers) : _quantizers, t = input.Shape[2];
        var residual = input;
        Tensor<T>? output = null, commitment = null;
        var allCodes = new int[nq, t];
        for (int q = 0; q < nq; q++)
        {
            int[] codes;
            Tensor<T> code;
            using (new NoGradScope<T>())
            {
                var rows = Rows(Detach(residual));
                if (IsTrainingMode && NumOps.ToDouble(_initialized[q][0]) == 0) KMeansInit(rows, q);
                codes = Nearest(rows, _embed[q]);
                code = Lookup(codes, _embed[q]);
                if (IsTrainingMode) UpdateCodebook(rows, codes, q);
            }
            // Straight-through: x + stop(q − x).
            var quantized = Engine.TensorAdd(residual, Engine.TensorSubtract(code, Detach(residual)));
            if (IsTrainingMode)
            {
                var d = Engine.TensorSubtract(residual, code);
                var squared = Engine.TensorMultiply(d, d);
                var term = _commitmentAsMean
                    ? Engine.ReduceMean(squared, new[] { 0, 1, 2 }, keepDims: false)
                    : Engine.ReduceSum(squared, new[] { 0, 1, 2 }, keepDims: false);
                commitment = commitment is null ? term : Engine.TensorAdd(commitment, term);
            }
            if (q == 0) FirstQuantized = quantized;
            residual = Engine.TensorSubtract(residual, quantized);
            output = output is null ? quantized : Engine.TensorAdd(output, quantized);
            for (int i = 0; i < t; i++) allCodes[q, i] = codes[i];
        }
        LastCodes = allCodes;
        CommitmentLoss = commitment is null || !_commitmentAsMean ? commitment : Engine.TensorMultiplyScalar(commitment, NumOps.FromDouble(1.0 / nq));
        return output!;
    }

    private static Tensor<T> Detach(Tensor<T> x) => new(x._shape, x.ToVector());

    /// <summary>The codes <c>[n_q, T]</c> of <c>[1, dim, T]</c> (no codebook update).</summary>
    public int[,] Encode(Tensor<T> x, int? quantizers = null)
    {
        int nq = Math.Min(quantizers ?? _quantizers, _quantizers), t = x.Shape[2];
        var codes = new int[nq, t];
        using var _ = new NoGradScope<T>();
        var residual = Rows(x);
        for (int q = 0; q < nq; q++)
        {
            var c = Nearest(residual, _embed[q]);
            for (int i = 0; i < t; i++) codes[q, i] = c[i];
            residual = Engine.TensorSubtract(residual, Engine.TensorMatMul(OneHot(c), _embed[q]));
        }
        return codes;
    }

    /// <summary>The sum of the code vectors <c>[1, dim, T]</c> of codes <c>[n_q, T]</c>.</summary>
    public Tensor<T> Decode(int[,] codes)
    {
        int nq = codes.GetLength(0), t = codes.GetLength(1);
        var y = new Tensor<T>(new[] { 1, _dim, t });
        for (int q = 0; q < nq; q++)
        {
            var row = new int[t];
            for (int i = 0; i < t; i++) row[i] = codes[q, i];
            y = Engine.TensorAdd(y, Lookup(row, _embed[q]));
        }
        return y;
    }

    Tensor<T> IResidualVectorQuantizer<T>.Forward(Tensor<T> latent) => Forward(latent);

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Quantizers"] = _quantizers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Bins"] = _bins.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
