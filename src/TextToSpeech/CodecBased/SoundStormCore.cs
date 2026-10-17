using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>How a masked acoustic token enters the network.</summary>
internal enum SoundStormMaskInput
{
    /// <summary>It contributes nothing: the embedding table's padding row, held at zero (Pheme's reference).</summary>
    Zero,

    /// <summary>A learned mask embedding, the row after the codes (MaskGIT; lucidrains/soundstorm-pytorch).</summary>
    MaskToken,
}

/// <summary>Which prompt prefix a training step keeps unmasked.</summary>
internal enum SoundStormPromptSampling
{
    /// <summary>SoundStorm §3.2: t ~ U{0, T − 1}, and positions t′ &gt; t may be masked.</summary>
    Paper,

    /// <summary>Pheme's <c>create_mask</c>: start ~ U{1, T − 2}, positions t ≥ start may be masked, and at least one
    /// is.</summary>
    PhemeReference,
}

/// <summary>
/// The sizes and conventions of a SoundStorm acoustic model (Borsos et al. 2023): <see cref="Levels"/> RVQ codebooks of
/// <see cref="CodebookSize"/> codes, conditioned frame by frame on tokens from a <see cref="ConditioningCodes"/>-entry
/// vocabulary.
/// </summary>
/// <param name="Levels">Q, the RVQ levels modeled.</param>
/// <param name="CodebookSize">C, the codes per level.</param>
/// <param name="EmbeddingRows">Rows of each level's embedding (C plus any special tokens).</param>
/// <param name="HeadOutputs">Outputs of each level's head.</param>
/// <param name="PaddingRow">The level embeddings' zero row, or −1 for none.</param>
/// <param name="ConditioningCodes">Rows of the conditioning embedding.</param>
/// <param name="ConditioningPaddingRow">The conditioning embedding's zero row, or −1 for none.</param>
/// <param name="Conformer">The bidirectional Conformer's sizes (§3.1).</param>
/// <param name="GlobalConditioningDim">Width of an utterance-level vector projected and added to every frame (Pheme's
/// speaker embedding), or 0 for none.</param>
/// <param name="GlobalConditioningDropout">Dropout on that vector, applied at inference too, as Pheme's reference does.</param>
/// <param name="MaskInput">How a masked token enters the network.</param>
/// <param name="Prompt">Which prompt prefix a training step keeps.</param>
/// <param name="FinerPromptLevelsVisible">Whether levels finer than the one being predicted are read in the prompt
/// (§3.2 masks finer levels only after the prompt: Y<sub>&gt;t,&gt;q</sub>). Pheme's reference reads levels up to the
/// current one only.</param>
/// <param name="GreedyLastIteration">Whether the last confidence-based iteration of a level picks the most likely
/// code instead of sampling (§3.3).</param>
internal sealed record SoundStormCoreConfiguration(
    int Levels,
    int CodebookSize,
    int EmbeddingRows,
    int HeadOutputs,
    int PaddingRow,
    int ConditioningCodes,
    int ConditioningPaddingRow,
    SoundStormConformerConfiguration Conformer,
    int GlobalConditioningDim,
    double GlobalConditioningDropout,
    SoundStormMaskInput MaskInput,
    SoundStormPromptSampling Prompt,
    bool FinerPromptLevelsVisible,
    bool GreedyLastIteration);

/// <summary>
/// The SoundStorm acoustic network and its masked training and iterative parallel decoding (Borsos et al. 2023,
/// §3.1–3.3), shared by <see cref="SoundStorm{T}"/> and <see cref="Pheme{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// The input sums, per frame, one embedding per RVQ level and the conditioning token's embedding, so the Conformer's
/// sequence is as long as the number of codec frames whatever the number of levels (§3.1). Each level has its own
/// dense head.
/// </para>
/// <para>
/// A training step samples a level q, a prompt prefix and a cosine-scheduled mask (p = cos u, u ~ U[0, π/2]) over the
/// positions after the prompt; level q is masked there, and so is every finer level. The loss is the cross-entropy on
/// level q's masked positions only (§3.2).
/// </para>
/// <para>
/// Decoding fills the levels coarse to fine (§3.3). A level given one iteration takes the most likely code everywhere
/// in one pass; a level given more runs MaskGIT (Chang et al. 2022): sample the masked positions, keep the most
/// confident on a cosine schedule, and re-mask the rest. The confidence adds Gumbel noise annealed linearly to zero,
/// as MaskGIT's reference does; SoundStorm's paper does not state it.
/// </para>
/// </remarks>
internal sealed class SoundStormCore<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IEngine _engine;
    private readonly SoundStormCoreConfiguration _c;
    private readonly List<LayerBase<T>> _layers;

    /// <summary>Each level's embedding table.</summary>
    public List<TiedEmbeddingLayer<T>> LevelEmbeddings { get; } = new();

    /// <summary>The conditioning tokens' embedding table.</summary>
    public TiedEmbeddingLayer<T> ConditioningEmbedding { get; }

    /// <summary>The projection of the utterance-level vector, when the configuration has one.</summary>
    public DenseLayer<T>? GlobalProjection { get; }

    /// <summary>The bidirectional Conformer.</summary>
    public SoundStormConformer<T> Conformer { get; }

    /// <summary>Each level's output head.</summary>
    public List<DenseLayer<T>> Heads { get; } = new();

    /// <summary>The configuration.</summary>
    public SoundStormCoreConfiguration Configuration => _c;

    /// <summary>The token a masked position carries in the code arrays this class reads.</summary>
    public int MaskId => _c.MaskInput == SoundStormMaskInput.MaskToken ? _c.CodebookSize : _c.PaddingRow;

    /// <summary>
    /// Builds the network into <paramref name="layers"/> (embeddings, the global projection, the Conformer, the heads, in
    /// that order) and draws every weight from <paramref name="normal"/>: linear and convolution weights and the
    /// embeddings, with zero biases and padding rows, and layer norms at one and zero.
    /// </summary>
    public SoundStormCore(IEngine engine, List<LayerBase<T>> layers, SoundStormCoreConfiguration configuration, Func<double> normal)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _layers = layers ?? throw new ArgumentNullException(nameof(layers));
        _c = configuration ?? throw new ArgumentNullException(nameof(configuration));
        if (normal is null) throw new ArgumentNullException(nameof(normal));
        if (_c.MaskInput == SoundStormMaskInput.MaskToken && _c.EmbeddingRows <= _c.CodebookSize)
            throw new ArgumentException("A mask token needs an embedding row after the codes.", nameof(configuration));
        if (_c.MaskInput == SoundStormMaskInput.Zero && _c.PaddingRow < 0)
            throw new ArgumentException("Zeroed masked inputs need a padding row.", nameof(configuration));

        int dim = _c.Conformer.Dim;
        for (int l = 0; l < _c.Levels; l++)
        {
            var embedding = Own(new TiedEmbeddingLayer<T>(_c.EmbeddingRows, dim));
            embedding.Reinitialize(normal, _c.PaddingRow >= 0 ? _c.PaddingRow : null);
            LevelEmbeddings.Add(embedding);
        }
        ConditioningEmbedding = Own(new TiedEmbeddingLayer<T>(_c.ConditioningCodes, dim));
        ConditioningEmbedding.Reinitialize(normal, _c.ConditioningPaddingRow >= 0 ? _c.ConditioningPaddingRow : null);
        if (_c.GlobalConditioningDim > 0)
            GlobalProjection = Own(new DenseLayer<T>(dim, new IdentityActivation<T>() as IActivationFunction<T>));
        Conformer = new SoundStormConformer<T>(engine, layers, _c.Conformer);
        for (int l = 0; l < _c.Levels; l++)
            Heads.Add(Own(new DenseLayer<T>(_c.HeadOutputs, new IdentityActivation<T>() as IActivationFunction<T>)));
        InitializeWeights(normal);
    }

    private TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
    {
        _layers.Add(layer);
        return layer;
    }

    // Linear and Conv1d weights from the normal draw with zero biases; LayerNorm at one and zero.
    private void InitializeWeights(Func<double> normal)
    {
        foreach (var layer in _layers)
        {
            switch (layer)
            {
                case DenseLayer<T> dense:
                {
                    using (new NoGradScope<T>()) dense.Forward(new Tensor<T>(new[] { 1, InputWidth(dense) }));
                    var w = dense.GetWeights();
                    for (int i = 0; i < w.Length; i++) w[i] = NumOps.FromDouble(normal());
                    _engine.InvalidatePersistentTensor(w);
                    var b = dense.GetBiases();
                    for (int i = 0; i < b.Length; i++) b[i] = NumOps.Zero;
                    _engine.InvalidatePersistentTensor(b);
                    break;
                }
                case BiasFreeLinearLayer<T> linear:
                {
                    var values = new Vector<T>(linear.InputSize * linear.OutputSize);
                    for (int i = 0; i < values.Length; i++) values[i] = NumOps.FromDouble(normal());
                    linear.SetParameters(values);
                    break;
                }
                case NormedConv1DLayer<T> conv:
                    conv.Reinitialize(normal);
                    break;
            }
        }
    }

    private int InputWidth(DenseLayer<T> dense)
    {
        var c = _c.Conformer;
        if (ReferenceEquals(dense, GlobalProjection)) return _c.GlobalConditioningDim;
        if (Heads.Contains(dense)) return c.Dim;
        foreach (var block in Conformer.Blocks)
        {
            if (ReferenceEquals(dense, block.Ff1In) || ReferenceEquals(dense, block.Ff2In)) return c.Dim;
            if (ReferenceEquals(dense, block.Ff1Out) || ReferenceEquals(dense, block.Ff2Out)) return c.Dim * c.FeedForwardMultiplier;
            if (ReferenceEquals(dense, block.ToOut)) return c.Heads * c.HeadDim;
        }
        throw new InvalidOperationException("Unknown dense layer in the SoundStorm network.");
    }

    // ---------------------------------------------------------------- network

    /// <summary>
    /// The logits <c>[frames, HeadOutputs]</c> of level <paramref name="level"/>'s head. <paramref name="codes"/> is
    /// <c>[Levels, frames]</c>; position (l, t) is read when <paramref name="visible"/>[l, t] is true and enters as
    /// masked otherwise.
    /// </summary>
    public Tensor<T> Logits(int[,] codes, bool[,] visible, int[] conditioning, int level, Tensor<T>? global,
        bool training, Random random)
    {
        int frames = conditioning.Length, dim = _c.Conformer.Dim;
        Tensor<T>? sum = null;
        for (int l = 0; l < _c.Levels; l++)
        {
            bool any = false;
            for (int t = 0; t < frames && !any; t++) any = visible[l, t];
            // A level with nothing to read is skipped rather than summed as zeros, except under a learned mask token,
            // where its masked positions still carry the mask embedding.
            if (!any && _c.MaskInput == SoundStormMaskInput.Zero) continue;
            if (!any && l > level) continue;
            var ids = new Tensor<T>(new[] { frames });
            Tensor<T>? keep = _c.MaskInput == SoundStormMaskInput.Zero ? new Tensor<T>(new[] { frames, dim }) : null;
            for (int t = 0; t < frames; t++)
            {
                bool read = visible[l, t];
                int id = read ? codes[l, t] : MaskId;
                ids[t] = NumOps.FromDouble(id);
                // The padding row contributes zero and gets no gradient; so does a masked position under Zero.
                if (keep is not null && read && id != _c.PaddingRow)
                    for (int d = 0; d < dim; d++) keep[t, d] = NumOps.One;
            }
            var embedded = LevelEmbeddings[l].Forward(ids);
            if (keep is not null) embedded = _engine.TensorMultiply(embedded, keep);
            sum = sum is null ? embedded : _engine.TensorAdd(sum, embedded);
        }
        var conditioningIds = new Tensor<T>(new[] { frames });
        Tensor<T>? conditioningKeep = _c.ConditioningPaddingRow >= 0 ? new Tensor<T>(new[] { frames, dim }) : null;
        for (int t = 0; t < frames; t++)
        {
            conditioningIds[t] = NumOps.FromDouble(conditioning[t]);
            if (conditioningKeep is not null && conditioning[t] != _c.ConditioningPaddingRow)
                for (int d = 0; d < dim; d++) conditioningKeep[t, d] = NumOps.One;
        }
        var conditioned = ConditioningEmbedding.Forward(conditioningIds);
        if (conditioningKeep is not null) conditioned = _engine.TensorMultiply(conditioned, conditioningKeep);
        var x = sum is null ? conditioned : _engine.TensorAdd(sum, conditioned);
        if (GlobalProjection is not null && global is not null)
        {
            var dropped = T5Seq2Seq<T>.Dropout(_engine, _engine.Reshape(global, new[] { 1, _c.GlobalConditioningDim }),
                _c.GlobalConditioningDropout, random);
            var projected = GlobalProjection.Forward(dropped);                                                 // [1, dim]
            x = _engine.TensorAdd(x, _engine.TensorTile(projected, new[] { frames, 1 }));
        }
        var hidden = Conformer.Forward(x, training, random);
        return Heads[level].Forward(hidden);
    }

    /// <summary>Which positions of <c>[Levels, frames]</c> a step predicting <paramref name="level"/> reads: every
    /// coarser level, level <paramref name="level"/> outside <paramref name="masked"/>, and the finer levels inside the
    /// prompt (<paramref name="promptFrames"/>) when the configuration reads them there.</summary>
    public bool[,] Visibility(int frames, int level, bool[]? masked, int promptFrames)
    {
        var visible = new bool[_c.Levels, frames];
        for (int l = 0; l < _c.Levels; l++)
            for (int t = 0; t < frames; t++)
                visible[l, t] = l < level
                    || (l == level && (masked is null || !masked[t]))
                    || (l > level && _c.FinerPromptLevelsVisible && t < promptFrames);
        return visible;
    }

    // masking_logic.schedule("cosine"): cos(r · π / 2) for r ~ U[0, 1), i.e. cos u for u ~ U[0, π/2), clipped below.
    internal static double CosineSchedule(double ratio) => Math.Min(1.0, Math.Max(1e-6, Math.Cos(ratio * Math.PI / 2)));

    // ---------------------------------------------------------------- training (§3.2)

    /// <summary>
    /// One training example's objective: draws the level, prompt and mask from <paramref name="random"/> now and
    /// returns a function computing the mean cross-entropy over level q's masked positions (zero when none is masked).
    /// </summary>
    public Func<Tensor<T>> TrainingObjective(int[,] codes, int[] conditioning, Tensor<T>? global, bool training, Random random)
    {
        int frames = conditioning.Length;
        if (codes.GetLength(0) != _c.Levels || codes.GetLength(1) != frames)
            throw new ArgumentException($"Expected codes [{_c.Levels}, {frames}].", nameof(codes));
        if (frames < 3)
            throw new ArgumentException("SoundStorm training needs at least three frames (a prompt and a masked span).", nameof(codes));

        int level = random.Next(_c.Levels);
        int firstMaskable;
        if (_c.Prompt == SoundStormPromptSampling.PhemeReference)
            firstMaskable = 1 + random.Next(frames - 2);                                  // randint(1, min_len − 1)
        else
            firstMaskable = random.Next(frames) + 1;                                      // t ~ U{0, T − 1}; mask t′ > t
        double ratio = CosineSchedule(random.NextDouble());
        var masked = new bool[frames];
        bool any = false;
        for (int t = firstMaskable; t < frames; t++)
            any |= masked[t] = random.NextDouble() < ratio;
        if (!any && _c.Prompt == SoundStormPromptSampling.PhemeReference)
            masked[firstMaskable + random.Next(frames - firstMaskable)] = true;
        var positions = Enumerable.Range(0, frames)
            .Where(t => masked[t] && codes[level, t] != _c.PaddingRow && codes[level, t] < _c.HeadOutputs).ToArray();

        // Finer levels are masked after the prompt (Y>t,>q) and read inside it when the configuration says so.
        var visible = Visibility(frames, level, masked, firstMaskable);
        return () =>
        {
            var logits = Logits(codes, visible, conditioning, level, global, training, random);                 // [T, V]
            var logProbabilities = _engine.TensorLogSoftmax(logits, axis: -1);
            var target = new Tensor<T>(logits._shape);
            foreach (int t in positions) target[t, codes[level, t]] = NumOps.FromDouble(-1.0 / positions.Length);
            return _engine.ReduceSum(_engine.TensorMultiply(logProbabilities, target), new[] { 0, 1 }, keepDims: false);
        };
    }

    // ---------------------------------------------------------------- decoding (§3.3)

    /// <summary>
    /// Fills <paramref name="codes"/> <c>[Levels, frames]</c> from <paramref name="promptFrames"/> on, coarse level first;
    /// positions still to decode hold <see cref="MaskId"/>. <paramref name="iterations"/>(level) is 1 for one greedy
    /// pass, or the number of MaskGIT iterations.
    /// </summary>
    public void Decode(int[,] codes, int[] conditioning, int promptFrames, Func<int, int> iterations, Tensor<T>? global,
        Random random)
    {
        int frames = conditioning.Length;
        for (int level = 0; level < _c.Levels; level++)
        {
            int steps = iterations(level);
            if (steps <= 1)
            {
                // The level is read wherever it already holds a code: Pheme's reference re-decodes the prompt's last
                // frame (start_t) while still reading its code.
                var unknown = new bool[frames];
                for (int t = 0; t < frames; t++) unknown[t] = codes[level, t] == MaskId;
                var visible = Visibility(frames, level, unknown, promptFrames);
                var logits = Logits(codes, visible, conditioning, level, global, training: false, random);
                for (int t = promptFrames; t < frames; t++) codes[level, t] = ArgMax(logits, t);
                continue;
            }
            MaskGit(codes, conditioning, promptFrames, level, steps, global, random);
        }
    }

    private static int ArgMax(Tensor<T> logits, int t)
    {
        int best = 0;
        double bestValue = double.NegativeInfinity;
        for (int v = 0; v < logits.Shape[1]; v++)
        {
            double value = NumOps.ToDouble(logits[t, v]);
            if (value > bestValue) { bestValue = value; best = v; }
        }
        return best;
    }

    // MaskGIT (Chang et al. 2022) on one level, as Pheme's TTSConformer.multi_step_inference runs it.
    private void MaskGit(int[,] codes, int[] conditioning, int promptFrames, int level, int steps, Tensor<T>? global,
        Random random)
    {
        int frames = conditioning.Length;
        var unknownNow = new bool[frames];
        for (int t = promptFrames; t < frames; t++)
        {
            unknownNow[t] = true;
            codes[level, t] = MaskId;
        }
        int maskedAtStart = frames - promptFrames;
        var final = new int[frames];
        for (int t = 0; t < frames; t++) final[t] = codes[level, t];
        for (int step = 0; step < steps; step++)
        {
            // Under zeroed masking a position is unknown while it holds the mask id, even when a step sampled that id
            // (the padding row is a head output too), as Pheme's reference tests current == mask.
            if (_c.MaskInput == SoundStormMaskInput.Zero)
                for (int t = 0; t < frames; t++) unknownNow[t] = codes[level, t] == MaskId;
            var visible = Visibility(frames, level, unknownNow, promptFrames);
            var logits = Logits(codes, visible, conditioning, level, global, training: false, random);
            int vocabulary = logits.Shape[1];
            var sampled = new int[frames];
            var confidence = new double[frames];
            int unknown = 0;
            double ratio = (step + 1.0) / steps, temperature = 1.0 * (1.0 - ratio);
            bool greedy = _c.GreedyLastIteration && step == steps - 1;
            for (int t = 0; t < frames; t++)
            {
                var row = new double[vocabulary];
                double max = double.NegativeInfinity;
                for (int v = 0; v < vocabulary; v++) max = Math.Max(max, row[v] = NumOps.ToDouble(logits[t, v]));
                double total = 0;
                for (int v = 0; v < vocabulary; v++) total += row[v] = Math.Exp(row[v] - max);
                int choice;
                if (greedy)
                {
                    choice = 0;
                    for (int v = 1; v < vocabulary; v++) if (row[v] > row[choice]) choice = v;
                }
                else
                {
                    // Categorical(logits).sample()
                    double draw = random.NextDouble() * total;
                    choice = vocabulary - 1;
                    for (int v = 0; v < vocabulary; v++)
                    {
                        draw -= row[v];
                        if (draw <= 0) { choice = v; break; }
                    }
                }
                bool isUnknown = unknownNow[t];
                sampled[t] = isUnknown ? choice : codes[level, t];
                if (isUnknown && !greedy)
                {
                    unknown++;
                    double gumbel = -Math.Log(-Math.Log(Math.Max(1e-20, random.NextDouble())));
                    confidence[t] = Math.Log(row[sampled[t]] / total) + temperature * gumbel;
                }
                else
                {
                    if (isUnknown) unknown++;
                    confidence[t] = double.PositiveInfinity;
                }
            }
            final = sampled;
            if (greedy) break;
            int toMask = (int)Math.Floor(maskedAtStart * CosineSchedule(ratio));
            toMask = Math.Max(1, Math.Min(unknown - 1, toMask));
            // mask_by_random_topk: everything below the toMask-th smallest confidence is masked again.
            var order = Enumerable.Range(0, frames).OrderBy(t => confidence[t]).ToArray();
            double cutOff = confidence[order[Math.Min(toMask, frames - 1)]];
            for (int t = 0; t < frames; t++)
            {
                unknownNow[t] = confidence[t] < cutOff;
                codes[level, t] = unknownNow[t] ? MaskId : sampled[t];
            }
        }
        for (int t = 0; t < frames; t++) codes[level, t] = final[t];
    }

    // ---------------------------------------------------------------- weights

    /// <summary>Loads the network from a PyTorch state dict laid out as Pheme's <c>TTSConformer</c>
    /// (<c>embedding.{l}</c>, the conditioning table, the global projection, <c>conformer</c>, <c>heads.{l}</c>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string conditioningName, string globalName)
    {
        int dim = _c.Conformer.Dim;
        for (int l = 0; l < LevelEmbeddings.Count; l++)
            LevelEmbeddings[l].LoadTable(read($"embedding.{l}.weight", new[] { _c.EmbeddingRows, dim }));
        ConditioningEmbedding.LoadTable(read($"{conditioningName}.weight", new[] { _c.ConditioningCodes, dim }));
        if (GlobalProjection is not null)
            TorchParameters.Linear(_engine, GlobalProjection, _c.GlobalConditioningDim, dim,
                read($"{globalName}.weight", new[] { dim, _c.GlobalConditioningDim }), read($"{globalName}.bias", new[] { dim }));
        Conformer.LoadTorchWeights(_engine, "conformer", read);
        for (int l = 0; l < Heads.Count; l++)
            TorchParameters.Linear(_engine, Heads[l], dim, _c.HeadOutputs, read($"heads.{l}.weight", new[] { _c.HeadOutputs, dim }),
                read($"heads.{l}.bias", new[] { _c.HeadOutputs }));
    }
}
