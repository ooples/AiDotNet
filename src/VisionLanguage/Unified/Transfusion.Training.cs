using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Transfusion's training objective: <c>L = L_LM + lambda * L_DDPM</c> (Zhou et al. 2024,
/// arXiv:2408.11039, Eq. 4), and the sampler that reads back what it trains.
/// </summary>
/// <remarks>
/// <para>
/// The paper trains ONE transformer on a sequence that mixes discrete text tokens with the
/// continuous patches of an image, and applies a different loss to each modality:
/// next-token cross-entropy on the text positions, and the DDPM noise-prediction loss
/// (Ho et al. 2020) on the image patches, combined with a balancing coefficient
/// <c>lambda</c> -- "We set the lambda coefficient in the Transfusion objective (Equation 4)
/// to 5 following preliminary experiments" (Sec. 4.1, Optimization).
/// </para>
/// <para><b>Attention.</b> Transfusion "appl[ies] causal attention to every element in the
/// sequence, and bidirectional attention within the elements of each individual image"
/// (Sec. 3, Transfusion Attention), and its ablation finds the intra-image bidirectional
/// attention important. With the paper's training layout -- text first, then the image block
/// -- each loss reads rows under exactly one of those two masks: a text row sees only
/// earlier text (causal), and a patch row sees all of the text plus every patch of its own
/// image. "Within an image" is a bound in both directions, so the bidirectional pass runs
/// over <c>[text, BOI, patches]</c> and stops there: a patch row must not see the EOI row
/// that FOLLOWS its image. This objective therefore runs the shared trunk twice, once causal
/// over the full sequence for the LM term and once bidirectional over that prefix for the
/// diffusion term, and reads each term from the pass whose mask the paper specifies for
/// those rows. It is the same weights both times, so the gradients add exactly as Eq. 4
/// requires; one fused pass would need a per-row block mask inside the attention layer,
/// which the shared layer does not take yet.
/// </para>
/// <para><b>Markers.</b> The paper surrounds each image with BOI / EOI tokens (Sec. 3, Data
/// Representation) and computes "LM loss ... per token", with one exception stated in its
/// footnote 6: "When the input is a BOI token, we do not compute any loss." Both markers are
/// therefore scored as tokens -- the last caption row predicts BOI, the last patch row
/// predicts EOI -- and the BOI row itself predicts nothing. Without this the two marker
/// embeddings would be registered, counted and serialized but never receive a gradient.
/// </para>
/// <para><b>Image representation.</b> Patches are pixel patches, flattened and projected by
/// the paper's "simple linear layer". The paper's canonical setting patchifies a pretrained
/// VAE's latents instead -- "While our canonical setting uses a VAE following latent
/// diffusion models, we were also able to demonstrate Transfusion using raw pixel
/// representations in preliminary experiments" (Sec. 3, footnote 1) -- and the objective is
/// representation-agnostic between the two: it needs <c>x_0</c>, not specifically a latent.
/// A latent path here would have to invent a VAE this model does not own, and a randomly
/// initialized VAE's latents carry less signal than the pixels do.
/// </para>
/// <para><b>Noise schedule.</b> The cosine schedule of Nichol and Dhariwal (2021), which the
/// paper adopts: <c>sqrt(alphaBar_t) ~ cos(t/T * pi/2)</c> with their <c>s = 0.008</c>
/// offset, over <see cref="TransfusionOptions.DiffusionTimesteps"/> (1000) timesteps.
/// </para>
/// </remarks>
public partial class Transfusion<T>
{
    // Nichol & Dhariwal (2021) cosine-schedule offset, as cited by Zhou et al. 2024.
    private const double CosineScheduleOffset = 0.008;

    // Parameter-carrying pieces of the mixed-modal path. They are NOT appended to Layers:
    // Layers is the sequential vision/decoder chain that PredictCore walks, and an embedding
    // or a vocabulary head in the middle of that chain would break Predict. The parameter
    // generator surfaces layer FIELDS through GetExtraTrainableLayers, so these are still
    // counted, trained, serialized and cloned.
    private EmbeddingLayer<T>? _textEmbedding;
    private DenseLayer<T>? _patchProjection;
    private DenseLayer<T>? _lmHead;
    private DenseLayer<T>? _noiseHead;
    private Random? _trainingRandom;
    private int _patchSide;
    private int _patchDim;
    private int _patchGridSide;

    /// <summary>Builds the mixed-modal training heads (native mode only).</summary>
    private void InitializeTrainingHeads()
    {
        int tokens = Math.Max(1, _options.NumVisualTokens);
        int grid = Math.Max(1, (int)Math.Round(Math.Sqrt(tokens)));
        _patchSide = Math.Max(1, _options.ImageSize / grid);
        _patchDim = 3 * _patchSide * _patchSide;
        // Ceiling, not floor: when the patch side does not divide the image, a floored grid
        // drops the bottom and right border pixels (ImageSize 32 with 9 visual tokens gives a
        // 10-pixel patch and loses two rows and two columns). One extra row and column cover
        // the whole image, with the overhanging part of the last patch zero-padded.
        _patchGridSide = Math.Max(1, (_options.ImageSize + _patchSide - 1) / _patchSide);

        // IdentityActivation implements both activation interfaces, so the element-wise one is
        // named explicitly: the untyped new() makes the DenseLayer constructor ambiguous.
        IActivationFunction<T> identity = new IdentityActivation<T>();
        _textEmbedding = new EmbeddingLayer<T>(_options.VocabSize, _options.DecoderDim);
        _patchProjection = new DenseLayer<T>(_options.DecoderDim, identity);
        _lmHead = new DenseLayer<T>(_options.VocabSize, identity);
        _noiseHead = new DenseLayer<T>(_patchDim, identity);
        _trainingRandom = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// The four parameter-carrying heads of the mixed-modal path as non-null references, so the
    /// paths that use them neither repeat a null check per use nor suppress one.
    /// </summary>
    private (EmbeddingLayer<T> Text, DenseLayer<T> Patch, DenseLayer<T> Lm, DenseLayer<T> Noise)
        MixedModalHeads()
    {
        if (_textEmbedding is null || _patchProjection is null
            || _lmHead is null || _noiseHead is null)
        {
            throw new InvalidOperationException(
                "Transfusion's mixed-modal heads are only built in native mode.");
        }

        return (_textEmbedding, _patchProjection, _lmHead, _noiseHead);
    }

    /// <summary>
    /// One optimizer step on the paper's objective. <paramref name="expected"/> is the caption's
    /// token ids when it is rank-1; any other shape carries no text, so the sequence is the image
    /// alone and the LM term scores only the EOI marker.
    /// </summary>
    internal T TrainMixedModal(Tensor<T> image, Tensor<T> expected)
    {
        var heads = MixedModalHeads();
        int[] caption = expected.Rank == 1
            ? NeuralNetworkBase<T>.TokenIds(expected, ClampTokenId, _options.MaxSequenceLength)
            : [];

        // The timestep and the noise are drawn ONCE per step, not inside the objective: a
        // line-search optimizer re-evaluates the closure, and a fresh draw per evaluation would
        // compare two different objectives (TrainWithCustomObjective's recomputation contract).
        var rng = _trainingRandom ?? Random;
        int timestep = rng.Next(Math.Max(1, _options.DiffusionTimesteps));
        var patches = Patchify(PreprocessImage(image));
        var noise = GaussianLike(patches, rng);

        return TrainWithCustomObjective(
            image,
            expected,
            (_, _) => MixedModalObjective(heads, caption, patches, noise, timestep),
            _optimizer);
    }

    private int ClampTokenId(int id) => Math.Min(Math.Max(id, 0), _options.VocabSize - 1);

    /// <summary>Gets the kind of objective this model optimizes: two losses at once.</summary>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind =>
        TrainingObjectiveKind.MultiTask;

    /// <summary>The supplied target is the caption when it is rank-1, and carries no text
    /// otherwise; either way the objective reads it directly.</summary>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(
        Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <summary>
    /// <c>L = L_LM + lambda * L_DDPM</c> averaged over a fixed quadrature of three timesteps with
    /// the noise seeded per timestep, under a no-gradient scope and with no optimizer step.
    /// </summary>
    /// <remarks>
    /// A training step draws ONE random timestep, so two reported step losses cannot show a trend;
    /// this returns the same value for the same parameters, which is what makes a before/after
    /// comparison meaningful. It is also the only honest loss for this model: <c>Train</c>
    /// minimizes the mixed-modal objective and never regresses <c>Predict(input)</c> onto the
    /// target, so measuring that regression would measure a quantity nothing optimizes.
    /// </remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException(
                "Transfusion's training objective is only defined in native mode.");
        }

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int timesteps = Math.Max(1, _options.DiffusionTimesteps);
        double total = 0.0;
        for (int quarter = 1; quarter <= 3; quarter++)
            total += EvaluateMixedObjective(input, target, quarter * timesteps / 4);
        return NumOps.FromDouble(total / 3.0);
    }

    /// <summary>
    /// The objective's value for one fixed timestep, with the noise drawn from a generator seeded
    /// by that timestep, and no optimizer step. Two calls that differ only in
    /// <see cref="TransfusionOptions.DiffusionLossWeight"/> therefore differ only through
    /// <c>lambda</c>, which is what makes <c>L = L_LM + lambda * L_DDPM</c> checkable.
    /// </summary>
    internal double EvaluateMixedObjective(Tensor<T> image, Tensor<T> expected, int timestep)
    {
        var heads = MixedModalHeads();
        int[] caption = expected.Rank == 1
            ? NeuralNetworkBase<T>.TokenIds(expected, ClampTokenId, _options.MaxSequenceLength)
            : [];
        var patches = Patchify(PreprocessImage(image));
        var noise = GaussianLike(patches, RandomHelper.CreateSeededRandom(timestep));
        var loss = MixedModalObjective(heads, caption, patches, noise, timestep);
        return NumOps.ToDouble(loss[0]);
    }

    /// <summary>
    /// Test-only hook: the parameters of the two output heads on their own. The whole-model
    /// vector is dominated by the shared trunk, so it cannot tell "the language head moved"
    /// from "some trunk weight moved".
    /// </summary>
    internal (Vector<T> Language, Vector<T> Noise) MixedModalHeadParametersForTest()
    {
        var heads = MixedModalHeads();
        return (heads.Lm.GetParameters(), heads.Noise.GetParameters());
    }
    /// <summary>
    /// <c>L_LM + lambda * L_DDPM</c> over one mixed text-image sequence, on the tape.
    /// </summary>
    private Tensor<T> MixedModalObjective(
        (EmbeddingLayer<T> Text, DenseLayer<T> Patch, DenseLayer<T> Lm, DenseLayer<T> Noise) heads,
        int[] caption,
        Tensor<T> patches,
        Tensor<T> noise,
        int timestep)
    {
        if (!_options.EnableDiffusionLoss && caption.Length == 0)
        {
            throw new InvalidOperationException(
                "EnableDiffusionLoss is off and the target carries no text tokens, so Transfusion's "
                    + "objective has no term to minimize. Pass a rank-1 target of caption token ids, "
                    + "or leave EnableDiffusionLoss on.");
        }

        double alphaBar = CosineAlphaBar(timestep, Math.Max(1, _options.DiffusionTimesteps));

        // x_t = sqrt(alphaBar) * x_0 + sqrt(1 - alphaBar) * eps   (Ho et al. 2020, Eq. 2)
        var noisy = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(patches, NumOps.FromDouble(Math.Sqrt(alphaBar))),
            Engine.TensorMultiplyScalar(
                noise, NumOps.FromDouble(Math.Sqrt(Math.Max(0.0, 1.0 - alphaBar)))));

        Tensor<T>? textRows = caption.Length > 0
            ? heads.Text.Forward(TokenRow(caption))
            : null;

        var loss = LanguageModelTerm(heads, textRows, noisy, timestep, caption);
        if (_options.EnableDiffusionLoss)
        {
            var predicted = PredictPatchNoise(heads, textRows, noisy, timestep);
            loss = Engine.TensorAdd(
                loss,
                Engine.TensorMultiplyScalar(
                    MeanSquaredError(predicted, noise),
                    NumOps.FromDouble(_options.DiffusionLossWeight)));
        }

        return loss;
    }

    /// <summary>
    /// Mean next-token cross-entropy over the caption and the two image markers, read under the
    /// causal mask: text row <c>t</c> predicts token <c>t + 1</c>, the last text row predicts BOI,
    /// and the last patch row predicts EOI. The BOI row predicts nothing -- "When the input is a
    /// BOI token, we do not compute any loss" (Sec. 3, Objective, footnote 6).
    /// </summary>
    private Tensor<T> LanguageModelTerm(
        (EmbeddingLayer<T> Text, DenseLayer<T> Patch, DenseLayer<T> Lm, DenseLayer<T> Noise) heads,
        Tensor<T>? textRows,
        Tensor<T> noisy,
        int timestep,
        int[] caption)
    {
        int d = _options.DecoderDim;
        int patchCount = noisy.Shape[1];
        int textLength = caption.Length;
        var patchRows = PatchRows(heads, noisy, timestep);
        var sequence = textRows is null
            ? Engine.TensorConcatenate(
                [EmbeddedToken(heads, BoiTokenId), patchRows, EmbeddedToken(heads, EoiTokenId)],
                axis: 1)
            : Engine.TensorConcatenate(
                [textRows, EmbeddedToken(heads, BoiTokenId), patchRows,
                 EmbeddedToken(heads, EoiTokenId)],
                axis: 1);
        var hidden = RunTrunk(WithPositions(sequence), causal: true);

        // The rows that predict a token, in sequence order: every text row, then the last patch
        // row. Row textLength is the BOI input and is skipped; the EOI row predicts nothing
        // because nothing follows the image in a training sequence.
        var lastPatchRow = Engine.TensorSlice(hidden, [0, textLength + patchCount, 0], [1, 1, d]);
        var scored = textLength > 0
            ? Engine.TensorConcatenate(
                [Engine.TensorSlice(hidden, [0, 0, 0], [1, textLength, d]), lastPatchRow], axis: 1)
            : lastPatchRow;
        var logits = Engine.Reshape(
            heads.Lm.Forward(scored), [textLength + 1, _options.VocabSize]);

        var targets = new int[textLength + 1];
        for (int i = 1; i < textLength; i++) targets[i - 1] = caption[i];
        if (textLength > 0) targets[textLength - 1] = ClampTokenId(BoiTokenId);
        targets[textLength] = ClampTokenId(EoiTokenId);
        return NeuralNetworkBase<T>.TokenCrossEntropy(logits, targets);
    }

    /// <summary>
    /// The noise the trunk predicts for one image's patches, read under the bidirectional mask
    /// (Ho et al. 2020, Eq. 3). Training and generation both go through here, so the heads a
    /// training step moves are the heads the sampler reads, and the sequence stops at the last
    /// patch so that no patch row attends to the EOI row that follows the image.
    /// </summary>
    private Tensor<T> PredictPatchNoise(
        (EmbeddingLayer<T> Text, DenseLayer<T> Patch, DenseLayer<T> Lm, DenseLayer<T> Noise) heads,
        Tensor<T>? textRows,
        Tensor<T> noisy,
        int timestep)
    {
        int d = _options.DecoderDim;
        int patchCount = noisy.Shape[1];
        int textLength = textRows is null ? 0 : textRows.Shape[1];
        var patchRows = PatchRows(heads, noisy, timestep);
        var sequence = textRows is null
            ? Engine.TensorConcatenate([EmbeddedToken(heads, BoiTokenId), patchRows], axis: 1)
            : Engine.TensorConcatenate(
                [textRows, EmbeddedToken(heads, BoiTokenId), patchRows], axis: 1);
        var hidden = RunTrunk(WithPositions(sequence), causal: false);
        var patchHidden = Engine.TensorSlice(
            hidden, [0, textLength + 1, 0], [1, patchCount, d]);
        return heads.Noise.Forward(patchHidden);
    }

    /// <summary>
    /// Samples one image's patches from noise with the schedule, conditioning, patchifier and
    /// noise head the objective trains: DDIM (Song et al. 2021) with <c>eta = 0</c> over a
    /// strided subset of the trained cosine schedule.
    /// </summary>
    internal Tensor<T> SampleImagePatches(int[] caption, int steps)
    {
        var heads = MixedModalHeads();
        int timesteps = Math.Max(1, _options.DiffusionTimesteps);
        steps = Math.Max(1, Math.Min(steps, timesteps));
        var rng = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        Tensor<T>? textRows = caption.Length > 0 ? heads.Text.Forward(TokenRow(caption)) : null;
        int patchCount = _patchGridSide * _patchGridSide;
        var x = GaussianLike(new Tensor<T>([1, patchCount, _patchDim]), rng);

        for (int step = steps; step >= 1; step--)
        {
            int t = (step * timesteps / steps) - 1;
            int previous = ((step - 1) * timesteps / steps) - 1;
            double alphaBar = CosineAlphaBar(t, timesteps);
            double alphaBarPrevious = previous < 0 ? 1.0 : CosineAlphaBar(previous, timesteps);

            var predicted = PredictPatchNoise(heads, textRows, x, Math.Max(0, t));
            // x_0 = (x_t - sqrt(1 - alphaBar) * eps) / sqrt(alphaBar), then
            // x_{t-1} = sqrt(alphaBarPrev) * x_0 + sqrt(1 - alphaBarPrev) * eps.
            var denoised = Engine.TensorMultiplyScalar(
                Engine.TensorSubtract(
                    x,
                    Engine.TensorMultiplyScalar(
                        predicted, NumOps.FromDouble(Math.Sqrt(Math.Max(0.0, 1.0 - alphaBar))))),
                NumOps.FromDouble(1.0 / Math.Sqrt(Math.Max(alphaBar, 1e-12))));
            x = Engine.TensorAdd(
                Engine.TensorMultiplyScalar(
                    denoised, NumOps.FromDouble(Math.Sqrt(alphaBarPrevious))),
                Engine.TensorMultiplyScalar(
                    predicted,
                    NumOps.FromDouble(Math.Sqrt(Math.Max(0.0, 1.0 - alphaBarPrevious)))));
        }

        return x;
    }

    /// <summary>The patchifier's rows for one image, conditioned on the timestep.</summary>
    private Tensor<T> PatchRows(
        (EmbeddingLayer<T> Text, DenseLayer<T> Patch, DenseLayer<T> Lm, DenseLayer<T> Noise) heads,
        Tensor<T> noisy,
        int timestep) =>
        Engine.TensorAdd(
            heads.Patch.Forward(noisy),
            TimestepRows(timestep, noisy.Shape[1], _options.DecoderDim));

    /// <summary>One marker row: the token's embedding, as a <c>[1, 1, d]</c> sequence row.</summary>
    private Tensor<T> EmbeddedToken(
        (EmbeddingLayer<T> Text, DenseLayer<T> Patch, DenseLayer<T> Lm, DenseLayer<T> Noise) heads,
        int id) =>
        heads.Text.Forward(TokenRow([ClampTokenId(id)]));

    /// <summary>Adds the sinusoidal position encoding for the sequence's own length.</summary>
    private Tensor<T> WithPositions(Tensor<T> sequence) =>
        Engine.TensorAdd(
            sequence, SinusoidalPositions(sequence.Shape[1], _options.DecoderDim));

    /// <summary>Mean squared error between two equally shaped tensors, on the tape.</summary>
    private Tensor<T> MeanSquaredError(Tensor<T> predicted, Tensor<T> target)
    {
        var difference = Engine.TensorSubtract(predicted, target);
        var squared = Engine.TensorMultiply(difference, difference);
        return Engine.TensorMultiplyScalar(
            Engine.ReduceSum(squared, null), NumOps.FromDouble(1.0 / squared.Length));
    }
    /// <summary>
    /// The shared transformer trunk: every layer after the vision patch embedding, run over the
    /// mixed sequence with the mask the caller's loss term requires. The flag is restored so a
    /// later <c>Predict</c> sees the attention configuration it was built with.
    /// </summary>
    private Tensor<T> RunTrunk(Tensor<T> sequence, bool causal)
    {
        var attention = new List<(MultiHeadAttentionLayer<T> Layer, bool Previous)>();
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
        {
            if (Layers[i] is MultiHeadAttentionLayer<T> mha)
            {
                attention.Add((mha, mha.UseCausalMask));
                mha.UseCausalMask = causal;
            }
        }

        try
        {
            var current = sequence;
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
                current = Layers[i].Forward(current);
            return current;
        }
        finally
        {
            foreach (var (layer, previous) in attention)
                layer.UseCausalMask = previous;
        }
    }

    /// <summary>Flattens an image into <c>[1, P, 3 * patchSide * patchSide]</c> row-major patches.</summary>
    /// <remarks>
    /// The grid is ceiled, so every pixel of the image lands in exactly one patch even when the
    /// patch side does not divide the image; the part of a border patch that hangs over the edge
    /// is zero-padded. A floored grid silently dropped those border rows and columns.
    /// </remarks>
    private Tensor<T> Patchify(Tensor<T> image)
    {
        var shape = image._shape;
        int channels = shape.Length >= 3 ? shape[^3] : 3;
        int height = shape[^2];
        int width = shape[^1];
        if (height != _options.ImageSize || width != _options.ImageSize)
        {
            // The grid the objective builds comes from the image in hand, while the sampler and
            // Unpatchify use the configured square grid, so a differently sized image would train
            // one patch sequence and generate from another without ever saying so.
            throw new ArgumentException(
                $"Transfusion is configured for {_options.ImageSize}x{_options.ImageSize} images; "
                    + $"got {height}x{width}. Resize the image before training on it.",
                nameof(image));
        }

        int rows = Math.Max(1, (height + _patchSide - 1) / _patchSide);
        int columns = Math.Max(1, (width + _patchSide - 1) / _patchSide);
        var patches = new Tensor<T>([1, rows * columns, _patchDim]);
        var source = image.ToVector();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                int patch = (r * columns) + c;
                int slot = 0;
                for (int ch = 0; ch < Math.Min(3, channels); ch++)
                {
                    for (int y = 0; y < _patchSide; y++)
                    {
                        for (int x = 0; x < _patchSide; x++)
                        {
                            int pixelY = (r * _patchSide) + y;
                            int pixelX = (c * _patchSide) + x;
                            int index = (((ch * height) + pixelY) * width) + pixelX;
                            bool inside = pixelY < height && pixelX < width
                                && index < source.Length;
                            patches[0, patch, slot++] = inside ? source[index] : NumOps.Zero;
                        }
                    }
                }
            }
        }
        return patches;
    }

    /// <summary>
    /// The inverse of <see cref="Patchify"/>, resampled to <paramref name="outputSize"/>: the
    /// patch grid is written back to pixels, the normalization <see cref="Patchify"/> read the
    /// image under is undone, and the result is clamped to the <c>[0, 1]</c> range an image
    /// tensor carries. The flat interleaved-RGB layout is the one this model has always returned.
    /// </summary>
    private Tensor<T> Unpatchify(Tensor<T> patches, int outputSize)
    {
        double[] mean = _options.ImageMean.Length >= 3
            ? _options.ImageMean
            : [0.48145466, 0.4578275, 0.40821073];
        double[] std = _options.ImageStd.Length >= 3
            ? _options.ImageStd
            : [0.26862954, 0.26130258, 0.27577711];
        int imageSize = Math.Max(1, _options.ImageSize);
        int patchCount = patches.Shape[1];
        var result = new Tensor<T>([outputSize * outputSize * 3]);
        for (int y = 0; y < outputSize; y++)
        {
            for (int x = 0; x < outputSize; x++)
            {
                // Nearest-neighbour from the trained resolution; the grid may cover slightly
                // more than the image, and the padding past the border is never sampled.
                int sourceY = Math.Min(imageSize - 1, y * imageSize / outputSize);
                int sourceX = Math.Min(imageSize - 1, x * imageSize / outputSize);
                int patch = ((sourceY / _patchSide) * _patchGridSide) + (sourceX / _patchSide);
                if (patch >= patchCount) continue;
                int withinY = sourceY % _patchSide;
                int withinX = sourceX % _patchSide;
                int pixel = ((y * outputSize) + x) * 3;
                for (int ch = 0; ch < 3; ch++)
                {
                    int slot = (ch * _patchSide * _patchSide) + (withinY * _patchSide) + withinX;
                    double value = (NumOps.ToDouble(patches[0, patch, slot]) * std[ch]) + mean[ch];
                    result[pixel + ch] = NumOps.FromDouble(Math.Min(1.0, Math.Max(0.0, value)));
                }
            }
        }
        return result;
    }

    /// <summary>Standard normal noise shaped like <paramref name="like"/>.</summary>
    private Tensor<T> GaussianLike(Tensor<T> like, Random rng)
    {
        var noise = new Tensor<T>(like._shape);
        var span = noise.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
        {
            // Box-Muller: two uniforms in (0, 1] give one standard normal draw.
            double u1 = Math.Max(rng.NextDouble(), 1e-12);
            double u2 = rng.NextDouble();
            span[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return noise;
    }

    /// <summary>
    /// <c>alphaBar_t = f(t) / f(0)</c> with <c>f(t) = cos^2(((t/T) + s) / (1 + s) * pi/2)</c> --
    /// the cosine schedule of Nichol and Dhariwal (2021) that the paper adopts.
    /// </summary>
    internal static double CosineAlphaBar(int timestep, int timesteps)
    {
        double F(double step)
        {
            double value = Math.Cos(((step / timesteps) + CosineScheduleOffset)
                / (1.0 + CosineScheduleOffset) * (Math.PI / 2.0));
            return value * value;
        }

        return Math.Min(1.0, Math.Max(1e-8, F(timestep) / F(0.0)));
    }

    /// <summary>Token ids as the <c>[1, S]</c> tensor the embedding layer indexes with.</summary>
    private Tensor<T> TokenRow(int[] ids)
    {
        var row = new Tensor<T>([1, ids.Length]);
        for (int i = 0; i < ids.Length; i++) row[0, i] = NumOps.FromDouble(ids[i]);
        return row;
    }

    /// <summary>The sinusoidal timestep embedding, repeated over <paramref name="rows"/> rows.</summary>
    private Tensor<T> TimestepRows(int timestep, int rows, int width)
    {
        var embedded = new Tensor<T>([1, rows, width]);
        int half = Math.Max(1, width / 2);
        for (int i = 0; i < width; i++)
        {
            int pair = i % half;
            double frequency = Math.Exp(-Math.Log(10000.0) * pair / half);
            double value = i < half
                ? Math.Sin(timestep * frequency)
                : Math.Cos(timestep * frequency);
            var typed = NumOps.FromDouble(value);
            for (int r = 0; r < rows; r++) embedded[0, r, i] = typed;
        }
        return embedded;
    }

    /// <summary>Sinusoidal position encoding (Vaswani et al. 2017) for the mixed sequence.</summary>
    private Tensor<T> SinusoidalPositions(int length, int width)
    {
        var positions = new Tensor<T>([1, length, width]);
        for (int p = 0; p < length; p++)
        {
            for (int i = 0; i < width; i++)
            {
                // The exponent is 2 * floor(i / 2) / d_model: a sine and a cosine SHARE one
                // frequency, so the integer division is the formula, not a lost fraction.
                int pair = i / 2;
                double frequency = Math.Pow(10000.0, -2.0 * pair / width);
                double value = i % 2 == 0 ? Math.Sin(p * frequency) : Math.Cos(p * frequency);
                positions[0, p, i] = NumOps.FromDouble(value);
            }
        }
        return positions;
    }

    /// <summary>The beginning-of-image marker id (Sec. 3, Data Representation).</summary>
    private int BoiTokenId => _options.BeginningOfImageTokenId ?? Math.Max(0, _options.VocabSize - 2);

    /// <summary>The end-of-image marker id.</summary>
    private int EoiTokenId => _options.EndOfImageTokenId ?? Math.Max(0, _options.VocabSize - 1);
}