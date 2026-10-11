using AiDotNet.Interfaces;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Transfusion's training objective: <c>L = L_LM + lambda * L_DDPM</c> (Zhou et al. 2024,
/// arXiv:2408.11039, Eq. 4).
/// </summary>
/// <remarks>
/// <para>
/// The paper trains ONE transformer on a sequence that mixes discrete text tokens with the
/// continuous latent patches of an image, and applies a different loss to each modality:
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
/// image (bidirectional, the image being last). This objective therefore runs the shared
/// trunk twice, once causal for the LM term and once bidirectional for the diffusion term,
/// and reads each term from the pass whose mask the paper specifies for those rows. It is
/// the same weights and the same sequence both times, so the gradients add exactly as
/// Eq. 4 requires; one fused pass would need a per-row block mask inside the attention
/// layer, which the shared layer does not take yet.
/// </para>
/// <para><b>Noise schedule.</b> The cosine schedule of Nichol and Dhariwal (2021), which the
/// paper adopts: <c>sqrt(alphaBar_t) ~ cos(t/T * pi/2)</c> with their <c>s = 0.008</c>
/// offset, over <see cref="TransfusionOptions.DiffusionTimesteps"/> (1000) timesteps.
/// </para>
/// <para><b>Patchification</b> is the paper's "simple linear layer" over flattened patches,
/// with the image surrounded by BOI / EOI marker rows as in Sec. 3, Data Representation.
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

    /// <summary>Builds the mixed-modal training heads (native mode only).</summary>
    private void InitializeTrainingHeads()
    {
        int tokens = Math.Max(1, _options.NumVisualTokens);
        int grid = Math.Max(1, (int)Math.Round(Math.Sqrt(tokens)));
        _patchSide = Math.Max(1, _options.ImageSize / grid);
        _patchDim = 3 * _patchSide * _patchSide;

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
    /// One optimizer step on the paper's objective. <paramref name="expected"/> is the caption's
    /// token ids when it is rank-1; any other shape carries no text, so the sequence is the image
    /// alone and the LM term has no tokens to score (the diffusion term is unchanged).
    /// </summary>
    internal T TrainMixedModal(Tensor<T> image, Tensor<T> expected)
    {
        var embedding = _textEmbedding ?? throw new InvalidOperationException(
            "Transfusion's mixed-modal training heads are only built in native mode.");

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
            (_, _) => MixedModalObjective(caption, patches, noise, timestep),
            _optimizer);
    }

    private int ClampTokenId(int id) => Math.Min(Math.Max(id, 0), _options.VocabSize - 1);

    /// <summary>
    /// Test-only hook: the objective's value for one fixed timestep, with the noise drawn from a
    /// generator seeded by that timestep, and no optimizer step. Two calls that differ only in
    /// <see cref="TransfusionOptions.DiffusionLossWeight"/> therefore differ only through
    /// <c>lambda</c>, which is what makes <c>L = L_LM + lambda * L_DDPM</c> checkable.
    /// </summary>
    internal double MixedObjectiveValueForTest(Tensor<T> image, Tensor<T> expected, int timestep)
    {
        int[] caption = expected.Rank == 1
            ? NeuralNetworkBase<T>.TokenIds(expected, ClampTokenId, _options.MaxSequenceLength)
            : [];
        var patches = Patchify(PreprocessImage(image));
        var noise = GaussianLike(patches, RandomHelper.CreateSeededRandom(timestep));
        var loss = MixedModalObjective(caption, patches, noise, timestep);
        return NumOps.ToDouble(loss[0]);
    }

    /// <summary>
    /// <c>L_LM + lambda * L_DDPM</c> over one mixed text-image sequence, on the tape.
    /// </summary>
    private Tensor<T> MixedModalObjective(int[] caption, Tensor<T> patches, Tensor<T> noise, int timestep)
    {
        int d = _options.DecoderDim;
        int patchCount = patches.Shape[1];
        double alphaBar = CosineAlphaBar(timestep, Math.Max(1, _options.DiffusionTimesteps));

        // x_t = sqrt(alphaBar) * x_0 + sqrt(1 - alphaBar) * eps   (Ho et al. 2020, Eq. 2)
        var noisy = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(patches, NumOps.FromDouble(Math.Sqrt(alphaBar))),
            Engine.TensorMultiplyScalar(noise, NumOps.FromDouble(Math.Sqrt(Math.Max(0.0, 1.0 - alphaBar)))));

        // [1, P, d] patch rows, conditioned on the timestep the way every DDPM denoiser is.
        var patchRows = Engine.TensorAdd(
            _patchProjection!.Forward(noisy),
            TimestepRows(timestep, patchCount, d));

        // Marker rows: the paper surrounds the image with BOI / EOI tokens.
        var boi = embeddedToken(BoiTokenId);
        var eoi = embeddedToken(EoiTokenId);

        Tensor<T>? textRows = caption.Length > 0 ? _textEmbedding!.Forward(TokenRow(caption)) : null;
        var sequence = textRows is null
            ? Engine.TensorConcatenate([boi, patchRows, eoi], axis: 1)
            : Engine.TensorConcatenate([textRows, boi, patchRows, eoi], axis: 1);
        sequence = Engine.TensorAdd(sequence, SinusoidalPositions(sequence.Shape[1], d));

        int textLength = caption.Length;
        var loss = Engine.TensorMultiplyScalar(
            DiffusionTerm(sequence, textLength, patchCount, noise),
            NumOps.FromDouble(_options.DiffusionLossWeight));

        // The LM term needs at least two tokens: row t predicts token t + 1.
        if (textLength >= 2)
            loss = Engine.TensorAdd(LanguageModelTerm(sequence, caption), loss);

        return loss;

        Tensor<T> embeddedToken(int id) => _textEmbedding!.Forward(TokenRow([ClampTokenId(id)]));
    }

    /// <summary>Mean next-token cross-entropy over the caption, read under the causal mask.</summary>
    private Tensor<T> LanguageModelTerm(Tensor<T> sequence, int[] caption)
    {
        var hidden = RunTrunk(sequence, causal: true);
        // Rows 0 .. S-2 predict caption[1 .. S-1].
        var textHidden = Engine.TensorSlice(
            hidden, [0, 0, 0], [1, caption.Length - 1, _options.DecoderDim]);
        var logits = Engine.Reshape(
            _lmHead!.Forward(textHidden), [caption.Length - 1, _options.VocabSize]);
        return NeuralNetworkBase<T>.TokenCrossEntropy(logits, caption[1..]);
    }

    /// <summary>
    /// Mean squared error between the predicted and the true noise over the image's patches,
    /// read under the bidirectional mask (Ho et al. 2020, Eq. 3).
    /// </summary>
    private Tensor<T> DiffusionTerm(Tensor<T> sequence, int textLength, int patchCount, Tensor<T> noise)
    {
        var hidden = RunTrunk(sequence, causal: false);
        // The patch rows sit after the text and the BOI marker.
        var patchHidden = Engine.TensorSlice(
            hidden, [0, textLength + 1, 0], [1, patchCount, _options.DecoderDim]);
        var predicted = _noiseHead!.Forward(patchHidden);
        var difference = Engine.TensorSubtract(predicted, noise);
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
    private Tensor<T> Patchify(Tensor<T> image)
    {
        var shape = image._shape;
        int channels = shape.Length >= 3 ? shape[^3] : 3;
        int height = shape[^2];
        int width = shape[^1];
        int rows = Math.Max(1, height / _patchSide);
        int columns = Math.Max(1, width / _patchSide);
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
                            patches[0, patch, slot++] =
                                index < source.Length ? source[index] : NumOps.Zero;
                        }
                    }
                }
            }
        }
        return patches;
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
                double frequency = Math.Pow(10000.0, -2.0 * (i / 2) / width);
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
