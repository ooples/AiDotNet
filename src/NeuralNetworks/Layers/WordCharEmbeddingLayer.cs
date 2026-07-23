using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Paper-faithful word + character embedding front-end for sequence-labeling NER models
/// (Lample et al., NAACL 2016, "Neural Architectures for Named Entity Recognition", §3).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This composite layer turns a sentence of integer token/character indices into the dense
/// per-token representation that the downstream word-level BiLSTM-CRF consumes. It implements
/// the two complementary embedding streams from Lample et al. (2016):
/// <list type="number">
/// <item><b>Word embeddings:</b> a learnable lookup table (initializable from pretrained GloVe
/// vectors) maps each word index to a dense vector — word identity and, when pretrained, semantic
/// similarity.</item>
/// <item><b>Character embeddings:</b> a small bidirectional LSTM reads each word's character
/// sequence and produces a fixed-size vector per word — morphology (capitalization, prefixes/suffixes,
/// the shape of out-of-vocabulary words) that the word lookup alone cannot represent.</item>
/// </list>
/// The two streams are concatenated per token, exactly as in the paper, giving a representation
/// of size <c>wordEmbeddingDim + charHiddenDim</c>.
/// </para>
/// <para>
/// <b>Implementation note — embeddings as linear layers.</b> An embedding lookup is mathematically a
/// linear projection of a one-hot index vector, so each lookup table is implemented as a no-frills
/// <see cref="DenseLayer{T}"/> applied to a one-hot encoding. This is deliberate: it routes the table
/// through the same gradient-tape-tracked matmul that trains every other dense layer, so the word and
/// character tables actually learn end-to-end. (A raw integer gather is faster for large vocabularies
/// but, in this engine's eager training path, does not propagate gradients back to the table — which
/// would leave the embeddings frozen at their random initialization.) For very large vocabularies the
/// one-hot matmul costs more memory than a gather; for typical NER vocabularies it is fine.
/// </para>
/// <para>
/// <b>Why this matters:</b> feeding a BiLSTM-CRF pre-computed, identity-only embeddings (or hash-derived
/// vectors) yields a model that can only memorize the training vocabulary, hallucinates on unseen words,
/// and — with a process-randomized hash — produces non-deterministic output. Owning real learnable word
/// and character tables is the standard fix.
/// </para>
/// <para>
/// <b>Input contract.</b> A single packed integer tensor of shape <c>[sequenceLength, 1 + maxWordLength]</c>
/// per sentence: column 0 holds the word index, columns 1..maxWordLength hold that word's character
/// indices (zero-padded). Index 0 is reserved for padding. Output shape is
/// <c>[sequenceLength, wordEmbeddingDim + charHiddenDim]</c>.
/// </para>
/// <para>
/// <b>For Beginners:</b> a network can't read text directly — words must become numbers. This layer is
/// the "reading" front-end: it looks up a learned vector for each word AND spells each word out
/// letter-by-letter through a tiny LSTM, so the model can also guess at words it never saw. It glues the
/// two together for every word and hands the result to the rest of the network.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true)]
public class WordCharEmbeddingLayer<T> : LayerBase<T>
{
    private readonly DenseLayer<T> _wordEmbedding;
    private readonly DenseLayer<T> _charEmbedding;
    private readonly BidirectionalLayer<T> _charBiLstm;

    private readonly int _wordVocabSize;
    private readonly int _charVocabSize;
    private readonly int _maxWordLength;
    private readonly int _wordEmbeddingDim;
    private readonly int _charHiddenDim;

    /// <summary>
    /// Gets the dimensionality of the per-token output vector
    /// (<c>wordEmbeddingDim + charHiddenDim</c>) that feeds the downstream word-level BiLSTM.
    /// </summary>
    public int OutputEmbeddingDim => _wordEmbeddingDim + _charHiddenDim;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new <see cref="WordCharEmbeddingLayer{T}"/>.
    /// </summary>
    /// <param name="wordVocabSize">Number of word types in the word vocabulary (index 0 = padding).</param>
    /// <param name="wordEmbeddingDim">Word embedding dimension (Lample et al. use 100, matching GloVe-100d).</param>
    /// <param name="charVocabSize">Number of distinct characters in the character vocabulary (index 0 = padding).</param>
    /// <param name="charEmbeddingDim">Character embedding dimension (Lample et al. use 25).</param>
    /// <param name="charHiddenDim">Hidden units per direction of the character BiLSTM (Lample et al. use 25).</param>
    /// <param name="sequenceLength">Fixed sentence length in tokens (matches the CRF's sequence dimension).</param>
    /// <param name="maxWordLength">Maximum number of characters per word; longer words are truncated.</param>
    public WordCharEmbeddingLayer(
        int wordVocabSize,
        int wordEmbeddingDim,
        int charVocabSize,
        int charEmbeddingDim,
        int charHiddenDim,
        int sequenceLength,
        int maxWordLength)
        : base(
            [sequenceLength, 1 + maxWordLength],
            [sequenceLength, wordEmbeddingDim + charHiddenDim],
            (IActivationFunction<T>)new IdentityActivation<T>())
    {
        if (wordVocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(wordVocabSize));
        if (wordEmbeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(wordEmbeddingDim));
        if (charVocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(charVocabSize));
        if (charEmbeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(charEmbeddingDim));
        if (charHiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(charHiddenDim));
        if (sequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(sequenceLength));
        if (maxWordLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxWordLength));

        _wordVocabSize = wordVocabSize;
        _charVocabSize = charVocabSize;
        _maxWordLength = maxWordLength;
        _wordEmbeddingDim = wordEmbeddingDim;
        _charHiddenDim = charHiddenDim;

        // Embedding tables implemented as linear layers over one-hot indices (see class remarks):
        // this keeps the tables on the gradient tape so they train end-to-end.
        _wordEmbedding = new DenseLayer<T>(wordEmbeddingDim, (IActivationFunction<T>?)new IdentityActivation<T>());
        _charEmbedding = new DenseLayer<T>(charEmbeddingDim, (IActivationFunction<T>?)new IdentityActivation<T>());

        // Character-level bidirectional LSTM. mergeMode=true (element-wise add of the two
        // directions) keeps the per-word character representation at charHiddenDim.
        var charLstm = new LSTMLayer<T>(charHiddenDim);
        _charBiLstm = new BidirectionalLayer<T>(charLstm, mergeMode: true,
            (IActivationFunction<T>?)new IdentityActivation<T>());

        // Register children so TapeTrainingStep.CollectParameters discovers and trains the
        // word/char embedding tables and the character BiLSTM end-to-end.
        RegisterSubLayer(_wordEmbedding);
        RegisterSubLayer(_charEmbedding);
        RegisterSubLayer(_charBiLstm);
    }

    /// <summary>
    /// Gets the underlying word-embedding linear layer, exposed so a model can initialize it from
    /// pretrained vectors. The weight matrix has shape <c>[wordVocabSize, wordEmbeddingDim]</c>;
    /// row <c>i</c> is the embedding of word index <c>i</c>.
    /// </summary>
    public DenseLayer<T> WordEmbedding => _wordEmbedding;

    /// <summary>
    /// Runs the word + character embedding front-end.
    /// </summary>
    /// <param name="input">Packed integer tensor of shape <c>[sequenceLength, 1 + maxWordLength]</c>:
    /// column 0 is the word index, columns 1.. are the word's character indices (zero-padded).</param>
    /// <returns>Per-token representation of shape <c>[sequenceLength, wordEmbeddingDim + charHiddenDim]</c>.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank != 2)
            throw new ArgumentException(
                $"WordCharEmbeddingLayer expects rank-2 packed input [sequenceLength, 1 + maxWordLength]; got rank {input.Rank}.",
                nameof(input));

        int seq = input.Shape[0];
        int packedWidth = input.Shape[1];
        if (packedWidth != 1 + _maxWordLength)
            throw new ArgumentException(
                $"WordCharEmbeddingLayer expects packed width {1 + _maxWordLength} (1 word index + {_maxWordLength} char indices); got {packedWidth}.",
                nameof(input));

        // Build one-hot encodings (constants; no gradient w.r.t. indices). The linear layers then
        // project them, which is an embedding lookup that the gradient tape can train. PAD (index 0)
        // is masked: padded sequence rows and padded character slots must NOT contribute trainable
        // features, and they must NOT be averaged into the character pooling, or short words and
        // padding rows would learn a spurious PAD embedding.
        var wordOneHot = new Tensor<T>([seq, _wordVocabSize]);
        var charOneHot = new Tensor<T>([seq, _maxWordLength, _charVocabSize]);
        var tokenMask = new Tensor<T>([seq, OutputEmbeddingDim]); // 1 on real tokens, 0 on PAD rows
        var charCounts = new int[seq];                            // real (non-PAD) chars per word
        for (int s = 0; s < seq; s++)
        {
            int wordId = NormalizeIndex(input[s, 0], _wordVocabSize);
            if (wordId > 0) // 0 == PAD: leave the row's one-hot empty and the mask zero
            {
                wordOneHot[s, wordId] = NumOps.One;
                for (int d = 0; d < OutputEmbeddingDim; d++)
                    tokenMask[s, d] = NumOps.One;
            }

            for (int c = 0; c < _maxWordLength; c++)
            {
                int charId = NormalizeIndex(input[s, 1 + c], _charVocabSize);
                if (charId > 0) // 0 == PAD char: skip
                {
                    charOneHot[s, c, charId] = NumOps.One;
                    charCounts[s]++;
                }
            }
        }

        // Word stream: [seq, wordVocab] -> [seq, wordEmbeddingDim].
        var wordFeat = _wordEmbedding.Forward(wordOneHot);

        // Character stream: [seq, maxWordLength, charVocab] -> [seq, maxWordLength, charEmbeddingDim]
        // -> char BiLSTM (words as batch, characters as time) -> [seq, maxWordLength, charHiddenDim]
        // -> mean-pool over the REAL characters only -> [seq, charHiddenDim].
        // Pool, mask and concat go through the tape-tracked op family (TensorMultiply / ReduceSum /
        // TensorConcatenate) so the gradient flows back into the character BiLSTM and both embedding
        // tables. (Engine.ReduceMax / Engine.Concat are NOT autodiff nodes — using them here would
        // silently sever the gradient and freeze the embeddings.)
        var charEmb = _charEmbedding.Forward(charOneHot);
        var charBi = _charBiLstm.Forward(charEmb);

        // Zero the padded character positions before pooling, and divide by the real char count so a
        // 3-letter word isn't diluted by maxWordLength-3 zero rows.
        // charBi has shape [seq, maxWordLength, charHiddenDim] (BiLSTM output, merged directions).
        var charMask = new Tensor<T>([seq, _maxWordLength, _charHiddenDim]);
        for (int s = 0; s < seq; s++)
            for (int c = 0; c < Math.Min(charCounts[s], _maxWordLength); c++)
                for (int d = 0; d < _charHiddenDim; d++)
                    charMask[s, c, d] = NumOps.One;

        var charSum = Engine.ReduceSum(Engine.TensorMultiply(charBi, charMask), [1], keepDims: false);
        var charScale = new Tensor<T>([seq, _charHiddenDim]);
        for (int s = 0; s < seq; s++)
        {
            T inv = NumOps.FromDouble(1.0 / Math.Max(1, charCounts[s]));
            for (int d = 0; d < _charHiddenDim; d++)
                charScale[s, d] = inv;
        }
        var charFeat = Engine.TensorMultiply(charSum, charScale);

        // Concatenate the two streams per token, then mask PAD rows to zero: [seq, wordEmbeddingDim + charHiddenDim].
        var fused = Engine.TensorConcatenate([wordFeat, charFeat], axis: 1);
        return Engine.TensorMultiply(fused, tokenMask);
    }

    private int NormalizeIndex(T value, int vocabSize)
    {
        int idx = (int)Math.Round(NumOps.ToDouble(value));
        if (idx < 0)
            throw new ArgumentOutOfRangeException(nameof(value), "Packed indices must be non-negative.");
        // Out-of-range positive ids map to UNK (index 1), NOT PAD (index 0) — an unseen token is
        // "unknown", not "absent". Falls back to PAD only for a degenerate size-1 vocabulary.
        if (idx >= vocabSize) return vocabSize > 1 ? 1 : 0;
        return idx;
    }

    /// <inheritdoc/>
    /// <remarks>Concatenates child parameters in a fixed order: word table, char table, char BiLSTM.</remarks>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(_wordEmbedding.GetParameters(), _charEmbedding.GetParameters()),
            _charBiLstm.GetParameters());
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int wordLen = _wordEmbedding.GetParameters().Length;
        int charLen = _charEmbedding.GetParameters().Length;
        int biLen = _charBiLstm.GetParameters().Length;

        if (parameters.Length != wordLen + charLen + biLen)
            throw new ArgumentException(
                $"Expected {wordLen + charLen + biLen} parameters, but got {parameters.Length}.",
                nameof(parameters));

        _wordEmbedding.SetParameters(parameters.Slice(0, wordLen));
        _charEmbedding.SetParameters(parameters.Slice(wordLen, charLen));
        _charBiLstm.SetParameters(parameters.Slice(wordLen + charLen, biLen));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _wordEmbedding.UpdateParameters(learningRate);
        _charEmbedding.UpdateParameters(learningRate);
        _charBiLstm.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _wordEmbedding.ResetState();
        _charEmbedding.ResetState();
        _charBiLstm.ResetState();
    }
}
