using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Paper-faithful text + 2D-layout embedding front-end for layout-aware document models
/// (Xu et al., KDD 2020, "LayoutLM: Pre-training of Text and Layout for Document Image
/// Understanding", §3.2 "Model Architecture").
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LayoutLM's contribution over BERT is that a token carries WHERE it sits on the page, not just
/// what it says. The paper builds one embedding block that sums a token's identity, its position in
/// the reading order, and six numbers derived from its bounding box:
/// </para>
/// <code>
/// E(t_i) = Word(t_i) + Pos1D(i)
///        + X(x0_i) + Y(y0_i) + X(x1_i) + Y(y1_i)
///        + W(x1_i - x0_i) + H(y1_i - y0_i)
/// </code>
/// <para>
/// The x-axis table is SHARED between the left and right edges, and the y-axis table between the
/// top and bottom edges — the paper uses "two embedding tables to embed x-axis and y-axis features
/// separately", so a coordinate means the same thing whichever corner it came from. Width and
/// height get their own tables because a box's SIZE is a different feature from its POSITION: two
/// boxes can start at the same x and be a caption and a paragraph.
/// </para>
/// <para><b>Why this is a layer and not model-level tensors.</b> The five layout tables previously
/// lived as fields on each layout-aware model, were allocated and randomly initialized, were
/// counted as parameters, serialized into every checkpoint and updated by the optimizer — and were
/// never read by any forward pass. Weights that train but do not participate are worse than absent
/// ones: they consume memory and checkpoint bytes and they make the parameter count describe a
/// model that does not exist. As a layer the tables sit in <c>Layers</c>, so they are discovered by
/// <c>TapeTrainingStep.CollectParameters</c>, counted by the parameter automation, and picked up by
/// BOTH the inference forward and the tape-based training forward with no override on either.
/// </para>
/// <para><b>Learned, not sinusoidal, positions.</b> This layer replaces the
/// <see cref="EmbeddingLayer{T}"/> + <see cref="PositionalEncodingLayer{T}"/> pair that used to sit
/// at the front of these stacks. <c>PositionalEncodingLayer</c> is
/// <c>SupportsTraining =&gt; false</c> — fixed sinusoids — whereas LayoutLM inherits BERT's LEARNED
/// position embeddings. The sinusoidal layer was therefore not merely a duplicate of the dead
/// <c>_positionEmbeddings</c> field; it was the wrong kind of position embedding.
/// </para>
/// <para><b>For Beginners:</b> A plain language model sees a document as one long line of words, so
/// it cannot tell a table cell from a footnote. This layer hands the model the coordinates of the
/// box each word was printed in, so "500" in the top-right corner of an invoice can be learned to
/// mean something different from "500" in the body text. You give it the word IDs and, optionally,
/// the boxes; it gives back one vector per word that mixes both.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.SequenceModeling)]
// A rank-1 token sequence is the shape the generated tests drive, because it is the rank whose
// output contract is statically known (see the TensorLayout note below). The packed rank-2 path is
// covered explicitly by LayoutEmbeddingLayerTests.
[LayerProperty(IsTrainable = true, ChangesShape = true,
    TestInputShape = "8", TestConstructorArgs = "64, 16, 32, 64")]
// Two layouts are declared because two are statically determined. Axis 0/1 is Time - the token's
// place in the reading order - and the trailing axis is Features on both sides, but it means
// different things: going in it is the PACKED column (token id, then the four box coordinates),
// coming out it is the summed embedding. That is why the output is Fixed, not Same.
//
// Rank 2 is deliberately absent. It is the one rank this layer cannot resolve statically: [seq, 5]
// is a packed sequence and produces rank 2, while [batch, seq] is batched token ids and produces
// rank 3, and only the runtime width separates them. OutputAxesFor returns null there rather than
// assert one of the two - a contract that is right half the time is worse than no contract,
// because shape resolution would propagate it without ever being told it had guessed.
[TensorLayout(TensorAxis.Time,
    Direction = TensorLayoutDirection.Input,
    Note = "Token ids only; the 2D layout terms contribute nothing without boxes.")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Trailing axis is 5 for packed (tokenId, x0, y0, x1, y1) or 1 for token ids only.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class LayoutEmbeddingLayer<T> : LayerBase<T>, IShapeContract
{
    /// <summary>
    /// Width of a packed input row: one token ID followed by the four bounding-box coordinates
    /// <c>(x0, y0, x1, y1)</c>.
    /// </summary>
    public const int PackedRowWidth = 5;

    /// <summary>
    /// Width of a boxes-only row when width and height are derived: <c>(x0, y0, x1, y1)</c>.
    /// </summary>
    public const int BoxOnlyRowWidth = 4;

    /// <summary>
    /// Width of a boxes-only row that carries width and height explicitly:
    /// <c>(x0, y0, x1, y1, w, h)</c>. This is the six-feature layout vector LiLT and LayoutLM both
    /// describe; supplying it directly lets a caller whose OCR already reports box extents skip the
    /// subtraction, and lets a caller with a non-axis-aligned box give a real extent rather than one
    /// implied by two corners.
    /// </summary>
    public const int BoxWithExtentRowWidth = 6;

    private readonly bool _includeTokens;

    /// <summary>
    /// Row width this instance consumes: five with a token column, four in boxes-only mode. A
    /// boxes-only layer also accepts <see cref="BoxWithExtentRowWidth"/> rows, which carry width and
    /// height instead of leaving them to be derived.
    /// </summary>
    public int RowWidth => _includeTokens ? PackedRowWidth : BoxOnlyRowWidth;

    private readonly EmbeddingLayer<T>? _wordEmbeddings;
    private readonly EmbeddingLayer<T> _positionEmbeddings;
    private readonly EmbeddingLayer<T> _xEmbeddings;
    private readonly EmbeddingLayer<T> _yEmbeddings;
    private readonly EmbeddingLayer<T> _widthEmbeddings;
    private readonly EmbeddingLayer<T> _heightEmbeddings;

    private readonly int _vocabSize;
    private readonly int _hiddenDim;
    private readonly int _maxSequenceLength;
    private readonly int _maxPosition2D;

    /// <inheritdoc />
    /// <remarks>
    /// Rank 2 returns null on purpose - see the note on the class. The other two ranks replace the
    /// index axis with the embedding width, which is a constructor argument and so is
    /// <c>Fixed</c> rather than derived from anything on the input.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_hiddenDim <= 0) return null;

        return inputRank switch
        {
            1 => new[]
            {
                new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hiddenDim)),
            },
            3 => new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hiddenDim)),
            },
            _ => null,
        };
    }

    /// <summary>Dimensionality of the per-token vector this layer produces.</summary>
    public int EmbeddingDim => _hiddenDim;

    /// <summary>Size of the token vocabulary the word table covers.</summary>
    public int VocabularySize => _vocabSize;

    /// <summary>Longest sequence the learned 1D position table can index; later tokens reuse the last row.</summary>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <summary>Number of coordinate buckets each 2D table covers.</summary>
    public int MaxPosition2D => _maxPosition2D;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Every table is sized from constructor arguments alone — vocabulary, embedding width, sequence
    /// limit and coordinate-grid size — so none of them has to wait for an input shape.
    /// </summary>
    protected override bool ParametersAreConstructionSized => true;

    /// <summary>
    /// Declares that this layer consumes integer indices, not continuous features, so callers and
    /// the conformance suite generate legal IDs instead of random reals.
    /// </summary>
    /// <remarks>
    /// A packed row mixes two domains - column 0 indexes the vocabulary, columns 1-4 index the
    /// coordinate grid - and a single domain cannot say that. The NARROWER of the two is reported
    /// for packed shapes, because a value legal in the smaller range is legal in both, whereas
    /// reporting the wider one would call a coordinate of 20000 acceptable and then saturate it.
    /// </remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape)
    {
        bool packed = inputShape is not null
            && inputShape.Length >= 2
            && inputShape[inputShape.Length - 1] == RowWidth;

        // Boxes-only rows carry no token column, so the coordinate grid is the whole domain.
        if (!_includeTokens) return LayerInputDomain.Indices(_maxPosition2D);

        return LayerInputDomain.Indices(packed ? Math.Min(_vocabSize, _maxPosition2D) : _vocabSize);
    }

    /// <summary>
    /// Initializes a new <see cref="LayoutEmbeddingLayer{T}"/>.
    /// </summary>
    /// <param name="vocabSize">Token vocabulary size (30522 for BERT-base, which LayoutLM builds on).</param>
    /// <param name="hiddenDim">Embedding width; every table shares it because the terms are summed (768 for base).</param>
    /// <param name="maxSequenceLength">Longest token sequence, sizing the 1D position table (512 for BERT).</param>
    /// <param name="maxPosition2D">
    /// Number of distinct coordinate buckets. The paper normalizes every box onto a 0-1000 grid so
    /// that page size drops out; 1024 is the usual table size, leaving headroom above 1000.
    /// </param>
    /// <param name="includeTokens">
    /// <c>true</c> (the default) for the LayoutLM-style block: a packed <c>[seq, 5]</c> row whose
    /// first column is a token ID, summing text and layout together.
    /// <para>
    /// <c>false</c> for a boxes-only block consuming <c>[seq, 4]</c>, which is what LiLT needs: its
    /// layout stream carries NO text, and keeping the two apart is the paper's contribution -- it is
    /// what lets one pre-trained layout encoder pair with any language's text encoder. Feeding
    /// tokens into that stream would destroy the language-independence the model exists for.
    /// No word table is allocated in this mode, so it is not a dead parameter either.
    /// </para>
    /// </param>
    public LayoutEmbeddingLayer(
        [LayerState] int vocabSize = 30522,
        [LayerState] int hiddenDim = 768,
        [LayerState] int maxSequenceLength = 512,
        [LayerState] int maxPosition2D = 1024,
        [LayerState] bool includeTokens = true)
        // Declared shapes mirror EmbeddingLayer's [1] -> [embeddingDim]: what this layer fixes is the
        // WIDTH of each token's vector, not how many tokens arrive. Declaring the sequence axis as
        // maxSequenceLength instead claimed a [32, 16] output for an 8-token input and the
        // conformance check rightly rejected it ("output length 128 is not a multiple of ... 512").
        : base([1], [hiddenDim])
    {
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
        if (maxPosition2D <= 0) throw new ArgumentOutOfRangeException(nameof(maxPosition2D));

        _vocabSize = vocabSize;
        _hiddenDim = hiddenDim;
        _maxSequenceLength = maxSequenceLength;
        _maxPosition2D = maxPosition2D;
        _includeTokens = includeTokens;

        _wordEmbeddings = includeTokens ? new EmbeddingLayer<T>(vocabSize, hiddenDim) : null;
        _positionEmbeddings = new EmbeddingLayer<T>(maxSequenceLength, hiddenDim);
        _xEmbeddings = new EmbeddingLayer<T>(maxPosition2D, hiddenDim);
        _yEmbeddings = new EmbeddingLayer<T>(maxPosition2D, hiddenDim);
        _widthEmbeddings = new EmbeddingLayer<T>(maxPosition2D, hiddenDim);
        _heightEmbeddings = new EmbeddingLayer<T>(maxPosition2D, hiddenDim);

        // Register the children so TapeTrainingStep.CollectParameters walks into them: an
        // unregistered sub-layer is invisible to the optimizer and trains silently never.
        if (_wordEmbeddings is not null) RegisterSubLayer(_wordEmbeddings);
        RegisterSubLayer(_positionEmbeddings);
        RegisterSubLayer(_xEmbeddings);
        RegisterSubLayer(_yEmbeddings);
        RegisterSubLayer(_widthEmbeddings);
        RegisterSubLayer(_heightEmbeddings);
    }

    /// <summary>
    /// Gets the token embedding table, exposed so a model can load pretrained vectors.
    /// Null in boxes-only mode, where no word table exists.
    /// </summary>
    public EmbeddingLayer<T>? WordEmbeddings => _wordEmbeddings;

    /// <summary>
    /// Embeds a token sequence, adding the 2D layout terms when bounding boxes are supplied.
    /// </summary>
    /// <param name="input">
    /// Either a plain token-ID sequence or a packed token+box tensor, distinguished by the size of
    /// the LAST axis:
    /// <list type="bullet">
    /// <item><c>[seq]</c> — token IDs only.</item>
    /// <item><c>[seq, 5]</c> — packed: <c>(tokenId, x0, y0, x1, y1)</c> per token.</item>
    /// <item><c>[batch, seq]</c> or <c>[batch, seq, 1]</c> — batched token IDs only.</item>
    /// <item><c>[batch, seq, 5]</c> — batched packed.</item>
    /// </list>
    /// Tokens-only input is the beginner default and keeps the layer behaving exactly like a
    /// BERT embedding block; the layout terms simply do not contribute.
    /// </param>
    /// <returns>Per-token embeddings, the input shape with the index axis replaced by <c>hiddenDim</c>.</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Bring the WHOLE block up, not just the tables this particular input happens to touch. The
        // sub-tables allocate lazily, so a tokens-only forward would materialize word + position and
        // leave x/y/w/h empty — and the save path deliberately records only what exists while the
        // restore path materializes everything reachable. That asymmetry is fine for a layer that is
        // all-or-nothing and fatal for one that is half up: a checkpoint written after a tokens-only
        // forward carried 1536 values and the restore expected 5632. These six tables are one
        // embedding block in the paper and there is no meaningful state where four of them are absent.
        EnsureParametersMaterialized();

        int rank = input.Rank;
        if (rank < 1)
            throw new ArgumentException("LayoutEmbeddingLayer expects at least a rank-1 token sequence.", nameof(input));

        int lastAxis = input.Shape[rank - 1];

        // In boxes-only mode a row is the four coordinates and there is no token column at all, so
        // the "packed" row is one slot narrower and the word lookup below is skipped entirely.
        // A boxes-only row is four coordinates, or six when the caller supplies the extents too.
        bool extentsGiven = !_includeTokens && rank >= 2 && lastAxis == BoxWithExtentRowWidth;
        bool packed = (rank >= 2 && lastAxis == RowWidth) || extentsGiven;

        if (!_includeTokens && !packed)
            throw new ArgumentException(
                $"A boxes-only LayoutEmbeddingLayer expects rows of {BoxOnlyRowWidth} coordinates " +
                $"(x0, y0, x1, y1) or {BoxWithExtentRowWidth} (x0, y0, x1, y1, w, h); got a trailing " +
                $"axis of {lastAxis}. There is no token column in this mode, so a bare index sequence " +
                "has nothing to look up.",
                nameof(input));

        int rowWidth = extentsGiven ? BoxWithExtentRowWidth : RowWidth;

        // Leading shape = the index grid, i.e. everything except a packed row / trailing singleton.
        int[] leading;
        if (packed || (rank >= 2 && lastAxis == 1))
        {
            leading = new int[rank - 1];
            for (int i = 0; i < rank - 1; i++) leading[i] = input.Shape[i];
        }
        else
        {
            leading = new int[rank];
            for (int i = 0; i < rank; i++) leading[i] = input.Shape[i];
        }

        int gridSize = 1;
        for (int i = 0; i < leading.Length; i++) gridSize *= leading[i];

        // The sequence axis is the last leading axis: [seq] -> seq, [batch, seq] -> seq.
        int seqLen = leading.Length == 0 ? 1 : leading[leading.Length - 1];

        // A packed row advances rowWidth slots per token; every other form is one index per token.
        int stride = packed ? rowWidth : 1;

        // Offset of the first coordinate within a row: past the token column when there is one.
        int boxBase = _includeTokens ? 1 : 0;
        var flat = input.Data.Span;

        var positions = new Tensor<T>(leading);
        var positionSpan = positions.Data.Span;

        Tensor<T>? tokenIds = _includeTokens ? new Tensor<T>(leading) : null;

        for (int i = 0; i < gridSize; i++)
        {
            if (tokenIds is not null) tokenIds.Data.Span[i] = flat[i * stride];
            // Reading order within each sequence; wraps per batch row.
            positionSpan[i] = NumOps.FromDouble(Math.Min(i % seqLen, _maxSequenceLength - 1));
        }

        // Reading-order position always applies. Word identity only when there is a token column --
        // in boxes-only mode the block is pure layout by design.
        var embedded = _positionEmbeddings.Forward(positions);
        if (_wordEmbeddings is not null && tokenIds is not null)
            embedded = Engine.TensorAdd(embedded, _wordEmbeddings.Forward(tokenIds));

        if (!packed)
        {
            // No boxes supplied — the layout terms contribute nothing rather than contributing a
            // wrong constant. Adding a fixed index-0 embedding here would inject the same learned
            // vector into every token and teach the model a bias that means "no layout given".
            return embedded;
        }

        var x0 = new Tensor<T>(leading);
        var y0 = new Tensor<T>(leading);
        var x1 = new Tensor<T>(leading);
        var y1 = new Tensor<T>(leading);
        var widths = new Tensor<T>(leading);
        var heights = new Tensor<T>(leading);

        var x0Span = x0.Data.Span;
        var y0Span = y0.Data.Span;
        var x1Span = x1.Data.Span;
        var y1Span = y1.Data.Span;
        var widthSpan = widths.Data.Span;
        var heightSpan = heights.Data.Span;

        for (int i = 0; i < gridSize; i++)
        {
            int b = i * rowWidth + boxBase;
            int left = ClampCoordinate(flat[b]);
            int top = ClampCoordinate(flat[b + 1]);
            int right = ClampCoordinate(flat[b + 2]);
            int bottom = ClampCoordinate(flat[b + 3]);

            x0Span[i] = NumOps.FromDouble(left);
            y0Span[i] = NumOps.FromDouble(top);
            x1Span[i] = NumOps.FromDouble(right);
            y1Span[i] = NumOps.FromDouble(bottom);

            // Size, not position. Taken from the row when the caller supplied it, otherwise derived
            // from the corners -- and by magnitude, because a corner-swapped box (right < left) still
            // has a real extent and clamping that difference to zero would erase it.
            int w = extentsGiven ? ClampCoordinate(flat[b + 4]) : Math.Abs(right - left);
            int h = extentsGiven ? ClampCoordinate(flat[b + 5]) : Math.Abs(bottom - top);

            widthSpan[i] = NumOps.FromDouble(Math.Min(w, _maxPosition2D - 1));
            heightSpan[i] = NumOps.FromDouble(Math.Min(h, _maxPosition2D - 1));
        }

        // Both corners go through the SAME axis table, per the paper.
        embedded = Engine.TensorAdd(embedded, _xEmbeddings.Forward(x0));
        embedded = Engine.TensorAdd(embedded, _yEmbeddings.Forward(y0));
        embedded = Engine.TensorAdd(embedded, _xEmbeddings.Forward(x1));
        embedded = Engine.TensorAdd(embedded, _yEmbeddings.Forward(y1));
        embedded = Engine.TensorAdd(embedded, _widthEmbeddings.Forward(widths));
        embedded = Engine.TensorAdd(embedded, _heightEmbeddings.Forward(heights));

        return embedded;
    }

    /// <summary>
    /// Maps a raw coordinate onto the table's index range. OCR boxes arrive in page pixels or on the
    /// paper's 0-1000 grid, and a stray value must not throw out of a lookup that the caller cannot
    /// see — so out-of-range coordinates saturate at the edge of the page, which is where they are.
    /// </summary>
    private int ClampCoordinate(T value)
    {
        double raw = NumOps.ToDouble(value);
        if (double.IsNaN(raw)) return 0;

        int index = (int)Math.Round(raw);
        if (index < 0) return 0;
        if (index >= _maxPosition2D) return _maxPosition2D - 1;
        return index;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _wordEmbeddings?.UpdateParameters(learningRate);
        _positionEmbeddings.UpdateParameters(learningRate);
        _xEmbeddings.UpdateParameters(learningRate);
        _yEmbeddings.UpdateParameters(learningRate);
        _widthEmbeddings.UpdateParameters(learningRate);
        _heightEmbeddings.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _wordEmbeddings?.ResetState();
        _positionEmbeddings.ResetState();
        _xEmbeddings.ResetState();
        _yEmbeddings.ResetState();
        _widthEmbeddings.ResetState();
        _heightEmbeddings.ResetState();
    }
}
