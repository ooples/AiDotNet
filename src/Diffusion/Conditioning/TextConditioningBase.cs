using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.Validation;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Base class for text conditioning modules used in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Inherits <see cref="NeuralNetworkBase{T}"/> so concrete subclasses use the
/// codebase's golden "Architecture.Layers ? : LayerHelper.CreateDefaultXxxTextLayers"
/// pattern — the same shape that the recent CLAP / AST / PANNs rewrites established.
/// The forward path walks the inherited <see cref="NeuralNetworkBase{T}.Layers"/>
/// list via <see cref="NeuralNetworkBase{T}.Predict"/>, so engine fast-paths,
/// gradient tape, and weight-streaming auto-detect (#1222, merged) all apply
/// automatically.
/// </para>
/// <para>
/// Tokenisation is delegated to an injected <see cref="ITokenizer"/> from
/// <c>src/Tokenization</c>. Each concrete subclass constructs its paper-
/// canonical tokenizer (CLIP byte-level BPE, T5 SentencePiece-Unigram,
/// Gemma SentencePiece, etc.) in its constructor; this base class never
/// fabricates token IDs from characters.
/// </para>
/// </remarks>
public abstract class TextConditioningBase<T> : NeuralNetworkBase<T>, IConditioningModule<T>
{
    /// <summary>The injected tokenizer for this conditioner's text input.</summary>
    protected ITokenizer Tokenizer { get; }

    /// <inheritdoc />
    public int EmbeddingDimension { get; }

    /// <inheritdoc />
    public ConditioningType ConditioningType => ConditioningType.Text;

    /// <inheritdoc />
    public abstract bool ProducesPooledOutput { get; }

    /// <inheritdoc />
    public int MaxSequenceLength { get; }

    /// <summary>Vocabulary size of the configured tokenizer.</summary>
    public int VocabSize => Tokenizer.VocabularySize;

    /// <summary>
    /// Initialises a new text conditioning module.
    /// </summary>
    /// <param name="architecture">The neural network architecture descriptor.</param>
    /// <param name="tokenizer">The paper-canonical tokenizer for this conditioner's variant.</param>
    /// <param name="maxSequenceLength">Maximum token sequence length.</param>
    /// <param name="embeddingDimension">Output embedding dimension.</param>
    /// <param name="lossFunction">Optional loss function; defaults to MSE for inference-only paths.</param>
    protected TextConditioningBase(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer tokenizer,
        int maxSequenceLength,
        int embeddingDimension,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        Guard.NotNull(tokenizer);
        Tokenizer = tokenizer;
        MaxSequenceLength = maxSequenceLength;
        EmbeddingDimension = embeddingDimension;
    }

    /// <summary>
    /// Layer-stack hook for concrete subclasses. Override to return the paper-
    /// faithful layer stack from the appropriate <see cref="LayerHelper{T}"/>
    /// factory.
    /// </summary>
    protected abstract IEnumerable<ILayer<T>> CreateDefaultLayers();

    /// <summary>
    /// Wires the layer stack at construction time. Uses the codebase's golden
    /// pattern: user-supplied <see cref="NeuralNetworkArchitecture{T}.Layers"/>
    /// wins if non-empty; otherwise fall back to the subclass's paper-faithful
    /// <see cref="CreateDefaultLayers"/> default.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is { Count: > 0 })
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(CreateDefaultLayers());
    }

    // === IConditioningModule ===

    /// <inheritdoc />
    public virtual Tensor<T> Encode(Tensor<T> input) =>
        EncodeCompiled(input, () => RunLayerStack(input));

    /// <inheritdoc />
    public virtual Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null) =>
        RunLayerStack(tokenIds);

    /// <summary>
    /// PyTorch-style direct forward pass: walk the inherited
    /// <see cref="NeuralNetworkBase{T}.Layers"/> list in order and let each
    /// layer's <c>Forward</c> handle its own shape contract. Deliberately
    /// bypasses <see cref="NeuralNetworkBase{T}.Predict"/>'s auto-batch-promote /
    /// auto-squeeze logic — that path infers an "unbatched rank" from the
    /// architecture's <c>InputSize</c> / <c>InputHeight</c> / <c>InputWidth</c>
    /// and squeezes the layer-stack output's leading unit dim when it thinks
    /// the caller passed an unbatched sample. Token-ID conditioner inputs
    /// are always <c>[batch, seqLen]</c> and the layer stack returns
    /// <c>[batch, seqLen, embeddingDim]</c>; we want that rank-3 shape
    /// preserved verbatim, not silently squeezed to rank-2.
    /// </summary>
    private Tensor<T> RunLayerStack(Tensor<T> input)
    {
        // Lazy layer init: defer materialising the layer stack until the
        // first forward pass. Cheap construction of large variants (T5-XXL,
        // CLIP-bigG, Qwen2-7B) — caller pays only when they actually run
        // a forward, not when they merely build the object. Matches the
        // PyTorch convention where nn.Module subclasses build their
        // submodules eagerly but weights are lazy.
        if (Layers.Count == 0) InitializeLayers();
        var x = input;
        foreach (var layer in Layers) x = layer.Forward(x);
        return x;
    }

    /// <summary>
    /// Default pooling: mean over the sequence axis. Concrete subclasses
    /// whose paper specifies a different pool (e.g. CLIP's EOS-position
    /// pooling, BERT-style [CLS] pooling) override this method.
    /// </summary>
    public virtual Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings) =>
        MeanPool(sequenceEmbeddings);

    /// <inheritdoc />
    public virtual Tensor<T> GetUnconditionalEmbedding(int batchSize)
    {
        var emptyTexts = new string[batchSize];
        for (int i = 0; i < batchSize; i++) emptyTexts[i] = string.Empty;
        var tokenIds = TokenizeBatch(emptyTexts);
        return EncodeText(tokenIds);
    }

    /// <inheritdoc />
    public virtual Tensor<T> Tokenize(string text)
    {
        var result = Tokenizer.Encode(text, BuildEncodingOptions());
        return BuildTokenTensor(new List<TokenizationResult> { result });
    }

    /// <inheritdoc />
    public virtual Tensor<T> TokenizeBatch(string[] texts)
    {
        Guard.NotNull(texts);
        var resultList = Tokenizer.EncodeBatch(new List<string>(texts), BuildEncodingOptions());
        return BuildTokenTensor(resultList);
    }

    /// <summary>
    /// Per-conditioner encoding options. Subclasses can override to alter
    /// padding / truncation / special-token policy; the default produces
    /// max-length padding to <see cref="MaxSequenceLength"/>.
    /// </summary>
    protected virtual EncodingOptions BuildEncodingOptions() => new EncodingOptions
    {
        MaxLength = MaxSequenceLength,
        Padding = true,
        Truncation = true,
        AddSpecialTokens = true,
    };

    /// <summary>
    /// Packs <see cref="TokenizationResult"/>s into a <c>[batch, seqLen]</c>
    /// token-ID tensor for <see cref="EmbeddingLayer{T}"/>.
    /// </summary>
    private Tensor<T> BuildTokenTensor(List<TokenizationResult> results)
    {
        int batchSize = results.Count;
        int seqLen = MaxSequenceLength;
        var data = new Vector<T>(batchSize * seqLen);
        for (int b = 0; b < batchSize; b++)
        {
            var ids = results[b].TokenIds;
            int written = Math.Min(ids.Count, seqLen);
            for (int i = 0; i < written; i++)
                data[b * seqLen + i] = NumOps.FromDouble(ids[i]);
        }
        return new Tensor<T>(new[] { batchSize, seqLen }, data);
    }

    // === Compile host (#1272 W2) ===

    private CompiledModelHost<T>? _compileHost;
    private int _compileStructureVersion;

    private CompiledModelHost<T> EnsureCompileHost() =>
        _compileHost ??= new CompiledModelHost<T>(modelIdentity: GetType().Name);

    /// <summary>
    /// Routes <paramref name="eagerEncode"/> through the conditioner's compile
    /// host so the second + Nth call at the same input shape replays a cached
    /// compiled plan.
    /// </summary>
    protected Tensor<T> EncodeCompiled(Tensor<T> input, System.Func<Tensor<T>> eagerEncode) =>
        EnsureCompileHost().Predict(input, _compileStructureVersion, eagerEncode);

    /// <summary>Async overload of <see cref="EncodeCompiled"/>.</summary>
    protected System.Threading.Tasks.ValueTask<Tensor<T>> EncodeCompiledAsync(
        Tensor<T> input,
        System.Func<Tensor<T>> eagerEncode,
        System.Threading.CancellationToken cancellationToken = default) =>
        EnsureCompileHost().PredictAsync(input, _compileStructureVersion, eagerEncode, cancellationToken);

    /// <summary>Invalidates any cached compiled plan after layer-graph mutations.</summary>
    protected void InvalidateConditionerCompiledPlans()
    {
        _compileStructureVersion++;
        _compileHost?.Invalidate();
    }

    /// <summary>Mean-pools <c>[B, S, D]</c> to <c>[B, D]</c> via the engine's reduce kernel.</summary>
    protected Tensor<T> MeanPool(Tensor<T> sequenceEmbeddings)
    {
        int rank = sequenceEmbeddings.Shape.Length;
        if (rank != 3)
            throw new ArgumentException(
                $"MeanPool expects rank-3 [B, S, D]; got rank {rank}.",
                nameof(sequenceEmbeddings));
        int seqLen = sequenceEmbeddings.Shape[1];
        var summed = Engine.ReduceSum(sequenceEmbeddings, new[] { 1 }, keepDims: false);
        return Engine.TensorDivideScalar(summed, NumOps.FromDouble(seqLen));
    }

    // === NeuralNetworkBase abstract surface ===

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = (int)layer.ParameterCount;
            if (count == 0) continue;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T>
    {
        Name = GetType().Name,
        Description = $"{GetType().Name} — text conditioning module for diffusion models.",
        AdditionalInfo = new Dictionary<string, object>
        {
            ["VocabSize"] = VocabSize,
            ["EmbeddingDimension"] = EmbeddingDimension,
            ["MaxSequenceLength"] = MaxSequenceLength,
            ["TokenizerType"] = Tokenizer.GetType().Name,
            ["NumLayers"] = Layers.Count,
        }
    };

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(VocabSize);
        writer.Write(EmbeddingDimension);
        writer.Write(MaxSequenceLength);
        writer.Write(Tokenizer.GetType().AssemblyQualifiedName ?? Tokenizer.GetType().FullName ?? "");
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        VerifyEqual(reader.ReadInt32(), VocabSize, nameof(VocabSize));
        VerifyEqual(reader.ReadInt32(), EmbeddingDimension, nameof(EmbeddingDimension));
        VerifyEqual(reader.ReadInt32(), MaxSequenceLength, nameof(MaxSequenceLength));
        string persistedTokenizerType = reader.ReadString();
        string currentTokenizerType = Tokenizer.GetType().AssemblyQualifiedName ?? Tokenizer.GetType().FullName ?? "";
        if (!string.Equals(persistedTokenizerType, currentTokenizerType, StringComparison.Ordinal))
            throw new InvalidOperationException(
                $"Persisted tokenizer type '{persistedTokenizerType}' does not match current '{currentTokenizerType}'.");
    }

    private static void VerifyEqual<TValue>(TValue persisted, TValue current, string name)
        where TValue : IEquatable<TValue>
    {
        if (!persisted.Equals(current))
            throw new InvalidOperationException(
                $"Persisted {name} = {persisted} does not match constructor option {current}.");
    }
}
