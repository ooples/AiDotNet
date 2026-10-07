using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A token embedding whose table also projects hidden states back to token logits (weight tying, Press and Wolf 2017),
/// as T5's shared input/output embedding does.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><see cref="LayerBase{T}.Forward(Tensor{T})"/> looks rows up for token ids (any shape of integer-valued ids;
/// the output adds a trailing <c>dimension</c> axis); <see cref="Logits"/> multiplies hidden states by the transposed
/// table. Both use the same parameter, so its gradient collects from both ends.</para>
/// <para>Rows start at N(0, 1), as <c>nn.Embedding</c> does; T5 initializes its shared table at N(0, 1) too
/// (initializer factor 1).</para>
/// <para><b>For Beginners:</b> The table that turns a word id into a vector is reused, transposed, to score every word
/// for the next position, so the model learns one set of word vectors instead of two.</para>
/// </remarks>
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 3", TestConstructorArgs = "5, 4")]
[TensorPort("input", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.TokenIds, MaxExclusiveMember = "_vocabularySize")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.Features)]
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class TiedEmbeddingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _vocabularySize;
    private readonly int _dimension;

    [TrainableParameter(Role = PersistentTensorRole.Embeddings)]
    private Tensor<T> _table;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="vocabularySize">Number of tokens (rows of the table).</param>
    /// <param name="dimension">Width of each embedding.</param>
    public TiedEmbeddingLayer([LayerState] int vocabularySize, [LayerState] int dimension)
        : base(new[] { 1 }, new[] { dimension })
    {
        if (vocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (dimension <= 0) throw new ArgumentOutOfRangeException(nameof(dimension));
        _vocabularySize = vocabularySize;
        _dimension = dimension;
        _table = new Tensor<T>(new[] { vocabularySize, dimension });
        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        for (int i = 0; i < _table.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            _table[i] = NumOps.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        RegisterTrainableParameter(_table, PersistentTensorRole.Embeddings);
    }

    /// <summary>Number of tokens.</summary>
    public int VocabularySize => _vocabularySize;

    /// <summary>Embedding width.</summary>
    public int Dimension => _dimension;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank switch
    {
        1 => new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_dimension)),
        },
        2 => new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_dimension)),
        },
        _ => null,
    };

    /// <summary>The table <c>[vocabulary, dimension]</c>.</summary>
    internal Tensor<T> Table => _table;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var flat = Engine.Reshape(input, new[] { input.Length });
        var rows = Engine.TensorEmbeddingLookupFromFloatIndices(_table, flat);           // [n, dimension]
        var shape = new int[input.Rank + 1];
        for (int i = 0; i < input.Rank; i++) shape[i] = input.Shape[i];
        shape[input.Rank] = _dimension;
        return Engine.Reshape(rows, shape);
    }

    /// <summary>Token logits <c>[..., vocabulary]</c> of hidden states <c>[..., dimension]</c>: <c>h · tableᵀ</c>.</summary>
    public Tensor<T> Logits(Tensor<T> hidden)
    {
        if (hidden.Shape[hidden.Rank - 1] != _dimension)
            throw new ArgumentException($"Expected a trailing dimension of {_dimension}.", nameof(hidden));
        int rows = hidden.Length / _dimension;
        var flat = Engine.Reshape(hidden, new[] { rows, _dimension });
        var logits = Engine.TensorMatMul(flat, Engine.TensorTranspose(_table));          // [rows, vocabulary]
        var shape = hidden.Shape.ToArray();
        shape[shape.Length - 1] = _vocabularySize;
        return Engine.Reshape(logits, shape);
    }

    /// <summary>Redraws every entry from <paramref name="sample"/> (row-major) and zeroes row
    /// <paramref name="zeroRow"/> when given, as <c>nn.Embedding(padding_idx=...)</c> starts its padding row.</summary>
    internal void Reinitialize(Func<double> sample, int? zeroRow = null)
    {
        for (int i = 0; i < _table.Length; i++) _table[i] = NumOps.FromDouble(sample());
        if (zeroRow is int row)
            for (int d = 0; d < _dimension; d++) _table[row, d] = NumOps.Zero;
        Engine.InvalidatePersistentTensor(_table);
    }

    /// <summary>Loads the table from a row-major <c>[vocabulary, dimension]</c> array.</summary>
    internal void LoadTable(double[] values)
    {
        if (values.Length != _table.Length)
            throw new ArgumentException($"Expected {_table.Length} values, got {values.Length}.", nameof(values));
        for (int i = 0; i < values.Length; i++) _table[i] = NumOps.FromDouble(values[i]);
        Engine.InvalidatePersistentTensor(_table);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        metadata["VocabularySize"] = _vocabularySize.ToString(invariant);
        metadata["Dimension"] = _dimension.ToString(invariant);
        return metadata;
    }
}
