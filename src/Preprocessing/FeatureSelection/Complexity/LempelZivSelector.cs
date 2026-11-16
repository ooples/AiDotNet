using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Complexity;

/// <summary>
/// Lempel-Ziv Complexity based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Lempel-Ziv complexity, which measures
/// the number of distinct patterns in a discretized sequence.
/// </para>
/// <para><b>For Beginners:</b> Lempel-Ziv complexity counts how many unique
/// patterns appear when reading a sequence. It's related to compression -
/// sequences that compress well have low complexity (predictable patterns),
/// while complex sequences have many unique patterns and don't compress well.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LempelZivSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _complexityValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? ComplexityValues => _complexityValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LempelZivSelector(
        int nFeaturesToSelect = 10,
        int nBins = 2,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Normalized complexity bound: n / log_k(n) where k = alphabet size
        double normFactor = n / (Math.Log(n) / Math.Log(_nBins));

        var columnSymbols = Symbols(Tensor<T>.FromMatrix(data), _nBins);

        // Compute Lempel-Ziv complexity of each column's integer symbol sequence
        var complexity = new int[p];
        for (int j = 0; j < p; j++)
        {
            var sequence = new int[n];
            Array.Copy(columnSymbols, j * n, sequence, 0, n);
            complexity[j] = ComputeLZComplexity(sequence);
        }

        var complexityValues = NormalizeComplexity(complexity, normFactor);
        _complexityValues = complexityValues.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(complexityValues, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// Discretizes every column to integer symbols, column-major: trunc((x - min) / range * (nBins - 1)),
    /// capped at nBins - 1; a (near-)constant column is all symbol 0. Half widens to double first, since its
    /// range and numerators overflow at 65504 for representable inputs.
    /// </summary>
    internal static int[] Symbols(Tensor<T> x, int nBins)
    {
        if (PreprocessingTensorOps.IsNarrowFloat<T>())
            return LempelZivSelector<double>.Symbols(x.Cast<double>(), nBins);

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var min = engine.Reshape(PreprocessingTensorOps.ColumnMin(x), new[] { 1, p });
        var range = engine.TensorSubtract(engine.Reshape(PreprocessingTensorOps.ColumnMax(x), new[] { 1, p }), min);
        var scaled = engine.TensorMultiplyScalar(
            engine.TensorDivide(engine.TensorSubtract(x, engine.TensorBroadcastTo(min, new[] { n, p })),
                engine.TensorBroadcastTo(range, new[] { n, p })),
            ops.FromDouble(nBins - 1));
        var bins = engine.TensorMin(engine.TensorTrunc(scaled), ops.FromDouble(nBins - 1));
        var constant = engine.TensorBroadcastTo(engine.TensorLessThan(range, PreprocessingTensorOps.Tolerance<T>(1e-10)), new[] { n, p });
        var symbols = engine.TensorWhere(constant, new Tensor<T>(new[] { n, p }), bins);
        return engine.TensorTranspose(symbols).Cast<int>().AsSpan().ToArray();
    }

    /// <summary>
    /// Divides the raw complexities by the theoretical bound in double and casts only the result: the bound
    /// itself passes 65504 for Half once n reaches about two million rows.
    /// </summary>
    internal static Tensor<T> NormalizeComplexity(int[] complexity, double normFactor)
    {
        if (normFactor <= 0)
            return new Tensor<T>(new[] { complexity.Length });
        var raw = new Tensor<int>((int[])complexity.Clone(), new[] { complexity.Length }).Cast<double>();
        return AiDotNetEngine.Current.TensorDivideScalar(raw, normFactor).Cast<T>();
    }

    private int ComputeLZComplexity(int[] sequence)
    {
        int n = sequence.Length;
        if (n == 0) return 0;
        if (n == 1) return 1;

        var dictionary = new HashSet<string>();
        int complexity = 0;
        int i = 0;

        while (i < n)
        {
            int length = 1;
            string word = sequence[i].ToString();

            // Extend word while it exists in dictionary
            while (i + length <= n)
            {
                var sb = new System.Text.StringBuilder();
                for (int k = i; k < i + length; k++)
                    sb.Append(sequence[k]).Append(",");
                word = sb.ToString();

                if (!dictionary.Contains(word) || i + length == n)
                    break;

                length++;
            }

            dictionary.Add(word);
            complexity++;
            i += length;
        }

        return complexity;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LempelZivSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LempelZivSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LempelZivSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
