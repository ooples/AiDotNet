using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Maps tabular columns onto finite attribute domains for the marginal-based private synthesizers
/// (PrivBayes, AIM): equal-width bins for numeric columns, the category index for categorical ones.
/// </summary>
/// <remarks>
/// <para>
/// Both papers assume each attribute's domain is public. When public bounds are supplied the bin edges do
/// not depend on the private data and values outside them fall into the end bins. Without bounds the
/// observed minimum and maximum are used, and those two numbers per column are released without noise.
/// </para>
/// <para>
/// Everything runs as whole-tensor engine ops generic in T; codes are integer tensors.
/// </para>
/// </remarks>
internal sealed class DomainDiscretizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Tensor<T> _binLower;
    private readonly Tensor<T> _binWidth;
    private readonly Tensor<T>[] _categoryValues;

    private DomainDiscretizer(int[] cardinality, bool[] isCategorical, Tensor<T> binLower, Tensor<T> binWidth,
        Tensor<T>[] categoryValues)
    {
        Cardinality = cardinality;
        IsCategorical = isCategorical;
        _binLower = binLower;
        _binWidth = binWidth;
        _categoryValues = categoryValues;
    }

    /// <summary>Domain size of every attribute.</summary>
    public int[] Cardinality { get; }

    /// <summary>Whether each attribute is categorical.</summary>
    public bool[] IsCategorical { get; }

    /// <summary>
    /// Learns the domains and encodes the data. <paramref name="codes"/> receives one integer code tensor
    /// of length n per column.
    /// </summary>
    public static DomainDiscretizer<T> Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int numBins,
        Vector<T>? lowerBounds, Vector<T>? upperBounds, out Tensor<int>[] codes)
    {
        var engine = AiDotNetEngine.Current;
        int n = data.Rows;
        int d = data.Columns;
        numBins = Math.Max(1, numBins);

        var x = Tensor<T>.FromMatrix(data);
        var lower = ResolveBound(lowerBounds, "DomainLowerBounds", d,
            () => engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(x), new[] { 0 }, false)));
        var upper = ResolveBound(upperBounds, "DomainUpperBounds", d,
            () => engine.ReduceMax(x, new[] { 0 }, false));

        // Width = (hi - lo) / bins; a constant column (width below 1e-10) uses width 1.
        var width = engine.TensorDivideScalar(engine.TensorSubtract(upper, lower), NumOps.FromDouble(numBins));
        width = engine.TensorWhere(engine.TensorGreaterThan(width, NumOps.FromDouble(1e-10)), width,
            Filled(new[] { d }, NumOps.One));

        var shape = new[] { n, d };
        var scaled = engine.TensorDivide(engine.TensorSubtract(x, Broadcast(lower, shape)), Broadcast(width, shape));
        var bins = engine.TensorClamp(engine.TensorFloor(scaled), NumOps.Zero, NumOps.FromDouble(numBins - 1));
        var binsByColumn = engine.TensorTranspose(bins).Contiguous().Cast<int>();

        codes = new Tensor<int>[d];
        var cardinality = new int[d];
        var isCategorical = new bool[d];
        var categoryValues = new Tensor<T>[d];
        for (int j = 0; j < d; j++)
        {
            if (columns[j].IsCategorical)
            {
                var column = Tensor<T>.FromVector(data.GetColumn(j));
                var values = engine.TensorUnique(column, true);
                isCategorical[j] = true;
                categoryValues[j] = values;
                cardinality[j] = values.Length;
                codes[j] = engine.TensorSearchSorted(values, column, false);
            }
            else
            {
                categoryValues[j] = new Tensor<T>(new[] { 0 });
                cardinality[j] = numBins;
                codes[j] = binsByColumn.GetSlice(j);
            }
        }

        return new DomainDiscretizer<T>(cardinality, isCategorical, lower, width, categoryValues);
    }

    /// <summary>
    /// Turns sampled codes back into values: a uniform position inside the bin for numeric columns, the
    /// category's own value for categorical columns. The uniform draws use one seed from
    /// <paramref name="random"/>.
    /// </summary>
    public Matrix<T> Decode(Tensor<int>[] codes, int rows, Random random)
    {
        var engine = AiDotNetEngine.Current;
        int d = codes.Length;
        var shape = new[] { rows, d };
        var position = engine.TensorAdd(
            engine.TensorStack(codes, 1).Cast<T>(),
            engine.TensorRandomUniformRange(shape, NumOps.Zero, NumOps.One, random.Next()));
        var numeric = engine.TensorAdd(
            engine.TensorMultiply(position, Broadcast(_binWidth, shape)),
            Broadcast(_binLower, shape));

        if (!IsCategorical.Any(c => c))
        {
            return numeric.ToMatrix();
        }

        var transposed = engine.TensorTranspose(numeric).Contiguous();
        var columns = new Tensor<T>[d];
        for (int j = 0; j < d; j++)
        {
            columns[j] = IsCategorical[j]
                ? engine.TensorIndexSelect(_categoryValues[j], codes[j], 0)
                : transposed.GetSlice(j);
        }

        return engine.TensorStack(columns, 1).ToMatrix();
    }

    /// <summary>
    /// Counts of every cell of the joint domain of <paramref name="attributes"/> (in the given order,
    /// first attribute most significant), zero cells included, shaped by their domain sizes.
    /// </summary>
    public Tensor<T> Counts(Tensor<int>[] codes, IReadOnlyList<int> attributes)
    {
        var engine = AiDotNetEngine.Current;
        int rows = codes[0].Length;
        var cell = ConfigIndex(codes, attributes, rows);
        var shape = attributes.Count == 0 ? new[] { 1 } : attributes.Select(a => Cardinality[a]).ToArray();
        int cells = shape.Aggregate(1, (acc, c) => acc * c);
        return engine.TensorBinCount(cell, cells).Cast<T>().Reshape(shape);
    }

    /// <summary>Mixed-radix index of each row's configuration of <paramref name="attributes"/> (zero when empty).</summary>
    public Tensor<int> ConfigIndex(Tensor<int>[] codes, IReadOnlyList<int> attributes, int rows)
    {
        var engine = AiDotNetEngine.Current;
        var index = new Tensor<int>(new[] { rows });
        foreach (int a in attributes)
        {
            index = engine.TensorAdd(engine.TensorMultiplyScalar(index, Cardinality[a]), codes[a]);
        }

        return index;
    }

    private static Tensor<T> ResolveBound(Vector<T>? bound, string name, int d, Func<Tensor<T>> fromData)
    {
        if (bound is null) return fromData();
        if (bound.Length != d)
        {
            throw new ArgumentException(
                $"{name} has {bound.Length} entries but the data has {d} columns.", name);
        }

        return Tensor<T>.FromVector(bound);
    }

    private static Tensor<T> Broadcast(Tensor<T> row, int[] shape)
        => AiDotNetEngine.Current.TensorBroadcastTo(row.Reshape(new[] { 1, row.Length }), shape).Contiguous();

    /// <summary>A tensor of the given shape with every element set to <paramref name="value"/>.</summary>
    public static Tensor<T> Filled(int[] shape, T value)
    {
        var tensor = new Tensor<T>(shape);
        AiDotNetEngine.Current.TensorFill(tensor, value);
        return tensor;
    }
}
