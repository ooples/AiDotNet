using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Helpers;

/// <summary>
/// Whole-array building blocks shared by the explainers: path interpolation, path integration,
/// finite-difference gradients and row/tensor conversions, all expressed as engine operations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class ExplainerTensorOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// A [1, n] tensor holding a copy of <paramref name="row"/>.
    /// </summary>
    public static Tensor<T> RowTensor(Vector<T> row)
    {
        return new Tensor<T>(new[] { 1, row.Length }, row.Clone());
    }

    /// <summary>
    /// A 1-row matrix holding a copy of <paramref name="row"/>.
    /// </summary>
    public static Matrix<T> RowMatrix(Vector<T> row)
    {
        var matrix = new Matrix<T>(1, row.Length);
        AiDotNetEngine.Current.SetRow(matrix, 0, row);
        return matrix;
    }

    /// <summary>
    /// Index of the largest entry (first one on ties).
    /// </summary>
    public static int ArgMax(Vector<T> values)
    {
        if (values.Length == 0)
            throw new ArgumentException("Cannot take the arg max of an empty vector.", nameof(values));

        return AiDotNetEngine.Current.TensorArgMax(values.AsTensor(), 0).GetFlat(0);
    }

    /// <summary>
    /// Rows baseline + alpha_s * (input - baseline) for each alpha_s, as an [alphas, n] tensor.
    /// </summary>
    public static Tensor<T> InterpolationPath(Vector<T> baseline, Vector<T> input, Tensor<T> alphas)
    {
        var engine = AiDotNetEngine.Current;
        var offsets = engine.TensorOuter(alphas, engine.Subtract(input, baseline).AsTensor());
        var baselineRows = engine.TensorBroadcastTo(baseline.AsTensor().Reshape(new[] { 1, baseline.Length }), offsets.Shape);
        return engine.TensorAdd(offsets, baselineRows);
    }

    /// <summary>
    /// alpha_s = s / steps for s = 0..steps (both end points), as a [steps + 1] tensor.
    /// </summary>
    public static Tensor<T> UniformAlphasWithEndpoints(int steps)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorDivideScalar(engine.TensorArange<T>(0, steps + 1, 1), NumOps.FromDouble(steps));
    }

    /// <summary>
    /// Trapezoid-rule average of gradients sampled at s / steps for s = 0..steps: the end points weigh 0.5,
    /// interior points 1, and the sum is divided by steps.
    /// </summary>
    public static Vector<T> TrapezoidAverage(IReadOnlyList<Vector<T>> gradients)
    {
        var engine = AiDotNetEngine.Current;
        int count = gradients.Count;
        int steps = count - 1;
        var weights = engine.Fill(count, NumOps.Divide(NumOps.One, NumOps.FromDouble(steps)));
        T halfWeight = NumOps.Divide(NumOps.FromDouble(0.5), NumOps.FromDouble(steps));
        weights[0] = halfWeight;
        weights[steps] = halfWeight;
        return WeightedSum(gradients, weights);
    }

    /// <summary>
    /// Sum over i of weights[i] * vectors[i].
    /// </summary>
    public static Vector<T> WeightedSum(IReadOnlyList<Vector<T>> vectors, Vector<T> weights)
    {
        var engine = AiDotNetEngine.Current;
        var stacked = Stack(vectors);
        return engine.TensorMatMul(weights.AsTensor().Reshape(new[] { 1, vectors.Count }), stacked).ToVector();
    }

    /// <summary>
    /// Element-wise mean of equally long vectors.
    /// </summary>
    public static Vector<T> Mean(IReadOnlyList<Vector<T>> vectors)
    {
        return AiDotNetEngine.Current.ReduceMean(Stack(vectors), new[] { 0 }, false).ToVector();
    }

    /// <summary>
    /// Equally long vectors as the rows of a [count, length] tensor.
    /// </summary>
    public static Tensor<T> Stack(IReadOnlyList<Vector<T>> vectors)
    {
        var rows = new Tensor<T>[vectors.Count];
        for (int i = 0; i < rows.Length; i++)
        {
            rows[i] = vectors[i].AsTensor();
        }

        return AiDotNetEngine.Current.TensorStack(rows, 0);
    }

    /// <summary>
    /// Central-difference gradient (f(x + eps e_j) - f(x - eps e_j)) / (2 eps) of output
    /// <paramref name="outputIndex"/> for each of the first <paramref name="count"/> features, evaluating the
    /// plus then the minus input for each feature in turn.
    /// </summary>
    public static Vector<T> CentralDifferenceGradient(
        Func<Vector<T>, Vector<T>> predict, Vector<T> input, int outputIndex, double epsilon, int count)
    {
        var engine = AiDotNetEngine.Current;
        T eps = NumOps.FromDouble(epsilon);
        var inputTensor = input.AsTensor();
        var plusAll = engine.TensorAddScalar(inputTensor, eps);
        var minusAll = engine.TensorSubtractScalar(inputTensor, eps);
        var positions = engine.TensorArange<int>(0, input.Length, 1);

        var plusOutputs = new Tensor<T>[count];
        var minusOutputs = new Tensor<T>[count];
        for (int j = 0; j < count; j++)
        {
            var isFeature = engine.TensorEqScalar(positions, j);
            plusOutputs[j] = predict(engine.TensorWhere(isFeature, plusAll, inputTensor).ToVector()).AsTensor();
            minusOutputs[j] = predict(engine.TensorWhere(isFeature, minusAll, inputTensor).ToVector()).AsTensor();
        }

        if (count == 0)
            return new Vector<T>(0);

        int outputs = plusOutputs[0].Length;
        if (outputIndex < 0 || outputIndex >= outputs)
            throw new ArgumentOutOfRangeException(nameof(outputIndex),
                $"Output index {outputIndex} is out of bounds for prediction with {outputs} outputs.");

        var plus = engine.TensorSliceAxis(engine.TensorStack(plusOutputs, 0), 1, outputIndex);
        var minus = engine.TensorSliceAxis(engine.TensorStack(minusOutputs, 0), 1, outputIndex);
        return engine.TensorDivideScalar(engine.TensorSubtract(plus, minus), NumOps.FromDouble(2 * epsilon)).ToVector();
    }

    /// <summary>
    /// "+" for entries at or above zero and "" for negative ones (a negative value prints its own sign).
    /// </summary>
    public static string[] SignPrefixes(IEnumerable<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var vector = new Vector<T>(values).AsTensor();
        var atOrAboveZero = engine.TensorGreaterEqual(vector, engine.TensorFullLike(vector, NumOps.Zero)).Cast<int>().ToVector();
        return Enumerable.Select(atOrAboveZero, flag => flag != 0 ? "+" : "").ToArray();
    }
}
