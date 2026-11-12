namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates or corrects MRI bias field (intensity non-uniformity).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BiasFieldCorrection<T> : ImageAugmenterBase<T>
{
    public double MinCoeff { get; }
    public double MaxCoeff { get; }
    public int Order { get; }

    public BiasFieldCorrection(double minCoeff = 0.0, double maxCoeff = 0.3,
        int order = 3, double probability = 0.5) : base(probability)
    {
        MinCoeff = minCoeff; MaxCoeff = maxCoeff; Order = order;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Generate polynomial bias field: for each term 0 < i + j <= Order in row-major (i, j) order the
        // context drew GetRandomDouble(MinCoeff, MaxCoeff) then GetRandomBool(), i.e. two NextDouble
        // draws per term; draw them in that order and scatter them into the [Order + 1, Order + 1] grid
        var engine = Engine;
        T two = NumOps.FromDouble(2.0);
        int side = Order + 1;
        var termCells = Enumerable.Range(0, side * side)
            .Where(cell => cell / side + cell % side > 0 && cell / side + cell % side <= Order)
            .ToArray();
        var coeffs = new Tensor<T>([side * side]);
        if (termCells.Length > 0)
        {
            var draws = AugmentationRandom<T>.Unit(context, [termCells.Length, 2]);
            var magnitude = engine.TensorAddScalar(
                engine.TensorMultiplyScalar(engine.TensorSliceAxis(draws, 1, 0), NumOps.FromDouble(MaxCoeff - MinCoeff)),
                NumOps.FromDouble(MinCoeff));

            // GetRandomBool() is NextDouble() < 0.5: +1 when true, -1 otherwise
            var sign = engine.TensorSubtractScalar(
                engine.TensorMultiplyScalar(engine.TensorLessThan(engine.TensorSliceAxis(draws, 1, 1), NumOps.FromDouble(0.5)), two),
                NumOps.One);
            coeffs = engine.TensorScatter(coeffs, new Tensor<int>(termCells, [termCells.Length]),
                engine.TensorMultiply(magnitude, sign), 0);
        }

        // Polynomial bias field over normalized coordinates in [-1, 1]
        var ny = engine.Reshape(engine.TensorSubtractScalar(
            engine.TensorDivideScalar(engine.TensorMultiplyScalar(engine.TensorArange<T>(0, data.Height, 1), two), NumOps.FromDouble(data.Height)), NumOps.One),
            [data.Height, 1]);
        var nx = engine.Reshape(engine.TensorSubtractScalar(
            engine.TensorDivideScalar(engine.TensorMultiplyScalar(engine.TensorArange<T>(0, data.Width, 1), two), NumOps.FromDouble(data.Width)), NumOps.One),
            [1, data.Width]);

        // bias = 1 + sum_ij coeffs[i, j] * ny^i * nx^j = 1 + Y * C * X with Y[:, i] = ny^i and X[j, :] = nx^j
        // (coeffs is zero outside 0 < i + j <= Order)
        var rowPowers = new Tensor<T>[Order + 1];
        var colPowers = new Tensor<T>[Order + 1];
        rowPowers[0] = ImageTensor<T>.Constant(ny, 1.0);
        colPowers[0] = ImageTensor<T>.Constant(nx, 1.0);
        for (int p = 1; p <= Order; p++)
        {
            rowPowers[p] = engine.TensorMultiply(rowPowers[p - 1], ny);
            colPowers[p] = engine.TensorMultiply(colPowers[p - 1], nx);
        }

        var field = engine.TensorMatMul(
            engine.TensorMatMul(engine.TensorConcatenate(rowPowers, 1), engine.Reshape(coeffs, [side, side])),
            engine.TensorConcatenate(colPowers, 0));
        var bias = engine.TensorClamp(engine.TensorAddScalar(field, NumOps.One), NumOps.FromDouble(0.5), NumOps.FromDouble(1.5)); // Clamp bias field

        var scaled = engine.TensorMultiply(data.ToPlanarTensor(), engine.Reshape(bias, [1, data.Height, data.Width]));
        result.SetFromPlanarTensor(engine.TensorClamp(scaled, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_coeff"] = MinCoeff; p["max_coeff"] = MaxCoeff; p["order"] = Order;
        return p;
    }
}
