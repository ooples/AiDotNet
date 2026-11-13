namespace AiDotNet.Regression;

/// <summary>
/// Batched building blocks shared by the neural survival models (<see cref="DeepSurv{T}"/> and
/// <see cref="DeepHit{T}"/>): every activation is an [n, width] tensor and every step is a whole-array
/// engine operation, so neither model walks samples or units one element at a time.
/// </summary>
internal static class SurvivalNetworkOps
{
    /// <summary>
    /// Upper bound on the elements of one pairwise comparison block, so concordance counting over a
    /// large evaluation set keeps its intermediate [rows, n] masks bounded.
    /// </summary>
    private const long PairwiseBlockElements = 1L << 24;

    /// <summary>
    /// Uniform(-scale, scale) weights, drawn from <paramref name="random"/> in row-major order -- the
    /// same draw order as a nested row/column loop, so a seeded model initializes identically.
    /// </summary>
    public static Matrix<T> UniformWeights<T>(Random random, int rows, int columns, T scale)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var unit = Tensor<T>.FromMatrix(Matrix<T>.CreateRandom(random, rows, columns));

        // (u * 2 - 1) * scale
        var centered = engine.TensorSubtractScalar(engine.TensorMultiplyScalar(unit, numOps.FromDouble(2.0)), numOps.One);
        return engine.TensorMultiplyScalar(centered, scale).ToMatrix();
    }

    /// <summary>
    /// The He-style scale sqrt(2 / fanIn) used by both models' initializers.
    /// </summary>
    public static T HeScale<T>(int fanIn)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Sqrt(numOps.Divide(numOps.FromDouble(2.0), numOps.FromDouble(fanIn)));
    }

    /// <summary>
    /// An inverted-dropout mask of shape [rows, columns]: <paramref name="keepScale"/> where the unit is
    /// kept and 0 where it is dropped. One uniform draw per cell in row-major order; a unit is kept when
    /// its draw is not below <paramref name="dropoutRate"/>.
    /// </summary>
    public static Tensor<T> DropoutMask<T>(Random random, int rows, int columns, T dropoutRate, T keepScale)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var unit = Tensor<T>.FromMatrix(Matrix<T>.CreateRandom(random, rows, columns));
        var keep = engine.ScalarMinusTensor(numOps.One, engine.TensorLessThan(unit, dropoutRate));
        return engine.TensorMultiplyScalar(keep, keepScale);
    }

    /// <summary>
    /// Per-column mean and population standard deviation; a column whose deviation is below 1e-10 gets
    /// a deviation of 1 so it standardizes to a constant 0 instead of dividing by zero.
    /// </summary>
    public static (Vector<T> Mean, Vector<T> Std) ColumnMeanAndStd<T>(Matrix<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = x.Rows;
        int columns = x.Columns;
        var data = Tensor<T>.FromMatrix(x);

        var mean = engine.ReduceMean(data, new[] { 0 }, keepDims: false).ToVector();
        var centered = engine.TensorSubtract(data, BroadcastRow(mean, n));
        var variance = engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, keepDims: false);
        var std = engine.TensorSqrt(variance);
        var degenerate = engine.TensorLessThan(std, numOps.FromDouble(1e-10));
        std = engine.TensorWhere(degenerate, Tensor<T>.CreateDefault(new[] { columns }, numOps.One), std);

        return (mean, std.ToVector());
    }

    /// <summary>
    /// The rows of <paramref name="x"/> at <paramref name="indices"/>, cut to the first
    /// <paramref name="numFeatures"/> columns and standardized with the training statistics, as an
    /// [indices.Length, numFeatures] tensor. Without statistics (an untrained model) the rows are raw.
    /// </summary>
    public static Tensor<T> StandardizedRows<T>(Matrix<T> x, int[] indices, int numFeatures, Vector<T>? mean, Vector<T>? std)
    {
        var engine = AiDotNetEngine.Current;
        int n = indices.Length;
        var source = x.Columns == numFeatures ? x : x.GetSubMatrix(0, 0, x.Rows, numFeatures);
        var rows = engine.TensorIndexSelect(Tensor<T>.FromMatrix(source), new Tensor<int>(indices, new[] { n }), 0);

        if (mean is null || std is null || mean.Length != numFeatures || std.Length != numFeatures)
        {
            return rows;
        }

        return engine.TensorDivide(
            engine.TensorSubtract(rows, BroadcastRow(mean, n)),
            BroadcastRow(std, n));
    }

    /// <summary>
    /// A [rows, values.Length] tensor whose every row is <paramref name="values"/>.
    /// </summary>
    public static Tensor<T> BroadcastRow<T>(Vector<T> values, int rows)
    {
        return AiDotNetEngine.Current.TensorBroadcastTo(
            new Tensor<T>(new[] { 1, values.Length }, values), new[] { rows, values.Length });
    }

    /// <summary>
    /// A [values.Length, columns] tensor whose every column is <paramref name="values"/>.
    /// </summary>
    public static Tensor<T> BroadcastColumn<T>(Vector<T> values, int columns)
    {
        return AiDotNetEngine.Current.TensorBroadcastTo(
            new Tensor<T>(new[] { values.Length, 1 }, values), new[] { values.Length, columns });
    }

    /// <summary>
    /// The affine map input * W + b for a whole batch: [rows, in] times [in, out] plus the bias row.
    /// </summary>
    public static Tensor<T> Linear<T>(Tensor<T> input, int rows, Matrix<T> weights, Vector<T> biases)
    {
        var engine = AiDotNetEngine.Current;
        var product = engine.TensorMatMul(input, Tensor<T>.FromMatrix(weights));
        return engine.TensorAdd(product, BroadcastRow(biases, rows));
    }

    /// <summary>
    /// Backward pass of <see cref="Linear{T}"/>: given d(loss)/d(output) for the batch, returns the
    /// weight gradient input^T * d, the bias gradient (column sums of d), and d * W^T for the layer below.
    /// </summary>
    public static (Matrix<T> WeightGradient, Vector<T> BiasGradient, Tensor<T> InputDelta) LinearBackward<T>(
        Tensor<T> input, Tensor<T> outputDelta, Matrix<T> weights)
    {
        var engine = AiDotNetEngine.Current;
        var weightGradient = engine.TensorMatMul(engine.TensorTranspose(input), outputDelta).ToMatrix();
        var biasGradient = engine.ReduceSum(outputDelta, new[] { 0 }, keepDims: false).ToVector();
        var inputDelta = engine.TensorMatMul(outputDelta, engine.TensorTranspose(Tensor<T>.FromMatrix(weights)));
        return (weightGradient, biasGradient, inputDelta);
    }

    /// <summary>
    /// Bias-corrected Adam coefficients for one optimizer step (beta1 0.9, beta2 0.999, eps 1e-8).
    /// </summary>
    public readonly struct AdamCoefficients<T>
    {
        public AdamCoefficients(double learningRate, double l2, int step)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            LearningRate = numOps.FromDouble(learningRate);
            L2 = numOps.FromDouble(l2);
            Beta1 = numOps.FromDouble(0.9);
            Beta2 = numOps.FromDouble(0.999);
            OneMinusBeta1 = numOps.Subtract(numOps.One, Beta1);
            OneMinusBeta2 = numOps.Subtract(numOps.One, Beta2);
            Epsilon = numOps.FromDouble(1e-8);
            T stepT = numOps.FromDouble(step);
            Correction1 = numOps.Subtract(numOps.One, numOps.Power(Beta1, stepT));
            Correction2 = numOps.Subtract(numOps.One, numOps.Power(Beta2, stepT));
        }

        public T LearningRate { get; }
        public T L2 { get; }
        public T Beta1 { get; }
        public T Beta2 { get; }
        public T OneMinusBeta1 { get; }
        public T OneMinusBeta2 { get; }
        public T Epsilon { get; }
        public T Correction1 { get; }
        public T Correction2 { get; }
    }

    /// <summary>
    /// One Adam step on a weight matrix, optionally with the L2 penalty added to the gradient
    /// (g + l2 * w), writing the parameter and both moment estimates back into the live storage.
    /// </summary>
    public static void AdamStep<T>(Matrix<T> parameter, Matrix<T> gradient, Matrix<T> m, Matrix<T> v,
        AdamCoefficients<T> c, bool applyL2)
    {
        var (p, mNew, vNew) = AdamTensors(Tensor<T>.FromMatrix(parameter), Tensor<T>.FromMatrix(gradient),
            Tensor<T>.FromMatrix(m), Tensor<T>.FromMatrix(v), c, applyL2);
        p.CopyTo(parameter.AsWritableSpan());
        mNew.CopyTo(m.AsWritableSpan());
        vNew.CopyTo(v.AsWritableSpan());
    }

    /// <summary>
    /// One Adam step on a parameter vector, writing the parameter and both moments back in place.
    /// </summary>
    public static void AdamStep<T>(Vector<T> parameter, Vector<T> gradient, Vector<T> m, Vector<T> v,
        AdamCoefficients<T> c, bool applyL2)
    {
        var (p, mNew, vNew) = AdamTensors(Tensor<T>.FromVector(parameter), Tensor<T>.FromVector(gradient),
            Tensor<T>.FromVector(m), Tensor<T>.FromVector(v), c, applyL2);
        p.CopyTo(parameter.AsWritableSpan());
        mNew.CopyTo(m.AsWritableSpan());
        vNew.CopyTo(v.AsWritableSpan());
    }

    private static (Tensor<T> Parameter, Tensor<T> M, Tensor<T> V) AdamTensors<T>(
        Tensor<T> parameter, Tensor<T> gradient, Tensor<T> m, Tensor<T> v, AdamCoefficients<T> c, bool applyL2)
    {
        var engine = AiDotNetEngine.Current;
        var g = applyL2
            ? engine.TensorAdd(gradient, engine.TensorMultiplyScalar(parameter, c.L2))
            : gradient;

        var mNew = engine.TensorAdd(engine.TensorMultiplyScalar(m, c.Beta1), engine.TensorMultiplyScalar(g, c.OneMinusBeta1));
        var vNew = engine.TensorAdd(
            engine.TensorMultiplyScalar(v, c.Beta2),
            engine.TensorMultiply(engine.TensorMultiplyScalar(g, c.OneMinusBeta2), g));

        // lr * (m / c1) / (sqrt(v / c2) + eps)
        var numerator = engine.TensorMultiplyScalar(engine.TensorDivideScalar(mNew, c.Correction1), c.LearningRate);
        var denominator = engine.TensorAddScalar(engine.TensorSqrt(engine.TensorDivideScalar(vNew, c.Correction2)), c.Epsilon);
        var updated = engine.TensorSubtract(parameter, engine.TensorDivide(numerator, denominator));
        return (updated, mNew, vNew);
    }

    /// <summary>
    /// Per-row sum of |W| for the first layer, normalized to sum to one when the total is positive --
    /// the first-layer weight-magnitude proxy both models report as feature importance.
    /// </summary>
    public static Vector<T> FirstLayerImportances<T>(Matrix<T>? firstLayer, int numFeatures)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        if (firstLayer is null || numFeatures == 0 || firstLayer.Rows < numFeatures)
        {
            return new Vector<T>(numFeatures);
        }

        var rows = firstLayer.Rows == numFeatures
            ? firstLayer
            : firstLayer.GetSubMatrix(0, 0, numFeatures, firstLayer.Columns);
        var importances = engine.ReduceSum(
            engine.TensorAbs(Tensor<T>.FromMatrix(rows)), new[] { 1 }, keepDims: false).ToVector();
        T total = engine.Sum(importances);
        return numOps.GreaterThan(total, numOps.Zero) ? engine.Divide(importances, total) : importances;
    }

    /// <summary>
    /// Harrell's concordance counts over every ordered pair (i, j) with an observed event for i and
    /// times[i] &lt; times[j]. A pair is concordant when the scores rank i as failing first and
    /// discordant when they rank it the other way; equal scores count as neither.
    /// </summary>
    /// <param name="times">Observed times.</param>
    /// <param name="events">Event codes; zero means censored.</param>
    /// <param name="scores">Per-subject score.</param>
    /// <param name="higherScoreFailsFirst">True for risk scores, false for predicted times.</param>
    public static (int Concordant, int Discordant) ConcordanceCounts<T>(
        Vector<T> times, Vector<T> events, Vector<T> scores, bool higherScoreFailsFirst)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = times.Length;
        if (n == 0)
        {
            return (0, 0);
        }

        var observed = engine.ScalarMinusTensor(
            numOps.One, engine.TensorEquals(Tensor<T>.FromVector(events), numOps.Zero)).ToVector();
        int blockRows = (int)Math.Max(1L, Math.Min(n, PairwiseBlockElements / n));
        int concordant = 0;
        int discordant = 0;

        for (int start = 0; start < n; start += blockRows)
        {
            int count = Math.Min(blockRows, n - start);
            var timeI = BroadcastColumn(times.GetSubVector(start, count), n);
            var timeJ = BroadcastRow(times, count);
            var scoreI = BroadcastColumn(scores.GetSubVector(start, count), n);
            var scoreJ = BroadcastRow(scores, count);

            var comparable = engine.TensorMultiply(
                engine.TensorLessThan(timeI, timeJ),
                BroadcastColumn(observed.GetSubVector(start, count), n));

            var iFirst = higherScoreFailsFirst
                ? engine.TensorGreaterThan(scoreI, scoreJ)
                : engine.TensorLessThan(scoreI, scoreJ);
            var jFirst = higherScoreFailsFirst
                ? engine.TensorLessThan(scoreI, scoreJ)
                : engine.TensorGreaterThan(scoreI, scoreJ);

            concordant += engine.TensorCountNonzero(engine.TensorMultiply(comparable, iFirst));
            discordant += engine.TensorCountNonzero(engine.TensorMultiply(comparable, jFirst));
        }

        return (concordant, discordant);
    }
}
