namespace AiDotNet.Classification.TimeSeries;

/// <summary>
/// The linear head ROCKET and MiniRocket put on their features: scikit-learn's
/// <c>StandardScaler(with_mean=False)</c> followed by <c>RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))</c>,
/// the pipeline sktime's <c>RocketClassifier</c> and <c>MiniRocketClassifier</c> build.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The targets are +1/-1 indicators (one column for two classes, one per class otherwise), the intercept is
/// fitted without penalty, and the ridge strength is the candidate with the smallest leave-one-out squared error,
/// computed in closed form from one eigendecomposition (scikit-learn's efficient GCV): the hat matrix of the
/// centered problem is <c>H = B diag(f(e, alpha)) B'</c> for an eigenbasis <c>B</c>, plus <c>1/n</c> for the
/// intercept, and the leave-one-out residual is <c>(y_i - yhat_i) / (1 - H_ii)</c>.</para>
/// <para>The fitted parameters are laid out as the per-output coefficients over the raw (unscaled) features,
/// output by output, followed by one intercept per output.</para>
/// </remarks>
internal static class RidgeClassifierCv<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>scikit-learn's zero-scale threshold: a feature whose scale is below 10 machine epsilons keeps scale 1.</summary>
    private const double ZeroScale = 10 * 2.220446049250313e-16;

    /// <summary>The candidate ridge strengths: <c>np.logspace(-3, 3, 10)</c>.</summary>
    internal static double[] DefaultAlphas { get; } =
        Enumerable.Range(0, 10).Select(i => Math.Pow(10.0, -3.0 + (6.0 * i / 9.0))).ToArray();

    /// <summary>The number of outputs (target columns) for a class count.</summary>
    internal static int OutputCount(int classCount) => classCount == 2 ? 1 : classCount;

    /// <summary>Fits the scaled, cross-validated ridge classifier and returns its parameter vector.</summary>
    /// <param name="features">The training features, one row per sample.</param>
    /// <param name="labels">The class label of each sample.</param>
    /// <param name="classLabels">The distinct class labels in order; with two classes the last is the positive one.</param>
    /// <param name="alphas">The candidate ridge strengths, in the order ties are broken (first wins).</param>
    internal static Vector<T> Fit(Matrix<T> features, Vector<T> labels, Vector<T> classLabels, IReadOnlyList<double> alphas)
    {
        var engine = AiDotNetEngine.Current;
        int n = features.Rows, p = features.Columns, classes = classLabels.Length;
        int outputs = OutputCount(classes);
        int candidates = alphas.Count;
        var x = new Tensor<T>(new[] { n, p }, features);

        // +1 / -1 targets: the positive (last) class for two classes, one indicator column per class otherwise.
        var targetClasses = outputs == 1
            ? engine.TensorSlice(classLabels.AsTensor(), new[] { classes - 1 }, new[] { 1 })
            : classLabels.AsTensor();
        var matches = engine.TensorEquals(
            engine.TensorBroadcastTo(engine.Reshape(labels.AsTensor(), new[] { n, 1 }), new[] { n, outputs }),
            engine.TensorBroadcastTo(engine.Reshape(targetClasses, new[] { 1, outputs }), new[] { n, outputs }));
        var y = engine.TensorAddScalar(engine.TensorMultiplyScalar(matches, NumOps.FromDouble(2.0)), NumOps.Negate(NumOps.One));

        // StandardScaler(with_mean=False): divide each feature by its population standard deviation. Centering for
        // the unpenalized intercept commutes with the scaling, so the centered scaled features come out directly.
        var featureShape = new[] { n, p };
        var mean = engine.ReduceMean(x, new[] { 0 }, true);
        var centeredRaw = engine.TensorSubtract(x, engine.TensorBroadcastTo(mean, featureShape));
        var rawScale = engine.TensorSqrt(engine.ReduceMean(engine.TensorMultiply(centeredRaw, centeredRaw), new[] { 0 }, true));
        // At or below the threshold, not strictly below: in a type that cannot represent ZeroScale (Half rounds it to
        // zero) an exactly constant feature must still get scale one instead of dividing by zero.
        var ones = engine.TensorOnesLike(rawScale);
        var scale = engine.TensorWhere(
            engine.TensorLessEqual(rawScale, engine.TensorMultiplyScalar(ones, NumOps.FromDouble(ZeroScale))), ones, rawScale);
        var centered = engine.TensorDivide(centeredRaw, engine.TensorBroadcastTo(scale, featureShape));
        var scaledMean = engine.TensorDivide(mean, scale);
        var yMean = engine.ReduceMean(y, new[] { 0 }, true);
        var yCentered = engine.TensorSubtract(y, engine.TensorBroadcastTo(yMean, new[] { n, outputs }));

        // Eigen route: the Gram matrix (n <= p) or the covariance matrix (n > p), whichever is smaller.
        Tensor<T> basis;        // B: hat matrix H = B diag(fitShrink) B'
        Tensor<T> coefficientBasis;  // C: coefficients W = C diag(coefficientShrink) B' y
        Tensor<T> eigenvalues;
        bool gram = n <= p;
        if (gram)
        {
            var (values, vectors) = LinalgDispatch<T>.Eigh(engine.TensorMatMul(centered, engine.TensorTranspose(centered)));
            eigenvalues = values;
            basis = vectors;
            coefficientBasis = engine.TensorMatMul(engine.TensorTranspose(centered), vectors);
        }
        else
        {
            var (values, vectors) = LinalgDispatch<T>.Eigh(engine.TensorMatMul(engine.TensorTranspose(centered), centered));
            eigenvalues = values;
            basis = engine.TensorMatMul(centered, vectors);
            coefficientBasis = vectors;
        }

        int m = eigenvalues.Length;
        // The candidates are a fixed table of constants, converted to T once in bulk.
        var alphaValues = new Tensor<double>(alphas.ToArray(), new[] { candidates }).Cast<T>();
        var grid = new[] { m, candidates };
        var lambdaGrid = engine.TensorBroadcastTo(engine.Reshape(eigenvalues, new[] { m, 1 }), grid);
        var inverse = engine.TensorDivide(
            engine.TensorOnesLike(lambdaGrid),
            engine.TensorAdd(lambdaGrid, engine.TensorBroadcastTo(engine.Reshape(alphaValues, new[] { 1, candidates }), grid)));
        var fitShrink = gram ? engine.TensorMultiply(lambdaGrid, inverse) : inverse;

        // Leave-one-out residuals for every candidate at once: [n, outputs, candidates].
        var projected = engine.TensorMatMul(engine.TensorTranspose(basis), yCentered);
        var hatDiagonal = engine.TensorAddScalar(
            engine.TensorMatMul(engine.TensorMultiply(basis, basis), fitShrink), NumOps.FromDouble(1.0 / n));
        var cube = new[] { m, outputs, candidates };
        var shrunk = engine.TensorMultiply(
            engine.TensorBroadcastTo(engine.Reshape(fitShrink, new[] { m, 1, candidates }), cube),
            engine.TensorBroadcastTo(engine.Reshape(projected, new[] { m, outputs, 1 }), cube));
        var sampleCube = new[] { n, outputs, candidates };
        var fitted = engine.Reshape(
            engine.TensorMatMul(basis, engine.Reshape(shrunk, new[] { m, outputs * candidates })), sampleCube);
        var residual = engine.TensorDivide(
            engine.TensorSubtract(engine.TensorBroadcastTo(engine.Reshape(yCentered, new[] { n, outputs, 1 }), sampleCube), fitted),
            engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorSubtract(engine.TensorOnesLike(hatDiagonal), hatDiagonal), new[] { n, 1, candidates }),
                sampleCube));
        var errors = engine.ReduceSum(engine.TensorMultiply(residual, residual), new[] { 0, 1 }, false);

        // The first candidate with the smallest error (scikit-learn keeps the first best score).
        var best = FirstIndexOf(errors, engine.TensorMinValue(errors));
        var bestAlpha = engine.TensorTake(alphaValues, best);

        var coefficientShrink = engine.TensorDivide(
            engine.TensorOnesLike(eigenvalues),
            engine.TensorAdd(eigenvalues, engine.TensorBroadcastTo(bestAlpha, new[] { m })));
        var weights = engine.TensorMatMul(
            coefficientBasis,
            engine.TensorMultiply(engine.TensorBroadcastTo(engine.Reshape(coefficientShrink, new[] { m, 1 }), new[] { m, outputs }), projected));

        // Fold the scaling into the coefficients so prediction reads raw features: w_raw = w / scale and
        // b = mean(y) - mean(x / scale) . w.
        var intercepts = engine.TensorSubtract(yMean, engine.TensorMatMul(scaledMean, weights));
        var rawWeights = engine.TensorDivide(
            weights, engine.TensorBroadcastTo(engine.Reshape(scale, new[] { p, 1 }), new[] { p, outputs }));
        var parameters = engine.TensorConcatenate(new[]
        {
            engine.Reshape(engine.TensorTranspose(rawWeights), new[] { outputs * p }),
            engine.Reshape(intercepts, new[] { outputs })
        }, 0);
        return parameters.ToVector();
    }

    /// <summary>Predicts class labels with a parameter vector produced by <see cref="Fit"/>.</summary>
    internal static Vector<T> Predict(Matrix<T> features, Vector<T> parameters, Vector<T> classLabels, int featureCount)
    {
        var engine = AiDotNetEngine.Current;
        int n = features.Rows, classes = classLabels.Length;
        int outputs = OutputCount(classes);
        if (features.Columns != featureCount)
        {
            throw new ArgumentException(
                $"Expected {featureCount} features per row, got {features.Columns}.", nameof(features));
        }

        if (parameters.Length != outputs * (featureCount + 1))
        {
            throw new ArgumentException(
                $"Expected {outputs * (featureCount + 1)} parameters, got {parameters.Length}.", nameof(parameters));
        }

        var all = parameters.AsTensor();
        var weights = engine.Reshape(engine.TensorSlice(all, new[] { 0 }, new[] { outputs * featureCount }), new[] { outputs, featureCount });
        var intercepts = engine.Reshape(engine.TensorSlice(all, new[] { outputs * featureCount }, new[] { outputs }), new[] { 1, outputs });
        var scores = engine.TensorAdd(
            engine.TensorMatMul(new Tensor<T>(new[] { n, featureCount }, features), engine.TensorTranspose(weights)),
            engine.TensorBroadcastTo(intercepts, new[] { n, outputs }));

        var labels = classLabels.AsTensor();
        if (outputs == 1)
        {
            // A positive decision value is the last class; zero is the first (scikit-learn's LinearClassifierMixin:
            // scores > 0).
            var positive = engine.TensorGreaterThan(engine.Reshape(scores, new[] { n }), new Tensor<T>(new[] { n }));
            var last = engine.TensorBroadcastTo(engine.TensorSlice(labels, new[] { classes - 1 }, new[] { 1 }), new[] { n });
            var first = engine.TensorBroadcastTo(engine.TensorSlice(labels, new[] { 0 }, new[] { 1 }), new[] { n });
            return engine.TensorWhere(positive, last, first).ToVector();
        }

        return engine.TensorTake(labels, FirstArgMaxPerRow(scores)).ToVector();
    }

    /// <summary>The index of each row's first maximum: the earliest maximal entry carries the largest weight.</summary>
    internal static Tensor<int> FirstArgMaxPerRow(Tensor<T> rows)
    {
        var engine = AiDotNetEngine.Current;
        int count = rows.Shape[0], width = rows.Shape[1];
        var shape = new[] { count, width };
        var rowMax = engine.TensorBroadcastTo(engine.ReduceMax(rows, new[] { 1 }, true), shape);
        return engine.TensorArgMax(engine.TensorMultiply(engine.TensorEquals(rows, rowMax), DescendingWeights(width, count)), 1);
    }

    /// <summary>The (1-element) index of the first entry of a 1-D tensor equal to <paramref name="value"/>.</summary>
    private static Tensor<int> FirstIndexOf(Tensor<T> values, T value)
    {
        var engine = AiDotNetEngine.Current;
        int width = values.Length;
        var hits = engine.Reshape(engine.TensorEquals(values, value), new[] { 1, width });
        return engine.Reshape(engine.TensorArgMax(engine.TensorMultiply(hits, DescendingWeights(width, 1)), 1), new[] { 1 });
    }

    /// <summary>Rows of (width, width - 1, ..., 1).</summary>
    private static Tensor<T> DescendingWeights(int width, int rows)
    {
        var engine = AiDotNetEngine.Current;
        var descending = engine.TensorAddScalar(
            engine.TensorNegate(engine.TensorArange<T>(0, width, 1)), NumOps.FromDouble(width));
        return engine.TensorBroadcastTo(engine.Reshape(descending, new[] { 1, width }), new[] { rows, width });
    }
}
