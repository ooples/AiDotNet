using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.Regression;

/// <summary>
/// Implements Radial Basis Function (RBF) Regression, a technique that uses radial basis functions
/// as the basis for approximating complex nonlinear relationships between inputs and outputs.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Radial Basis Function Regression works by transforming the input space using a set of radial basis functions,
/// each centered at a different point. These functions produce a response that depends on the distance from the
/// input to the center point. The model then combines these responses linearly to make predictions.
/// </para>
/// <para>
/// The algorithm first selects a set of centers (typically using k-means clustering), computes the RBF features
/// for each input point, and then solves a linear regression problem to find the optimal weights.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Think of RBF regression as placing a set of "bell curves" at strategic locations in your input space.
/// Each curve gives a strong response when an input is close to its center and a weak response when it's far away.
/// The model predicts by combining these responses with learned weights. This approach is particularly good at
/// modeling complex, non-linear relationships in data.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an RBF regression with radial basis function features
/// var options = new RadialBasisFunctionOptions();
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train with RBF centers and weighted linear combination
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new RadialBasisFunctionRegression&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict for a new sample
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 6, 7 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Kernel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Radial Basis Functions", "https://doi.org/10.1017/CBO9780511543241")]
public partial class RadialBasisFunctionRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Configuration options for the radial basis function regression model.
    /// </summary>
    /// <value>
    /// Contains settings like the number of centers, gamma parameter, and random seed.
    /// </value>
    private readonly RadialBasisFunctionOptions _options;

    /// <summary>
    /// Relative scale factor for the adaptive ridge penalty (fraction of mean diagonal magnitude).
    /// </summary>
    private const double StabilityLambdaScale = 1e-6;

    /// <summary>
    /// Absolute floor for the ridge penalty when diagonal magnitude is near zero.
    /// </summary>
    private const double MinimumStabilityLambda = 1e-6;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The centers of the radial basis functions.
    /// </summary>
    /// <value>
    /// A matrix where each row represents a center point in the input space.
    /// </value>
    [Buffer]
    private Matrix<T> _centers;

    /// <summary>
    /// The weights used to combine the radial basis function outputs.
    /// </summary>
    /// <value>
    /// A vector of weights, including a bias term.
    /// </value>
    [Buffer]
    private Vector<T> _weights;

    /// <summary>
    /// Initializes a new instance of the RadialBasisFunctionRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the RBF regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the RBF regression model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public RadialBasisFunctionRegression(RadialBasisFunctionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new RadialBasisFunctionOptions();
        _centers = Matrix<T>.Empty();
        _weights = Vector<T>.Empty();
    }

    /// <summary>
    /// Optimizes the model parameters based on the training data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method implements the core of the RBF regression algorithm. The steps are:
    /// 1. Select centers using k-means clustering
    /// 2. Compute RBF features for each input point
    /// 3. Apply regularization to the RBF features
    /// 4. Solve a linear regression problem to find the optimal weights
    /// 5. Apply regularization to the weights
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is the main training method where the model learns from your data. It first finds good locations
    /// for the "bell curves" (centers) using a clustering algorithm, then calculates how each input point
    /// responds to these centers. Finally, it solves a linear equation to find the best weights for combining
    /// these responses to predict the target values.
    /// </para>
    /// </remarks>
    /// <summary>
    /// RBF solves analytically — no optimizer parameter injection.
    /// </summary>
        /// <remarks>
    /// Expressed as a capability, not as a count. A zero ParameterCount also suppresses
    /// injection -- that is why this was written that way -- but it overloads a COUNT to carry
    /// a CAPABILITY: the model does have parameters (the base getter returns its coefficients
    /// and intercept), so the count contradicted the vector and anything pairing the two by
    /// length saw parameters the model claimed not to have.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

    /// <summary>
    /// Returns all features since RBF uses distance-based kernels across all features.
    /// </summary>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        int numFeatures = _centers.Columns > 0 ? _centers.Columns : 0;
        return Enumerable.Range(0, numFeatures);
    }

    /// <inheritdoc />
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Auto-scale gamma if using the default value of 1.0
        if (Math.Abs(_options.Gamma - 1.0) < 1e-10)
        {
            T totalVar = RegressionEngineOps2.TotalColumnVariance(x);
            T columns = NumOps.FromDouble(x.Columns);
            T gamma = NumOps.GreaterThan(totalVar, NumOps.FromDouble(1e-10))
                ? NumOps.Divide(NumOps.One, NumOps.Multiply(columns, NumOps.Divide(totalVar, columns)))
                : NumOps.One;
            // The option is a plain configuration number, so the fitted scalar is stored back into it.
            _options.Gamma = NumOps.ToDouble(gamma);
        }

        // Select centers
        _centers = SelectCenters(x);

        // Compute RBF features
        Matrix<T> rbfFeatures = ComputeRBFFeatures(x);

        // Solve for weights using linear regression
        // Note: Regularization is applied within SolveLinearRegression via ridge penalty
        _weights = SolveLinearRegression(rbfFeatures, y);
    }

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method transforms the input data using the RBF features and then applies the learned weights
    /// to make predictions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// After training, this method is used to make predictions on new data. It first transforms each input
    /// example using the radial basis functions (calculating how close it is to each center), then combines
    /// these transformed values using the learned weights to produce the final prediction.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        Matrix<T> rbfFeatures = ComputeRBFFeatures(input);
        // RBF features are computed directly - no transformation needed
        return rbfFeatures.Multiply(_weights);
    }

    /// <summary>
    /// Predicts the value for a single input vector.
    /// </summary>
    /// <param name="input">The input feature vector.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// This method transforms a single input vector using the RBF features and then applies the learned weights
    /// to make a prediction.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This is the core prediction function for a single example. It calculates how the input responds to each
    /// radial basis function (center), then combines these responses using the learned weights to produce
    /// the final prediction.
    /// </para>
    /// </remarks>
    protected override T PredictSingle(Vector<T> input)
    {
        Vector<T> rbfFeatures = ComputeRBFFeaturesSingle(input);
        // RBF features are computed directly - no transformation needed
        return rbfFeatures.DotProduct(_weights);
    }

    /// <summary>
    /// Selects centers for the radial basis functions using k-means clustering.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <returns>A matrix where each row represents a center point.</returns>
    /// <remarks>
    /// <para>
    /// This method implements k-means clustering to select centers for the radial basis functions.
    /// The steps are:
    /// 1. Initialize centers randomly
    /// 2. Iterate until convergence or maximum iterations:
    ///    a. Assign each point to the nearest center
    ///    b. Recompute centers as the mean of assigned points
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method finds good locations for the "bell curves" (centers) by grouping similar data points together
    /// and placing a center at the middle of each group. It uses an algorithm called k-means clustering,
    /// which iteratively assigns points to the nearest center and then updates the centers based on these assignments.
    /// </para>
    /// </remarks>
    private Matrix<T> SelectCenters(Matrix<T> x)
    {
        int numCenters = Math.Min(_options.NumberOfCenters, x.Rows);

        // Deterministic k-means++ farthest-point seeding: centers are a pure
        // function of X (no RNG). This is essential for invariants like
        // scaling/translation-equivariance, where two model instances trained
        // on the same X with differently-scaled y must produce predictions
        // that scale linearly with y. Random init breaks that property
        // because two separate models pick different starting centers and
        // converge to different local minima, even though the math says
        // weights = (XᵀX + λI)⁻¹ Xᵀy is linear in y for any *fixed* X-derived
        // feature matrix.
        //
        // Algorithm: first center is x[0]; each subsequent center is the
        // point with the largest min-distance to the centers selected so far
        // (argmax of d²-min over current centers). This is the deterministic
        // farthest-point variant of k-means++ initialization.
        var centers = new Matrix<T>(numCenters, x.Columns);
        centers.SetRow(0, x.GetRow(0));

        if (numCenters > 1)
        {
            // minDistSq[i] = squared distance from x[i] to its closest current center.
            var firstDistance = RegressionEngineOps2.PairwiseDistances(x, centers.GetSubMatrix(0, 0, 1, x.Columns)).GetColumn(0);
            var minDistSq = Engine.Multiply(firstDistance, firstDistance);

            for (int c = 1; c < numCenters; c++)
            {
                // First point with the largest min-distance (a strict > scan keeps the earliest).
                int farthest = Engine.TensorArgMax(minDistSq.AsTensor(), 0)[0];
                centers.SetRow(c, x.GetRow(farthest));

                // Update minDistSq with the new center's contribution.
                var distance = RegressionEngineOps2.PairwiseDistances(x, centers.GetSubMatrix(c, 0, 1, x.Columns)).GetColumn(0);
                minDistSq = Engine.Min(minDistSq, Engine.Multiply(distance, distance));
            }
        }

        // Perform K-means clustering. With deterministic init above and
        // deterministic empty-cluster fallback below, the entire SelectCenters
        // path is now a pure function of X.
        const int maxIterations = 100;
        var assignments = new int[x.Rows];
        var newCenters = new Matrix<T>(numCenters, x.Columns);
        var ones = Engine.Fill(x.Rows, NumOps.One).AsTensor();
        var xTensor = Tensor<T>.FromMatrix(x);

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Assign points to nearest center (first nearest on ties, like a strict < scan)
            var distances = RegressionEngineOps2.PairwiseDistances(x, centers);
            var nearest = Engine.TensorArgMin(Tensor<T>.FromMatrix(distances), 1).ToArray();
            bool changed = !nearest.SequenceEqual(assignments);
            assignments = nearest;

            if (!changed)
            {
                break; // Convergence reached
            }

            // Compute new centers: per-cluster sums of the assigned rows and the cluster sizes
            var assignmentIndex = new Tensor<int>(assignments, new[] { assignments.Length });
            var sums = Engine.TensorIndexAdd(
                Engine.TensorZerosLike(Tensor<T>.FromMatrix(newCenters)), 0, assignmentIndex, xTensor).ToMatrix();
            var countTensor = Engine.TensorIndexAdd(
                Engine.FillZero<T>(numCenters).AsTensor(), 0, assignmentIndex, ones);
            var occupied = Engine.TensorNonzero(countTensor).ToArray();
            var nonEmpty = new bool[numCenters];
            foreach (int cluster in occupied)
            {
                nonEmpty[cluster] = true;
            }

            // Means of the non-empty clusters; empty clusters keep their (zero) sums here.
            var safeCounts = Engine.TensorWhere(nonEmpty, countTensor, Engine.TensorOnesLike(countTensor));
            var means = Engine.TensorDivide(
                Tensor<T>.FromMatrix(sums),
                Engine.TensorBroadcastTo(Engine.Reshape(safeCounts, new[] { numCenters, 1 }), new[] { numCenters, x.Columns })).ToMatrix();
            newCenters = means;

            for (int i = 0; i < numCenters; i++)
            {
                if (nonEmpty[i])
                {
                    continue;
                }

                // Deterministic empty-cluster fallback: reseed with the
                // point farthest from any current non-empty center, which
                // mirrors the k-means++ seeding above and keeps training
                // a pure function of X. The scan sees the clusters before i
                // already averaged and those after i still as running sums.
                var reference = new List<int>();
                for (int c = 0; c < numCenters; c++)
                {
                    if (c != i && nonEmpty[c])
                    {
                        reference.Add(c);
                    }
                }

                int farthest = 0;
                if (reference.Count > 0)
                {
                    var before = reference.Where(c => c < i).ToList();
                    var after = reference.Where(c => c > i).ToList();
                    var referenceRows = new List<Tensor<T>>();
                    if (before.Count > 0)
                    {
                        referenceRows.Add(Tensor<T>.FromMatrix(RegressionEngineOps2.GatherRows(means, before)));
                    }

                    if (after.Count > 0)
                    {
                        referenceRows.Add(Tensor<T>.FromMatrix(RegressionEngineOps2.GatherRows(sums, after)));
                    }

                    var referenceCenters = referenceRows.Count == 1
                        ? referenceRows[0].ToMatrix()
                        : Engine.TensorConcatenate(referenceRows.ToArray(), 0).ToMatrix();
                    var toReference = Tensor<T>.FromMatrix(RegressionEngineOps2.PairwiseDistances(x, referenceCenters));
                    var minDistToCenter = Engine.TensorAmin(toReference, new[] { 1 }, keepDims: false);
                    farthest = Engine.TensorArgMax(minDistToCenter, 0)[0];
                }

                newCenters.SetRow(i, x.GetRow(farthest));
            }

            centers = newCenters;
        }

        return centers;
    }

    /// <summary>
    /// Computes the RBF features for a matrix of input points.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <returns>A matrix of RBF features, including a bias term.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the RBF features for each input point by calculating the response of each
    /// radial basis function to the input.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method transforms your input data by calculating how close each point is to each center,
    /// then applying the radial basis function (a bell curve) to these distances. The result is a new
    /// representation of your data that captures non-linear relationships.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeRBFFeatures(Matrix<T> x)
    {
        // Features per row: [1 (bias), x1, ..., xp (linear), rbf1, ..., rbfK]
        int p = x.Columns;
        if (x.Rows == 0)
        {
            return new Matrix<T>(0, _centers.Rows + 1 + p);
        }

        var bias = Engine.TensorOnesLike(new Tensor<T>(new[] { x.Rows, 1 }));
        var distances = Tensor<T>.FromMatrix(RegressionEngineOps2.PairwiseDistances(x, _centers));
        T gamma = NumOps.FromDouble(_options.Gamma);
        var kernel = Engine.TensorExp(Engine.TensorNegate(
            Engine.TensorMultiplyScalar(Engine.TensorSquare(distances), gamma)));
        return Engine.TensorConcatenate(new[] { bias, Tensor<T>.FromMatrix(x), kernel }, 1).ToMatrix();
    }

    /// <summary>
    /// Computes the RBF features for a single input vector.
    /// </summary>
    /// <param name="x">The input feature vector.</param>
    /// <returns>A vector of RBF features, including a bias term.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the RBF features for a single input point by calculating the response of each
    /// radial basis function to the input.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method transforms a single input example by calculating how close it is to each center,
    /// then applying the radial basis function (a bell curve) to these distances. The first element
    /// is always 1, which serves as a bias term (intercept) in the model.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeRBFFeaturesSingle(Vector<T> x)
    {
        // Features: [1 (bias), x1, x2, ..., xp (linear), rbf1, rbf2, ..., rbfK]
        return ComputeRBFFeatures(RegressionEngineOps2.AsRowMatrix(x)).GetRow(0);
    }

    /// <summary>
    /// Calculates the Euclidean distance between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The Euclidean distance between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the Euclidean distance (straight-line distance) between two points in the input space.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Euclidean distance is the straight-line distance between two points in space, calculated using the
    /// Pythagorean theorem. This is used to determine how close an input point is to each center.
    /// </para>
    /// </remarks>

    /// <summary>
    /// Solves a linear regression problem to find the optimal weights using ridge regularization.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>The optimal weights vector.</returns>
    /// <remarks>
    /// <para>
    /// This method solves the linear regression problem using the normal equations approach with ridge
    /// regularization (Tikhonov regularization). The regularization term (lambda * I) is added to X^T * X
    /// to ensure numerical stability and prevent overfitting. This computes: w = (X^T * X + lambda * I)^-1 * X^T * y.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// After transforming the input data using radial basis functions, this method finds the best weights
    /// to combine these transformed features to predict the target values. It uses a mathematical technique
    /// called the "normal equations" with a small regularization penalty to find stable weights that
    /// minimize prediction error while avoiding numerical issues.
    /// </para>
    /// </remarks>
    private Vector<T> SolveLinearRegression(Matrix<T> x, Vector<T> y)
    {
        // Solve via SVD pseudoinverse first; if that returns null / non-finite
        // weights, OR weights so small they collapse predictions to zero
        // relative to the response magnitude (a symptom seen when a parallel
        // GPU SVD silently produces near-zero singular values on the second
        // model in TranslationEquivariance_ShiftingTargets_ShiftsPredictions),
        // fall back to a normal-equations ridge solve. The fallback is the
        // straightforward (XᵀX + λI)⁻¹·Xᵀy form with λ scaled to the trace
        // of XᵀX so it matches the design matrix's magnitude regardless of
        // input scale; that path doesn't depend on the GPU SVD kernel and
        // delivers the right-magnitude bias term needed to track a shifted Y.
        Vector<T>? weights = TrySvdSolve(x, y);
        if (weights == null || !ModelTestHelpers_AllFinite(weights) || SvdWeightsAreBad(x, y, weights))
        {
            weights = NormalEquationsRidgeSolve(x, y);
        }

        // Apply external regularization (if any) on the learned weight
        // vector — the pseudoinverse path doesn't build the normal-equations
        // matrix that the previous code was regularizing.
        if (Regularization != null)
        {
            weights = Regularization.Regularize(weights);
        }

        return weights;
    }

    private bool SvdWeightsAreBad(Matrix<T> x, Vector<T> y, Vector<T> weights)
    {
        // Decide whether the SVD-produced weights are bad enough that the
        // CPU-deterministic NormalEquationsRidgeSolve fallback should take over.
        // We check two distinct failure modes a parallel/GPU SVD can land in:
        //
        //   (A) Collapsed-to-zero predictions: max |pred| < 1% of max |y|.
        //       Originally surfaced on TranslationEquivariance_ShiftingTargets
        //       (a second model in a parallel run silently produced near-zero
        //       singular values and weights that mapped everything to ~0).
        //
        //   (B) Systematically wrong predictions: training R² < 0. This catches
        //       the case where the SVD path returns finite weights of plausible
        //       magnitude but the predictions are anti-correlated with y (the
        //       symptom on PR #1488 CI was Builder R²=-0.95 on linear data
        //       while the same model on a single-threaded local run produced
        //       R²=1.0). R²<0 on the training set means the solve underfits
        //       worse than predicting the mean — a clear "use the CPU
        //       fallback" signal regardless of solver internals.
        //
        // Both checks operate on the supplied (x, y) pair, so they cost a
        // single x · w mat-vec and one pass over y — negligible vs SVD itself.
        var preds = x.Multiply(weights);
        T maxPred = RegressionEngineOps2.MaxAbs(preds);
        T maxY = RegressionEngineOps2.MaxAbs(y);
        T meanY = y.Length == 0 ? NumOps.Zero : Engine.Mean(y);

        if (NumOps.GreaterThan(maxY, NumOps.FromDouble(1e-10)) &&
            NumOps.LessThan(maxPred, NumOps.Multiply(maxY, NumOps.FromDouble(0.01))))
            return true;

        T ssRes = RegressionEngineOps2.SumOfSquares(Engine.Subtract(y, preds));
        T ssTot = RegressionEngineOps2.SumOfSquares(RegressionEngineOps2.AddScalar(y, NumOps.Negate(meanY)));
        if (NumOps.GreaterThan(ssTot, NumOps.FromDouble(1e-12)))
        {
            T trainR2 = NumOps.Subtract(NumOps.One, NumOps.Divide(ssRes, ssTot));
            if (NumOps.LessThan(trainR2, NumOps.Zero))
                return true;
        }

        return false;
    }

    private Vector<T>? TrySvdSolve(Matrix<T> x, Vector<T> y)
    {
        try
        {
            var svd = new AiDotNet.DecompositionMethods.MatrixDecomposition.SvdDecomposition<T>(x);

            // Largest singular value, ignoring NaN and never below zero.
            T sigmaMax = NumOps.Zero;
            int k = svd.S.Length;
            if (k > 0)
            {
                var s = svd.S.AsTensor();
                var sNoNan = Engine.TensorWhere(Engine.TensorIsNan(s), Engine.TensorZerosLike(s), s);
                T largest = Engine.TensorMaxValue(sNoNan);
                if (NumOps.GreaterThan(largest, sigmaMax))
                    sigmaMax = largest;
            }

            if (!NumOps.GreaterThan(sigmaMax, NumOps.Zero) || NumOps.IsNaN(sigmaMax))
            {
                // Caller falls back to normal-equations ridge solve.
                return null;
            }

            // Tikhonov-regularized SVD solve: weights = V · diag(σ / (σ² + λ²)) · Uᵀ · y.
            // Unlike a hard tolerance-based pseudoinverse this smoothly damps
            // small singular values instead of zeroing them. Small λ ≈ 1e-6 · σ_max
            // gives a stable solve while barely biasing the well-conditioned directions.
            T lambda = NumOps.Multiply(sigmaMax, NumOps.FromDouble(1e-6));
            T lambdaSq = NumOps.Multiply(lambda, lambda);

            // weights = Σ_i Vt[i, :] · (U[:, i] · y) · σ_i / (σ_i² + λ²)
            var sigmas = svd.S;
            var damped = Engine.Divide(
                sigmas, RegressionEngineOps2.AddScalar(Engine.Multiply(sigmas, sigmas), lambdaSq));
            var uUsed = svd.U.Columns == k ? svd.U : svd.U.GetSubMatrix(0, 0, svd.U.Rows, k);
            var vtUsed = svd.Vt.Rows == k ? svd.Vt : svd.Vt.GetSubMatrix(0, 0, k, svd.Vt.Columns);
            var coefficients = Engine.Multiply(Engine.MatrixVectorMultiply(Engine.MatrixTranspose(uUsed), y), damped);
            return Engine.MatrixVectorMultiply(Engine.MatrixTranspose(vtUsed), coefficients);
        }
        catch (Exception ex) when (
            ex is InvalidOperationException ||
            ex is ArithmeticException)
        {
            // Narrow the catch so unrelated bugs (NullReference, argument
            // mismatch, etc.) bubble up instead of forcing the caller into
            // the normal-equations fallback under the wrong circumstances.
            System.Diagnostics.Debug.WriteLine(
                $"[RadialBasisFunctionRegression] SVD path failed, falling back: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private Vector<T> NormalEquationsRidgeSolve(Matrix<T> x, Vector<T> y)
    {
        // Normal-equations ridge OLS: w = (XᵀX + λI)⁻¹ Xᵀy.
        // Slower-condition path used as fallback when SVD fails — λ chosen
        // proportional to the trace of XᵀX so it scales sensibly with the
        // magnitude of the design matrix without manual tuning.
        var xT = x.Transpose();
        var xTx = xT.Multiply(x);
        T trace = xTx.Rows == 0 ? NumOps.Zero : Engine.Sum(xTx.Diagonal());
        T lambdaRidge = NumOps.Divide(
            NumOps.Multiply(trace, NumOps.FromDouble(1e-6)),
            NumOps.FromDouble(Math.Max(1, xTx.Rows)));
        // Floor lambda at a tiny absolute value so a near-zero design matrix
        // (trace ≈ 0) doesn't leave xTx singular when we go to invert it.
        // The minimum is the same threshold normal-equations ridge solvers
        // pick to keep the diagonal numerically nonzero.
        T minLambda = NumOps.FromDouble(1e-12);
        if (NumOps.LessThan(lambdaRidge, minLambda))
            lambdaRidge = minLambda;
        xTx = RegressionEngineOps2.AddDiagonal(xTx, lambdaRidge);
        var xTy = xT.Multiply(y);
        return xTx.Inverse().Multiply(xTy);
    }

    private static bool ModelTestHelpers_AllFinite(Vector<T> v)
    {
        return RegressionEngineOps2.AllFinite(v);
    }

    /// <summary>
    /// Creates a new instance of the radial basis function regression model with the same options.
    /// </summary>
    /// <returns>A new instance of the RBF regression model with the same configuration but no trained parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the radial basis function regression model with the same 
    /// configuration options as the current instance, but without copying the trained parameters (centers and weights).
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model configuration without 
    /// any learned parameters. It's like getting a blank template with the same settings.
    /// 
    /// Think of it like getting a fresh copy of a form with all the same fields and settings,
    /// but without any of the data filled in. The new model has the same:
    /// - Number of centers
    /// - Gamma parameter (controls how quickly the influence of each center drops off)
    /// - Regularization settings
    /// - Other configuration options
    /// 
    /// But it doesn't have the learned centers or weights from training.
    /// 
    /// This is mainly used internally by the framework when performing operations like
    /// cross-validation or creating ensembles of similar models.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a new instance with the same options and regularization
        return new RadialBasisFunctionRegression<T>(_options, Regularization);
    }
}
