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
/// var options = new RadialBasisFunctionRegressionOptions&lt;double&gt;();
/// var model = new RadialBasisFunctionRegression&lt;double&gt;(options);
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = Matrix&lt;double&gt;.Build.Dense(6, 2, new double[] {
///     1, 2,  3, 4,  5, 6,  7, 8,  9, 10,  11, 12 });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train with RBF centers and weighted linear combination
/// model.Train(features, targets);
///
/// // Predict for a new sample
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 6, 7 });
/// var prediction = model.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Kernel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Radial Basis Functions", "https://doi.org/10.1017/CBO9780511543241")]
public class RadialBasisFunctionRegression<T> : NonLinearRegressionBase<T>
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
    private Matrix<T> _centers;

    /// <summary>
    /// The weights used to combine the radial basis function outputs.
    /// </summary>
    /// <value>
    /// A vector of weights, including a bias term.
    /// </value>
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
    public override long ParameterCount => 0;

    /// <summary>
    /// Returns all features since RBF uses distance-based kernels across all features.
    /// </summary>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        int numFeatures = _centers.Columns > 0 ? _centers.Columns : 0;
        return Enumerable.Range(0, numFeatures);
    }

    /// <summary>
    /// Deep copy via serialization to preserve private _centers and _weights.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new RadialBasisFunctionRegression<T>(_options, Regularization);
        clone.Deserialize(Serialize());
        return clone;
    }

    public override IFullModel<T, Matrix<T>, Vector<T>> DeepCopy() => Clone();

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Auto-scale gamma if using the default value of 1.0
        if (Math.Abs(_options.Gamma - 1.0) < 1e-10)
        {
            double totalVar = 0;
            for (int j = 0; j < x.Columns; j++)
            {
                double mean = 0;
                for (int i = 0; i < x.Rows; i++)
                    mean += NumOps.ToDouble(x[i, j]);
                mean /= x.Rows;
                double variance = 0;
                for (int i = 0; i < x.Rows; i++)
                {
                    double d = NumOps.ToDouble(x[i, j]) - mean;
                    variance += d * d;
                }
                totalVar += variance / x.Rows;
            }
            _options.Gamma = totalVar > 1e-10 ? 1.0 / (x.Columns * (totalVar / x.Columns)) : 1.0;
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
            var minDistSq = new double[x.Rows];
            for (int i = 0; i < x.Rows; i++)
            {
                T d = VectorHelper.EuclideanDistance(x.GetRow(i), centers.GetRow(0));
                double dd = NumOps.ToDouble(d);
                minDistSq[i] = dd * dd;
            }

            for (int c = 1; c < numCenters; c++)
            {
                int farthest = 0;
                double farthestDist = minDistSq[0];
                for (int i = 1; i < x.Rows; i++)
                {
                    if (minDistSq[i] > farthestDist)
                    {
                        farthestDist = minDistSq[i];
                        farthest = i;
                    }
                }
                centers.SetRow(c, x.GetRow(farthest));

                // Update minDistSq with the new center's contribution.
                for (int i = 0; i < x.Rows; i++)
                {
                    T d = VectorHelper.EuclideanDistance(x.GetRow(i), centers.GetRow(c));
                    double dd = NumOps.ToDouble(d);
                    double dSq = dd * dd;
                    if (dSq < minDistSq[i])
                        minDistSq[i] = dSq;
                }
            }
        }

        // Perform K-means clustering. With deterministic init above and
        // deterministic empty-cluster fallback below, the entire SelectCenters
        // path is now a pure function of X.
        const int maxIterations = 100;
        var assignments = new int[x.Rows];
        var newCenters = new Matrix<T>(numCenters, x.Columns);

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            bool changed = false;

            // Assign points to nearest center
            for (int i = 0; i < x.Rows; i++)
            {
                int nearestCenter = 0;
                T minDistance = VectorHelper.EuclideanDistance(x.GetRow(i), centers.GetRow(0));

                for (int j = 1; j < numCenters; j++)
                {
                    T distance = VectorHelper.EuclideanDistance(x.GetRow(i), centers.GetRow(j));
                    if (NumOps.LessThan(distance, minDistance))
                    {
                        minDistance = distance;
                        nearestCenter = j;
                    }
                }

                if (assignments[i] != nearestCenter)
                {
                    assignments[i] = nearestCenter;
                    changed = true;
                }
            }

            if (!changed)
            {
                break; // Convergence reached
            }

            // Compute new centers
            var counts = new int[numCenters];
            for (int i = 0; i < numCenters; i++)
            {
                newCenters.SetRow(i, new Vector<T>(x.Columns));
            }

            for (int i = 0; i < x.Rows; i++)
            {
                int assignment = assignments[i];
                newCenters.SetRow(assignment, newCenters.GetRow(assignment).Add(x.GetRow(i)));
                counts[assignment]++;
            }

            for (int i = 0; i < numCenters; i++)
            {
                if (counts[i] > 0)
                {
                    newCenters.SetRow(i, newCenters.GetRow(i).Divide(NumOps.FromDouble(counts[i])));
                }
                else
                {
                    // Deterministic empty-cluster fallback: reseed with the
                    // point farthest from any current non-empty center, which
                    // mirrors the k-means++ seeding above and keeps training
                    // a pure function of X.
                    int farthest = 0;
                    double farthestDist = -1;
                    for (int rowIdx = 0; rowIdx < x.Rows; rowIdx++)
                    {
                        double minDistToCenter = double.MaxValue;
                        for (int c = 0; c < numCenters; c++)
                        {
                            if (c == i || counts[c] == 0) continue;
                            T d = VectorHelper.EuclideanDistance(x.GetRow(rowIdx), newCenters.GetRow(c));
                            double dd = NumOps.ToDouble(d);
                            if (dd < minDistToCenter) minDistToCenter = dd;
                        }
                        if (minDistToCenter > farthestDist)
                        {
                            farthestDist = minDistToCenter;
                            farthest = rowIdx;
                        }
                    }
                    newCenters.SetRow(i, x.GetRow(farthest));
                }
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
        int p = x.Columns;
        var rbfFeatures = new Matrix<T>(x.Rows, _centers.Rows + 1 + p);

        for (int i = 0; i < x.Rows; i++)
        {
            var row = ComputeRBFFeaturesSingle(x.GetRow(i));
            for (int j = 0; j < row.Length; j++)
            {
                rbfFeatures[i, j] = row[j];
            }
        }

        return rbfFeatures;
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
        int p = x.Length;
        var features = new Vector<T>(_centers.Rows + 1 + p);
        features[0] = NumOps.One; // Bias term

        // Add linear features for extrapolation capability
        for (int j = 0; j < p; j++)
            features[1 + j] = x[j];

        for (int i = 0; i < _centers.Rows; i++)
        {
            T distance = VectorHelper.EuclideanDistance(x, _centers.GetRow(i));
            features[1 + p + i] = RbfKernel(distance);
        }

        return features;
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
    /// Applies the radial basis function kernel to a distance value.
    /// </summary>
    /// <param name="distance">The distance value.</param>
    /// <returns>The result of applying the RBF kernel to the distance.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the Gaussian radial basis function kernel, which is defined as exp(-gamma * distance^2).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This function creates the "bell curve" shape of the radial basis function. It takes a distance value
    /// and returns a value between 0 and 1, where 1 means the input is exactly at the center (distance = 0)
    /// and values close to 0 mean the input is far from the center. The gamma parameter controls how quickly
    /// the function drops off with distance.
    /// </para>
    /// </remarks>
    private T RbfKernel(T distance)
    {
        T gamma = NumOps.FromDouble(_options.Gamma);
        return NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, NumOps.Multiply(distance, distance))));
    }

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
        double maxPred = 0;
        for (int i = 0; i < preds.Length; i++)
            maxPred = Math.Max(maxPred, Math.Abs(NumOps.ToDouble(preds[i])));

        double maxY = 0;
        double meanY = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            maxY = Math.Max(maxY, Math.Abs(yi));
            meanY += yi;
        }
        meanY /= Math.Max(1, y.Length);

        if (maxY > 1e-10 && maxPred < maxY * 0.01)
            return true;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            double pi = NumOps.ToDouble(preds[i]);
            double r = yi - pi;
            ssRes += r * r;
            double t = yi - meanY;
            ssTot += t * t;
        }
        if (ssTot > 1e-12)
        {
            double trainR2 = 1.0 - ssRes / ssTot;
            if (trainR2 < 0)
                return true;
        }

        return false;
    }

    private Vector<T>? TrySvdSolve(Matrix<T> x, Vector<T> y)
    {
        try
        {
            var svd = new AiDotNet.DecompositionMethods.MatrixDecomposition.SvdDecomposition<T>(x);

            T sigmaMax = NumOps.Zero;
            for (int i = 0; i < svd.S.Length; i++)
            {
                if (NumOps.GreaterThan(svd.S[i], sigmaMax))
                    sigmaMax = svd.S[i];
            }

            double sigmaMaxD = NumOps.ToDouble(sigmaMax);
            if (!(sigmaMaxD > 0) || double.IsNaN(sigmaMaxD))
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

            var weights = new Vector<T>(svd.Vt.Columns);
            for (int i = 0; i < svd.S.Length; i++)
            {
                T sigma = svd.S[i];
                T sigmaSq = NumOps.Multiply(sigma, sigma);
                T damped = NumOps.Divide(sigma, NumOps.Add(sigmaSq, lambdaSq));

                Vector<T> uCol = svd.U.GetColumn(i);
                T coeff = NumOps.Multiply(uCol.DotProduct(y), damped);
                Vector<T> vtRow = svd.Vt.GetRow(i);
                weights = weights.Add(vtRow.Multiply(coeff));
            }
            return weights;
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
        T trace = NumOps.Zero;
        for (int i = 0; i < xTx.Rows; i++)
            trace = NumOps.Add(trace, xTx[i, i]);
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
        for (int i = 0; i < xTx.Rows; i++)
            xTx[i, i] = NumOps.Add(xTx[i, i], lambdaRidge);
        var xTy = xT.Multiply(y);
        return xTx.Inverse().Multiply(xTy);
    }

    private static bool ModelTestHelpers_AllFinite(Vector<T> v)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < v.Length; i++)
        {
            double d = ops.ToDouble(v[i]);
            if (double.IsNaN(d) || double.IsInfinity(d))
                return false;
        }
        return true;
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters, including base class data, options, centers, and weights.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize RBF specific data
        writer.Write(_options.NumberOfCenters);
        writer.Write(_options.Gamma);
        writer.Write(_options.Seed ?? -1);

        // Serialize centers
        writer.Write(_centers.Rows);
        writer.Write(_centers.Columns);
        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_centers[i, j]));
            }
        }

        // Serialize weights
        writer.Write(_weights.Length);
        for (int i = 0; i < _weights.Length; i++)
        {
            writer.Write(Convert.ToDouble(_weights[i]));
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array, including base class data,
    /// options, centers, and weights.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize RBF specific data
        _options.NumberOfCenters = reader.ReadInt32();
        _options.Gamma = reader.ReadDouble();
        int seed = reader.ReadInt32();
        _options.Seed = seed == -1 ? null : seed;

        // Deserialize centers
        int centerRows = reader.ReadInt32();
        int centerColumns = reader.ReadInt32();
        _centers = new Matrix<T>(centerRows, centerColumns);
        for (int i = 0; i < centerRows; i++)
        {
            for (int j = 0; j < centerColumns; j++)
            {
                _centers[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize weights
        int weightsLength = reader.ReadInt32();
        _weights = new Vector<T>(weightsLength);
        for (int i = 0; i < weightsLength; i++)
        {
            _weights[i] = NumOps.FromDouble(reader.ReadDouble());
        }
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
