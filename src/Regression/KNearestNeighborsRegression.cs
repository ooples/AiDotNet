using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;

namespace AiDotNet.Regression;

/// <summary>
/// Implements K-Nearest Neighbors algorithm for regression, which predicts target values
/// by averaging the values of the K closest training examples.
/// </summary>
/// <remarks>
/// <para>
/// K-Nearest Neighbors (KNN) is a non-parametric and instance-based learning algorithm that makes
/// predictions based on the similarity between the input and training samples. For regression, it computes
/// the average of the target values of the K nearest neighbors to the query point. The algorithm doesn't
/// build an explicit model but instead stores all training examples and performs computations at prediction time.
/// </para>
/// <para><b>For Beginners:</b> K-Nearest Neighbors is like asking your neighbors for advice.
/// 
/// Imagine you want to guess the price of a house:
/// - You look at the K most similar houses to yours (the "nearest neighbors") 
/// - You take the average of their prices as your prediction
/// 
/// The "K" is just how many neighbors you consider. If K=3, you look at the 3 most similar houses.
/// 
/// This approach is:
/// - Simple to understand: similar inputs should have similar outputs
/// - Makes no assumptions about the data's structure
/// - Works well when similar examples in your data actually have similar target values
/// 
/// Unlike most machine learning algorithms, KNN doesn't "learn" patterns during training - it simply
/// remembers all examples and does the real work at prediction time by finding similar examples.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a KNN regression model with K=3 nearest neighbors
/// var options = new KNearestNeighborsOptions();
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Store all training examples (lazy learner)
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new KNearestNeighborsRegression&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict by averaging the K nearest neighbors' values
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 6, 7 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Nearest Neighbor Pattern Classification", "https://doi.org/10.1109/TIT.1967.1053964")]
public partial class KNearestNeighborsRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Configuration options for the K-Nearest Neighbors algorithm.
    /// </summary>
    private readonly KNearestNeighborsOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Matrix containing the feature vectors of the training samples.
    /// </summary>
    [Buffer]
    private Matrix<T> _xTrain;

    /// <summary>
    /// Vector containing the target values of the training samples.
    /// </summary>
    [Buffer]
    private Vector<T> _yTrain;

    /// <summary>
    /// Initializes a new instance of the <see cref="KNearestNeighborsRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration options for the KNN algorithm.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new K-Nearest Neighbors regression model with the specified options and
    /// regularization strategy. If no options are provided, default values are used. If no regularization
    /// is specified, no regularization is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new K-Nearest Neighbors model.
    /// 
    /// The key setting is K - the number of neighbors to consider when making predictions:
    /// - Smaller K values (like 1 or 2): More sensitive to noise in the data
    /// - Larger K values (like 10 or 20): Smoother predictions but might miss important patterns
    /// 
    /// If you don't specify any options, the model will use reasonable default settings.
    /// 
    /// Example:
    /// ```csharp
    /// // Create a KNN model with default settings
    /// var knn = new KNearestNeighborsRegression&lt;double&gt;();
    /// 
    /// // Create a KNN model with custom options
    /// var options = new KNearestNeighborsOptions { K = 5 };
    /// var customKnn = new KNearestNeighborsRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public KNearestNeighborsRegression(KNearestNeighborsOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new KNearestNeighborsOptions();
        _xTrain = new Matrix<T>(0, 0);
        _yTrain = new Vector<T>(0);
        SoftKNNTemperature = NumOps.One; // Default temperature = 1.0
    }

    /// <summary>
    /// KNN is a lazy learner — no optimizer parameter injection.
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
    /// KNN uses all features (distance-based, no coefficient pruning).
    /// </summary>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        return Enumerable.Range(0, _xTrain.Columns > 0 ? _xTrain.Columns : 0);
    }

    /// <summary>
    /// Optimizes the KNN model by storing the training data for later use in predictions.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <remarks>
    /// <para>
    /// This method "trains" the KNN model by storing the training data for later use during prediction.
    /// Unlike many other machine learning algorithms, KNN doesn't build a parametric model during training.
    /// Instead, it simply stores the training data and uses it to compute predictions at runtime by
    /// finding the K nearest neighbors to each query point.
    /// </para>
    /// <para><b>For Beginners:</b> KNN doesn't really "learn" during training - it just memorizes the examples.
    /// 
    /// While most machine learning models try to extract patterns during training, KNN takes a different approach:
    /// 1. It simply stores all the training examples (both features and target values)
    /// 2. When asked to make a prediction, it does the actual work of finding similar examples
    /// 
    /// Think of it like studying for an exam by memorizing all the examples in a textbook, 
    /// rather than trying to understand the underlying rules. When given a new problem,
    /// you solve it by finding the most similar examples from the ones you memorized.
    /// 
    /// This is why KNN is sometimes called a "lazy learner" - it doesn't do much work during training,
    /// but has to work harder at prediction time.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Note: KNN is a distance-based method that doesn't use regularization in the traditional sense.
        // The training data is stored as-is for neighbor finding.
        _xTrain = x;
        _yTrain = y;
    }

    /// <summary>
    /// Predicts target values for the provided input features using the trained KNN model.
    /// </summary>
    /// <param name="input">A matrix where each row represents a sample to predict and each column represents a feature.</param>
    /// <returns>A vector of predicted values corresponding to each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts target values for new input data by finding the K nearest neighbors from the
    /// training data for each input sample and computing the average of their target values. Unlike
    /// parametric models, KNN is a distance-based method that does not apply data regularization
    /// transformations. The prediction relies entirely on the stored training data and distance calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to make predictions on new data.
    /// 
    /// For each input example, it:
    /// 1. Calculates how similar the new example is to each training example (using distance)
    /// 2. Finds the K most similar training examples
    /// 3. Takes the average of their target values as the prediction
    /// 
    /// This method handles multiple inputs at once, making a separate prediction for each one.
    /// 
    /// Example:
    /// ```csharp
    /// // Make predictions
    /// var predictions = knn.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_xTrain.Rows == 0 || _yTrain.Length == 0)
        {
            throw new InvalidOperationException("The model must be trained before making predictions. Call Train() first.");
        }

        // Note: KNN is a distance-based method - no data transformation is applied
        return new Vector<T>(input.GetRows().Select(PredictSingle));
    }

    /// <summary>
    /// Predicts the target value for a single input feature vector.
    /// </summary>
    /// <param name="input">The feature vector of the sample to predict.</param>
    /// <returns>The predicted value for the input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts the target value for a single input feature vector by finding the K nearest
    /// neighbors from the training data and computing the average of their target values. The distance
    /// between the input and each training sample is computed using Euclidean distance.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a prediction for a single new data point.
    /// 
    /// The prediction process works like this:
    /// 1. Calculate the distance between the new point and every training example
    /// 2. Find the K training examples with the smallest distances (the nearest neighbors)
    /// 3. Calculate the average of their target values
    /// 4. Return this average as the prediction
    /// 
    /// For example, if you want to predict a house price and K=3, this method would:
    /// - Find the 3 most similar houses from the training data
    /// - Calculate the average of their prices
    /// - Return that average as the predicted price
    /// </para>
    /// </remarks>
    protected override T PredictSingle(Vector<T> input)
    {
        var engine = AiDotNetEngine.Current;
        var distanceVector = CalculateDistances(input);

        // Stable ordering: equal distances keep the lower training index first
        var nearestNeighbors = distanceVector
            .Select((distance, index) => (index, distance))
            .OrderBy(x => x.distance, Comparer<T>.Default)
            .Take(_options.K)
            .ToList();
        var neighborIndices = nearestNeighbors.Select(nb => nb.index).ToArray();

        // Fit a local OLS model among the K nearest neighbors for better extrapolation
        int k = nearestNeighbors.Count;
        int p = input.Length;

        // If enough neighbors for OLS, use local linear fit
        if (k > p + 1)
        {
            // Build local X (intercept column first) and y
            var localX = _xTrain.GetRows(neighborIndices).AddConstantColumn(NumOps.One);
            var localY = RegressionEngineOps.Gather(_yTrain, neighborIndices);

            // Solve (X'X + ridge I) beta = X'y for local coefficients
            var localXt = engine.MatrixTranspose(localX);
            var xTx = engine.MatrixMultiply(localXt, localX);
            var xTy = engine.MatrixVectorMultiply(localXt, localY);
            xTx = engine.MatrixAdd(xTx, Matrix<T>.CreateDiagonal(engine.Fill(xTx.Rows, NumOps.FromDouble(1e-8))));
            var coefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, MatrixDecompositionType.Lu);

            // Predict: intercept + sum(coeff * input)
            T intercept = coefficients[0];
            return NumOps.Add(intercept, engine.DotProduct(RegressionEngineOps2.Slice(coefficients, 1, p), input));
        }

        // Fallback: inverse-distance weighting, w = 1/d (1e10 for d < 1e-10)
        if (k == 0)
        {
            return NumOps.Divide(NumOps.Zero, NumOps.Zero);
        }

        var neighborDistances = Tensor<T>.FromVector(RegressionEngineOps.Gather(distanceVector, neighborIndices));
        var weights = engine.TensorWhere(
            engine.TensorLessThan(neighborDistances, NumOps.FromDouble(1e-10)),
            Tensor<T>.FromVector(engine.Fill(k, NumOps.FromDouble(1e10))),
            engine.TensorReciprocal(neighborDistances)).ToVector();
        T weightedSum = engine.DotProduct(weights, RegressionEngineOps.Gather(_yTrain, neighborIndices));
        T totalWeight = engine.Sum(weights);

        return NumOps.Divide(weightedSum, totalWeight);
    }

    /// <summary>
    /// Calculates the Euclidean distance from a feature vector to every training sample.
    /// </summary>
    /// <param name="input">The query feature vector.</param>
    /// <returns>The Euclidean distance to each training row, in training order.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector dimension differs from the training data.</exception>
    private Vector<T> CalculateDistances(Vector<T> input)
    {
        if (_xTrain.Rows == 0)
        {
            return new Vector<T>(0);
        }

        if (input.Length != _xTrain.Columns)
        {
            throw new ArgumentException(
                $"Feature vectors must have the same dimension. Expected {input.Length} features but got {_xTrain.Columns}. " +
                "Ensure the input features match the training data dimensions.");
        }

        return AiDotNetEngine.Current.Sqrt(RegressionEngineOps.RowSquaredDistances(_xTrain, input));
    }

    /// <summary>
    /// Gets the model type of the K-Nearest Neighbors Regression model.
    /// </summary>
    /// <returns>The model type enumeration value.</returns>

    /// <summary>
    /// Creates a new instance of the KNearestNeighborsRegression with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new KNearestNeighborsRegression instance with the same options and regularization as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the KNearestNeighborsRegression model with the same configuration options
    /// and regularization settings as the current instance. This is useful for model cloning, ensemble methods, or
    /// cross-validation scenarios where multiple instances of the same model with identical configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model's blueprint.
    /// 
    /// When you need multiple versions of the same type of model with identical settings:
    /// - This method creates a new, empty model with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new model has the same settings but no trained data
    /// - This is useful for techniques that need multiple models, like cross-validation
    /// 
    /// For example, when testing your model on different subsets of data,
    /// you'd want each test to use a model with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new KNearestNeighborsRegression<T>(_options, Regularization);
    }

    // ===== Soft KNN Support for JIT Compilation =====

    /// <summary>
    /// Gets or sets whether to use soft (differentiable) KNN mode for JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable soft KNN mode with attention-weighted outputs for JIT support;
    /// <c>false</c> (default) for traditional hard K-nearest neighbors.
    /// </value>
    /// <remarks>
    /// <para><b>Soft KNN:</b> Instead of selecting exactly K nearest neighbors and averaging their
    /// labels, soft KNN computes attention weights over ALL training samples based on distances.
    /// This makes the algorithm differentiable and JIT-compilable.</para>
    /// <para><b>Formula:</b> weights = softmax(-distances / temperature)</para>
    /// <para><b>Output:</b> weighted_output = sum(weights * labels)</para>
    /// <para><b>Trade-offs:</b></para>
    /// <list type="bullet">
    /// <item><description>Soft KNN is differentiable and JIT-compilable</description></item>
    /// <item><description>Results are smooth approximations of hard K selection</description></item>
    /// <item><description>Lower temperature = sharper attention (closer to hard K selection)</description></item>
    /// <item><description>Higher temperature = softer attention (considers more neighbors)</description></item>
    /// </list>
    /// <para><b>Computational Note:</b> Soft KNN computes attention over ALL training samples,
    /// which can be expensive for large training sets. The JIT-compiled version embeds all
    /// support vectors as constants, so the computation graph size scales with training set size.</para>
    /// </remarks>
    public bool UseSoftKNN { get; set; } = false;

    /// <summary>
    /// Gets or sets the temperature parameter for soft KNN mode.
    /// </summary>
    /// <value>
    /// The temperature for softmax attention. Lower values produce sharper attention.
    /// Default is 1.0.
    /// </value>
    public T SoftKNNTemperature { get; set; }
}
