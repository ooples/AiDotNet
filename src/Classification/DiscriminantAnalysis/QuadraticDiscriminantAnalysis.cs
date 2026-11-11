using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
namespace AiDotNet.Classification.DiscriminantAnalysis;

/// <summary>
/// Quadratic Discriminant Analysis classifier.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// QDA is similar to LDA but allows each class to have its own covariance matrix,
/// which creates quadratic (curved) decision boundaries.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Quadratic Discriminant Analysis (QDA) is a classification technique that:
///
/// 1. Models each class with its own Gaussian distribution
/// 2. Each class has its own covariance matrix (unlike LDA)
/// 3. Decision boundaries are quadratic (curved) instead of linear
///
/// When to use QDA over LDA:
/// - When classes have different covariance structures
/// - When you have enough samples per class to estimate covariance reliably
/// - When decision boundaries are naturally curved
///
/// Trade-offs:
/// - More flexible than LDA (can capture curved boundaries)
/// - Needs more parameters (separate covariance per class)
/// - More prone to overfitting with small datasets
/// - Computationally more expensive than LDA
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create QDA classifier with per-class covariance matrices
/// var options = new DiscriminantAnalysisOptions&lt;double&gt;();
///
/// // Prepare training data with classes that have different covariance structures
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.1 }, { 1.2, 0.9 }, { 0.8, 1.0 }, { 5.0, 5.1 }, { 5.2, 4.9 }, { 4.8, 5.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train to learn per-class Gaussian distributions with quadratic boundaries
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new QuadraticDiscriminantAnalysis&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Classify new sample using quadratic discriminant function
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.1, 1.0 } });
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("The Use of Multiple Measurements in Taxonomic Problems", "https://doi.org/10.1111/j.1469-1809.1936.tb02137.x")]
public partial class QuadraticDiscriminantAnalysis<T> : ProbabilisticClassifierBase<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The flattened class means, as LinearDiscriminantAnalysis. PackParameters and UnpackParameters ARE the previous
    /// GetParameters and SetParameters, renamed rather than rewritten, so the layout is
    /// byte-for-byte unchanged. What changes is that the COUNT now folds this same
    /// component instead of ClassifierBase's NumFeatures * NumClasses formula, which
    /// described the classifier's shape rather than what it serializes.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VariableLengthParameterSource<T>(
            () => PackParameters().Length,
            PackParameters,
            UnpackParameters));
    }
    /// <summary>
    /// Gets the QDA-specific options.
    /// </summary>
    protected new DiscriminantAnalysisOptions<T> Options => (DiscriminantAnalysisOptions<T>)base.Options;

    /// <summary>
    /// QDA computes parameters from per-class covariance matrices during training.
    /// It does not support flat parameter initialization.
    /// </summary>
    public override bool SupportsParameterInitialization => false;

    /// <summary>
    /// Class means for each class.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _classMeans = new Matrix<T>(0, 0);

    /// <summary>
    /// Covariance matrix for each class.
    /// </summary>
    private Matrix<T>[] _classCovariances = Array.Empty<Matrix<T>>();

    /// <summary>
    /// Inverse of covariance matrix for each class.
    /// </summary>
    private Matrix<T>[] _classCovarianceInverses = Array.Empty<Matrix<T>>();

    /// <summary>
    /// Log determinant of covariance matrix for each class.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _classLogDets = new Vector<T>(0);

    /// <summary>
    /// Class priors (prior probabilities).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _classPriors = new Vector<T>(0);

    /// <summary>
    /// Initializes a new instance of the QuadraticDiscriminantAnalysis class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public QuadraticDiscriminantAnalysis(DiscriminantAnalysisOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new DiscriminantAnalysisOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the QDA classifier on the provided data.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Compute class means
        _classMeans = ComputeClassMeans(x, y);

        // Compute class priors
        _classPriors = ComputeClassPriors(y);

        // Compute per-class covariance matrices
        _classCovariances = new Matrix<T>[NumClasses];
        _classCovarianceInverses = new Matrix<T>[NumClasses];
        var logDets = new T[NumClasses];

        for (int c = 0; c < NumClasses; c++)
        {
            _classCovariances[c] = ComputeClassCovariance(x, y, c);

            // Add regularization if specified
            if (Options.RegularizationParam > 0)
            {
                _classCovariances[c] = AddRegularization(_classCovariances[c]);
            }

            // Compute inverse and log determinant
            _classCovarianceInverses[c] = ComputeInverse(_classCovariances[c]);
            logDets[c] = ComputeLogDeterminant(_classCovariances[c]);
        }

        _classLogDets = Vector<T>.Wrap(logDets);
    }

    /// <summary>
    /// Computes the mean vector for each class.
    /// </summary>
    private Matrix<T> ComputeClassMeans(Matrix<T> x, Vector<T> y)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been fitted.");
        }

        if (x.Rows == 0)
        {
            return new Matrix<T>(NumClasses, NumFeatures);
        }

        // Sum features per class (membership^T * X), then divide by the class count (empty classes stay zero)
        var engine = Engine;
        var membership = ClassMembership(y);
        var sums = engine.TensorMatMul(engine.TensorTranspose(membership), Tensor<T>.FromMatrix(x));
        var counts = engine.ReduceSum(membership, new[] { 0 }, false).Reshape(NumClasses, 1);
        return engine.TensorDivide(sums, engine.TensorClampMin(counts, NumOps.One)).ToMatrix();
    }

    /// <summary>
    /// Computes class prior probabilities.
    /// </summary>
    private Vector<T> ComputeClassPriors(Vector<T> y)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been fitted.");
        }

        int n = y.Length;
        var counts = Engine.ReduceSum(ClassMembership(y), new[] { 0 }, false);
        return Engine.TensorDivideScalar(counts, NumOps.FromDouble(n)).ToVector();
    }

    /// <summary>
    /// Computes the covariance matrix for a specific class.
    /// </summary>
    private Matrix<T> ComputeClassCovariance(Matrix<T> x, Vector<T> y, int classIndex)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been fitted.");
        }

        var engine = Engine;
        if (x.Rows == 0)
        {
            return new Matrix<T>(NumFeatures, NumFeatures);
        }

        // Rows of this class, centered on the class mean; other rows are masked to zero
        var inClass = ClassMembership(y).Slice(1, classIndex, classIndex + 1);
        int count = engine.TensorCountNonzero(inClass);
        var mean = Tensor<T>.FromMatrix(_classMeans.GetSubMatrix(classIndex, 0, 1, NumFeatures));
        var centered = engine.TensorMultiply(engine.TensorSubtract(Tensor<T>.FromMatrix(x), mean), inClass);

        // Sum of outer products divided by (n - 1) for unbiased estimate
        T denominator = NumOps.FromDouble(Math.Max(1, count - 1));
        return engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), denominator).ToMatrix();
    }

    /// <summary>
    /// Adds regularization to the covariance matrix.
    /// </summary>
    private Matrix<T> AddRegularization(Matrix<T> covariance)
    {
        T regParam = NumOps.FromDouble(Options.RegularizationParam);

        var engine = Engine;
        return engine.TensorAdd(Tensor<T>.FromMatrix(covariance),
            engine.TensorMultiplyScalar(engine.TensorEye<T>(NumFeatures), regParam)).ToMatrix();
    }

    /// <summary>
    /// Computes the inverse of a matrix using Gaussian elimination.
    /// </summary>
    private Matrix<T> ComputeInverse(Matrix<T> matrix) => InvertMatrix(matrix);

    /// <summary>
    /// Computes the log determinant of a matrix using LU decomposition.
    /// </summary>
    private T ComputeLogDeterminant(Matrix<T> matrix)
    {
        if (matrix.Rows == 0)
        {
            return NumOps.Zero;
        }

        // log|det| = sum of log|pivot| of the partially pivoted LU factorization (pivots floored at 1e-12)
        var engine = Engine;
        var (lu, _) = Linalg.LuFactor(Tensor<T>.FromMatrix(matrix));
        var pivots = engine.TensorClampMin(engine.TensorAbs(engine.TensorDiagonal(lu)), NumOps.FromDouble(1e-12));
        return engine.TensorSum(engine.TensorLog(pivots));
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been fitted.");
        }

        return input.Rows == 0 ? new Vector<T>(0) : ClassIndicesToLabels(ArgMaxRows(probs));
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_classMeans is null || _classCovarianceInverses is null ||
            _classPriors is null || _classLogDets is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (input.Rows == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        var engine = Engine;
        var samples = Tensor<T>.FromMatrix(input.Columns == NumFeatures
            ? input
            : input.GetSubMatrix(0, 0, input.Rows, NumFeatures));

        // Per-class offset log(prior) - 0.5 * log|Sigma_c|
        T minusHalf = NumOps.FromDouble(-0.5);
        var offsets = engine.TensorAdd(engine.TensorLog(_classPriors.AsTensor()),
            engine.TensorMultiplyScalar(_classLogDets.AsTensor(), minusHalf)).ToArray();

        // Log posterior = offset_c - 0.5 * (x - mu_c)^T Sigma_c^(-1) (x - mu_c)
        var columns = new Vector<T>[NumClasses];
        for (int c = 0; c < NumClasses; c++)
        {
            var diff = engine.TensorSubtract(samples, Tensor<T>.FromMatrix(_classMeans.GetSubMatrix(c, 0, 1, NumFeatures)));
            var inverse = Tensor<T>.FromMatrix(_classCovarianceInverses[c]);
            var mahalanobis = engine.ReduceSum(
                engine.TensorMultiply(engine.TensorMatMul(diff, inverse), diff), new[] { 1 }, false);
            columns[c] = engine.TensorAddScalar(engine.TensorMultiplyScalar(mahalanobis, minusHalf), offsets[c]).ToVector();
        }

        // Convert to probabilities using softmax
        return engine.Softmax(Tensor<T>.FromMatrix(StackColumns(columns)), 1).ToMatrix();
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        if (probs.Rows == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        var engine = Engine;
        return engine.TensorLog(engine.TensorClampMin(Tensor<T>.FromMatrix(probs), NumOps.FromDouble(1e-15))).ToMatrix();
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["RegularizationParam"] = Options.RegularizationParam;
        return metadata;
    }

    /// <inheritdoc/>
    private Vector<T> PackParameters()
    {
        // QDA doesn't have a simple parameter vector like linear classifiers
        // Return flattened class means for serialization
        if (_classMeans is null)
        {
            return new Vector<T>(0);
        }

        return _classMeans.ToRowVector();
    }

    /// <inheritdoc/>
    private void UnpackParameters(Vector<T> parameters)
    {
        // QDA computes parameters from class statistics, not direct parameter setting.
        // Accept silently so the optimizer can initialize the model without crashing.
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        return CreateNewInstance();
    }

    private void SerializeMatrix(Dictionary<string, object> data, string name, Matrix<T>? matrix)
    {
        if (matrix is null) return;
        data[name] = matrix.ToRowVector().ToArray();
        data[$"{name}Rows"] = matrix.Rows;
        data[$"{name}Cols"] = matrix.Columns;
    }

    private Matrix<T>? DeserializeMatrix(JObject obj, string name)
    {
        var token = obj[name];
        if (token is null) return null;
        var arr = token.ToObject<T[]>() ?? Array.Empty<T>();
        int rows = obj[$"{name}Rows"]?.ToObject<int>() ?? 0;
        int cols = obj[$"{name}Cols"]?.ToObject<int>() ?? 0;
        if (rows <= 0 || cols <= 0) return null;
        if (arr.Length < rows * cols)
        {
            throw new InvalidOperationException(
                $"Deserialization failed: {name} array length {arr.Length} is less than expected {rows}x{cols}={rows * cols}.");
        }
        var values = new T[rows * cols];
        Array.Copy(arr, values, values.Length);
        return new Tensor<T>(values, new[] { rows, cols }).ToMatrix();
    }

    private void SerializeVector(Dictionary<string, object> data, string name, Vector<T>? vector)
    {
        if (vector is null) return;
        data[name] = vector.ToArray();
    }

    private Vector<T>? DeserializeVector(JObject obj, string name)
    {
        var token = obj[name];
        if (token is null) return null;
        var arr = token.ToObject<T[]>() ?? Array.Empty<T>();
        if (arr.Length == 0) return null;
        return Vector<T>.Wrap(arr);
    }
}
