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
/// Linear Discriminant Analysis classifier.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// LDA finds a linear combination of features that best separates classes.
/// It projects data onto a lower-dimensional space while maximizing class separation.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Linear Discriminant Analysis (LDA) is a classification technique that:
///
/// 1. Finds the directions that best separate the classes
/// 2. Projects the data onto these directions
/// 3. Classifies based on which class region the projection falls into
///
/// Key assumptions:
/// - Each class follows a Gaussian distribution
/// - All classes share the same covariance matrix
/// - Features are continuous
///
/// When to use LDA:
/// - When classes are well-separated
/// - When you have more samples than features
/// - When you want a simple, interpretable classifier
/// - As a dimensionality reduction technique
///
/// Trade-offs:
/// - Assumes shared covariance (use QDA if classes differ)
/// - Linear boundaries (may not work for complex patterns)
/// - Sensitive to outliers
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create LDA classifier for linear class separation
/// var options = new DiscriminantAnalysisOptions&lt;double&gt;();
///
/// // Prepare training data with two classes: 6 samples, 2 features
/// var features = new Matrix&lt;double&gt;(6, 2);
/// features[0, 0] = 1.0; features[0, 1] = 1.1;
/// features[1, 0] = 1.2; features[1, 1] = 0.9;
/// features[2, 0] = 0.8; features[2, 1] = 1.0;
/// features[3, 0] = 5.0; features[3, 1] = 5.1;
/// features[4, 0] = 5.2; features[4, 1] = 4.9;
/// features[5, 0] = 4.8; features[5, 1] = 5.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train to find optimal linear projection for class separation
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new LinearDiscriminantAnalysis&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Classify new sample using discriminant function
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 1.1; newSample[0, 1] = 1.0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("The Use of Multiple Measurements in Taxonomic Problems", "https://doi.org/10.1111/j.1469-1809.1936.tb02137.x")]
public partial class LinearDiscriminantAnalysis<T> : ProbabilisticClassifierBase<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The flattened class means. Its own comment says LDA has no simple parameter vector, but it DOES serialize the means and a test asserts they are non-empty after training -- deleting the surface on the strength of the comment was wrong. PackParameters and UnpackParameters ARE the previous
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
    /// LDA computes parameters from class covariance matrices during training.
    /// It does not support flat parameter initialization.
    /// </summary>
    public override bool SupportsParameterInitialization => false;

    /// <summary>
    /// Gets the LDA-specific options.
    /// </summary>
    protected new DiscriminantAnalysisOptions<T> Options => (DiscriminantAnalysisOptions<T>)base.Options;

    /// <summary>
    /// Class means for each class.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _classMeans = new Matrix<T>(0, 0);

    /// <summary>
    /// Pooled within-class covariance matrix (shared by all classes).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _pooledCovariance = new Matrix<T>(0, 0);

    /// <summary>
    /// Inverse of the pooled covariance matrix.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _covarianceInverse = new Matrix<T>(0, 0);

    /// <summary>
    /// Class priors (prior probabilities).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _classPriors = new Vector<T>(0);

    /// <summary>
    /// Initializes a new instance of the LinearDiscriminantAnalysis class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public LinearDiscriminantAnalysis(DiscriminantAnalysisOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new DiscriminantAnalysisOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the LDA classifier on the provided data.
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

        // Compute pooled within-class covariance
        _pooledCovariance = ComputePooledCovariance(x, y);

        // Add regularization if specified
        if (Options.RegularizationParam > 0)
        {
            _pooledCovariance = AddRegularization(_pooledCovariance);
        }

        // Compute inverse of covariance
        _covarianceInverse = ComputeInverse(_pooledCovariance);
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
        var counts = CountsToValues(engine.TensorClampMin(CountClasses(y), 1)).Reshape(NumClasses, 1);
        return engine.TensorDivide(sums, counts).ToMatrix();
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

        // Class counts in int (a sum of one-hot rows in T stops at 2048 for Half), divided in double
        int n = y.Length;
        return n == 0 ? new Vector<T>(NumClasses) : CountFractions(CountClasses(y), n).ToVector();
    }

    /// <summary>
    /// Computes the pooled within-class covariance matrix.
    /// </summary>
    private Matrix<T> ComputePooledCovariance(Matrix<T> x, Vector<T> y)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been fitted.");
        }

        var engine = Engine;
        var membership = ClassMembership(y);

        // Samples of a known class, centered on their class mean (rows of unknown labels become zero)
        var known = engine.ReduceSum(membership, new[] { 1 }, true);
        int totalCount = engine.TensorCountNonzero(known);
        var classMeanPerSample = engine.TensorMatMul(membership, Tensor<T>.FromMatrix(_classMeans));
        var centered = engine.TensorMultiply(
            engine.TensorSubtract(Tensor<T>.FromMatrix(x), classMeanPerSample), known);

        // Sum of (x - mean)(x - mean)^T divided by (n - k) where k is number of classes
        // When totalCount <= NumClasses, covariance may be rank-deficient
        bool insufficientSamples = totalCount <= NumClasses;
        T denominator = NumOps.FromDouble(Math.Max(1, totalCount - NumClasses));
        var covariance = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), denominator);

        // Add automatic regularization when samples are insufficient to ensure positive definiteness
        if (insufficientSamples)
        {
            covariance = engine.TensorAdd(covariance,
                engine.TensorMultiplyScalar(engine.TensorEye<T>(NumFeatures), NumOps.FromDouble(1e-6)));
        }

        return covariance.ToMatrix();
    }

    /// <summary>
    /// Adds regularization to the covariance matrix.
    /// </summary>
    private Matrix<T> AddRegularization(Matrix<T> covariance)
    {
        T regParam = NumOps.FromDouble(Options.RegularizationParam);

        // Add regularization to diagonal: Sigma + reg * I
        var engine = Engine;
        return engine.TensorAdd(Tensor<T>.FromMatrix(covariance),
            engine.TensorMultiplyScalar(engine.TensorEye<T>(NumFeatures), regParam)).ToMatrix();
    }

    /// <summary>
    /// Computes the inverse of a matrix using Gaussian elimination.
    /// </summary>
    private Matrix<T> ComputeInverse(Matrix<T> matrix) => InvertMatrix(matrix);

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
        if (_classMeans is null || _covarianceInverse is null || _classPriors is null || ClassLabels is null)
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
        var inverse = Tensor<T>.FromMatrix(_covarianceInverse);
        var logPriors = engine.TensorLog(_classPriors.AsTensor()).ToArray();
        T minusHalf = NumOps.FromDouble(-0.5);

        // Log posterior per class = log(prior) - 0.5 * (x - mu_c)^T Sigma^(-1) (x - mu_c)
        var columns = new Vector<T>[NumClasses];
        for (int c = 0; c < NumClasses; c++)
        {
            var diff = engine.TensorSubtract(samples, Tensor<T>.FromMatrix(_classMeans.GetSubMatrix(c, 0, 1, NumFeatures)));
            var mahalanobis = engine.ReduceSum(
                engine.TensorMultiply(engine.TensorMatMul(diff, inverse), diff), new[] { 1 }, false);
            columns[c] = engine.TensorAddScalar(engine.TensorMultiplyScalar(mahalanobis, minusHalf), logPriors[c]).ToVector();
        }

        // Convert to probabilities using softmax (log-sum-exp trick)
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
        return engine.TensorLog(engine.TensorClampMin(Tensor<T>.FromMatrix(probs), Tolerance(1e-15))).ToMatrix();
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
        // LDA doesn't have a simple parameter vector like linear classifiers
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
        // LDA computes parameters from class statistics, not direct parameter setting.
        // Accept silently so the optimizer can initialize the model without crashing.
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        // Return a fresh instance — LDA parameters come from training, not direct setting
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

/// <summary>
/// Configuration options for discriminant analysis classifiers.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class DiscriminantAnalysisOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the regularization parameter.
    /// </summary>
    /// <value>
    /// Value added to diagonal of covariance matrix for stability. Default is 0.0001.
    /// </value>
    public double RegularizationParam { get; set; } = 0.0001;
}
