using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Models.Parameters;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.SVM;



/// <summary>
/// Nu-Support Vector Classifier using the nu-SVM formulation.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Nu-SVC uses a different parameterization than standard SVC. Instead of the C parameter,
/// it uses nu which is an upper bound on the fraction of margin errors and a lower bound
/// on the fraction of support vectors.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Nu-SVC is an alternative way to control the SVM's complexity:
///
/// Standard SVC uses C:
/// - C controls the penalty for misclassification
/// - Hard to interpret: "what does C=1.0 mean?"
///
/// Nu-SVC uses nu:
/// - nu is between 0 and 1
/// - nu is approximately the fraction of support vectors
/// - nu is also an upper bound on training errors
/// - More intuitive: "I want about 30% support vectors" means nu=0.3
///
/// Use Nu-SVC when:
/// - You want more interpretable regularization
/// - You have a target for the error rate
/// - The C parameter in standard SVC is hard to tune
///
/// Note: Nu-SVC and standard SVC produce very similar results when
/// properly tuned, but nu can be easier to set intuitively.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Nu-SVC with intuitive regularization parameter
/// var options = new SVMOptions&lt;double&gt;();
///
/// // Prepare training data: 6 samples with 2 features
/// var features = new Matrix&lt;double&gt;(6, 2);
/// features[0, 0] = 1.0; features[0, 1] = 1.1;
/// features[1, 0] = 1.2; features[1, 1] = 0.9;
/// features[2, 0] = 0.8; features[2, 1] = 1.0;
/// features[3, 0] = 5.0; features[3, 1] = 5.1;
/// features[4, 0] = 5.2; features[4, 1] = 4.9;
/// features[5, 0] = 4.8; features[5, 1] = 5.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train with nu parameter controlling support vector fraction
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new NuSupportVectorClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict class for new sample
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 1.1; newSample[0, 1] = 1.0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.SVM)]
[ModelCategory(ModelCategory.Kernel)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("New Support Vector Algorithms", "https://doi.org/10.1162/089976600300015565", Year = 2000, Authors = "Bernhard Scholkopf, Alex J. Smola, Robert C. Williamson, Peter L. Bartlett")]
public partial class NuSupportVectorClassifier<T> : SVMBase<T>
{
    /// <summary>
    /// Stored training features.
    /// </summary>
    [Buffer(Availability = ParameterAvailability.Fit)]
    private Matrix<T>? _xTrain;

    /// <summary>
    /// Stored training labels (converted to +1/-1).
    /// </summary>
    [Buffer(Availability = ParameterAvailability.Fit)]
    private Vector<T>? _yTrain;

    /// <summary>
    /// Alpha coefficients.
    /// </summary>
    [FittedParameter]
    private Vector<T>? _alphas;

    /// <summary>
    /// Rho parameter (offset in the decision function).
    /// </summary>

    [FittedParameter]
    private T _rho;


    /// <summary>
    /// The nu parameter value.
    /// </summary>
    private double _nu;

    /// <summary>
    /// Returns the maximum of two values.
    /// </summary>
    private T Max(T a, T b)
    {
        return NumOps.Compare(a, b) >= 0 ? a : b;
    }

    /// <summary>
    /// Returns the minimum of two values.
    /// </summary>
    private T Min(T a, T b)
    {
        return NumOps.Compare(a, b) <= 0 ? a : b;
    }

    /// <summary>
    /// Initializes a new instance of the NuSupportVectorClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the Nu-SVC.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    /// <param name="nu">The nu parameter (0 to 1). Default is 0.5.</param>
    public NuSupportVectorClassifier(SVMOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        double nu = 0.5)
        : base(options, regularization)
    {
        _rho = NumOps.Zero;
        if (nu <= 0 || nu > 1)
        {
            throw new ArgumentException("Nu must be in (0, 1].", nameof(nu));
        }
        _nu = nu;
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the Nu-SVC on the provided data.
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

        // Store training data
        _xTrain = x.Clone();

        // Convert labels to +1/-1
        _yTrain = SignedTargets(y);

        // Train using Nu-SMO variant
        TrainNuSMO();
    }

    /// <summary>
    /// Nu-SMO training algorithm.
    /// </summary>
    private void TrainNuSMO()
    {
        if (_xTrain is null || _yTrain is null)
        {
            throw new InvalidOperationException("Training data has not been initialized.");
        }

        int n = _xTrain.Rows;
        _intercept = new Vector<T>(1);

        // The nu-SVC dual (Scholkopf et al., 2000; Chang and Lin, 2001) is
        //     minimize  ½·alphaᵀQ·alpha    subject to  yᵀalpha = 0,  sum(alpha) = nu,
        //                                              0 <= alpha_i <= 1/n
        // with Q_ij = y_i·y_j·K(x_i, x_j). It differs from C-SVC in carrying a SECOND equality
        // constraint fixing the total mass, and in having no linear term.
        //
        // The shared solver handles this in its same-label pairing mode: moving two multipliers of
        // the same class in opposite directions leaves both sums invariant, which is precisely
        // LIBSVM's Solver_NU. A feasible starting point is required, since all-zeros satisfies
        // yᵀalpha = 0 but not sum(alpha) = nu.
        //
        // This replaces an inlined loop that described itself as "simplified optimization -
        // gradient descent on alphas", picked its partner multiplier at RANDOM, and applied a
        // "simplified KKT check" that ignored the second constraint entirely.
        var linear = new Vector<T>(n);            // nu-SVC has no linear term
        T upperBound = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));
        var upperBounds = FilledVector(n, upperBound);

        var initialAlphas = BuildFeasibleNuStart(n, upperBound);

        var solver = new AiDotNet.Solvers.QuadraticProgramming.SequentialMinimalOptimizationSolver<T>(
            new SequentialMinimalOptimizationOptions
            {
                MaxIterations = Options.MaxIterations < 0
                    ? 1000000
                    : (int)Math.Min(int.MaxValue, (long)Options.MaxIterations * n),
                Tolerance = Options.Tolerance,
                RestrictPairsToSameLabel = true,
            });

        var (alphas, bias, _) = solver.Solve(
            (a, b) => ComputeKernel(GetRow(_xTrain, a), GetRow(_xTrain, b)),
            _yTrain, linear, upperBounds, initialAlphas);

        _alphas = alphas;
        _intercept[0] = bias;
        _rho = NumOps.Negate(bias);

        ExtractSupportVectors();
    }

    /// <summary>
    /// Builds a starting point satisfying both nu-SVC equality constraints.
    /// </summary>
    /// <remarks>
    /// Each class receives half of the total mass nu, spread evenly across its members and capped
    /// at the per-multiplier bound, so that sum(alpha) = nu and yᵀalpha = 0 both hold before the
    /// first step. Any leftover mass from the cap is redistributed to members still below it.
    /// </remarks>
    private Vector<T> BuildFeasibleNuStart(int n, T upperBound)
    {
        if (_yTrain is null)
        {
            throw new InvalidOperationException("_yTrain is null where it is required.");
        }

        // Labels are exactly +1 / -1 (see SignedTargets).
        int positiveCount = IndicesAbove(_yTrain, NumOps.Zero).Length;
        int negativeCount = n - positiveCount;

        double maximumFeasibleNu = 2.0 * Math.Min(positiveCount, negativeCount) / n;
        if (_nu > maximumFeasibleNu + 1e-12)
        {
            throw new ArgumentException(
                $"Nu={_nu} is infeasible for this class distribution. With {positiveCount} positive " +
                $"and {negativeCount} negative samples, Nu must be at most {maximumFeasibleNu}.",
                nameof(_nu));
        }

        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Every member of a class shares the same cap, so the even water-fill settles in one pass:
        // each member receives min(cap, (nu / 2) / classCount).
        double half = _nu / 2.0;
        T positiveShare = positiveCount == 0 ? NumOps.Zero : Min(upperBound, NumOps.FromDouble(half / positiveCount));
        T negativeShare = negativeCount == 0 ? NumOps.Zero : Min(upperBound, NumOps.FromDouble(half / negativeCount));
        var isPositive = Engine.TensorGreaterThan(_yTrain.AsTensor(), NumOps.Zero);
        return Engine.TensorWhere(isPositive,
            FilledVector(n, positiveShare).AsTensor(),
            FilledVector(n, negativeShare).AsTensor()).ToVector();
    }

    /// <summary>
    /// Decision values sum over alpha_i &gt; 1e-10 of alpha_i * y_i * kernel(x_i, x) - rho for every row.
    /// </summary>
    private Vector<T> ComputeDecisions(Matrix<T> input)
    {
        if (_xTrain is null || _yTrain is null || _alphas is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var active = IndicesAbove(_alphas, NumOps.FromDouble(1e-10));
        var coefficients = Engine.Multiply(SelectElements(_alphas, active), SelectElements(_yTrain, active));
        var train = active.Length == 0 ? new Matrix<T>(0, _xTrain.Columns) : _xTrain.GetRows(active);
        return KernelDecisions(input, train, coefficients, NumOps.Negate(_rho)).GetColumn(0);
    }

    /// <summary>
    /// Extracts support vectors.
    /// </summary>
    private void ExtractSupportVectors()
    {
        if (_xTrain is null || _yTrain is null || _alphas is null)
        {
            throw new InvalidOperationException("Training data has not been initialized.");
        }

        var svIndices = IndicesAbove(_alphas, NumOps.FromDouble(1e-10));
        if (svIndices.Length > 0)
        {
            var supportVectors = _xTrain.GetRows(svIndices);
            _supportVectors = supportVectors.Columns == NumFeatures
                ? supportVectors
                : SelectColumns(supportVectors, Enumerable.Range(0, NumFeatures).ToArray());
            var dualCoef = Engine.Multiply(SelectElements(_alphas, svIndices), SelectElements(_yTrain, svIndices));
            _dualCoef = dualCoef.AsTensor().Reshape(1, svIndices.Length).ToMatrix();
        }
    }

    /// <inheritdoc/>
    public override Matrix<T> DecisionFunction(Matrix<T> input)
    {
        if (input.Rows == 0)
        {
            return new Matrix<T>(0, 1);
        }

        return ComputeDecisions(input).AsTensor().Reshape(input.Rows, 1).ToMatrix();
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var decisions = DecisionFunction(input);
        if (input.Rows == 0)
        {
            return new Vector<T>(0);
        }

        // Non-negative decision -> positive class, otherwise the first class
        var isPositive = AtLeast(decisions.GetColumn(0), NumOps.Zero);
        var positive = FilledVector(input.Rows, ClassLabels[ClassLabels.Length - 1]).AsTensor();
        var negative = FilledVector(input.Rows, ClassLabels[0]).AsTensor();
        return Engine.TensorWhere(isPositive.AsTensor(), positive, negative).ToVector();
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        var decisions = DecisionFunction(input);
        if (input.Rows == 0 || NumClasses == 0)
        {
            return new Matrix<T>(input.Rows, NumClasses);
        }

        // Sigmoid transformation: p = 1 / (1 + exp(-decision)); columns [1 - p, p, 0, ...]
        var engine = Engine;
        var ones = new Tensor<T>(new[] { input.Rows, 1 });
        engine.TensorFill(ones, NumOps.One);
        var prob = engine.TensorDivide(ones,
            engine.TensorAddScalar(engine.TensorExp(engine.TensorNegate(Tensor<T>.FromMatrix(decisions))), NumOps.One));
        var columns = new List<Tensor<T>> { engine.ScalarMinusTensor(NumOps.One, prob) };
        if (NumClasses > 1)
        {
            columns.Add(prob);
        }

        if (NumClasses > 2)
        {
            columns.Add(new Tensor<T>(new[] { input.Rows, NumClasses - 2 }));
        }

        return (columns.Count == 1 ? columns[0] : engine.TensorConcatenate(columns.ToArray(), 1)).ToMatrix();
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Nu"] = _nu;
        metadata.AdditionalInfo["Rho"] = NumOps.ToDouble(_rho);
        return metadata;
    }

    private void SerializeMatrix(Dictionary<string, object> data, string name, Matrix<T>? matrix)
    {
        SerializeMatrixValues(data, name, matrix);
    }

    private Matrix<T>? DeserializeMatrix(JObject obj, string name)
    {
        return DeserializeMatrixValues(obj, name);
    }

    private void SerializeVector(Dictionary<string, object> data, string name, Vector<T>? vector)
    {
        SerializeVectorValues(data, name, vector);
    }

    private Vector<T>? DeserializeVector(JObject obj, string name)
    {
        return DeserializeVectorValues(obj, name);
    }
}
