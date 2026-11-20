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
/// Support Vector Classifier using kernel methods for non-linear classification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This implementation uses a simplified Sequential Minimal Optimization (SMO) algorithm
/// to find the optimal separating hyperplane in the kernel-induced feature space.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// The Support Vector Classifier (SVC) finds the best boundary between classes by:
///
/// 1. Looking at all training points
/// 2. Finding the points closest to the decision boundary (support vectors)
/// 3. Drawing a boundary that maximizes the margin to these support vectors
/// 4. Using a kernel trick to handle non-linear boundaries
///
/// Common use cases:
/// - Text classification (spam detection, sentiment analysis)
/// - Image classification
/// - Bioinformatics (protein classification)
/// - Any problem with clear separation between classes
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for supportvectorclassifier classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new SupportVectorClassifier&lt;double&gt;(
///         new SVMOptions&lt;double&gt;()));
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.SVM)]
[ModelCategory(ModelCategory.Kernel)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("A Training Algorithm for Optimal Margin Classifiers", "https://doi.org/10.1145/130385.130401", Year = 1992, Authors = "Bernhard E. Boser, Isabelle M. Guyon, Vladimir N. Vapnik")]
public partial class SupportVectorClassifier<T> : SVMBase<T>
{
    /// <summary>
    /// Stored training features.
    /// </summary>
    [Buffer(Availability = ParameterAvailability.Fit)]
    private Matrix<T>? _xTrain;

    /// <summary>
    /// Stored training labels (converted to +1/-1 for binary).
    /// </summary>
    [Buffer(Availability = ParameterAvailability.Fit)]
    private Vector<T>? _yTrain;

    /// <summary>
    /// Alpha coefficients from SMO algorithm.
    /// </summary>
    [FittedParameter]
    private Vector<T>? _alphas;

    /// <summary>
    /// Cached training rows as raw arrays. Populated once at the start of
    /// <see cref="TrainSMO"/> and reused across the SMO inner loop so the
    /// kernel/decision evaluation does not have to re-extract a Vector{T}
    /// per inner iteration. The Vector{T} indexer's deferred-materializer
    /// monitor was the dominant cost per the PR #1184 trace.
    /// </summary>
    private T[][]? _xTrainRows;

    /// <summary>
    /// Cached training labels as a raw array (same rationale as
    /// <see cref="_xTrainRows"/>).
    /// </summary>
    private T[]? _yTrainArr;

    /// <summary>
    /// Cached alpha coefficients as a raw array, refreshed alongside
    /// <see cref="_alphas"/> after each SMO inner-loop update.
    /// </summary>
    private T[]? _alphasArr;

    /// <summary>
    /// Random number generator for SMO.
    /// </summary>
    private Random? _random;

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
    /// Initializes a new instance of the SupportVectorClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the SVC.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public SupportVectorClassifier(SVMOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the SVC on the provided data.
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

        _random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Store training data
        _xTrain = x.Clone();

        // Convert labels to +1/-1 for binary classification
        _yTrain = SignedTargets(y);

        // Train using simplified SMO
        TrainSMO();
    }

    /// <summary>
    /// Simplified SMO algorithm for training.
    /// </summary>
    private void TrainSMO()
    {
        if (_xTrain is null || _yTrain is null)
        {
            throw new InvalidOperationException("Training data has not been initialized.");
        }

        int n = _xTrain.Rows;
        _intercept = new Vector<T>(1);

        // Materialise the training matrix and labels into raw arrays once. The kernel evaluations
        // dominate training, and each one previously hammered the Vector{T} indexer through the
        // deferred-materializer monitor (~99% of train wall-clock per the PR #1184 trace).
        _xTrainRows = new T[n][];
        for (int i = 0; i < n; i++)
            _xTrainRows[i] = GetRowArray(_xTrain, i);
        _yTrainArr = _yTrain.ToArray();

        // The SVM dual is
        //     minimize  ½·alphaᵀQ·alpha − 1ᵀalpha   subject to  yᵀalpha = 0,  0 ≤ alpha_i ≤ C
        // with Q_ij = y_i·y_j·K(x_i, x_j). It is solved by the library's shared Sequential Minimal
        // Optimization solver, which implements Platt's full algorithm: second-choice heuristic,
        // error cache, and alternation between non-bound and full sweeps.
        //
        // This replaces an inlined SIMPLIFIED SMO that chose the partner multiplier at RANDOM (the
        // CS229 teaching variant). Random pairing converges far more slowly and can stall on
        // problems the full heuristic handles, and the same simplified loop was duplicated across
        // the SVM family — one correct implementation now serves all of them.
        var linear = FilledVector(n, NumOps.Negate(NumOps.One));
        var upperBounds = FilledVector(n, NumOps.FromDouble(Options.C));

        int solverIterationBudget = Options.MaxIterations <= 0
            ? 1_000_000
            : (int)Math.Min((long)Options.MaxIterations * n, int.MaxValue);

        var solver = new AiDotNet.Solvers.QuadraticProgramming.SequentialMinimalOptimizationSolver<T>(
            new SequentialMinimalOptimizationOptions
            {
                MaxIterations = solverIterationBudget,
                Tolerance = Options.Tolerance,
            });

        var (alphas, bias, _) = solver.Solve(
            ComputeKernelCached, _yTrain, linear, upperBounds);

        _alphas = alphas;
        _alphasArr = alphas.ToArray();
        _intercept[0] = bias;

        // Extract support vectors
        ExtractSupportVectors();
    }

    /// <summary>
    /// Computes the prediction error for sample i.
    /// </summary>
    private T ComputeError(int i)
    {
        if (_xTrainRows is null || _yTrainArr is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }
        // Reuse the cached row array for sample i and route through the
        // array-based decision path so the inner kernel loop stays on plain
        // memory access — see TrainSMO for why.
        T prediction = ComputeDecisionFromArray(_xTrainRows[i]);
        return NumOps.Subtract(prediction, _yTrainArr[i]);
    }

    /// <summary>
    /// Computes the decision value for a sample (Vector overload — used by
    /// Predict on user-supplied inputs).
    /// </summary>
    private T ComputeDecision(Vector<T> x)
    {
        if (_xTrain is null || _yTrain is null || _alphas is null || _intercept is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }
        // Materialise the query vector once; cached rows already exist when
        // _xTrainRows is populated (post-Train), otherwise fall back to the
        // direct Vector<T> path. The fast path matters because Predict on
        // a test set runs ComputeDecision per sample.
        if (_xTrainRows is not null && _yTrainArr is not null && _alphasArr is not null)
        {
            return ComputeDecisionFromArray(x.ToArray());
        }

        return ActiveDecisions(x.AsTensor().Reshape(1, x.Length).ToMatrix()).ToArray()[0];
    }

    /// <summary>
    /// bias + sum over samples with alpha &gt; 0 of alpha_i * y_i * kernel(x_i, x) for every row.
    /// </summary>
    private Vector<T> ActiveDecisions(Matrix<T> input)
    {
        if (_xTrain is null || _yTrain is null || _alphas is null || _intercept is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var active = IndicesAbove(_alphas, NumOps.Zero);
        var coefficients = Engine.Multiply(SelectElements(_alphas, active), SelectElements(_yTrain, active));
        var train = active.Length == 0 ? new Matrix<T>(0, _xTrain.Columns) : _xTrain.GetRows(active);
        return KernelDecisions(input, train, coefficients, _intercept.ToArray()[0]).GetColumn(0);
    }

    /// <summary>
    /// Array-typed decision evaluation. Walks the cached training rows and
    /// the cached alpha/label arrays so no per-iteration Vector{T} indexer
    /// hits the deferred-materializer monitor.
    /// </summary>
    private T ComputeDecisionFromArray(T[] x)
    {
        if (_xTrainRows is null || _yTrainArr is null || _alphasArr is null || _intercept is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }
        return ActiveDecisions(Vector<T>.Wrap(x).AsTensor().Reshape(1, x.Length).ToMatrix()).ToArray()[0];
    }

    /// <summary>
    /// Dispatches the configured kernel to its array overload. Currently the
    /// hot SVC training path only exercises RBF; other kernels fall back via
    /// allocating Vector{T} wrappers (one alloc per call instead of one per
    /// element).
    /// </summary>
    private T ComputeKernelFromArrays(T[] x, T[] y)
    {
        return Options.Kernel switch
        {
            KernelType.RBF => ComputeRBFKernelArrays(x, y),
            _ => ComputeKernel(new Vector<T>(x), new Vector<T>(y))
        };
    }

    /// <summary>
    /// Computes kernel value with caching support.
    /// </summary>
    private T ComputeKernelCached(int i, int j)
    {
        if (_xTrainRows is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }
        return ComputeKernelFromArrays(_xTrainRows[i], _xTrainRows[j]);
    }

    /// <summary>
    /// Selects the second alpha for SMO.
    /// </summary>
    private int SelectSecondAlpha(int i, int n)
    {
        if (_random is null)
        {
            throw new InvalidOperationException("Random number generator not initialized.");
        }
        int j = _random.Next(n - 1);
        if (j >= i) j++;
        return j;
    }

    /// <summary>
    /// Clips alpha to bounds [L, H].
    /// </summary>
    private T ClipAlpha(T alpha, T L, T H)
    {
        if (NumOps.Compare(alpha, H) > 0) return H;
        if (NumOps.Compare(alpha, L) < 0) return L;
        return alpha;
    }

    /// <summary>
    /// Updates the intercept after alpha update.
    /// </summary>
    private void UpdateIntercept(int i, int j, T Ei, T Ej,
        T alphaINew, T alphaJNew, T alphaIOld, T alphaJOld, T C)
    {
        if (_yTrain is null || _intercept is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        T yi = _yTrain[i];
        T yj = _yTrain[j];
        T Kii = ComputeKernelCached(i, i);
        T Kjj = ComputeKernelCached(j, j);
        T Kij = ComputeKernelCached(i, j);

        T b1 = NumOps.Subtract(_intercept[0],
            NumOps.Add(Ei,
                NumOps.Add(
                    NumOps.Multiply(NumOps.Multiply(yi, NumOps.Subtract(alphaINew, alphaIOld)), Kii),
                    NumOps.Multiply(NumOps.Multiply(yj, NumOps.Subtract(alphaJNew, alphaJOld)), Kij)
                )
            )
        );

        T b2 = NumOps.Subtract(_intercept[0],
            NumOps.Add(Ej,
                NumOps.Add(
                    NumOps.Multiply(NumOps.Multiply(yi, NumOps.Subtract(alphaINew, alphaIOld)), Kij),
                    NumOps.Multiply(NumOps.Multiply(yj, NumOps.Subtract(alphaJNew, alphaJOld)), Kjj)
                )
            )
        );

        if (NumOps.Compare(alphaINew, NumOps.Zero) > 0 && NumOps.Compare(alphaINew, C) < 0)
        {
            _intercept[0] = b1;
        }
        else if (NumOps.Compare(alphaJNew, NumOps.Zero) > 0 && NumOps.Compare(alphaJNew, C) < 0)
        {
            _intercept[0] = b2;
        }
        else
        {
            _intercept[0] = NumOps.Divide(NumOps.Add(b1, b2), NumOps.FromDouble(2.0));
        }
    }

    /// <summary>
    /// Extracts support vectors from training data.
    /// </summary>
    private void ExtractSupportVectors()
    {
        if (_alphas is null || _xTrain is null || _yTrain is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var svIndices = IndicesAbove(_alphas, NumOps.FromDouble(1e-7));
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
        if (_xTrain == null || _alphas == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        if (input.Rows == 0)
        {
            return new Matrix<T>(0, 1);
        }

        var decisions = ActiveDecisions(input);
        return decisions.AsTensor().Reshape(input.Rows, 1).ToMatrix();
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        var decisions = DecisionFunction(input);
        if (input.Rows == 0 || NumClasses == 0)
        {
            return new Matrix<T>(input.Rows, NumClasses);
        }

        // Convert decision values to probabilities using sigmoid (Platt scaling approximation)
        var engine = Engine;
        var ones = new Tensor<T>(new[] { input.Rows, 1 });
        engine.TensorFill(ones, NumOps.One);
        var prob = engine.TensorDivide(ones,
            engine.TensorAddScalar(engine.TensorExp(engine.TensorNegate(Tensor<T>.FromMatrix(decisions))), NumOps.One));
        var rest = engine.ScalarMinusTensor(NumOps.One, prob);

        if (NumClasses == 2)
        {
            return engine.TensorConcatenate(new[] { rest, prob }, 1).ToMatrix();
        }

        // For multi-class, this is a simplified approach: the last class gets p, the others share 1 - p
        if (NumClasses == 1)
        {
            return prob.ToMatrix();
        }

        var remaining = engine.TensorDivideScalar(rest, NumOps.FromDouble(NumClasses - 1));
        var shared = engine.TensorTile(remaining, new[] { 1, NumClasses - 1 });
        return engine.TensorConcatenate(new[] { shared, prob }, 1).ToMatrix();
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
