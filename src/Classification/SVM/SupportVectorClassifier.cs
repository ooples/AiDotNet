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
/// // Use AiModelBuilder facade for supportvectorclassifier classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new SupportVectorClassifier&lt;double&gt;(
///         new SupportVectorClassifierOptions&lt;double&gt;()));
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
        _xTrain = new Matrix<T>(x.Rows, x.Columns);
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                _xTrain[i, j] = x[i, j];
            }
        }

        // Convert labels to +1/-1 for binary classification
        _yTrain = new Vector<T>(y.Length);
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        for (int i = 0; i < y.Length; i++)
        {
            if (NumOps.Compare(y[i], positiveClass) == 0)
            {
                _yTrain[i] = NumOps.One;
            }
            else
            {
                _yTrain[i] = NumOps.Negate(NumOps.One);
            }
        }

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
        var linear = new Vector<T>(n);
        var upperBounds = new Vector<T>(n);
        T penalty = NumOps.FromDouble(Options.C);
        for (int i = 0; i < n; i++)
        {
            linear[i] = NumOps.Negate(NumOps.One);
            upperBounds[i] = penalty;
        }

        var solver = new AiDotNet.Solvers.QuadraticProgramming.SequentialMinimalOptimizationSolver<T>(
            new SequentialMinimalOptimizationOptions
            {
                MaxIterations = Options.MaxIterations < 0 ? 1000000 : Options.MaxIterations * n,
                Tolerance = Options.Tolerance,
            },
            _random);

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
        // memory access â€” see TrainSMO for why.
        T prediction = ComputeDecisionFromArray(_xTrainRows[i]);
        return NumOps.Subtract(prediction, _yTrainArr[i]);
    }

    /// <summary>
    /// Computes the decision value for a sample (Vector overload â€” used by
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

        T sum = _intercept[0];
        for (int i = 0; i < _xTrain.Rows; i++)
        {
            if (NumOps.Compare(_alphas[i], NumOps.Zero) > 0)
            {
                T kernel = ComputeKernel(GetRow(_xTrain, i), x);
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(_alphas[i], _yTrain[i]), kernel));
            }
        }
        return sum;
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
        T sum = _intercept[0];
        var alphas = _alphasArr;
        var ys = _yTrainArr;
        var rows = _xTrainRows;
        var zero = NumOps.Zero;
        for (int i = 0; i < rows.Length; i++)
        {
            T alpha = alphas[i];
            if (NumOps.Compare(alpha, zero) > 0)
            {
                T kernel = ComputeKernelFromArrays(rows[i], x);
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(alpha, ys[i]), kernel));
            }
        }
        return sum;
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
        if (_xTrain is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }
        return ComputeKernel(GetRow(_xTrain, i), GetRow(_xTrain, j));
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

        var svIndices = new List<int>();
        for (int i = 0; i < _alphas.Length; i++)
        {
            if (NumOps.Compare(_alphas[i], NumOps.FromDouble(1e-7)) > 0)
            {
                svIndices.Add(i);
            }
        }

        if (svIndices.Count > 0)
        {
            _supportVectors = new Matrix<T>(svIndices.Count, NumFeatures);
            _dualCoef = new Matrix<T>(1, svIndices.Count);

            for (int i = 0; i < svIndices.Count; i++)
            {
                int idx = svIndices[i];
                for (int j = 0; j < NumFeatures; j++)
                {
                    _supportVectors[i, j] = _xTrain[idx, j];
                }
                _dualCoef[0, i] = NumOps.Multiply(_alphas[idx], _yTrain[idx]);
            }
        }
    }

    /// <inheritdoc/>
    public override Matrix<T> DecisionFunction(Matrix<T> input)
    {
        if (_xTrain == null || _alphas == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        var decisions = new Matrix<T>(input.Rows, 1);

        for (int i = 0; i < input.Rows; i++)
        {
            var sample = GetRow(input, i);
            decisions[i, 0] = ComputeDecision(sample);
        }

        return decisions;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        var decisions = DecisionFunction(input);
        var probabilities = new Matrix<T>(input.Rows, NumClasses);

        // Convert decision values to probabilities using sigmoid (Platt scaling approximation)
        for (int i = 0; i < input.Rows; i++)
        {
            T decision = decisions[i, 0];
            T prob = Sigmoid(decision);

            if (NumClasses == 2)
            {
                probabilities[i, 0] = NumOps.Subtract(NumOps.One, prob);
                probabilities[i, 1] = prob;
            }
            else
            {
                // For multi-class, this is a simplified approach
                probabilities[i, NumClasses - 1] = prob;
                T remaining = NumOps.Divide(NumOps.Subtract(NumOps.One, prob),
                    NumOps.FromDouble(NumClasses - 1));
                for (int c = 0; c < NumClasses - 1; c++)
                {
                    probabilities[i, c] = remaining;
                }
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Sigmoid function for probability estimation.
    /// </summary>
    private T Sigmoid(T x)
    {
        T negX = NumOps.Negate(x);
        T expNegX = NumOps.Exp(negX);
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new SupportVectorClassifier<T>(new SVMOptions<T>
        {
            C = Options.C,
            Kernel = Options.Kernel,
            Gamma = Options.Gamma,
            Degree = Options.Degree,
            Coef0 = Options.Coef0,
            Tolerance = Options.Tolerance,
            MaxIterations = Options.MaxIterations,
            Shrinking = Options.Shrinking,
            Probability = Options.Probability,
            Seed = Options.Seed,
            OneVsRest = Options.OneVsRest,
            CacheSize = Options.CacheSize
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (SupportVectorClassifier<T>)CreateNewInstance();

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;

        if (ClassLabels != null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (_xTrain != null)
        {
            clone._xTrain = new Matrix<T>(_xTrain.Rows, _xTrain.Columns);
            for (int i = 0; i < _xTrain.Rows; i++)
            {
                for (int j = 0; j < _xTrain.Columns; j++)
                {
                    clone._xTrain[i, j] = _xTrain[i, j];
                }
            }
        }

        if (_yTrain != null)
        {
            clone._yTrain = new Vector<T>(_yTrain.Length);
            for (int i = 0; i < _yTrain.Length; i++)
            {
                clone._yTrain[i] = _yTrain[i];
            }
        }

        if (_alphas != null)
        {
            clone._alphas = new Vector<T>(_alphas.Length);
            for (int i = 0; i < _alphas.Length; i++)
            {
                clone._alphas[i] = _alphas[i];
            }
        }

        if (_intercept != null)
        {
            clone._intercept = new Vector<T>(_intercept.Length);
            for (int i = 0; i < _intercept.Length; i++)
            {
                clone._intercept[i] = _intercept[i];
            }
        }

        if (_supportVectors != null)
        {
            clone._supportVectors = new Matrix<T>(_supportVectors.Rows, _supportVectors.Columns);
            for (int i = 0; i < _supportVectors.Rows; i++)
            {
                for (int j = 0; j < _supportVectors.Columns; j++)
                {
                    clone._supportVectors[i, j] = _supportVectors[i, j];
                }
            }
        }

        if (_dualCoef != null)
        {
            clone._dualCoef = new Matrix<T>(_dualCoef.Rows, _dualCoef.Columns);
            for (int i = 0; i < _dualCoef.Rows; i++)
            {
                for (int j = 0; j < _dualCoef.Columns; j++)
                {
                    clone._dualCoef[i, j] = _dualCoef[i, j];
                }
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumClasses", NumClasses },
            { "NumFeatures", NumFeatures },
            { "TaskType", (int)TaskType },
            { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() },
            { "RegularizationOptions", Regularization.GetOptions() },
            { "SVMOptions_C", Options.C },
            { "SVMOptions_Kernel", (int)Options.Kernel },
            { "SVMOptions_HasGamma", Options.Gamma.HasValue },
            { "SVMOptions_Degree", Options.Degree },
            { "SVMOptions_Coef0", Options.Coef0 },
            { "SVMOptions_Tolerance", Options.Tolerance },
            { "SVMOptions_MaxIterations", Options.MaxIterations },
            { "SVMOptions_Shrinking", Options.Shrinking },
            { "SVMOptions_Probability", Options.Probability },
            { "SVMOptions_OneVsRest", Options.OneVsRest }
        };

        if (Options.Gamma.HasValue)
            modelData["SVMOptions_GammaValue"] = Options.Gamma.Value;

        SerializeMatrix(modelData, "XTrain", _xTrain);
        SerializeVector(modelData, "YTrain", _yTrain);
        SerializeVector(modelData, "Alphas", _alphas);
        SerializeVector(modelData, "Intercept", _intercept);
        SerializeMatrix(modelData, "SupportVectors", _supportVectors);
        SerializeMatrix(modelData, "DualCoef", _dualCoef);

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata == null || modelMetadata.ModelData == null)
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<JObject>(modelDataString);

        if (modelDataObj == null)
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");

        NumClasses = modelDataObj["NumClasses"]?.ToObject<int>() ?? 0;
        NumFeatures = modelDataObj["NumFeatures"]?.ToObject<int>() ?? 0;
        TaskType = (ClassificationTaskType)(modelDataObj["TaskType"]?.ToObject<int>() ?? 0);

        var classLabelsToken = modelDataObj["ClassLabels"];
        if (classLabelsToken is not null)
        {
            var classLabelsAsDoubles = classLabelsToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (classLabelsAsDoubles.Length > 0)
            {
                ClassLabels = new Vector<T>(classLabelsAsDoubles.Length);
                for (int i = 0; i < classLabelsAsDoubles.Length; i++)
                    ClassLabels[i] = NumOps.FromDouble(classLabelsAsDoubles[i]);
            }
        }

        // Restore SVMOptions (kernel, C, gamma, etc.) - critical for correct predictions
        if (modelDataObj["SVMOptions_C"] is not null)
            Options.C = modelDataObj["SVMOptions_C"]?.ToObject<double>() ?? 1.0;
        if (modelDataObj["SVMOptions_Kernel"] is not null)
            Options.Kernel = (Enums.KernelType)(modelDataObj["SVMOptions_Kernel"]?.ToObject<int>() ?? 0);
        if (modelDataObj["SVMOptions_HasGamma"]?.ToObject<bool>() == true)
            Options.Gamma = modelDataObj["SVMOptions_GammaValue"]?.ToObject<double>();
        else
            Options.Gamma = null;
        if (modelDataObj["SVMOptions_Degree"] is not null)
            Options.Degree = modelDataObj["SVMOptions_Degree"]?.ToObject<int>() ?? 3;
        if (modelDataObj["SVMOptions_Coef0"] is not null)
            Options.Coef0 = modelDataObj["SVMOptions_Coef0"]?.ToObject<double>() ?? 0.0;
        if (modelDataObj["SVMOptions_Tolerance"] is not null)
            Options.Tolerance = modelDataObj["SVMOptions_Tolerance"]?.ToObject<double>() ?? 0.001;
        if (modelDataObj["SVMOptions_MaxIterations"] is not null)
            Options.MaxIterations = modelDataObj["SVMOptions_MaxIterations"]?.ToObject<int>() ?? 1000;
        if (modelDataObj["SVMOptions_Shrinking"] is not null)
            Options.Shrinking = modelDataObj["SVMOptions_Shrinking"]?.ToObject<bool>() ?? true;
        if (modelDataObj["SVMOptions_Probability"] is not null)
            Options.Probability = modelDataObj["SVMOptions_Probability"]?.ToObject<bool>() ?? false;
        if (modelDataObj["SVMOptions_OneVsRest"] is not null)
            Options.OneVsRest = modelDataObj["SVMOptions_OneVsRest"]?.ToObject<bool>() ?? false;

        _xTrain = DeserializeMatrix(modelDataObj, "XTrain");
        _yTrain = DeserializeVector(modelDataObj, "YTrain");
        _alphas = DeserializeVector(modelDataObj, "Alphas");
        _intercept = DeserializeVector(modelDataObj, "Intercept");
        _supportVectors = DeserializeMatrix(modelDataObj, "SupportVectors");
        _dualCoef = DeserializeMatrix(modelDataObj, "DualCoef");
    }

    private void SerializeMatrix(Dictionary<string, object> data, string name, Matrix<T>? matrix)
    {
        if (matrix is null) return;
        var arr = new double[matrix.Rows * matrix.Columns];
        int idx = 0;
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                arr[idx++] = NumOps.ToDouble(matrix[i, j]);
        data[name] = arr;
        data[$"{name}Rows"] = matrix.Rows;
        data[$"{name}Cols"] = matrix.Columns;
    }

    private Matrix<T>? DeserializeMatrix(JObject obj, string name)
    {
        var token = obj[name];
        if (token is null) return null;
        var arr = token.ToObject<double[]>() ?? Array.Empty<double>();
        int rows = obj[$"{name}Rows"]?.ToObject<int>() ?? 0;
        int cols = obj[$"{name}Cols"]?.ToObject<int>() ?? 0;
        if (rows <= 0 || cols <= 0) return null;
        if (arr.Length < rows * cols)
        {
            throw new InvalidOperationException(
                $"Deserialization failed: {name} array length {arr.Length} is less than expected {rows}x{cols}={rows * cols}.");
        }
        var matrix = new Matrix<T>(rows, cols);
        int idx = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = NumOps.FromDouble(arr[idx++]);
        return matrix;
    }

    private void SerializeVector(Dictionary<string, object> data, string name, Vector<T>? vector)
    {
        if (vector is null) return;
        var arr = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++)
            arr[i] = NumOps.ToDouble(vector[i]);
        data[name] = arr;
    }

    private Vector<T>? DeserializeVector(JObject obj, string name)
    {
        var token = obj[name];
        if (token is null) return null;
        var arr = token.ToObject<double[]>() ?? Array.Empty<double>();
        if (arr.Length == 0) return null;
        var vector = new Vector<T>(arr.Length);
        for (int i = 0; i < arr.Length; i++)
            vector[i] = NumOps.FromDouble(arr[i]);
        return vector;
    }
}
