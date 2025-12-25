using AiDotNet.Classification;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

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
public class SupportVectorClassifier<T> : SVMBase<T>
{
    /// <summary>
    /// Stored training features.
    /// </summary>
    private Matrix<T>? _xTrain;

    /// <summary>
    /// Stored training labels (converted to +1/-1 for binary).
    /// </summary>
    private Vector<T>? _yTrain;

    /// <summary>
    /// Alpha coefficients from SMO algorithm.
    /// </summary>
    private Vector<T>? _alphas;

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
    protected override ModelType GetModelType() => ModelType.SupportVectorClassifier;

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

        _random = Options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(Options.RandomState.Value)
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
        _alphas = new Vector<T>(n);
        _intercept = new Vector<T>(1);

        T C = NumOps.FromDouble(Options.C);
        T tolerance = NumOps.FromDouble(Options.Tolerance);
        int maxPasses = Options.MaxIterations < 0 ? 10000 : Options.MaxIterations;

        int passes = 0;
        while (passes < maxPasses)
        {
            int numChangedAlphas = 0;

            for (int i = 0; i < n; i++)
            {
                T Ei = ComputeError(i);
                T yi = _yTrain[i];
                T alphaI = _alphas[i];

                // Check KKT conditions
                T yiEi = NumOps.Multiply(yi, Ei);
                bool violatesKKT = (NumOps.Compare(yiEi, NumOps.Negate(tolerance)) < 0 && NumOps.Compare(alphaI, C) < 0)
                    || (NumOps.Compare(yiEi, tolerance) > 0 && NumOps.Compare(alphaI, NumOps.Zero) > 0);

                if (violatesKKT)
                {
                    // Select j randomly
                    int j = SelectSecondAlpha(i, n);

                    T Ej = ComputeError(j);
                    T yj = _yTrain[j];
                    T alphaJ = _alphas[j];

                    // Compute bounds
                    T L, H;
                    if (NumOps.Compare(yi, yj) != 0)
                    {
                        L = Max(NumOps.Zero, NumOps.Subtract(alphaJ, alphaI));
                        H = Min(C, NumOps.Add(NumOps.Subtract(C, alphaI), alphaJ));
                    }
                    else
                    {
                        L = Max(NumOps.Zero, NumOps.Subtract(NumOps.Add(alphaI, alphaJ), C));
                        H = Min(C, NumOps.Add(alphaI, alphaJ));
                    }

                    if (NumOps.Compare(L, H) >= 0)
                        continue;

                    // Compute eta
                    T Kii = ComputeKernelCached(i, i);
                    T Kjj = ComputeKernelCached(j, j);
                    T Kij = ComputeKernelCached(i, j);
                    T eta = NumOps.Subtract(NumOps.Multiply(NumOps.FromDouble(2.0), Kij),
                        NumOps.Add(Kii, Kjj));

                    if (NumOps.Compare(eta, NumOps.Zero) >= 0)
                        continue;

                    // Update alpha_j
                    T alphaJNew = NumOps.Subtract(alphaJ,
                        NumOps.Divide(NumOps.Multiply(yj, NumOps.Subtract(Ei, Ej)), eta));
                    alphaJNew = ClipAlpha(alphaJNew, L, H);

                    if (NumOps.Compare(NumOps.Abs(NumOps.Subtract(alphaJNew, alphaJ)),
                        NumOps.FromDouble(1e-5)) < 0)
                        continue;

                    // Update alpha_i
                    T alphaINew = NumOps.Add(alphaI,
                        NumOps.Multiply(NumOps.Multiply(yi, yj), NumOps.Subtract(alphaJ, alphaJNew)));

                    // Update intercept
                    UpdateIntercept(i, j, Ei, Ej, alphaINew, alphaJNew, alphaI, alphaJ, C);

                    _alphas[i] = alphaINew;
                    _alphas[j] = alphaJNew;

                    numChangedAlphas++;
                }
            }

            if (numChangedAlphas == 0)
            {
                passes++;
            }
            else
            {
                passes = 0;
            }
        }

        // Extract support vectors
        ExtractSupportVectors();
    }

    /// <summary>
    /// Computes the prediction error for sample i.
    /// </summary>
    private T ComputeError(int i)
    {
        if (_xTrain is null || _yTrain is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }
        T prediction = ComputeDecision(GetRow(_xTrain, i));
        return NumOps.Subtract(prediction, _yTrain[i]);
    }

    /// <summary>
    /// Computes the decision value for a sample.
    /// </summary>
    private T ComputeDecision(Vector<T> x)
    {
        if (_xTrain is null || _yTrain is null || _alphas is null || _intercept is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
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
            RandomState = Options.RandomState,
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
}
