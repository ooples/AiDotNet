using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

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
public class NuSupportVectorClassifier<T> : SVMBase<T>
{
    /// <summary>
    /// Stored training features.
    /// </summary>
    private Matrix<T>? _xTrain;

    /// <summary>
    /// Stored training labels (converted to +1/-1).
    /// </summary>
    private Vector<T>? _yTrain;

    /// <summary>
    /// Alpha coefficients.
    /// </summary>
    private Vector<T>? _alphas;

    /// <summary>
    /// Rho parameter (offset in the decision function).
    /// </summary>
    private T _rho = default!;

    /// <summary>
    /// The nu parameter value.
    /// </summary>
    private readonly double _nu;

    /// <summary>
    /// Random number generator.
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
        if (nu <= 0 || nu > 1)
        {
            throw new ArgumentException("Nu must be in (0, 1].", nameof(nu));
        }
        _nu = nu;
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.NuSupportVectorClassifier;

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

        // Convert labels to +1/-1
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

        // Train using Nu-SMO variant
        TrainNuSMO();
    }

    /// <summary>
    /// Nu-SMO training algorithm.
    /// </summary>
    private void TrainNuSMO()
    {
        if (_xTrain is null || _yTrain is null || _random is null)
        {
            throw new InvalidOperationException("Training data has not been initialized.");
        }

        int n = _xTrain.Rows;
        _alphas = new Vector<T>(n);
        _intercept = new Vector<T>(1);

        // Initialize alphas to satisfy constraints
        // sum_i alpha_i * y_i = 0
        // sum_i alpha_i = nu * n
        T nuN = NumOps.FromDouble(_nu * n);

        // Count positive and negative samples
        int posCount = 0;
        int negCount = 0;
        for (int i = 0; i < n; i++)
        {
            if (NumOps.Compare(_yTrain[i], NumOps.One) == 0)
            {
                posCount++;
            }
            else
            {
                negCount++;
            }
        }

        // Initialize alphas to satisfy sum(alpha) = nu*n and sum(alpha*y) = 0
        // For balanced initialization: alpha_pos = nu*n/(2*n_pos), alpha_neg = nu*n/(2*n_neg)
        if (posCount > 0 && negCount > 0)
        {
            T alphaPos = NumOps.Divide(NumOps.Divide(nuN, NumOps.FromDouble(2.0)), NumOps.FromDouble(posCount));
            T alphaNeg = NumOps.Divide(NumOps.Divide(nuN, NumOps.FromDouble(2.0)), NumOps.FromDouble(negCount));

            // Ensure alphas don't exceed 1/n (the upper bound for nu-SVC)
            T upperBound = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));
            alphaPos = Min(alphaPos, upperBound);
            alphaNeg = Min(alphaNeg, upperBound);

            for (int i = 0; i < n; i++)
            {
                if (NumOps.Compare(_yTrain[i], NumOps.One) == 0)
                {
                    _alphas[i] = alphaPos;
                }
                else
                {
                    _alphas[i] = alphaNeg;
                }
            }
        }

        // Simplified optimization - gradient descent on alphas
        T tolerance = NumOps.FromDouble(Options.Tolerance);
        int maxIter = Options.MaxIterations < 0 ? 1000 : Options.MaxIterations;
        T upperBoundAlpha = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));

        for (int iter = 0; iter < maxIter; iter++)
        {
            bool changed = false;

            for (int i = 0; i < n; i++)
            {
                T yi = _yTrain[i];
                T Ei = ComputeError(i);

                // Check if alpha can be updated
                T alphaI = _alphas[i];
                bool atLower = NumOps.Compare(alphaI, NumOps.Zero) <= 0;
                bool atUpper = NumOps.Compare(alphaI, upperBoundAlpha) >= 0;

                // Simplified KKT check
                T yiEi = NumOps.Multiply(yi, Ei);
                if ((NumOps.Compare(yiEi, NumOps.Negate(tolerance)) < 0 && !atUpper) ||
                    (NumOps.Compare(yiEi, tolerance) > 0 && !atLower))
                {
                    // Select j randomly
                    int j = _random.Next(n - 1);
                    if (j >= i) j++;

                    T yj = _yTrain[j];
                    T Ej = ComputeError(j);

                    // Simple gradient step
                    T Kii = ComputeKernel(GetRow(_xTrain, i), GetRow(_xTrain, i));
                    T Kjj = ComputeKernel(GetRow(_xTrain, j), GetRow(_xTrain, j));
                    T Kij = ComputeKernel(GetRow(_xTrain, i), GetRow(_xTrain, j));

                    T eta = NumOps.Subtract(NumOps.Add(Kii, Kjj), NumOps.Multiply(NumOps.FromDouble(2.0), Kij));
                    if (NumOps.Compare(eta, NumOps.FromDouble(1e-12)) < 0)
                    {
                        continue;
                    }

                    // Update alphas in pairs to maintain Nu-SVC constraint: sum(alpha_i * y_i) = 0
                    T alphaJ = _alphas[j];
                    T deltaAlpha = NumOps.Divide(
                        NumOps.Multiply(yi, NumOps.Subtract(Ej, Ei)),
                        eta);

                    // Compute bounds for alpha_i considering constraint maintenance
                    // If y_i == y_j: delta_alpha_j = -delta_alpha_i (to maintain sum constraint)
                    // If y_i != y_j: delta_alpha_j = delta_alpha_i
                    T deltaAlphaJ;
                    if (NumOps.Compare(NumOps.Multiply(yi, yj), NumOps.One) == 0)
                    {
                        // Same class: alpha_j decreases when alpha_i increases
                        deltaAlphaJ = NumOps.Negate(deltaAlpha);
                    }
                    else
                    {
                        // Different class: alpha_j changes same direction
                        deltaAlphaJ = deltaAlpha;
                    }

                    // Clip alpha_i
                    T newAlphaI = NumOps.Add(alphaI, deltaAlpha);
                    newAlphaI = Max(NumOps.Zero, Min(upperBoundAlpha, newAlphaI));
                    T actualDeltaI = NumOps.Subtract(newAlphaI, alphaI);

                    // Compute corresponding alpha_j
                    T newAlphaJ;
                    if (NumOps.Compare(NumOps.Multiply(yi, yj), NumOps.One) == 0)
                    {
                        newAlphaJ = NumOps.Subtract(alphaJ, actualDeltaI);
                    }
                    else
                    {
                        newAlphaJ = NumOps.Add(alphaJ, actualDeltaI);
                    }
                    newAlphaJ = Max(NumOps.Zero, Min(upperBoundAlpha, newAlphaJ));

                    if (NumOps.Compare(NumOps.Abs(actualDeltaI), tolerance) > 0)
                    {
                        _alphas[i] = newAlphaI;
                        _alphas[j] = newAlphaJ;
                        changed = true;
                    }
                }
            }

            if (!changed)
            {
                break;
            }
        }

        // Compute rho and intercept
        ComputeRhoAndIntercept();

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
            throw new InvalidOperationException("Training data has not been initialized.");
        }

        T decision = ComputeDecisionForSample(GetRow(_xTrain, i));
        return NumOps.Subtract(decision, _yTrain[i]);
    }

    /// <summary>
    /// Computes the decision value for a single sample.
    /// </summary>
    private T ComputeDecisionForSample(Vector<T> x)
    {
        if (_xTrain is null || _yTrain is null || _alphas is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        T sum = NumOps.Zero;
        for (int i = 0; i < _xTrain.Rows; i++)
        {
            if (NumOps.Compare(_alphas[i], NumOps.FromDouble(1e-10)) > 0)
            {
                T kernel = ComputeKernel(GetRow(_xTrain, i), x);
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(_alphas[i], _yTrain[i]), kernel));
            }
        }
        return NumOps.Subtract(sum, _rho);
    }

    /// <summary>
    /// Computes rho and intercept.
    /// </summary>
    private void ComputeRhoAndIntercept()
    {
        if (_xTrain is null || _yTrain is null || _alphas is null || _intercept is null)
        {
            throw new InvalidOperationException("Training data has not been initialized.");
        }

        // Compute rho as average of decision values on support vectors
        T sumRho = NumOps.Zero;
        int countSV = 0;
        T upperBound = NumOps.Divide(NumOps.One, NumOps.FromDouble(_xTrain.Rows));

        for (int i = 0; i < _xTrain.Rows; i++)
        {
            // Free support vectors (not at bounds)
            if (NumOps.Compare(_alphas[i], NumOps.FromDouble(1e-10)) > 0 &&
                NumOps.Compare(_alphas[i], NumOps.Subtract(upperBound, NumOps.FromDouble(1e-10))) < 0)
            {
                T decision = NumOps.Zero;
                for (int j = 0; j < _xTrain.Rows; j++)
                {
                    if (NumOps.Compare(_alphas[j], NumOps.FromDouble(1e-10)) > 0)
                    {
                        T kernel = ComputeKernel(GetRow(_xTrain, i), GetRow(_xTrain, j));
                        decision = NumOps.Add(decision, NumOps.Multiply(NumOps.Multiply(_alphas[j], _yTrain[j]), kernel));
                    }
                }
                sumRho = NumOps.Add(sumRho, NumOps.Subtract(decision, _yTrain[i]));
                countSV++;
            }
        }

        if (countSV > 0)
        {
            _rho = NumOps.Divide(sumRho, NumOps.FromDouble(countSV));
        }
        else
        {
            _rho = NumOps.Zero;
        }

        _intercept[0] = NumOps.Negate(_rho);
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

        var svIndices = new List<int>();
        for (int i = 0; i < _alphas.Length; i++)
        {
            if (NumOps.Compare(_alphas[i], NumOps.FromDouble(1e-10)) > 0)
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
        var decisions = new Matrix<T>(input.Rows, 1);
        for (int i = 0; i < input.Rows; i++)
        {
            decisions[i, 0] = ComputeDecisionForSample(GetRow(input, i));
        }
        return decisions;
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var decisions = DecisionFunction(input);
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            if (NumOps.Compare(decisions[i, 0], NumOps.Zero) >= 0)
            {
                predictions[i] = ClassLabels[ClassLabels.Length - 1];
            }
            else
            {
                predictions[i] = ClassLabels[0];
            }
        }

        return predictions;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        var decisions = DecisionFunction(input);
        var probabilities = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            // Sigmoid transformation
            T decision = decisions[i, 0];
            T negDecision = NumOps.Negate(decision);
            T expNeg = NumOps.Exp(negDecision);
            T prob = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));

            probabilities[i, 0] = NumOps.Subtract(NumOps.One, prob);
            if (NumClasses > 1)
            {
                probabilities[i, 1] = prob;
            }
        }

        return probabilities;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new NuSupportVectorClassifier<T>(new SVMOptions<T>
        {
            Kernel = Options.Kernel,
            Gamma = Options.Gamma,
            Degree = Options.Degree,
            Coef0 = Options.Coef0,
            Tolerance = Options.Tolerance,
            MaxIterations = Options.MaxIterations,
            RandomState = Options.RandomState
        }, null, _nu);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new NuSupportVectorClassifier<T>(new SVMOptions<T>
        {
            Kernel = Options.Kernel,
            Gamma = Options.Gamma,
            Degree = Options.Degree,
            Coef0 = Options.Coef0,
            Tolerance = Options.Tolerance,
            MaxIterations = Options.MaxIterations,
            RandomState = Options.RandomState
        }, null, _nu);

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;
        clone._rho = _rho;

        if (ClassLabels is not null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (_xTrain is not null)
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

        if (_yTrain is not null)
        {
            clone._yTrain = new Vector<T>(_yTrain.Length);
            for (int i = 0; i < _yTrain.Length; i++)
            {
                clone._yTrain[i] = _yTrain[i];
            }
        }

        if (_alphas is not null)
        {
            clone._alphas = new Vector<T>(_alphas.Length);
            for (int i = 0; i < _alphas.Length; i++)
            {
                clone._alphas[i] = _alphas[i];
            }
        }

        if (_intercept is not null)
        {
            clone._intercept = new Vector<T>(_intercept.Length);
            for (int i = 0; i < _intercept.Length; i++)
            {
                clone._intercept[i] = _intercept[i];
            }
        }

        if (_supportVectors is not null)
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

        if (_dualCoef is not null)
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
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Nu"] = _nu;
        metadata.AdditionalInfo["Rho"] = NumOps.ToDouble(_rho);
        return metadata;
    }
}
