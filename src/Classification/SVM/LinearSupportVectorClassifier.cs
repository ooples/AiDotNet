using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.SVM;

/// <summary>
/// Linear Support Vector Classifier optimized for linear classification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This implementation uses a primal formulation with stochastic gradient descent (SGD)
/// for efficient training on large datasets. Unlike the standard SVC which uses the kernel
/// trick, this classifier works directly in the original feature space.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Linear SVC is a simplified version of SVM that only draws straight lines to separate classes.
/// It's much faster to train than the regular SVC because it doesn't need to compute kernel
/// values between all pairs of training points.
///
/// Use Linear SVC when:
/// - You have a large dataset (thousands of samples)
/// - Your data is linearly separable or nearly so
/// - You need fast training and prediction
/// - You have high-dimensional data (many features)
///
/// Example use cases:
/// - Text classification (spam detection, sentiment)
/// - Document categorization
/// - High-dimensional bioinformatics data
/// </para>
/// </remarks>
public class LinearSupportVectorClassifier<T> : SVMBase<T>
{
    /// <summary>
    /// Weight vector for linear classification.
    /// </summary>
    private Vector<T>? _weights;

    /// <summary>
    /// Bias term (intercept) for the linear classifier.
    /// </summary>
    private T _bias = default!;

    /// <summary>
    /// Random number generator for SGD.
    /// </summary>
    private Random? _random;

    /// <summary>
    /// Initializes a new instance of the LinearSupportVectorClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the Linear SVC.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public LinearSupportVectorClassifier(SVMOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        // Override kernel to Linear for this classifier
        if (Options.Kernel != KernelType.Linear)
        {
            Options.Kernel = KernelType.Linear;
        }
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.LinearSupportVectorClassifier;

    /// <summary>
    /// Trains the Linear SVC on the provided data using SGD.
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

        // Convert labels to +1/-1 for binary classification
        var yBinary = new Vector<T>(y.Length);
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        for (int i = 0; i < y.Length; i++)
        {
            if (NumOps.Compare(y[i], positiveClass) == 0)
            {
                yBinary[i] = NumOps.One;
            }
            else
            {
                yBinary[i] = NumOps.Negate(NumOps.One);
            }
        }

        // Train using SGD with hinge loss
        TrainSGD(x, yBinary);

        // Set up support vectors (all training points for linear SVC)
        // In practice, we only store weights, but we can compute effective support vectors
        _intercept = new Vector<T>(1);
        _intercept[0] = _bias;
    }

    /// <summary>
    /// Trains using Stochastic Gradient Descent with hinge loss.
    /// </summary>
    private void TrainSGD(Matrix<T> x, Vector<T> y)
    {
        if (_random is null)
        {
            throw new InvalidOperationException("Random number generator not initialized.");
        }

        int n = x.Rows;
        int d = x.Columns;

        // Initialize weights to zeros
        _weights = new Vector<T>(d);
        _bias = NumOps.Zero;

        T C = NumOps.FromDouble(Options.C);
        T learningRate = NumOps.FromDouble(0.01);
        int maxIter = Options.MaxIterations < 0 ? 1000 : Options.MaxIterations;
        T tolerance = NumOps.FromDouble(Options.Tolerance);

        // Create indices for shuffling
        var indices = new int[n];
        for (int i = 0; i < n; i++)
        {
            indices[i] = i;
        }

        T prevLoss = NumOps.MaxValue;

        for (int epoch = 0; epoch < maxIter; epoch++)
        {
            // Shuffle indices
            ShuffleArray(indices);

            T epochLoss = NumOps.Zero;

            for (int idx = 0; idx < n; idx++)
            {
                int i = indices[idx];
                Vector<T> xi = GetRow(x, i);
                T yi = y[i];

                // Compute margin: y * (w · x + b)
                T margin = NumOps.Multiply(yi, ComputeLinearOutput(xi));

                // Compute loss and gradients
                if (NumOps.Compare(margin, NumOps.One) < 0)
                {
                    // Hinge loss is active: loss = 1 - margin
                    T hingeLoss = NumOps.Subtract(NumOps.One, margin);
                    epochLoss = NumOps.Add(epochLoss, hingeLoss);

                    // Update weights: w = w + lr * C * y * x
                    for (int j = 0; j < d; j++)
                    {
                        T grad = NumOps.Multiply(NumOps.Multiply(yi, xi[j]), C);
                        _weights[j] = NumOps.Add(_weights[j], NumOps.Multiply(learningRate, grad));
                    }

                    // Update bias: b = b + lr * C * y
                    _bias = NumOps.Add(_bias, NumOps.Multiply(learningRate, NumOps.Multiply(C, yi)));
                }

                // L2 regularization on weights (not bias)
                T lambda = NumOps.Divide(NumOps.One, NumOps.Multiply(C, NumOps.FromDouble(n)));
                for (int j = 0; j < d; j++)
                {
                    _weights[j] = NumOps.Multiply(_weights[j],
                        NumOps.Subtract(NumOps.One, NumOps.Multiply(learningRate, lambda)));
                }
            }

            // Check convergence
            T lossChange = NumOps.Abs(NumOps.Subtract(epochLoss, prevLoss));
            if (NumOps.Compare(lossChange, tolerance) < 0 && epoch > 10)
            {
                break;
            }

            prevLoss = epochLoss;

            // Decrease learning rate
            learningRate = NumOps.Divide(learningRate, NumOps.FromDouble(1.0 + 0.001 * epoch));
        }
    }

    /// <summary>
    /// Computes the linear output w · x + b.
    /// </summary>
    private T ComputeLinearOutput(Vector<T> x)
    {
        if (_weights is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        T output = _bias;
        for (int i = 0; i < _weights.Length; i++)
        {
            output = NumOps.Add(output, NumOps.Multiply(_weights[i], x[i]));
        }
        return output;
    }

    /// <summary>
    /// Shuffles an array in place.
    /// </summary>
    private void ShuffleArray(int[] array)
    {
        if (_random is null)
        {
            return;
        }

        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    /// <inheritdoc/>
    public override Matrix<T> DecisionFunction(Matrix<T> input)
    {
        if (_weights is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var decisions = new Matrix<T>(input.Rows, 1);
        for (int i = 0; i < input.Rows; i++)
        {
            decisions[i, 0] = ComputeLinearOutput(GetRow(input, i));
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
            // Positive decision -> positive class, negative -> negative class
            if (NumOps.Compare(decisions[i, 0], NumOps.Zero) >= 0)
            {
                predictions[i] = ClassLabels[ClassLabels.Length - 1]; // Positive class
            }
            else
            {
                predictions[i] = ClassLabels[0]; // Negative class
            }
        }

        return predictions;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        // Use Platt scaling approximation for probability estimates
        var decisions = DecisionFunction(input);
        var probabilities = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            // Sigmoid transformation: p = 1 / (1 + exp(-decision))
            T decision = decisions[i, 0];
            T negDecision = NumOps.Negate(decision);
            T expNeg = NumOps.Exp(negDecision);
            T prob = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNeg));

            // Probability for negative class
            probabilities[i, 0] = NumOps.Subtract(NumOps.One, prob);
            // Probability for positive class
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
        return new LinearSupportVectorClassifier<T>(new SVMOptions<T>
        {
            C = Options.C,
            Kernel = KernelType.Linear,
            Tolerance = Options.Tolerance,
            MaxIterations = Options.MaxIterations,
            RandomState = Options.RandomState
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new LinearSupportVectorClassifier<T>(new SVMOptions<T>
        {
            C = Options.C,
            Kernel = KernelType.Linear,
            Tolerance = Options.Tolerance,
            MaxIterations = Options.MaxIterations,
            RandomState = Options.RandomState
        });

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;

        if (ClassLabels is not null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (_weights is not null)
        {
            clone._weights = new Vector<T>(_weights.Length);
            for (int i = 0; i < _weights.Length; i++)
            {
                clone._weights[i] = _weights[i];
            }
        }

        clone._bias = _bias;

        if (_intercept is not null)
        {
            clone._intercept = new Vector<T>(_intercept.Length);
            for (int i = 0; i < _intercept.Length; i++)
            {
                clone._intercept[i] = _intercept[i];
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["Algorithm"] = "SGD";
        metadata.AdditionalInfo["WeightCount"] = _weights?.Length ?? 0;
        return metadata;
    }
}
