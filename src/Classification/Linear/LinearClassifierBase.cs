using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Linear;

/// <summary>
/// Provides a base implementation for linear classifiers.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Linear classifiers learn a linear decision function: f(x) = w * x + b
/// where w is the weight vector and b is the bias (intercept).
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Linear classifiers are one of the simplest forms of machine learning:
///
/// How they work:
/// 1. Each feature gets a weight (importance score)
/// 2. Multiply each feature by its weight and sum them up
/// 3. Add a bias term
/// 4. If the result is positive, predict one class; otherwise, the other
///
/// The training process adjusts the weights to correctly classify
/// training examples.
///
/// Advantages:
/// - Fast to train and predict
/// - Work well with many features
/// - Easy to interpret (weight = feature importance)
/// - Often surprisingly effective
/// </para>
/// </remarks>
public abstract class LinearClassifierBase<T> : ProbabilisticClassifierBase<T>
{
    /// <summary>
    /// Gets the linear classifier specific options.
    /// </summary>
    protected new LinearClassifierOptions<T> Options => (LinearClassifierOptions<T>)base.Options;

    /// <summary>
    /// The learned weight vector.
    /// </summary>
    protected Vector<T>? Weights { get; set; }

    /// <summary>
    /// The learned intercept (bias) term.
    /// </summary>
    protected T Intercept { get; set; } = default!;

    /// <summary>
    /// Random number generator for shuffling.
    /// </summary>
    protected Random? Random { get; set; }

    /// <summary>
    /// Initializes a new instance of the LinearClassifierBase class.
    /// </summary>
    /// <param name="options">Configuration options for the linear classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    protected LinearClassifierBase(LinearClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new LinearClassifierOptions<T>(), regularization, new HingeLoss<T>())
    {
        Intercept = NumOps.Zero;
    }

    /// <summary>
    /// Initializes the weights before training.
    /// </summary>
    protected virtual void InitializeWeights()
    {
        Weights = new Vector<T>(NumFeatures);
        for (int i = 0; i < NumFeatures; i++)
        {
            Weights[i] = NumOps.Zero;
        }
        Intercept = NumOps.Zero;

        Random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Computes the decision function value for a single sample.
    /// </summary>
    /// <param name="sample">The feature vector for a single sample.</param>
    /// <returns>The decision function value (w * x + b).</returns>
    protected T DecisionFunction(Vector<T> sample)
    {
        if (Weights is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        T result = Intercept;
        for (int j = 0; j < NumFeatures; j++)
        {
            result = NumOps.Add(result, NumOps.Multiply(Weights[j], sample[j]));
        }
        return result;
    }

    /// <summary>
    /// Computes decision function values for all samples.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A vector of decision function values.</returns>
    public Vector<T> DecisionFunctionBatch(Matrix<T> input)
    {
        var decisions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            var sample = new Vector<T>(input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                sample[j] = input[i, j];
            }
            decisions[i] = DecisionFunction(sample);
        }
        return decisions;
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (ClassLabels is null || Weights is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            var sample = new Vector<T>(input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                sample[j] = input[i, j];
            }

            T decision = DecisionFunction(sample);
            // For binary classification: positive -> class 1, negative -> class 0
            predictions[i] = NumOps.Compare(decision, NumOps.Zero) >= 0
                ? ClassLabels[ClassLabels.Length - 1]
                : ClassLabels[0];
        }
        return predictions;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        // Linear classifiers use sigmoid for binary classification
        // For multi-class (NumClasses > 2), use OneVsRestClassifier wrapper
        if (NumClasses > 2)
        {
            throw new NotSupportedException(
                $"PredictProbabilities only supports binary classification (NumClasses=2). " +
                $"Current NumClasses={NumClasses}. Use OneVsRestClassifier for multi-class problems.");
        }

        // Convert decision function to probabilities using sigmoid
        var probabilities = new Matrix<T>(input.Rows, NumClasses);
        var decisions = DecisionFunctionBatch(input);

        for (int i = 0; i < input.Rows; i++)
        {
            T prob = Sigmoid(decisions[i]);
            probabilities[i, 0] = NumOps.Subtract(NumOps.One, prob);
            if (NumClasses > 1)
            {
                probabilities[i, 1] = prob;
            }
        }
        return probabilities;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var logProbs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                // Clamp probability to avoid log(0)
                T p = probs[i, c];
                T minProb = NumOps.FromDouble(1e-15);
                if (NumOps.Compare(p, minProb) < 0)
                {
                    p = minProb;
                }
                logProbs[i, c] = NumOps.Log(p);
            }
        }
        return logProbs;
    }

    /// <summary>
    /// Computes the sigmoid function.
    /// </summary>
    protected T Sigmoid(T x)
    {
        // sigmoid(x) = 1 / (1 + exp(-x))
        T negX = NumOps.Negate(x);
        T expNegX = NumOps.Exp(negX);
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }

    /// <summary>
    /// Applies L1 regularization gradient to the weights.
    /// </summary>
    protected void ApplyL1Gradient(T learningRate, T alpha)
    {
        if (Weights is null) return;

        for (int j = 0; j < NumFeatures; j++)
        {
            // L1 gradient is sign(w) * alpha
            T sign = NumOps.Compare(Weights[j], NumOps.Zero) > 0
                ? NumOps.One
                : (NumOps.Compare(Weights[j], NumOps.Zero) < 0 ? NumOps.Negate(NumOps.One) : NumOps.Zero);
            T penalty = NumOps.Multiply(learningRate, NumOps.Multiply(alpha, sign));
            Weights[j] = NumOps.Subtract(Weights[j], penalty);
        }
    }

    /// <summary>
    /// Applies L2 regularization gradient to the weights.
    /// </summary>
    protected void ApplyL2Gradient(T learningRate, T alpha)
    {
        if (Weights is null) return;

        for (int j = 0; j < NumFeatures; j++)
        {
            // L2 gradient is 2 * alpha * w
            T penalty = NumOps.Multiply(learningRate, NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Multiply(alpha, Weights[j])));
            Weights[j] = NumOps.Subtract(Weights[j], penalty);
        }
    }

    /// <summary>
    /// Shuffles the training data indices.
    /// </summary>
    protected int[] ShuffleIndices(int n)
    {
        var indices = new int[n];
        for (int i = 0; i < n; i++)
        {
            indices[i] = i;
        }

        if (Options.Shuffle && Random is not null)
        {
            for (int i = n - 1; i > 0; i--)
            {
                int j = Random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        return indices;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        if (Weights is null)
        {
            return new Vector<T>(0);
        }

        // Return weights + intercept
        int length = Options.FitIntercept ? NumFeatures + 1 : NumFeatures;
        var parameters = new Vector<T>(length);
        for (int i = 0; i < NumFeatures; i++)
        {
            parameters[i] = Weights[i];
        }
        if (Options.FitIntercept)
        {
            parameters[NumFeatures] = Intercept;
        }
        return parameters;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedLength = Options.FitIntercept ? NumFeatures + 1 : NumFeatures;
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, got {parameters.Length}");
        }

        Weights = new Vector<T>(NumFeatures);
        for (int i = 0; i < NumFeatures; i++)
        {
            Weights[i] = parameters[i];
        }
        if (Options.FitIntercept)
        {
            Intercept = parameters[NumFeatures];
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = (LinearClassifierBase<T>)Clone();
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Compute gradients for linear classifier
        if (Weights is null)
        {
            return new Vector<T>(NumFeatures);
        }

        var gradients = new Vector<T>(NumFeatures);
        // Use decision function scores (continuous) instead of discrete predictions
        // for proper gradient computation
        var scores = DecisionFunctionBatch(input);

        for (int j = 0; j < NumFeatures; j++)
        {
            T grad = NumOps.Zero;
            for (int i = 0; i < input.Rows; i++)
            {
                T error = NumOps.Subtract(scores[i], target[i]);
                grad = NumOps.Add(grad, NumOps.Multiply(error, input[i, j]));
            }
            gradients[j] = NumOps.Divide(grad, NumOps.FromDouble(input.Rows));
        }

        return gradients;
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (Weights is null) return;

        for (int j = 0; j < NumFeatures && j < gradients.Length; j++)
        {
            T update = NumOps.Multiply(learningRate, gradients[j]);
            Weights[j] = NumOps.Subtract(Weights[j], update);
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["LearningRate"] = Options.LearningRate;
        metadata.AdditionalInfo["MaxIterations"] = Options.MaxIterations;
        metadata.AdditionalInfo["Penalty"] = Options.Penalty.ToString();
        metadata.AdditionalInfo["Loss"] = Options.Loss.ToString();
        metadata.AdditionalInfo["Alpha"] = Options.Alpha;
        return metadata;
    }
}
