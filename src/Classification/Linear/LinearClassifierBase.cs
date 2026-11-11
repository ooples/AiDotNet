using System.Text;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
namespace AiDotNet.Classification.Linear;

#pragma warning disable CS8601, CS8618 // Generic T defaults use default(T) - always used with value types

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
public abstract partial class LinearClassifierBase<T> : ProbabilisticClassifierBase<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>, IGradientComputable<T, Matrix<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The weight matrix followed by the intercepts, packed exactly as before. PackParameters and UnpackParameters ARE the previous
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
    protected T Intercept { get; set; } = default;

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

        // Use Engine for hardware-accelerated dot product: w · x + b
        return NumOps.Add(Intercept, Engine.DotProduct(Weights, sample));
    }

    /// <summary>
    /// Computes decision function values for all samples.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A vector of decision function values.</returns>
    public Vector<T> DecisionFunctionBatch(Matrix<T> input)
    {
        if (Weights is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (input.Rows == 0)
        {
            return new Vector<T>(0);
        }

        // w · x + b for every row in one matrix-vector product
        var products = Engine.MatrixVectorMultiply(input, Weights);
        return Engine.TensorAddScalar(products.AsTensor(), Intercept).ToVector();
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (ClassLabels is null || Weights is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        int n = input.Rows;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // For binary classification: non-negative decision -> last class, negative -> first class
        var engine = Engine;
        var negative = engine.TensorLessThan(DecisionFunctionBatch(input).AsTensor(), NumOps.Zero);
        return engine.TensorWhere(negative, FilledVector(n, ClassLabels[0]).AsTensor(),
            FilledVector(n, ClassLabels[ClassLabels.Length - 1]).AsTensor()).ToVector();
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

        if (input.Rows == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        // Convert decision function to probabilities using sigmoid
        var engine = Engine;
        var prob = engine.Sigmoid(DecisionFunctionBatch(input));
        var columns = new List<Vector<T>> { engine.ScalarMinusTensor(NumOps.One, prob.AsTensor()).ToVector() };
        if (NumClasses > 1)
        {
            columns.Add(prob);
        }

        return StackColumns(columns);
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        if (probs.Rows == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        // Clamp probability to avoid log(0)
        var engine = Engine;
        return engine.TensorLog(engine.TensorClampMin(Tensor<T>.FromMatrix(probs), NumOps.FromDouble(1e-15))).ToMatrix();
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
        if (Weights is null || Weights.Length == 0) return;

        // L1 gradient is sign(w) * alpha
        var penalty = Engine.Multiply(Engine.Sign(Weights), NumOps.Multiply(learningRate, alpha));
        Weights = Engine.Subtract(Weights, penalty);
    }

    /// <summary>
    /// Applies L2 regularization gradient to the weights.
    /// </summary>
    protected void ApplyL2Gradient(T learningRate, T alpha)
    {
        if (Weights is null || Weights.Length == 0) return;

        // L2 gradient is 2 * alpha * w
        T scale = NumOps.Multiply(learningRate, NumOps.Multiply(NumOps.FromDouble(2.0), alpha));
        Weights = Engine.Subtract(Weights, Engine.Multiply(Weights, scale));
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
    private Vector<T> PackParameters()
    {
        if (Weights is null)
        {
            return new Vector<T>(0);
        }

        // Return weights + intercept
        var weights = FitToLength(Weights, NumFeatures);
        return Options.FitIntercept
            ? Vector<T>.Concatenate(weights, Vector<T>.FromArray(new[] { Intercept }))
            : weights.Clone();
    }

    /// <inheritdoc/>
    private void UnpackParameters(Vector<T> parameters)
    {
        // When the model is untrained (NumFeatures == 0), infer NumFeatures from the
        // parameter vector. This is required for the optimizer's InitializeRandomSolution
        // which creates random parameters before the model has seen any training data.
        if (NumFeatures == 0 && parameters.Length > 0)
        {
            NumFeatures = Options.FitIntercept ? parameters.Length - 1 : parameters.Length;
        }

        int expectedLength = Options.FitIntercept ? NumFeatures + 1 : NumFeatures;
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, got {parameters.Length}");
        }

        Weights = NumFeatures == 0 ? new Vector<T>(0) : parameters.GetSubVector(0, NumFeatures).Clone();

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
    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Compute gradients for linear classifier
        if (Weights is null || input.Rows == 0)
        {
            return new Vector<T>(NumFeatures);
        }

        // Use decision function scores (continuous) instead of discrete predictions
        // for proper gradient computation
        var scores = DecisionFunctionBatch(input);

        // Compute error vector once: error = scores - target
        var errors = Engine.Subtract(scores, target);

        // Gradient: grad[j] = mean(error · x[:,j]) = (X^T error) / n
        T invN = NumOps.FromDouble(1.0 / input.Rows);
        var gradients = Engine.Multiply(Engine.MatrixVectorMultiply(Engine.MatrixTranspose(input), errors), invN);
        return FitToLength(gradients, NumFeatures);
    }

    /// <inheritdoc/>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (Weights is null || Weights.Length == 0) return;

        // Only the first min(NumFeatures, gradients.Length) weights are updated
        var update = Engine.Multiply(FitToLength(FitToLength(gradients, Math.Min(NumFeatures, gradients.Length)), Weights.Length), learningRate);
        Weights = Engine.Subtract(Weights, update);
    }

    /// <summary>
    /// Maps labels to +1 (the last class label) and -1 (every other label).
    /// </summary>
    /// <param name="y">The label vector.</param>
    /// <returns>The +1/-1 targets.</returns>
    private protected Vector<T> SignedBinaryTargets(Vector<T> y)
    {
        if (ClassLabels is null || y.Length == 0)
        {
            return new Vector<T>(y.Length);
        }

        var engine = Engine;
        var isPositive = engine.TensorEquals(y.AsTensor(), ClassLabels[ClassLabels.Length - 1]);
        return engine.TensorSubtractScalar(engine.TensorMultiplyScalar(isPositive, NumOps.FromDouble(2)), NumOps.One).ToVector();
    }

    /// <summary>
    /// The first <see cref="ClassifierBase{T}.NumFeatures"/> values of row <paramref name="row"/>.
    /// </summary>
    private protected Vector<T> GetSample(Matrix<T> x, int row)
    {
        return x.Columns == NumFeatures ? x.GetRow(row) : x.GetRowSegment(row, 0, NumFeatures);
    }

    /// <summary>
    /// w = w + scale * sample.
    /// </summary>
    private protected void AddScaledSample(Vector<T> sample, T scale)
    {
        if (Weights is null) return;
        Weights = Engine.Add(Weights, Engine.Multiply(sample, scale));
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
