using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
/// <example>
/// <code>
/// // Create linear SVC for fast large-scale classification
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
/// // Train using primal SGD in original feature space
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new LinearSupportVectorClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict class using learned linear hyperplane
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 1.1; newSample[0, 1] = 1.0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.SVM)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("A Training Algorithm for Optimal Margin Classifiers", "https://doi.org/10.1145/130385.130401")]
public partial class LinearSupportVectorClassifier<T> : SVMBase<T>
{
    /// <summary>
    /// Weight vector for linear classification.
    /// </summary>
    [FittedParameter]
    private Vector<T>? _weights;

    /// <summary>
    /// Bias term (intercept) for the linear classifier.
    /// </summary>

    [FittedParameter]
    private T _bias;


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
        _bias = NumOps.Zero;
        // Override kernel to Linear for this classifier
        if (Options.Kernel != KernelType.Linear)
        {
            Options.Kernel = KernelType.Linear;
        }
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

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

        _random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Convert labels to +1/-1 for binary classification
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        var yBinary = y.Length == 0
            ? new Vector<T>(0)
            : Engine.TensorSubtractScalar(
                Engine.TensorMultiplyScalar(Engine.TensorEquals(y.AsTensor(), positiveClass), NumOps.FromDouble(2.0)),
                NumOps.One).ToVector();

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
        var targets = y.ToArray();

        // L2 regularization strength on weights (not bias)
        T lambda = NumOps.Divide(NumOps.One, NumOps.Multiply(C, NumOps.FromDouble(n)));

        for (int epoch = 0; epoch < maxIter; epoch++)
        {
            // Shuffle indices
            ShuffleArray(indices);

            var sampleLosses = new T[n];

            for (int idx = 0; idx < n; idx++)
            {
                int i = indices[idx];
                sampleLosses[idx] = HingeStep(GetRow(x, i), targets[i], C, learningRate, lambda);
            }

            T epochLoss = n == 0 ? NumOps.Zero : Engine.Sum(Vector<T>.Wrap(sampleLosses));

            // Check convergence
            if (HasConverged(epochLoss, prevLoss, tolerance) && epoch > 10)
            {
                break;
            }

            prevLoss = epochLoss;

            // Decrease learning rate
            learningRate = DecayLearningRate(learningRate, epoch);
        }
    }

    /// <summary>
    /// Whether the epoch loss changed by less than the tolerance.
    /// </summary>
    private bool HasConverged(T epochLoss, T prevLoss, T tolerance)
    {
        return NumOps.Compare(NumOps.Abs(NumOps.Subtract(epochLoss, prevLoss)), tolerance) < 0;
    }

    /// <summary>
    /// learningRate / (1 + 0.001 * epoch).
    /// </summary>
    private T DecayLearningRate(T learningRate, int epoch)
    {
        return NumOps.Divide(learningRate, NumOps.FromDouble(1.0 + 0.001 * epoch));
    }

    /// <summary>
    /// Computes the linear output w · x + b.
    /// </summary>
    /// <summary>
    /// One SGD step on a sample; returns its hinge loss (0 when the margin is at least 1).
    /// </summary>
    private T HingeStep(Vector<T> xi, T yi, T C, T learningRate, T lambda)
    {
        if (_weights is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        // Compute margin: y * (w · x + b)
        T margin = NumOps.Multiply(yi, ComputeLinearOutput(xi));
        T loss = NumOps.Zero;

        if (NumOps.Compare(margin, NumOps.One) < 0)
        {
            // Hinge loss is active: loss = 1 - margin
            loss = NumOps.Subtract(NumOps.One, margin);

            // Update weights: w = w + lr * C * y * x; bias: b = b + lr * C * y
            T step = NumOps.Multiply(learningRate, NumOps.Multiply(C, yi));
            _weights = Engine.Add(_weights, Engine.Multiply(xi, step));
            _bias = NumOps.Add(_bias, step);
        }

        // L2 regularization: w = w * (1 - lr * lambda)
        _weights = Engine.Multiply(_weights, NumOps.Subtract(NumOps.One, NumOps.Multiply(learningRate, lambda)));
        return loss;
    }

    private T ComputeLinearOutput(Vector<T> x)
    {
        if (_weights is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        T output = _bias;
        output = NumOps.Add(output, Engine.DotProduct(_weights, x));
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

        // w · x + b for every row
        if (input.Rows == 0)
        {
            return new Matrix<T>(0, 1);
        }

        var scores = Engine.TensorAddScalar(
            Engine.TensorMatMul(Tensor<T>.FromMatrix(input), _weights.AsTensor().Reshape(_weights.Length, 1)), _bias);
        return scores.ToMatrix();
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

        // Positive decision -> positive class, negative -> negative class
        var engine = Engine;
        var isPositive = AtLeast(decisions.GetColumn(0), NumOps.Zero);
        var positive = FilledVector(input.Rows, ClassLabels[ClassLabels.Length - 1]).AsTensor();
        var negative = FilledVector(input.Rows, ClassLabels[0]).AsTensor();
        return engine.TensorWhere(isPositive.AsTensor(), positive, negative).ToVector();
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        // Use Platt scaling approximation for probability estimates
        var decisions = DecisionFunction(input);
        if (input.Rows == 0 || NumClasses == 0)
        {
            return new Matrix<T>(input.Rows, NumClasses);
        }

        // Sigmoid transformation: p = 1 / (1 + exp(-decision)); columns [1 - p, p, 0, ...]
        var engine = Engine;
        var decision = Tensor<T>.FromMatrix(decisions);
        var ones = new Tensor<T>(new[] { input.Rows, 1 });
        engine.TensorFill(ones, NumOps.One);
        var prob = engine.TensorDivide(ones, engine.TensorAddScalar(engine.TensorExp(engine.TensorNegate(decision)), NumOps.One));
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
        metadata.AdditionalInfo["Algorithm"] = "SGD";
        metadata.AdditionalInfo["WeightCount"] = _weights?.Length ?? 0;
        return metadata;
    }
}
