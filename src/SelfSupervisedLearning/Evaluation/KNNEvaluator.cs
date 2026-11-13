using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.Engines;
using AiDotNet.Validation;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// k-Nearest Neighbors (k-NN) evaluation for SSL representation quality.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> k-NN evaluation is a simple way to test SSL representations
/// without any training. We classify each test sample based on the labels of its k nearest
/// neighbors in the representation space.</para>
///
/// <para><b>Why k-NN evaluation?</b></para>
/// <list type="bullet">
/// <item>No training required - instant evaluation</item>
/// <item>Tests if similar images are close in representation space</item>
/// <item>Good sanity check during pretraining (run every few epochs)</item>
/// <item>Correlates well with linear evaluation accuracy</item>
/// </list>
///
/// <para><b>Typical protocol:</b></para>
/// <list type="number">
/// <item>Extract features from all training and test samples</item>
/// <item>For each test sample, find k nearest training samples</item>
/// <item>Predict class by majority voting among neighbors</item>
/// <item>Report top-1 accuracy</item>
/// </list>
///
/// <para><b>Common settings:</b> k=20, cosine similarity or L2 distance</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Simple Framework for Contrastive Learning of Visual Representations", "https://arxiv.org/abs/2002.05709", Year = 2020, Authors = "Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton")]
public partial class KNNEvaluator<T> : ModelBase<T, Tensor<T>, Tensor<T>>
{

    /// <inheritdoc />
    /// <remarks>The encoder being evaluated. Both hand-written surfaces forwarded straight to it.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_encoder);
    }
    // NumOps and Engine inherited from ModelBase

    private readonly INeuralNetwork<T> _encoder;
    private readonly int _k;
    private readonly bool _useCosine;
    private readonly double _temperature;

    [AiDotNet.Attributes.FittedParameter]
    private Tensor<T>? _trainFeatures;
    private int[]? _trainLabels;
    private int _numClasses;

    /// <summary>
    /// Gets the k value for k-NN.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Initializes a new instance of the KNNEvaluator class.
    /// </summary>
    /// <param name="encoder">The pretrained encoder.</param>
    /// <param name="k">Number of neighbors to consider (default: 20).</param>
    /// <param name="useCosine">Use cosine similarity instead of L2 distance (default: true).</param>
    /// <param name="temperature">Temperature for weighted voting (default: 0.07).</param>
    public KNNEvaluator(
        INeuralNetwork<T> encoder,
        int k = 20,
        bool useCosine = true,
        double temperature = 0.07)
    {
        Guard.NotNull(encoder);
        _encoder = encoder;
        _k = k;
        _useCosine = useCosine;
        _temperature = temperature;
    }

    /// <summary>
    /// Fits the k-NN classifier with training data.
    /// </summary>
    /// <param name="trainData">Training data [num_samples, features].</param>
    /// <param name="trainLabels">Training labels [num_samples].</param>
    public void Fit(Tensor<T> trainData, int[] trainLabels)
    {
        if (trainData is null) throw new ArgumentNullException(nameof(trainData));
        if (trainLabels is null) throw new ArgumentNullException(nameof(trainLabels));

        // Extract features using encoder
        _trainFeatures = ExtractFeatures(trainData);
        _trainLabels = trainLabels;
        _numClasses = trainLabels.Max() + 1;

        // Normalize features if using cosine similarity
        if (_useCosine)
        {
            _trainFeatures = L2Normalize(_trainFeatures);
        }
    }

    /// <summary>
    /// Evaluates k-NN classification on test data.
    /// </summary>
    /// <param name="testData">Test data [num_samples, features].</param>
    /// <param name="testLabels">Test labels [num_samples].</param>
    /// <returns>Top-1 accuracy.</returns>
    public double Evaluate(Tensor<T> testData, int[] testLabels)
    {
        if (_trainFeatures is null || _trainLabels is null)
            throw new InvalidOperationException("Must call Fit() before Evaluate()");

        var testFeatures = ExtractFeatures(testData);
        if (_useCosine)
        {
            testFeatures = L2Normalize(testFeatures);
        }

        var numTest = testFeatures.Shape[0];
        int correct = 0;

        for (int i = 0; i < numTest; i++)
        {
            var predicted = PredictSingle(testFeatures, i);
            if (predicted == testLabels[i])
                correct++;
        }

        return (double)correct / numTest;
    }

    /// <summary>
    /// Evaluates k-NN with different k values.
    /// </summary>
    /// <param name="testData">Test data.</param>
    /// <param name="testLabels">Test labels.</param>
    /// <param name="kValues">List of k values to evaluate.</param>
    /// <returns>Dictionary of k value to accuracy.</returns>
    public Dictionary<int, double> EvaluateMultipleK(
        Tensor<T> testData, int[] testLabels, int[] kValues)
    {
        if (_trainFeatures is null || _trainLabels is null)
            throw new InvalidOperationException("Must call Fit() before Evaluate()");

        var testFeatures = ExtractFeatures(testData);
        if (_useCosine)
        {
            testFeatures = L2Normalize(testFeatures);
        }

        var numTest = testFeatures.Shape[0];
        var dim = testFeatures.Shape[1];

        var results = new Dictionary<int, double>();
        var maxK = kValues.Max();

        // Pre-compute all distances
        var correct = new int[kValues.Length];

        for (int i = 0; i < numTest; i++)
        {
            // Compute distances to all training samples
            var distances = ComputeDistanceRow(testFeatures, i, _trainFeatures, dim)
                .Select((distance, index) => (index, distance))
                .ToArray();

            // Sort by distance (ascending for L2, descending for cosine similarity)
            if (_useCosine)
            {
                distances = [.. distances.OrderByDescending(d => NumOps.ToDouble(d.distance))];
            }
            else
            {
                distances = [.. distances.OrderBy(d => NumOps.ToDouble(d.distance))];
            }

            // Evaluate for each k value
            for (int ki = 0; ki < kValues.Length; ki++)
            {
                var k = kValues[ki];
                var neighbors = distances.Take(k).ToArray();
                var predicted = VotePrediction(neighbors);

                if (predicted == testLabels[i])
                    correct[ki]++;
            }
        }

        for (int ki = 0; ki < kValues.Length; ki++)
        {
            results[kValues[ki]] = (double)correct[ki] / numTest;
        }

        return results;
    }

    /// <summary>
    /// Predicts classes for multiple test samples.
    /// </summary>
    /// <param name="testData">Test data.</param>
    /// <returns>Predicted class labels.</returns>
    public int[] PredictClasses(Tensor<T> testData)
    {
        if (_trainFeatures is null || _trainLabels is null)
            throw new InvalidOperationException("Must call Fit() before Predict()");

        var testFeatures = ExtractFeatures(testData);
        if (_useCosine)
        {
            testFeatures = L2Normalize(testFeatures);
        }

        var numTest = testFeatures.Shape[0];
        var predictions = new int[numTest];

        for (int i = 0; i < numTest; i++)
        {
            predictions[i] = PredictSingle(testFeatures, i);
        }

        return predictions;
    }

    private int PredictSingle(Tensor<T> testFeatures, int testIdx)
    {
        var trainFeatures = _trainFeatures ?? throw new InvalidOperationException("Training features have not been initialized. Call Fit() first.");
        var dim = testFeatures.Shape[1];

        // Compute distances to all training samples
        var distances = ComputeDistanceRow(testFeatures, testIdx, trainFeatures, dim)
            .Select((distance, index) => (index, distance))
            .ToArray();

        // Sort by distance
        if (_useCosine)
        {
            // For cosine similarity, larger is better
            distances = [.. distances.OrderByDescending(d => NumOps.ToDouble(d.distance))];
        }
        else
        {
            // For L2 distance, smaller is better
            distances = [.. distances.OrderBy(d => NumOps.ToDouble(d.distance))];
        }

        // Get k nearest neighbors
        var neighbors = distances.Take(_k).ToArray();

        return VotePrediction(neighbors);
    }

    private int VotePrediction((int index, T distance)[] neighbors)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var trainLabels = _trainLabels ?? throw new InvalidOperationException("Training labels have not been initialized. Call Fit() first.");

        // Weighted voting based on distance
        var neighborDistances = new Vector<T>(neighbors.Select(n => n.distance)).AsTensor();
        var neighborLabels = neighbors.Select(n => trainLabels[n.index]).ToArray();

        var weights = _useCosine
            // Weight by similarity (higher is better)
            ? Engine.TensorExp(Engine.TensorDivideScalar(neighborDistances, NumOps.FromDouble(_temperature)))
            // Weight by inverse distance
            : Engine.Reciprocal(Engine.TensorAddScalar(neighborDistances, ObjectiveOps.Epsilon<T>(1e-8)).AsVector()).AsTensor();

        var votes = Engine.ScatterAdd(
            weights, new Tensor<int>(neighborLabels, new[] { neighborLabels.Length }), 0, _numClasses);

        // Return class with highest vote (the first one on ties)
        return Engine.TensorArgMax(votes, 0).ToArray()[0];
    }

    /// <summary>
    /// Distances from row <paramref name="aIdx"/> of <paramref name="a"/> to every row of <paramref name="b"/>,
    /// over the first <paramref name="dim"/> features: the dot product (cosine, features already normalized)
    /// or the L2 distance.
    /// </summary>
    private Vector<T> ComputeDistanceRow(Tensor<T> a, int aIdx, Tensor<T> b, int dim)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int numB = b.Shape[0];
        var aRow = Engine.TensorSlice(a, new[] { aIdx, 0 }, new[] { 1, dim });
        var bRows = b.Shape[1] == dim ? b : Engine.TensorSlice(b, new[] { 0, 0 }, new[] { numB, dim });

        if (_useCosine)
        {
            // Cosine similarity (features are already normalized)
            return Engine.TensorMatMul(bRows, Engine.TensorTranspose(aRow)).Reshape(numB).AsVector();
        }

        // L2 distance
        var diff = Engine.TensorSubtract(aRow, bRows);
        var squared = Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 1 }, keepDims: false);
        return Engine.TensorSqrt(squared).AsVector();
    }

    private Tensor<T> ExtractFeatures(Tensor<T> input)
    {
        return _encoder.Predict(input);
    }

    private Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        // x / sqrt(sum(x^2) + 1e-8) per row
        var sumSquared = Engine.ReduceSum(Engine.TensorMultiply(tensor, tensor), new[] { 1 }, keepDims: true);
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(sumSquared, ObjectiveOps.Epsilon<T>(1e-8)));
        return Engine.TensorDivide(tensor, norm);
    }

    #region ModelBase Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input) => _encoder.Predict(input);

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput) { }

    /// <inheritdoc />
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var copy = DeepCopy();
        ((IParameterizable<T, Tensor<T>, Tensor<T>>)copy).SetParameters(parameters);
        return copy;
    }

    #endregion
}
