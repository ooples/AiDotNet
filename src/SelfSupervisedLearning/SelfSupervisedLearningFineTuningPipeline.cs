using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Pipeline for fine-tuning SSL pretrained encoders on downstream tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> After SSL pretraining, you typically want to fine-tune
/// the encoder on a specific task with labeled data. This pipeline handles the
/// fine-tuning process with proper learning rate schedules and evaluation.</para>
///
/// <para><b>Fine-tuning strategies:</b></para>
/// <list type="bullet">
/// <item><b>Full fine-tuning:</b> Update all parameters</item>
/// <item><b>Linear probing:</b> Freeze encoder, train only classifier</item>
/// <item><b>Gradual unfreezing:</b> Unfreeze layers progressively</item>
/// </list>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Simple Framework for Contrastive Learning of Visual Representations", "https://arxiv.org/abs/2002.05709", Year = 2020, Authors = "Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton")]
public class SelfSupervisedLearningFineTuningPipeline<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly INeuralNetwork<T> _encoder;
    private readonly int _encoderOutputDim;
    private readonly int _numClasses;

    private FineTuningConfig _config;
    private T[] _classifierWeights;
    private T[] _classifierBias;

    /// <summary>
    /// Event raised for progress updates.
    /// </summary>
    #pragma warning disable CS0067
    public event Action<int, int, double>? OnProgress;
    #pragma warning restore CS0067

    /// <summary>
    /// Initializes a new fine-tuning pipeline.
    /// </summary>
    /// <param name="encoder">Pretrained encoder to fine-tune.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="numClasses">Number of classes for classification.</param>
    public SelfSupervisedLearningFineTuningPipeline(
        INeuralNetwork<T> encoder,
        int encoderOutputDim,
        int numClasses)
    {
        Guard.NotNull(encoder);
        _encoder = encoder;
        _encoderOutputDim = encoderOutputDim;
        _numClasses = numClasses;
        _config = new FineTuningConfig();

        // Initialize classifier
        var rng = RandomHelper.Shared;
        var scale = Math.Sqrt(2.0 / encoderOutputDim);

        _classifierWeights = new T[encoderOutputDim * numClasses];
        _classifierBias = new T[numClasses];

        for (int i = 0; i < _classifierWeights.Length; i++)
        {
            _classifierWeights[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
        }
    }

    /// <summary>
    /// Configures fine-tuning parameters.
    /// </summary>
    public SelfSupervisedLearningFineTuningPipeline<T> WithConfig(Action<FineTuningConfig> configure)
    {
        configure(_config);
        return this;
    }

    /// <summary>
    /// Sets the fine-tuning strategy.
    /// </summary>
    public SelfSupervisedLearningFineTuningPipeline<T> WithStrategy(FineTuningStrategy strategy)
    {
        _config.Strategy = strategy;
        return this;
    }

    /// <summary>
    /// Evaluates the model on test data.
    /// </summary>
    public double Evaluate(Tensor<T> testData, int[] testLabels)
    {
        var numSamples = testData.Shape[0];
        var batchSize = 256;
        int correct = 0;

        for (int i = 0; i < numSamples; i += batchSize)
        {
            int actualBatch = Math.Min(batchSize, numSamples - i);

            var batchData = ExtractBatch(testData, i, actualBatch);
            var batchLabels = testLabels.Skip(i).Take(actualBatch).ToArray();

            var features = _encoder.Predict(batchData);
            var logits = ComputeLogits(features);

            correct += ComputeCorrect(logits, batchLabels);
        }

        return (double)correct / numSamples;
    }

    private Tensor<T> ComputeLogits(Tensor<T> features)
    {
        // logits = features[:, :encoderOutputDim] W + b
        return Engine.TensorAdd(
            Engine.TensorMatMul(FeatureColumns(features), ClassifierWeightsView),
            ClassifierBiasView.Reshape(1, _numClasses));
    }

    /// <summary>Zero-copy [encoderOutputDim, numClasses] view of the classifier weights.</summary>
    private Tensor<T> ClassifierWeightsView =>
        Tensor<T>.FromMemory(_classifierWeights.AsMemory(), new[] { _encoderOutputDim, _numClasses });

    /// <summary>Zero-copy [numClasses] view of the classifier bias.</summary>
    private Tensor<T> ClassifierBiasView => Vector<T>.Wrap(_classifierBias).AsTensor();

    /// <summary>The first <c>encoderOutputDim</c> columns of a [batch, features] tensor.</summary>
    private Tensor<T> FeatureColumns(Tensor<T> features)
    {
        var batchSize = features.Shape[0];
        return features.Rank == 2 && features.Shape[1] == _encoderOutputDim
            ? features
            : Engine.TensorSlice(features, new[] { 0, 0 }, new[] { batchSize, _encoderOutputDim });
    }

    /// <summary>
    /// Row max, exp(logit - max) and its row sum over the first numClasses logits.
    /// </summary>
    private (Tensor<T> Max, Tensor<T> Exps, Tensor<T> SumExp) ShiftedExponentials(Tensor<T> logits)
    {
        var batchSize = logits.Shape[0];
        var classLogits = Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { batchSize, _numClasses });
        var maxLogit = Engine.ReduceMax(classLogits, new[] { 1 }, keepDims: true);
        var exps = Engine.TensorExp(Engine.TensorSubtract(classLogits, maxLogit));
        return (maxLogit, exps, Engine.ReduceSum(exps, new[] { 1 }, keepDims: true));
    }

    /// <summary>One-hot [batch, numClasses] targets for the labels.</summary>
    private Tensor<T> LabelTargets(int[] labels, int batchSize)
    {
        return Engine.TensorOneHot<T>(new Tensor<int>(labels.Take(batchSize).ToArray(), new[] { batchSize }), _numClasses);
    }

    private T ComputeCrossEntropyLoss(Tensor<T> logits, int[] labels)
    {
        var batchSize = logits.Shape[0];

        // log p_true = (trueLogit - max) - log(sum exp(logit - max))
        var (maxLogit, _, sumExp) = ShiftedExponentials(logits);
        var classLogits = Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { batchSize, _numClasses });
        var trueLogit = Engine.ReduceSum(
            Engine.TensorWhere(LabelTargets(labels, batchSize), classLogits, new Tensor<T>(new[] { batchSize, _numClasses })),
            new[] { 1 }, keepDims: true);
        var logProb = Engine.TensorSubtract(Engine.TensorSubtract(trueLogit, maxLogit), Engine.TensorLog(sumExp));
        T totalLoss = NumOps.Negate(Engine.TensorSum(logProb));

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    private Tensor<T> ComputeGradients(Tensor<T> logits, int[] labels)
    {
        var batchSize = logits.Shape[0];

        // Softmax probabilities minus the one-hot target
        var (_, exps, sumExp) = ShiftedExponentials(logits);
        var probs = Engine.TensorDivide(exps, sumExp);
        return Engine.TensorSubtract(probs, LabelTargets(labels, batchSize));
    }

    private void UpdateClassifier(Tensor<T> features, Tensor<T> gradLogits, double lr)
    {
        var batchSize = features.Shape[0];
        var lrT = NumOps.FromDouble(lr / batchSize);

        // dW = features^T gradLogits, db = sum_b gradLogits
        var gradWeights = Engine.TensorMatMul(Engine.TensorTranspose(FeatureColumns(features)), gradLogits);
        Engine.TensorSubtract(ClassifierWeightsView, Engine.TensorMultiplyScalar(gradWeights, lrT))
            .AsSpan().CopyTo(_classifierWeights);

        var gradBias = Engine.ReduceSum(gradLogits, new[] { 0 }, keepDims: false);
        Engine.TensorSubtract(ClassifierBiasView, Engine.TensorMultiplyScalar(gradBias, lrT))
            .AsSpan().CopyTo(_classifierBias);
    }

    private Tensor<T> ComputeFeatureGradients(Tensor<T> gradLogits)
    {
        // dFeatures = gradLogits W^T
        return Engine.TensorMatMul(gradLogits, Engine.TensorTranspose(ClassifierWeightsView));
    }

    private void UpdateEncoder(double lr)
    {
        var lrT = NumOps.FromDouble(lr);
        var grads = new Vector<T>(_encoder.GetParameterGradients());
        var pars = _encoder.GetParameters();
        _encoder.UpdateParameters(Engine.Subtract(pars, Engine.Multiply(grads, lrT)));
    }

    private int ComputeCorrect(Tensor<T> logits, int[] labels)
    {
        var batchSize = logits.Shape[0];

        // Predicted class per row is the first index of the row maximum
        var classLogits = Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { batchSize, _numClasses });
        var predicted = Engine.TensorArgMax(classLogits, 1).ToArray();
        return predicted.Zip(labels, (p, label) => p == label ? 1 : 0).Sum();
    }

    private double GetLearningRate(int epoch, int totalEpochs)
    {
        var baseLr = _config.LearningRate ?? 0.01;
        return baseLr * 0.5 * (1 + Math.Cos(Math.PI * epoch / totalEpochs));
    }

    private bool ShouldFreezeEncoder(int epoch, int totalEpochs)
    {
        return _config.Strategy switch
        {
            FineTuningStrategy.LinearProbing => true,
            FineTuningStrategy.GradualUnfreezing => epoch < totalEpochs / 4,
            _ => false
        };
    }

    private Tensor<T> ExtractBatch(Tensor<T> data, int startIdx, int batchSize)
    {
        var dim = data.Shape[1];
        return Engine.TensorSlice(data, new[] { startIdx, 0 }, new[] { batchSize, dim });
    }
}
