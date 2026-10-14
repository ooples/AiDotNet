using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// FT-Transformer implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// FTTransformerClassifier applies the FT-Transformer architecture to multi-class
/// classification problems. It uses the [CLS] token output with a linear classification head
/// and softmax activation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this model when you want to predict categories
/// (like spam/not spam, customer segments, product categories, etc.).
///
/// How it works:
/// 1. Features are tokenized and processed by transformer layers
/// 2. The [CLS] token captures information from all features
/// 3. A linear layer maps the [CLS] representation to class logits
/// 4. Softmax converts logits to probabilities
///
/// Example:
/// <code>
/// // Create classifier for 10 features predicting 3 classes
/// var options = new FTTransformerOptions&lt;double&gt; { EmbeddingDimension = 128, NumLayers = 2 };
/// var classifier = new FTTransformerClassifier&lt;double&gt;(10, 3, options);
///
/// // Forward pass
/// var input = new Tensor&lt;double&gt;([32, 10]); // batch of 32 samples, 10 features
/// var probabilities = classifier.PredictProbabilities(input);
/// var predictions = classifier.Predict(input);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Revisiting Deep Learning Models for Tabular Data",
    "https://arxiv.org/abs/2106.11959",
    Year = 2021,
    Authors = "Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A.")]
public class FTTransformerClassifier<T> : FTTransformerBase<T>
{
    private readonly int _numClasses;
    private readonly FullyConnectedLayer<T> _classificationHead;

    // Cache for backward pass
    [Scratch]
    private Tensor<T>? _clsOutputCache;
    [Scratch]
    private Tensor<T>? _logitsCache;
    [Scratch]
    private Tensor<T>? _probabilitiesCache;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Initializes a new instance of the FTTransformerClassifier class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="options">Model configuration options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creating a classifier:
    /// - numNumericalFeatures: How many number columns in your input data
    /// - numClasses: How many categories to predict (2 for binary, more for multi-class)
    /// - options: Model configuration (see FTTransformerOptions for details)
    ///
    /// For binary classification, numClasses should be 2.
    /// </para>
    /// </remarks>
    public FTTransformerClassifier(
        int numNumericalFeatures,
        int numClasses,
        FTTransformerOptions<T>? options = null)
        : base(numNumericalFeatures, options)
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2", nameof(numClasses));
        }

        _numClasses = numClasses;

        // Classification head: Linear layer from embedding dimension to number of classes. It emits raw
        // logits (softmax is applied separately); the paper's ReLU comes before it (ActivateForHead).
        _classificationHead = new FullyConnectedLayer<T>(
            EmbeddingDimension,
            numClasses,
            new IdentityActivation<T>());
    }

    /// <summary>
    /// Performs the forward pass to get class logits.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>Class logits tensor [batch_size, num_classes].</returns>
    public Tensor<T> Forward(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        // Get [CLS] representation from backbone
        var clsOutput = ForwardBackbone(numericalFeatures, categoricalIndices);
        _clsOutputCache = clsOutput;

        // Apply classification head: Linear(ReLU(LayerNorm(CLS)))
        var logits = _classificationHead.Forward(ActivateForHead(clsOutput));
        _logitsCache = logits;

        return logits;
    }

    /// <summary>
    /// Performs the forward pass with numerical features only.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <returns>Class logits tensor [batch_size, num_classes].</returns>
    public Tensor<T> Forward(Tensor<T> numericalFeatures)
    {
        return Forward(numericalFeatures, null);
    }

    /// <summary>
    /// Predicts class probabilities using softmax.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>Probability tensor [batch_size, num_classes].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the probability of each class:
    /// - Output values are between 0 and 1
    /// - Values in each row sum to 1
    /// - Higher values indicate more confident predictions
    ///
    /// Example output for 3 classes: [0.7, 0.2, 0.1] means:
    /// - 70% chance of class 0
    /// - 20% chance of class 1
    /// - 10% chance of class 2
    /// </para>
    /// </remarks>
    public Tensor<T> PredictProbabilities(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        var logits = Forward(numericalFeatures, categoricalIndices);
        var probabilities = ApplySoftmax(logits);
        _probabilitiesCache = probabilities;
        return probabilities;
    }

    /// <summary>
    /// Predicts class probabilities with numerical features only.
    /// </summary>
    public Tensor<T> PredictProbabilities(Tensor<T> numericalFeatures)
    {
        return PredictProbabilities(numericalFeatures, null);
    }

    /// <summary>
    /// Predicts the most likely class for each sample.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>Predicted class indices [batch_size].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the most likely class for each sample:
    /// - Output is an array of class indices (0, 1, 2, etc.)
    /// - The class with highest probability is selected
    ///
    /// Example: If you have 3 samples and 4 classes, output might be [2, 0, 1]
    /// meaning sample 0 is class 2, sample 1 is class 0, sample 2 is class 1.
    /// </para>
    /// </remarks>
    public int[] Predict(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        var probabilities = PredictProbabilities(numericalFeatures, categoricalIndices);
        int batchSize = probabilities.Shape[0];
        var predictions = new int[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            int maxIdx = 0;
            var maxProb = probabilities[b * _numClasses + 0];

            for (int c = 1; c < _numClasses; c++)
            {
                var prob = probabilities[b * _numClasses + c];
                if (NumOps.Compare(prob, maxProb) > 0)
                {
                    maxProb = prob;
                    maxIdx = c;
                }
            }

            predictions[b] = maxIdx;
        }

        return predictions;
    }

    /// <summary>
    /// Predicts class indices with numerical features only.
    /// </summary>
    public int[] Predict(Tensor<T> numericalFeatures)
    {
        return Predict(numericalFeatures, null);
    }

    /// <summary>
    /// Computes the cross-entropy loss.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities [batch_size, num_classes].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <returns>The average cross-entropy loss.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cross-entropy loss measures how wrong the predictions are:
    /// - Lower loss = better predictions
    /// - Loss of 0 = perfect predictions
    /// - Penalizes confident wrong predictions more heavily
    ///
    /// Formula: -log(probability of correct class)
    /// </para>
    /// </remarks>
    public T ComputeCrossEntropyLoss(Tensor<T> probabilities, int[] targets)
    {
        int batchSize = probabilities.Shape[0];
        var totalLoss = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-15);  // Numerical stability

        for (int b = 0; b < batchSize; b++)
        {
            int targetClass = targets[b];
            var prob = probabilities[b * _numClasses + targetClass];
            var safeProp = NumOps.Add(prob, epsilon);  // Avoid log(0)
            var loss = NumOps.Negate(NumOps.Log(safeProp));
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Performs a single training step.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>The training loss for this step.</returns>
    public T TrainStep(
        Tensor<T> numericalFeatures,
        int[] targets,
        T learningRate,
        Matrix<int>? categoricalIndices = null)
    {
        // Collect every trainable tensor as a LIVE field instance — tokenizer
        // (custom), then backbone encoder layers + final LayerNorm + the
        // classification head (via the tape param collector). ComputeGradients
        // maps gradients back to exactly these instances, and we apply SGD to
        // them in place so the model actually updates.
        var trainable = new System.Collections.Generic.List<Tensor<T>>(Tokenizer.GetTrainableTensors());
        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<T>>();
        foreach (var encoderLayer in EncoderLayers) layers.Add(encoderLayer);
        layers.Add(FinalLayerNorm);
        layers.Add(_classificationHead);
        trainable.AddRange(AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers));

        // Reverse-mode AD over the whole network: the forward (tokenizer +
        // transformer + head) and the cross-entropy below are Engine ops, so a
        // forward under the tape records them and ComputeGradients fills in the
        // gradient for every leaf tensor we collected. (Previously TrainStep had
        // only a "// Backward pass" comment, so UpdateParameters threw "Backward
        // pass must be called before updating parameters" — training never ran.)
        // Plain SGD at the caller's rate, the update this step has always applied, as one step on the shared fused
        // session: the compiled plan when it applies, the eager tape otherwise. Categorical features are looked up
        // by integer index on the host, which a plan would bake in, so a batch carrying them stays eager.
        T lossValue;
        if (categoricalIndices is null)
        {
            var stepper = _tapeStepper ??= new AiDotNet.Training.TapeTrainingStepper<T>(this);
            lossValue = stepper.Step(new AiDotNet.Training.FusedTrainingStepRequest<T>
            {
                Layers = Array.Empty<AiDotNet.Interfaces.ITrainableLayer<T>>(),
                Input = numericalFeatures,
                Target = OneHot(targets, numericalFeatures.Shape[0]),
                Forward = features => Forward(features, null),
                ComputeLoss = SoftmaxCrossEntropyLoss,
                Optimizer = SgdAt(NumOps.ToDouble(learningRate)),
                ExtraParameters = trainable,
            });
        }
        else
        {
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>();
            var logits = Forward(numericalFeatures, categoricalIndices);
            var lossTensor = SoftmaxCrossEntropyLoss(logits, OneHot(targets, logits.Shape[0]));
            var grads = tape.ComputeGradients(lossTensor, trainable.ToArray());
            foreach (var t in trainable)
            {
                if (t is null || !grads.TryGetValue(t, out var g) || g is null) continue;
                Engine.TensorSubtractInPlace(t, Engine.TensorMultiplyScalar(g, learningRate));
            }
            lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
        }

        ResetState();
        return lossValue;
    }

    // The class targets as a one-hot [batch, classes] tensor: data the step carries, not a closure.
    private Tensor<T> OneHot(int[] targets, int batch)
    {
        var onehot = new Tensor<T>([batch, _numClasses]);
        for (int b = 0; b < batch; b++)
            onehot[b * _numClasses + targets[b]] = NumOps.One;
        return onehot;
    }

    private AiDotNet.Training.TapeTrainingStepper<T>? _tapeStepper;

    /// <summary>Training steps run on the fused compiled plan.</summary>
    internal int FusedTrainingStepCount => _tapeStepper?.FusedStepCount ?? 0;

    /// <summary>Training steps run on the eager tape.</summary>
    internal int EagerTrainingStepCount => _tapeStepper?.EagerStepCount ?? 0;
    private AiDotNet.Optimizers.StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>? _sgd;
    private double _sgdRate = double.NaN;

    // Plain SGD at a rate, without the clipping the optimizer options enable by default (this step never clipped).
    private AiDotNet.Optimizers.StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>> SgdAt(double rate)
    {
        if (_sgd is null || _sgdRate != rate)
        {
            _sgd = new AiDotNet.Optimizers.StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(null,
                new StochasticGradientDescentOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = rate,
                    EnableGradientClipping = false,
                });
            _sgdRate = rate;
        }
        return _sgd;
    }

    /// <summary>
    /// Tape-tracked softmax cross-entropy on logits [B, C] against a one-hot target [B, C].
    /// CE = −mean_b Σ_c onehot[b,c]·log(softmax(logits)[b,c]). Built from Engine tensor ops, with the one-hot
    /// target a step input (no gradient), so reverse-mode AD and a compiled plan both differentiate it cleanly.
    /// </summary>
    private Tensor<T> SoftmaxCrossEntropyLoss(Tensor<T> logits, Tensor<T> onehot)
    {
        var logProbs = Engine.TensorLogSoftmax(logits, axis: -1);
        var weighted = Engine.TensorMultiply(logProbs, onehot);
        var perRow = Engine.ReduceSum(weighted, new[] { 1 }, keepDims: false);
        var meanVal = Engine.ReduceMean(perRow, new[] { 0 }, keepDims: false);
        return Engine.TensorMultiplyScalar(meanVal, NumOps.FromDouble(-1.0));
    }

    /// <summary>
    /// Computes classification accuracy.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>Accuracy as a value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Accuracy is the fraction of correct predictions:
    /// - Accuracy = (number correct) / (total samples)
    /// - 1.0 = 100% correct, 0.0 = 0% correct
    /// </para>
    /// </remarks>
    public T ComputeAccuracy(Tensor<T> numericalFeatures, int[] targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
        int correct = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            if (predictions[i] == targets[i])
            {
                correct++;
            }
        }

        return NumOps.FromDouble((double)correct / predictions.Length);
    }

    /// <summary>
    /// Computes the macro F1 score across all classes.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>The macro F1 score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> F1 score balances precision and recall:
    /// - F1 = 2 * (precision * recall) / (precision + recall)
    /// - Macro F1 averages F1 scores across all classes
    /// - Good for imbalanced datasets where accuracy can be misleading
    /// </para>
    /// </remarks>
    public T ComputeF1Score(Tensor<T> numericalFeatures, int[] targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
        var totalF1 = NumOps.Zero;

        for (int c = 0; c < _numClasses; c++)
        {
            int truePositives = 0;
            int falsePositives = 0;
            int falseNegatives = 0;

            for (int i = 0; i < predictions.Length; i++)
            {
                if (predictions[i] == c && targets[i] == c)
                    truePositives++;
                else if (predictions[i] == c && targets[i] != c)
                    falsePositives++;
                else if (predictions[i] != c && targets[i] == c)
                    falseNegatives++;
            }

            double precision = truePositives + falsePositives > 0
                ? (double)truePositives / (truePositives + falsePositives) : 0.0;
            double recall = truePositives + falseNegatives > 0
                ? (double)truePositives / (truePositives + falseNegatives) : 0.0;
            double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

            totalF1 = NumOps.Add(totalF1, NumOps.FromDouble(f1));
        }

        return NumOps.Divide(totalF1, NumOps.FromDouble(_numClasses));
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        return Engine.Softmax(logits, -1);
    }

    /// <summary>
    /// Updates all parameters including the classification head.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        base.UpdateParameters(learningRate);
        _classificationHead.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public override void ResetState()
    {
        base.ResetState();
        _clsOutputCache = null;
        _logitsCache = null;
        _probabilitiesCache = null;
        _classificationHead.ResetState();
    }
}
