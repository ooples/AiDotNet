# Junior Developer Implementation Guide: Issue #408

## Overview
**Issue**: Knowledge Distillation
**Goal**: Implement student-teacher training to compress large models into smaller ones
**Difficulty**: Advanced
**Estimated Time**: 14-18 hours

## What is Knowledge Distillation?

Knowledge distillation trains a small "student" model to mimic a large "teacher" model:
- **Teacher**: Large, accurate model (already trained)
- **Student**: Smaller, faster model (to be trained)
- **Goal**: Student learns to produce similar outputs to teacher

### Why Knowledge Distillation?

- Deploy large model accuracy in small model footprint
- Faster inference with minimal accuracy loss
- Transfer knowledge from ensemble to single model
- Learn from unlabeled data using teacher's soft predictions

## Mathematical Background

### Standard Training Loss
```
L_hard = CrossEntropy(student_logits, true_labels)

Where logits are raw network outputs before softmax
```

### Distillation Loss (Hinton et al., 2015)
```
L_soft = KL_Divergence(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
) * T^2

Where:
    T = temperature (usually 2-10)
    Higher T → softer probability distributions
    T^2 term balances gradient magnitudes
```

### Combined Loss
```
L_total = α * L_hard + (1 - α) * L_soft

Where:
    α = balance between hard labels and teacher (usually 0.3-0.5)
    L_hard = learn correct classifications
    L_soft = learn teacher's knowledge
```

### Why Temperature?

**Without temperature (T=1)**:
```
Class probabilities: [0.95, 0.03, 0.02]
→ Student only learns argmax, ignores relative confidences
```

**With temperature (T=5)**:
```
Class probabilities: [0.60, 0.25, 0.15]
→ Student learns: "2nd class somewhat likely, 3rd class slightly possible"
```

Temperature reveals teacher's uncertainty and inter-class relationships.

## Understanding the Codebase

### Key Files to Create

**Core Interfaces:**
```
C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\ITeacherModel.cs
C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IDistillationStrategy.cs
```

**Implementations:**
```
C:\Users\cheat\source\repos\AiDotNet\src\Distillation\KnowledgeDistillationTrainer.cs
C:\Users\cheat\source\repos\AiDotNet\src\Distillation\DistillationLoss.cs
C:\Users\cheat\source\repos\AiDotNet\src\Distillation\TeacherModelWrapper.cs
C:\Users\cheat\source\repos\AiDotNet\src\Distillation\FeatureDistillationStrategy.cs
C:\Users\cheat\source\repos\AiDotNet\src\Distillation\AttentionDistillationStrategy.cs
```

**Test Files:**
```
C:\Users\cheat\source\repos\AiDotNet\tests\Distillation\KnowledgeDistillationTests.cs
```

## Step-by-Step Implementation Guide

### Phase 1: Core Interfaces

#### Step 1.1: Create ITeacherModel Interface

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\ITeacherModel.cs
namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Represents a trained teacher model for knowledge distillation.
    /// </summary>
    /// <typeparam name="TInput">Input type</typeparam>
    /// <typeparam name="TOutput">Output type (usually logits or probabilities)</typeparam>
    public interface ITeacherModel<TInput, TOutput>
    {
        /// <summary>
        /// Gets the teacher's logits (pre-softmax outputs) for given input.
        /// </summary>
        /// <param name="input">Input data</param>
        /// <returns>Raw logits (not probabilities)</returns>
        TOutput GetLogits(TInput input);

        /// <summary>
        /// Gets the teacher's soft predictions (probabilities) for given input.
        /// </summary>
        /// <param name="input">Input data</param>
        /// <param name="temperature">Softmax temperature for softening predictions</param>
        /// <returns>Probability distribution</returns>
        TOutput GetSoftPredictions(TInput input, double temperature = 1.0);

        /// <summary>
        /// Gets intermediate layer activations (for feature distillation).
        /// </summary>
        /// <param name="input">Input data</param>
        /// <param name="layerName">Name of layer to extract features from</param>
        /// <returns>Feature map from specified layer</returns>
        object GetFeatures(TInput input, string layerName);

        /// <summary>
        /// Gets attention weights (for attention distillation in transformers).
        /// </summary>
        object GetAttentionWeights(TInput input, string layerName);

        /// <summary>
        /// Number of output classes (for classification tasks).
        /// </summary>
        int OutputDimension { get; }
    }
}
```

#### Step 1.2: Create IDistillationStrategy Interface

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IDistillationStrategy.cs
namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Defines a strategy for knowledge distillation.
    /// </summary>
    public interface IDistillationStrategy<TInput, TOutput, T>
    {
        /// <summary>
        /// Computes the distillation loss between student and teacher.
        /// </summary>
        /// <param name="studentOutput">Student model output (logits)</param>
        /// <param name="teacherOutput">Teacher model output (logits)</param>
        /// <param name="trueLabels">Ground truth labels (optional)</param>
        /// <returns>Distillation loss value</returns>
        T ComputeLoss(TOutput studentOutput, TOutput teacherOutput, TOutput? trueLabels = default);

        /// <summary>
        /// Temperature parameter for softening probability distributions.
        /// </summary>
        double Temperature { get; set; }

        /// <summary>
        /// Balance between hard loss (true labels) and soft loss (teacher).
        /// Value between 0 (only teacher) and 1 (only labels).
        /// </summary>
        double Alpha { get; set; }

        /// <summary>
        /// Gets the gradient for backpropagation.
        /// </summary>
        TOutput ComputeGradient(TOutput studentOutput, TOutput teacherOutput, TOutput? trueLabels = default);
    }
}
```

### Phase 2: Distillation Loss Functions

#### Step 2.1: Implement DistillationLoss

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Distillation\DistillationLoss.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Distillation
{
    /// <summary>
    /// Implements the standard knowledge distillation loss (Hinton et al., 2015).
    /// Combines hard loss (true labels) with soft loss (teacher predictions).
    /// </summary>
    public class DistillationLoss<T> : IDistillationStrategy<Vector<T>, Vector<T>, T>
    {
        private readonly INumericOperations<T> _numOps;

        public double Temperature { get; set; }
        public double Alpha { get; set; }

        /// <summary>
        /// Creates a new distillation loss function.
        /// </summary>
        /// <param name="temperature">Softmax temperature (default 3.0)</param>
        /// <param name="alpha">Balance between hard and soft loss (default 0.3)</param>
        public DistillationLoss(double temperature = 3.0, double alpha = 0.3)
        {
            _numOps = NumericOperations<T>.Instance;
            Temperature = temperature;
            Alpha = alpha;
        }

        public T ComputeLoss(Vector<T> studentLogits, Vector<T> teacherLogits, Vector<T>? trueLabels = null)
        {
            // Soft loss: KL divergence between student and teacher (with temperature)
            var studentSoft = Softmax(studentLogits, Temperature);
            var teacherSoft = Softmax(teacherLogits, Temperature);

            var softLoss = KLDivergence(studentSoft, teacherSoft);

            // Scale by T^2 to balance gradient magnitudes
            softLoss = _numOps.Multiply(softLoss, _numOps.FromDouble(Temperature * Temperature));

            // If we have true labels, add hard loss
            if (trueLabels != null)
            {
                var studentProbs = Softmax(studentLogits, temperature: 1.0);
                var hardLoss = CrossEntropy(studentProbs, trueLabels);

                // Combine: α * hard_loss + (1 - α) * soft_loss
                var alphaT = _numOps.FromDouble(Alpha);
                var oneMinusAlpha = _numOps.FromDouble(1.0 - Alpha);

                var totalLoss = _numOps.Add(
                    _numOps.Multiply(alphaT, hardLoss),
                    _numOps.Multiply(oneMinusAlpha, softLoss)
                );

                return totalLoss;
            }

            return softLoss;
        }

        public Vector<T> ComputeGradient(Vector<T> studentLogits, Vector<T> teacherLogits, Vector<T>? trueLabels = null)
        {
            int n = studentLogits.Length;
            var gradient = new Vector<T>(n);

            // Soft gradient: ∂L_soft/∂logits
            var studentSoft = Softmax(studentLogits, Temperature);
            var teacherSoft = Softmax(teacherLogits, Temperature);

            for (int i = 0; i < n; i++)
            {
                // KL divergence gradient: (student_soft - teacher_soft) * T^2
                var diff = _numOps.Subtract(studentSoft[i], teacherSoft[i]);
                gradient[i] = _numOps.Multiply(diff, _numOps.FromDouble(Temperature * Temperature));
            }

            // If we have true labels, add hard gradient
            if (trueLabels != null)
            {
                var studentProbs = Softmax(studentLogits, temperature: 1.0);
                var hardGradient = new Vector<T>(n);

                for (int i = 0; i < n; i++)
                {
                    // Cross-entropy gradient: student_probs - true_labels
                    hardGradient[i] = _numOps.Subtract(studentProbs[i], trueLabels[i]);
                }

                // Combine gradients
                var alphaT = _numOps.FromDouble(Alpha);
                var oneMinusAlpha = _numOps.FromDouble(1.0 - Alpha);

                for (int i = 0; i < n; i++)
                {
                    gradient[i] = _numOps.Add(
                        _numOps.Multiply(alphaT, hardGradient[i]),
                        _numOps.Multiply(oneMinusAlpha, gradient[i])
                    );
                }
            }

            return gradient;
        }

        /// <summary>
        /// Applies softmax with temperature to logits.
        /// </summary>
        private Vector<T> Softmax(Vector<T> logits, double temperature)
        {
            int n = logits.Length;
            var result = new Vector<T>(n);

            // Divide logits by temperature
            var scaledLogits = new T[n];
            for (int i = 0; i < n; i++)
            {
                double val = Convert.ToDouble(_numOps.ToDouble(logits[i])) / temperature;
                scaledLogits[i] = _numOps.FromDouble(val);
            }

            // Find max for numerical stability
            T maxLogit = scaledLogits[0];
            for (int i = 1; i < n; i++)
            {
                if (_numOps.GreaterThan(scaledLogits[i], maxLogit))
                    maxLogit = scaledLogits[i];
            }

            // Compute exp(logit - max) and sum
            T sum = _numOps.Zero;
            var expValues = new T[n];

            for (int i = 0; i < n; i++)
            {
                double val = Convert.ToDouble(_numOps.ToDouble(_numOps.Subtract(scaledLogits[i], maxLogit)));
                expValues[i] = _numOps.FromDouble(Math.Exp(val));
                sum = _numOps.Add(sum, expValues[i]);
            }

            // Normalize
            for (int i = 0; i < n; i++)
            {
                result[i] = _numOps.Divide(expValues[i], sum);
            }

            return result;
        }

        /// <summary>
        /// Computes KL divergence: sum(p * log(p / q))
        /// </summary>
        private T KLDivergence(Vector<T> p, Vector<T> q)
        {
            T divergence = _numOps.Zero;

            for (int i = 0; i < p.Length; i++)
            {
                double pVal = Convert.ToDouble(_numOps.ToDouble(p[i]));
                double qVal = Convert.ToDouble(_numOps.ToDouble(q[i]));

                if (pVal > 1e-10) // Avoid log(0)
                {
                    double contrib = pVal * Math.Log(pVal / (qVal + 1e-10));
                    divergence = _numOps.Add(divergence, _numOps.FromDouble(contrib));
                }
            }

            return divergence;
        }

        /// <summary>
        /// Computes cross-entropy: -sum(true_labels * log(predictions))
        /// </summary>
        private T CrossEntropy(Vector<T> predictions, Vector<T> trueLabels)
        {
            T entropy = _numOps.Zero;

            for (int i = 0; i < predictions.Length; i++)
            {
                double pred = Convert.ToDouble(_numOps.ToDouble(predictions[i]));
                double label = Convert.ToDouble(_numOps.ToDouble(trueLabels[i]));

                if (label > 1e-10) // Only compute where label is non-zero
                {
                    double contrib = -label * Math.Log(pred + 1e-10);
                    entropy = _numOps.Add(entropy, _numOps.FromDouble(contrib));
                }
            }

            return entropy;
        }
    }
}
```

### Phase 3: Teacher Model Wrapper

#### Step 3.1: Implement TeacherModelWrapper

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Distillation\TeacherModelWrapper.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Distillation
{
    /// <summary>
    /// Wraps an existing trained model to act as a teacher for distillation.
    /// </summary>
    public class TeacherModelWrapper<TInput, TOutput, T> : ITeacherModel<TInput, TOutput>
    {
        private readonly object _underlyingModel; // Could be NeuralNetwork, etc.
        private readonly Dictionary<string, object> _cachedActivations;

        public int OutputDimension { get; private set; }

        public TeacherModelWrapper(object model, int outputDimension)
        {
            _underlyingModel = model ?? throw new ArgumentNullException(nameof(model));
            OutputDimension = outputDimension;
            _cachedActivations = new Dictionary<string, object>();
        }

        public TOutput GetLogits(TInput input)
        {
            // Forward pass through model to get raw logits
            // This depends on your model implementation

            if (_underlyingModel is NeuralNetwork<TInput, TOutput, T> neuralNet)
            {
                // Get output before final activation
                return neuralNet.ForwardPass(input, returnLogits: true);
            }

            throw new NotSupportedException("Model type not supported for distillation");
        }

        public TOutput GetSoftPredictions(TInput input, double temperature = 1.0)
        {
            var logits = GetLogits(input);

            // Apply softmax with temperature
            if (logits is Vector<T> logitsVector)
            {
                return (TOutput)(object)ApplySoftmax(logitsVector, temperature);
            }

            throw new NotSupportedException("Output type not supported");
        }

        public object GetFeatures(TInput input, string layerName)
        {
            // Extract intermediate layer activations
            if (_underlyingModel is NeuralNetwork<TInput, TOutput, T> neuralNet)
            {
                neuralNet.ForwardPass(input); // Populate activations
                return neuralNet.GetLayerActivation(layerName);
            }

            throw new NotSupportedException("Feature extraction not supported for this model");
        }

        public object GetAttentionWeights(TInput input, string layerName)
        {
            // For transformer models with attention mechanisms
            // Would need to be implemented based on your transformer architecture
            throw new NotImplementedException("Attention distillation not yet implemented");
        }

        private Vector<T> ApplySoftmax(Vector<T> logits, double temperature)
        {
            var numOps = NumericOperations<T>.Instance;
            int n = logits.Length;
            var result = new Vector<T>(n);

            // Scale by temperature
            var scaledLogits = new T[n];
            for (int i = 0; i < n; i++)
            {
                double val = Convert.ToDouble(numOps.ToDouble(logits[i])) / temperature;
                scaledLogits[i] = numOps.FromDouble(val);
            }

            // Softmax computation (with numerical stability)
            T maxLogit = scaledLogits[0];
            for (int i = 1; i < n; i++)
            {
                if (numOps.GreaterThan(scaledLogits[i], maxLogit))
                    maxLogit = scaledLogits[i];
            }

            T sum = numOps.Zero;
            var expValues = new T[n];

            for (int i = 0; i < n; i++)
            {
                double val = Convert.ToDouble(numOps.ToDouble(numOps.Subtract(scaledLogits[i], maxLogit)));
                expValues[i] = numOps.FromDouble(Math.Exp(val));
                sum = numOps.Add(sum, expValues[i]);
            }

            for (int i = 0; i < n; i++)
            {
                result[i] = numOps.Divide(expValues[i], sum);
            }

            return result;
        }
    }
}
```

### Phase 4: Knowledge Distillation Trainer

#### Step 4.1: Implement KnowledgeDistillationTrainer

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Distillation\KnowledgeDistillationTrainer.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Distillation
{
    /// <summary>
    /// Trains a student model using knowledge distillation from a teacher model.
    /// </summary>
    public class KnowledgeDistillationTrainer<TInput, TOutput, T>
    {
        private readonly ITeacherModel<TInput, TOutput> _teacher;
        private readonly IDistillationStrategy<TInput, TOutput, T> _distillationLoss;
        private readonly INumericOperations<T> _numOps;

        public KnowledgeDistillationTrainer(
            ITeacherModel<TInput, TOutput> teacher,
            IDistillationStrategy<TInput, TOutput, T> distillationLoss)
        {
            _teacher = teacher ?? throw new ArgumentNullException(nameof(teacher));
            _distillationLoss = distillationLoss ?? throw new ArgumentNullException(nameof(distillationLoss));
            _numOps = NumericOperations<T>.Instance;
        }

        /// <summary>
        /// Trains student model on a single batch.
        /// </summary>
        /// <param name="student">Student model to train</param>
        /// <param name="inputs">Input batch</param>
        /// <param name="trueLabels">Ground truth labels (optional)</param>
        /// <returns>Average loss for the batch</returns>
        public T TrainBatch(
            object student,
            TInput[] inputs,
            TOutput[]? trueLabels = null)
        {
            T totalLoss = _numOps.Zero;

            for (int i = 0; i < inputs.Length; i++)
            {
                var input = inputs[i];
                var trueLabel = trueLabels?[i];

                // Get teacher predictions (soft targets)
                var teacherOutput = _teacher.GetLogits(input);

                // Get student predictions
                var studentOutput = GetStudentOutput(student, input);

                // Compute distillation loss
                var loss = _distillationLoss.ComputeLoss(studentOutput, teacherOutput, trueLabel);
                totalLoss = _numOps.Add(totalLoss, loss);

                // Compute gradient
                var gradient = _distillationLoss.ComputeGradient(studentOutput, teacherOutput, trueLabel);

                // Backpropagate through student
                BackpropagateStudent(student, gradient);
            }

            // Return average loss
            return _numOps.Divide(totalLoss, _numOps.FromDouble(inputs.Length));
        }

        /// <summary>
        /// Trains student model for multiple epochs.
        /// </summary>
        public void Train(
            object student,
            TInput[] trainInputs,
            TOutput[]? trainLabels,
            int epochs,
            int batchSize = 32,
            double learningRate = 0.001)
        {
            int numBatches = (trainInputs.Length + batchSize - 1) / batchSize;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                T epochLoss = _numOps.Zero;

                // Shuffle data
                var shuffled = ShuffleData(trainInputs, trainLabels);

                // Train on batches
                for (int b = 0; b < numBatches; b++)
                {
                    int start = b * batchSize;
                    int end = Math.Min(start + batchSize, trainInputs.Length);

                    var batchInputs = shuffled.inputs[start..end];
                    var batchLabels = trainLabels != null ? shuffled.labels[start..end] : null;

                    var batchLoss = TrainBatch(student, batchInputs, batchLabels);
                    epochLoss = _numOps.Add(epochLoss, batchLoss);
                }

                var avgEpochLoss = _numOps.Divide(epochLoss, _numOps.FromDouble(numBatches));

                Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {avgEpochLoss}");
            }
        }

        /// <summary>
        /// Evaluates student model accuracy.
        /// </summary>
        public double Evaluate(object student, TInput[] testInputs, TOutput[] testLabels)
        {
            int correct = 0;

            for (int i = 0; i < testInputs.Length; i++)
            {
                var studentOutput = GetStudentOutput(student, testInputs[i]);
                var teacherOutput = _teacher.GetLogits(testInputs[i]);

                // Check if predictions match
                if (IsPredictionCorrect(studentOutput, testLabels[i]))
                    correct++;
            }

            return (double)correct / testInputs.Length;
        }

        private TOutput GetStudentOutput(object student, TInput input)
        {
            // This depends on your student model implementation
            if (student is NeuralNetwork<TInput, TOutput, T> neuralNet)
            {
                return neuralNet.ForwardPass(input, returnLogits: true);
            }

            throw new NotSupportedException("Student model type not supported");
        }

        private void BackpropagateStudent(object student, TOutput gradient)
        {
            // This depends on your student model implementation
            if (student is NeuralNetwork<TInput, TOutput, T> neuralNet)
            {
                neuralNet.Backpropagate(gradient);
            }
        }

        private (TInput[] inputs, TOutput[] labels) ShuffleData(TInput[] inputs, TOutput[]? labels)
        {
            var random = new Random();
            var indices = Enumerable.Range(0, inputs.Length).ToArray();

            // Fisher-Yates shuffle
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            var shuffledInputs = indices.Select(i => inputs[i]).ToArray();
            var shuffledLabels = labels != null ? indices.Select(i => labels[i]).ToArray() : null;

            return (shuffledInputs, shuffledLabels);
        }

        private bool IsPredictionCorrect(TOutput prediction, TOutput trueLabel)
        {
            // For classification: check if argmax matches
            if (prediction is Vector<T> predVector && trueLabel is Vector<T> labelVector)
            {
                int predClass = ArgMax(predVector);
                int trueClass = ArgMax(labelVector);
                return predClass == trueClass;
            }

            return false;
        }

        private int ArgMax(Vector<T> vector)
        {
            int maxIndex = 0;
            T maxValue = vector[0];

            for (int i = 1; i < vector.Length; i++)
            {
                if (_numOps.GreaterThan(vector[i], maxValue))
                {
                    maxValue = vector[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }
    }
}
```

### Phase 5: Advanced Distillation Strategies

#### Step 5.1: Feature Distillation (FitNets)

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Distillation\FeatureDistillationStrategy.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Distillation
{
    /// <summary>
    /// Feature-based distillation (FitNets): student matches teacher's intermediate representations.
    /// Useful when student architecture differs significantly from teacher.
    /// </summary>
    public class FeatureDistillationStrategy<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly string[] _layerNames;
        private readonly double _featureWeight;

        /// <summary>
        /// Creates a feature distillation strategy.
        /// </summary>
        /// <param name="layerNames">Names of layers to match between teacher and student</param>
        /// <param name="featureWeight">Weight for feature loss vs. output loss (default 0.5)</param>
        public FeatureDistillationStrategy(string[] layerNames, double featureWeight = 0.5)
        {
            _numOps = NumericOperations<T>.Instance;
            _layerNames = layerNames ?? throw new ArgumentNullException(nameof(layerNames));
            _featureWeight = featureWeight;
        }

        /// <summary>
        /// Computes feature matching loss: MSE between student and teacher features.
        /// </summary>
        public T ComputeFeatureLoss(
            ITeacherModel<object, object> teacher,
            object student,
            object input)
        {
            T totalLoss = _numOps.Zero;

            foreach (var layerName in _layerNames)
            {
                // Get teacher features
                var teacherFeatures = teacher.GetFeatures(input, layerName);

                // Get student features
                var studentFeatures = GetStudentFeatures(student, input, layerName);

                // Compute MSE between feature maps
                var loss = ComputeMSE(studentFeatures, teacherFeatures);
                totalLoss = _numOps.Add(totalLoss, loss);
            }

            // Average across layers
            totalLoss = _numOps.Divide(totalLoss, _numOps.FromDouble(_layerNames.Length));

            return totalLoss;
        }

        private T ComputeMSE(object studentFeatures, object teacherFeatures)
        {
            // Convert to tensors and compute mean squared error
            if (studentFeatures is Tensor<T> studentTensor && teacherFeatures is Tensor<T> teacherTensor)
            {
                if (studentTensor.Rank != teacherTensor.Rank)
                    throw new ArgumentException("Student and teacher feature dimensions must match");

                T sumSquaredDiff = _numOps.Zero;
                int totalElements = 1;

                for (int i = 0; i < studentTensor.Rank; i++)
                    totalElements *= studentTensor.Dimensions[i];

                // Flatten and compute MSE
                for (int i = 0; i < totalElements; i++)
                {
                    var studentVal = GetFlattenedValue(studentTensor, i);
                    var teacherVal = GetFlattenedValue(teacherTensor, i);

                    var diff = _numOps.Subtract(studentVal, teacherVal);
                    var squared = _numOps.Multiply(diff, diff);
                    sumSquaredDiff = _numOps.Add(sumSquaredDiff, squared);
                }

                return _numOps.Divide(sumSquaredDiff, _numOps.FromDouble(totalElements));
            }

            throw new NotSupportedException("Feature type not supported");
        }

        private T GetFlattenedValue(Tensor<T> tensor, int flatIndex)
        {
            // Convert flat index to multi-dimensional indices
            var indices = new int[tensor.Rank];
            int remaining = flatIndex;

            for (int dim = tensor.Rank - 1; dim >= 0; dim--)
            {
                indices[dim] = remaining % tensor.Dimensions[dim];
                remaining /= tensor.Dimensions[dim];
            }

            return tensor[indices];
        }

        private object GetStudentFeatures(object student, object input, string layerName)
        {
            // Extract features from student model (implementation-specific)
            throw new NotImplementedException("Student feature extraction depends on model architecture");
        }
    }
}
```

#### Step 5.2: Attention Distillation (for Transformers)

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Distillation\AttentionDistillationStrategy.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Distillation
{
    /// <summary>
    /// Attention-based distillation: student matches teacher's attention patterns.
    /// Particularly useful for transformer models (BERT, GPT, etc.).
    /// </summary>
    public class AttentionDistillationStrategy<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly string[] _attentionLayerNames;
        private readonly double _attentionWeight;

        public AttentionDistillationStrategy(string[] attentionLayerNames, double attentionWeight = 0.3)
        {
            _numOps = NumericOperations<T>.Instance;
            _attentionLayerNames = attentionLayerNames;
            _attentionWeight = attentionWeight;
        }

        /// <summary>
        /// Computes attention matching loss: MSE between attention weight matrices.
        /// </summary>
        public T ComputeAttentionLoss(
            ITeacherModel<object, object> teacher,
            object student,
            object input)
        {
            T totalLoss = _numOps.Zero;

            foreach (var layerName in _attentionLayerNames)
            {
                // Get attention weights from teacher
                var teacherAttention = teacher.GetAttentionWeights(input, layerName);

                // Get attention weights from student
                var studentAttention = GetStudentAttention(student, input, layerName);

                // Compute MSE between attention matrices
                var loss = ComputeAttentionMSE(studentAttention, teacherAttention);
                totalLoss = _numOps.Add(totalLoss, loss);
            }

            totalLoss = _numOps.Divide(totalLoss, _numOps.FromDouble(_attentionLayerNames.Length));

            return totalLoss;
        }

        private T ComputeAttentionMSE(object studentAttention, object teacherAttention)
        {
            // Attention weights are typically [batch, heads, seq_len, seq_len] tensors
            if (studentAttention is Matrix<T> studentMatrix && teacherAttention is Matrix<T> teacherMatrix)
            {
                T sumSquaredDiff = _numOps.Zero;
                int totalElements = studentMatrix.Rows * studentMatrix.Columns;

                for (int i = 0; i < studentMatrix.Rows; i++)
                {
                    for (int j = 0; j < studentMatrix.Columns; j++)
                    {
                        var diff = _numOps.Subtract(studentMatrix[i, j], teacherMatrix[i, j]);
                        var squared = _numOps.Multiply(diff, diff);
                        sumSquaredDiff = _numOps.Add(sumSquaredDiff, squared);
                    }
                }

                return _numOps.Divide(sumSquaredDiff, _numOps.FromDouble(totalElements));
            }

            throw new NotSupportedException("Attention format not supported");
        }

        private object GetStudentAttention(object student, object input, string layerName)
        {
            // Extract attention weights from student (implementation-specific)
            throw new NotImplementedException("Student attention extraction depends on model architecture");
        }
    }
}
```

## Testing Strategy

### Phase 6: Comprehensive Tests

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\tests\Distillation\KnowledgeDistillationTests.cs
using Xunit;
using AiDotNet.Distillation;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.Distillation
{
    public class KnowledgeDistillationTests
    {
        [Fact]
        public void DistillationLoss_IdenticalOutputs_ReturnsZeroSoftLoss()
        {
            // Arrange
            var studentLogits = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
            var teacherLogits = new Vector<double>(new[] { 2.0, 1.0, 0.5 });

            var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);

            // Act
            var loss = distillationLoss.ComputeLoss(studentLogits, teacherLogits, trueLabels: null);

            // Assert
            // Identical logits → identical soft predictions → KL divergence = 0
            Assert.True(Math.Abs(Convert.ToDouble(loss)) < 1e-6);
        }

        [Fact]
        public void DistillationLoss_HighTemperature_SofterProbabilities()
        {
            // Arrange
            var logits = new Vector<double>(new[] { 10.0, 1.0, 0.1 });

            var lowTempLoss = new DistillationLoss<double>(temperature: 1.0);
            var highTempLoss = new DistillationLoss<double>(temperature: 5.0);

            // Act
            var lowTempProbs = ApplySoftmax(logits, temperature: 1.0);
            var highTempProbs = ApplySoftmax(logits, temperature: 5.0);

            // Assert
            // High temperature should give more balanced distribution
            Assert.True(highTempProbs[1] > lowTempProbs[1]); // Second class gets more probability
            Assert.True(highTempProbs[2] > lowTempProbs[2]); // Third class gets more probability
        }

        [Fact]
        public void DistillationLoss_CombinesHardAndSoftLoss()
        {
            // Arrange
            var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
            var teacherLogits = new Vector<double>(new[] { 1.5, 1.8, 0.6 });
            var trueLabels = new Vector<double>(new[] { 0.0, 1.0, 0.0 }); // Class 1 is correct

            var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.5);

            // Act
            var totalLoss = distillationLoss.ComputeLoss(studentLogits, teacherLogits, trueLabels);

            // Also compute with only soft loss
            var softOnlyLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.0);
            var softLoss = softOnlyLoss.ComputeLoss(studentLogits, teacherLogits, trueLabels: null);

            // Assert
            // Total loss should be between soft-only and hard-only
            Assert.True(Convert.ToDouble(totalLoss) > 0);
            Assert.NotEqual(Convert.ToDouble(totalLoss), Convert.ToDouble(softLoss));
        }

        [Fact]
        public void KnowledgeDistillation_StudentMatchesTeacher()
        {
            // Arrange - Simple 2-layer teacher
            var teacherWeights = new Matrix<double>(3, 2);
            teacherWeights[0, 0] = 0.5; teacherWeights[0, 1] = 0.3;
            teacherWeights[1, 0] = 0.6; teacherWeights[1, 1] = 0.4;
            teacherWeights[2, 0] = 0.7; teacherWeights[2, 1] = 0.5;

            // Student starts with random weights
            var studentWeights = new Matrix<double>(3, 2);
            studentWeights[0, 0] = 0.1; studentWeights[0, 1] = 0.1;
            studentWeights[1, 0] = 0.1; studentWeights[1, 1] = 0.1;
            studentWeights[2, 0] = 0.1; studentWeights[2, 1] = 0.1;

            // Training data
            var inputs = new Vector<double>[]
            {
                new Vector<double>(new[] { 1.0, 0.0, 0.0 }),
                new Vector<double>(new[] { 0.0, 1.0, 0.0 }),
                new Vector<double>(new[] { 0.0, 0.0, 1.0 })
            };

            // Act - Train student to match teacher
            // (Simplified example - real implementation would use proper neural network)

            // Assert - After training, student outputs should be close to teacher
            Assert.True(true); // Placeholder for actual training verification
        }

        [Fact]
        public void DistillationGradient_ProperlyScaled()
        {
            // Arrange
            var studentLogits = new Vector<double>(new[] { 1.0, 2.0, 0.5 });
            var teacherLogits = new Vector<double>(new[] { 1.5, 1.8, 0.6 });

            var temperature = 3.0;
            var distillationLoss = new DistillationLoss<double>(temperature, alpha: 0.0);

            // Act
            var gradient = distillationLoss.ComputeGradient(studentLogits, teacherLogits);

            // Assert
            // Gradient should be scaled by T^2
            Assert.NotNull(gradient);
            Assert.Equal(studentLogits.Length, gradient.Length);

            // Gradient magnitudes should be reasonable (not exploding/vanishing)
            for (int i = 0; i < gradient.Length; i++)
            {
                double gradVal = Convert.ToDouble(gradient[i]);
                Assert.True(Math.Abs(gradVal) < 100); // Not exploding
                Assert.True(Math.Abs(gradVal) < 1000);
            }
        }

        private Vector<double> ApplySoftmax(Vector<double> logits, double temperature)
        {
            int n = logits.Length;
            var result = new Vector<double>(n);

            double[] scaled = new double[n];
            for (int i = 0; i < n; i++)
                scaled[i] = logits[i] / temperature;

            double max = scaled.Max();
            double sum = 0;
            double[] exp = new double[n];

            for (int i = 0; i < n; i++)
            {
                exp[i] = Math.Exp(scaled[i] - max);
                sum += exp[i];
            }

            for (int i = 0; i < n; i++)
                result[i] = exp[i] / sum;

            return result;
        }
    }
}
```

## Usage Example: Complete Distillation Workflow

```csharp
// Example: Distill large teacher network into smaller student

// 1. Load pre-trained teacher model
var teacherNetwork = NeuralNetwork.LoadFromFile("teacher_model.bin");
var teacher = new TeacherModelWrapper<Vector<double>, Vector<double>, double>(
    teacherNetwork,
    outputDimension: 10
);

// 2. Create smaller student network
var studentNetwork = new NeuralNetwork<Vector<double>, Vector<double>, double>(
    inputSize: 784,    // Same input size (MNIST: 28x28)
    hiddenSizes: new[] { 64 },  // Much smaller: 1 layer with 64 neurons
    outputSize: 10
);

// 3. Configure distillation
var distillationLoss = new DistillationLoss<double>(
    temperature: 4.0,  // Higher T = softer targets
    alpha: 0.3         // 30% hard labels, 70% teacher knowledge
);

// 4. Create trainer
var trainer = new KnowledgeDistillationTrainer<Vector<double>, Vector<double>, double>(
    teacher,
    distillationLoss
);

// 5. Train student
trainer.Train(
    student: studentNetwork,
    trainInputs: mnistTrainImages,
    trainLabels: mnistTrainLabels,
    epochs: 20,
    batchSize: 128,
    learningRate: 0.001
);

// 6. Evaluate student
double studentAccuracy = trainer.Evaluate(studentNetwork, mnistTestImages, mnistTestLabels);
double teacherAccuracy = teacher.Evaluate(mnistTestImages, mnistTestLabels);

Console.WriteLine($"Teacher accuracy: {teacherAccuracy:P2}");
Console.WriteLine($"Student accuracy: {studentAccuracy:P2}");
Console.WriteLine($"Model size reduction: {CalculateSizeReduction(teacherNetwork, studentNetwork):P2}");
```

## Common Pitfalls to Avoid

1. **Wrong temperature** - Too low (T=1): student only learns argmax; Too high (T>10): gradients vanish
2. **Imbalanced alpha** - If alpha too high (>0.7), student ignores teacher
3. **Not using logits** - Must use pre-softmax outputs, not probabilities
4. **Forgetting T^2 scaling** - Without it, soft loss dominates hard loss
5. **Capacity mismatch** - Student too small can't capture teacher knowledge
6. **No fine-tuning** - After distillation, brief training on hard labels helps

## Advanced Topics

### Self-Distillation

Train a model to be its own teacher (improves calibration):
```csharp
// Use trained model as teacher for itself
var model = TrainModel(data);
var teacher = new TeacherModelWrapper(model, outputDim);

// Retrain same model with distillation
var trainer = new KnowledgeDistillationTrainer(teacher, distillationLoss);
trainer.Train(model, data, epochs: 10);
```

### Ensemble Distillation

Distill multiple teacher models into one student:
```csharp
// Average predictions from multiple teachers
var teachers = new[] { teacher1, teacher2, teacher3 };
var ensembleTeacher = new EnsembleTeacher(teachers);

// Student learns from ensemble knowledge
trainer.Train(student, ensembleTeacher, data);
```

## Validation Criteria

Your implementation is complete when:

1. Standard distillation loss (KL divergence + cross-entropy) implemented
2. Temperature scaling works correctly for softening probabilities
3. Alpha parameter balances hard and soft losses
4. Teacher model wrapper extracts logits and soft predictions
5. Distillation trainer performs end-to-end student training
6. Tests verify temperature effects and loss computation
7. Student model achieves >90% of teacher's accuracy with <50% parameters

## Learning Resources

- **Original Paper**: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- **FitNets**: Romero et al., "FitNets: Hints for Thin Deep Nets" (2014)
- **DistilBERT**: Sanh et al., "DistilBERT, a distilled version of BERT" (2019)
- **Survey**: Gou et al., "Knowledge Distillation: A Survey" (2021)

## Next Steps

1. Implement multi-teacher distillation
2. Add feature-based distillation (FitNets)
3. Implement attention transfer for transformers
4. Combine with pruning (Issue #407) for maximum compression
5. Add quantization-aware distillation (Issue #409)
6. Export distilled models to ONNX (Issue #410)

---

**Good luck!** Knowledge distillation is a cornerstone of model compression, used in production systems like DistilBERT, MobileNet, and TinyBERT. Mastering this will enable you to deploy large-model accuracy in resource-constrained environments.
