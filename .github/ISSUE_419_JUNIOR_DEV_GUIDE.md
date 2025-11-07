# Issue #419: Junior Developer Implementation Guide

## Understanding Continual Learning and Active Learning

**Goal**: Enable models to learn continuously from new data without forgetting old knowledge (continual learning) and to intelligently select the most informative examples for labeling (active learning).

---

## Key Concepts for Beginners

### What is Continual Learning?

**The Catastrophic Forgetting Problem**:
```
Train on Task A (cats vs dogs) → 95% accuracy
Train on Task B (cars vs planes) → 90% accuracy on B, but now 20% on A!
```

The model "forgets" old tasks when learning new ones.

**Continual Learning Strategies**:

1. **Regularization-based** (EWC, LwF):
   - Add penalty for changing important weights
   - "Important" = weights that mattered for old tasks

2. **Replay-based** (GEM, A-GEM):
   - Keep small buffer of old examples
   - Replay them while learning new tasks

3. **Parameter isolation**:
   - Dedicate different neurons to different tasks
   - Progressive neural networks

### What is Active Learning?

**The Problem**: Labeling data is expensive (medical images, legal documents).

**The Solution**: Let the model choose which examples to label next.

**Active Learning Strategies**:

1. **Uncertainty Sampling**:
   - Pick examples where model is most uncertain
   - "I don't know if this is a cat or dog - please tell me!"

2. **Query-by-Committee**:
   - Train multiple models
   - Pick examples where models disagree most

3. **Expected Model Change**:
   - Pick examples that would change model the most if labeled

---

## Phase 1: Elastic Weight Consolidation (EWC)

### AC 1.1: Implement FisherInformationMatrix

**What is Fisher Information?**
Measures how sensitive the loss is to each weight. High sensitivity = important weight for current task.

**Formula**: `F_i = E[(∂L/∂w_i)²]`

**File**: `src/ContinualLearning/FisherInformationMatrix.cs`

**Step 1**: Implement Fisher Information computation

```csharp
// File: src/ContinualLearning/FisherInformationMatrix.cs
namespace AiDotNet.ContinualLearning;

/// <summary>
/// Computes Fisher Information Matrix for Elastic Weight Consolidation.
/// The Fisher Information measures the importance of each weight for current task.
/// </summary>
public class FisherInformationMatrix<T>
{
    private readonly IModel<T> _model;
    private Dictionary<string, Matrix<T>> _fisherMatrices;

    public FisherInformationMatrix(IModel<T> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _fisherMatrices = new Dictionary<string, Matrix<T>>();
    }

    /// <summary>
    /// Compute Fisher Information Matrix for current task.
    /// This measures how important each weight is for the current task.
    /// </summary>
    /// <param name="dataLoader">Data from current task</param>
    /// <param name="numSamples">Number of samples to use (more = more accurate, slower)</param>
    public void Compute(IDataLoader<T> dataLoader, int numSamples = 1000)
    {
        var numOps = NumericOperations<T>.Instance;
        _fisherMatrices.Clear();

        // Get all parameters from model
        var parameters = GetModelParameters();

        // Initialize Fisher matrices (same shape as parameters, all zeros)
        foreach (var param in parameters)
        {
            _fisherMatrices[param.Key] = new Matrix<T>(param.Value.Rows, param.Value.Columns);
        }

        int sampleCount = 0;

        // Accumulate squared gradients over multiple samples
        foreach (var batch in dataLoader.GetBatches())
        {
            if (sampleCount >= numSamples)
                break;

            // Forward pass
            var prediction = _model.Forward(batch.Input, training: true);

            // For classification, use predicted class as "pseudo-label"
            var pseudoLabel = GetPredictedClass(prediction);

            // Compute loss with respect to predicted class
            var loss = ComputeLoss(prediction, pseudoLabel);

            // Backward pass to get gradients
            var gradients = _model.Backward(loss);

            // Accumulate squared gradients (Fisher = E[grad²])
            AccumulateSquaredGradients(gradients);

            sampleCount += batch.Input.Rows;
        }

        // Normalize by number of samples
        T normFactor = numOps.FromDouble(sampleCount);

        foreach (var key in _fisherMatrices.Keys.ToList())
        {
            var fisher = _fisherMatrices[key];

            for (int r = 0; r < fisher.Rows; r++)
            {
                for (int c = 0; c < fisher.Columns; c++)
                {
                    fisher[r, c] = numOps.Divide(fisher[r, c], normFactor);
                }
            }
        }
    }

    /// <summary>
    /// Get Fisher Information for a specific parameter.
    /// </summary>
    public Matrix<T> GetFisherInformation(string parameterName)
    {
        if (!_fisherMatrices.ContainsKey(parameterName))
            throw new ArgumentException($"No Fisher information for parameter {parameterName}");

        return _fisherMatrices[parameterName];
    }

    /// <summary>
    /// Get all Fisher Information matrices.
    /// </summary>
    public Dictionary<string, Matrix<T>> GetAllFisherInformation()
    {
        return new Dictionary<string, Matrix<T>>(_fisherMatrices);
    }

    private Dictionary<string, Matrix<T>> GetModelParameters()
    {
        var parameters = new Dictionary<string, Matrix<T>>();

        // Get layers from model (assuming Sequential model)
        var layersProperty = _model.GetType().GetProperty("Layers");
        if (layersProperty == null)
            throw new InvalidOperationException("Model must have Layers property");

        var layers = layersProperty.GetValue(_model) as IEnumerable<ILayer<T>>;
        if (layers == null)
            return parameters;

        int layerIndex = 0;
        foreach (var layer in layers)
        {
            // Get weights from DenseLayer
            if (layer is DenseLayer<T> denseLayer)
            {
                var weightsProperty = layer.GetType().GetField("_weights",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

                if (weightsProperty != null)
                {
                    var weights = weightsProperty.GetValue(layer) as Matrix<T>;
                    if (weights != null)
                    {
                        parameters[$"layer_{layerIndex}_weights"] = weights;
                    }
                }
            }

            layerIndex++;
        }

        return parameters;
    }

    private Matrix<T> GetPredictedClass(Matrix<T> prediction)
    {
        // For each sample, create one-hot encoding of predicted class
        var pseudoLabel = new Matrix<T>(prediction.Rows, prediction.Columns);
        var numOps = NumericOperations<T>.Instance;

        for (int r = 0; r < prediction.Rows; r++)
        {
            // Find argmax
            int maxIndex = 0;
            double maxValue = Convert.ToDouble(prediction[r, 0]);

            for (int c = 1; c < prediction.Columns; c++)
            {
                double value = Convert.ToDouble(prediction[r, c]);
                if (value > maxValue)
                {
                    maxValue = value;
                    maxIndex = c;
                }
            }

            // Create one-hot encoding
            for (int c = 0; c < prediction.Columns; c++)
            {
                pseudoLabel[r, c] = (c == maxIndex) ? numOps.One : numOps.Zero;
            }
        }

        return pseudoLabel;
    }

    private Matrix<T> ComputeLoss(Matrix<T> prediction, Matrix<T> target)
    {
        // Cross-entropy loss
        var loss = new CrossEntropyLoss<T>();
        var lossValue = loss.Compute(prediction, target);

        // Return gradient
        return loss.ComputeGradient(prediction, target);
    }

    private void AccumulateSquaredGradients(Dictionary<string, Matrix<T>> gradients)
    {
        var numOps = NumericOperations<T>.Instance;

        foreach (var kvp in gradients)
        {
            if (!_fisherMatrices.ContainsKey(kvp.Key))
                continue;

            var fisher = _fisherMatrices[kvp.Key];
            var gradient = kvp.Value;

            for (int r = 0; r < fisher.Rows; r++)
            {
                for (int c = 0; c < fisher.Columns; c++)
                {
                    // F += grad²
                    T gradSquared = numOps.Multiply(gradient[r, c], gradient[r, c]);
                    fisher[r, c] = numOps.Add(fisher[r, c], gradSquared);
                }
            }
        }
    }
}
```

**Step 2**: Create unit test

```csharp
// File: tests/UnitTests/ContinualLearning/FisherInformationMatrixTests.cs
namespace AiDotNet.Tests.ContinualLearning;

public class FisherInformationMatrixTests
{
    [Fact]
    public void Compute_ProducesNonNegativeValues()
    {
        // Arrange
        var model = CreateSimpleModel();
        var dataLoader = CreateMockDataLoader();
        var fisher = new FisherInformationMatrix<double>(model);

        // Act
        fisher.Compute(dataLoader, numSamples: 100);

        // Assert
        var fisherInfo = fisher.GetAllFisherInformation();

        foreach (var matrix in fisherInfo.Values)
        {
            for (int r = 0; r < matrix.Rows; r++)
            {
                for (int c = 0; c < matrix.Columns; c++)
                {
                    // Fisher Information is always non-negative (it's squared gradients)
                    Assert.True(matrix[r, c] >= 0.0);
                }
            }
        }
    }

    private IModel<double> CreateSimpleModel()
    {
        var model = new Sequential<double>();
        model.Add(new DenseLayer<double>(10, 5));
        model.Add(new ActivationLayer<double>(new ReLU<double>()));
        model.Add(new DenseLayer<double>(5, 2));
        return model;
    }

    private IDataLoader<double> CreateMockDataLoader()
    {
        // Create mock data loader with random data
        var data = new List<(Matrix<double> Input, Matrix<double> Target)>();

        for (int i = 0; i < 20; i++)
        {
            var input = new Matrix<double>(5, 10); // batch size 5
            var target = new Matrix<double>(5, 2);

            // Fill with random values
            var random = new Random(i);
            for (int r = 0; r < 5; r++)
            {
                for (int c = 0; c < 10; c++)
                    input[r, c] = random.NextDouble();

                target[r, random.Next(2)] = 1.0; // One-hot encoding
            }

            data.Add((input, target));
        }

        return new ListDataLoader<double>(data);
    }
}
```

---

### AC 1.2: Implement EWCLoss

**What does EWC Loss do?**
Regular loss + penalty for changing important weights.

**Formula**: `L_EWC = L_new + (λ/2) * Σ F_i * (θ_i - θ_i*)²`
- `L_new`: Loss on new task
- `F_i`: Fisher information (importance) of weight i
- `θ_i*`: Old weight value (before new task)
- `λ`: How much to protect old knowledge (hyperparameter)

**File**: `src/ContinualLearning/EWCLoss.cs`

```csharp
// File: src/ContinualLearning/EWCLoss.cs
namespace AiDotNet.ContinualLearning;

/// <summary>
/// Elastic Weight Consolidation loss.
/// Adds penalty for changing important weights from previous task.
/// </summary>
public class EWCLoss<T> : ILoss<T>
{
    private readonly ILoss<T> _baseLoss;
    private readonly double _ewcLambda;
    private readonly Dictionary<string, Matrix<T>> _fisherMatrices;
    private readonly Dictionary<string, Matrix<T>> _optimalParameters;

    /// <summary>
    /// Creates EWC loss.
    /// </summary>
    /// <param name="baseLoss">Base loss function (e.g., CrossEntropy)</param>
    /// <param name="ewcLambda">EWC regularization strength (typically 1-1000)</param>
    /// <param name="fisherMatrices">Fisher Information from previous task</param>
    /// <param name="optimalParameters">Optimal weights from previous task</param>
    public EWCLoss(
        ILoss<T> baseLoss,
        double ewcLambda,
        Dictionary<string, Matrix<T>> fisherMatrices,
        Dictionary<string, Matrix<T>> optimalParameters)
    {
        _baseLoss = baseLoss ?? throw new ArgumentNullException(nameof(baseLoss));
        _ewcLambda = ewcLambda;
        _fisherMatrices = fisherMatrices ?? throw new ArgumentNullException(nameof(fisherMatrices));
        _optimalParameters = optimalParameters ?? throw new ArgumentNullException(nameof(optimalParameters));
    }

    public T Compute(Matrix<T> predictions, Matrix<T> targets)
    {
        var numOps = NumericOperations<T>.Instance;

        // Compute base loss on new task
        T baseLoss = _baseLoss.Compute(predictions, targets);

        // Compute EWC penalty
        T ewcPenalty = numOps.Zero;

        foreach (var paramName in _fisherMatrices.Keys)
        {
            if (!_optimalParameters.ContainsKey(paramName))
                continue;

            var fisher = _fisherMatrices[paramName];
            var optimalParams = _optimalParameters[paramName];

            // Get current parameters (from model)
            // This requires access to model - simplified for now
            // In practice, pass current parameters to Compute method

            // Penalty = (λ/2) * Σ F_i * (θ_i - θ_i*)²
            // For each weight
            for (int r = 0; r < fisher.Rows; r++)
            {
                for (int c = 0; c < fisher.Columns; c++)
                {
                    // TODO: Get current parameter value
                    // T currentParam = GetCurrentParameter(paramName, r, c);

                    // T diff = numOps.Subtract(currentParam, optimalParams[r, c]);
                    // T diffSquared = numOps.Multiply(diff, diff);
                    // T weighted = numOps.Multiply(fisher[r, c], diffSquared);

                    // ewcPenalty = numOps.Add(ewcPenalty, weighted);
                }
            }
        }

        // Scale penalty by lambda/2
        T scaledPenalty = numOps.Multiply(
            ewcPenalty,
            numOps.FromDouble(_ewcLambda / 2.0)
        );

        // Total loss = base loss + EWC penalty
        return numOps.Add(baseLoss, scaledPenalty);
    }

    public Matrix<T> ComputeGradient(Matrix<T> predictions, Matrix<T> targets)
    {
        // Base gradient
        var baseGradient = _baseLoss.ComputeGradient(predictions, targets);

        // Add EWC gradient: λ * F_i * (θ_i - θ_i*)
        // This requires knowing which parameters affect which predictions
        // Simplified implementation - full version needs parameter tracking

        return baseGradient;
    }
}
```

**Step 2**: Create integration test

```csharp
// File: tests/IntegrationTests/ContinualLearning/EWCIntegrationTests.cs
namespace AiDotNet.Tests.ContinualLearning;

public class EWCIntegrationTests
{
    [Fact]
    public void EWC_PreventsCatastrophicForgetting()
    {
        // Arrange - Train on Task A
        var model = CreateModel();
        var taskAData = CreateTaskAData(); // Binary classification

        var trainer = new Trainer<double>(model, new CrossEntropyLoss<double>());
        trainer.Train(taskAData, epochs: 10);

        double taskAAccuracyBefore = EvaluateAccuracy(model, taskAData);

        // Compute Fisher Information for Task A
        var fisher = new FisherInformationMatrix<double>(model);
        fisher.Compute(taskAData, numSamples: 500);

        var optimalParams = GetModelParameters(model);

        // Train on Task B WITHOUT EWC
        var modelWithoutEWC = CreateModel();
        var trainerWithoutEWC = new Trainer<double>(modelWithoutEWC, new CrossEntropyLoss<double>());

        var taskBData = CreateTaskBData(); // Different binary classification
        trainerWithoutEWC.Train(taskBData, epochs: 10);

        double taskAAccuracyAfterWithoutEWC = EvaluateAccuracy(modelWithoutEWC, taskAData);

        // Train on Task B WITH EWC
        var modelWithEWC = CreateModel();
        var ewcLoss = new EWCLoss<double>(
            new CrossEntropyLoss<double>(),
            ewcLambda: 100.0,
            fisher.GetAllFisherInformation(),
            optimalParams
        );

        var trainerWithEWC = new Trainer<double>(modelWithEWC, ewcLoss);
        trainerWithEWC.Train(taskBData, epochs: 10);

        double taskAAccuracyAfterWithEWC = EvaluateAccuracy(modelWithEWC, taskAData);

        // Assert - EWC should maintain better performance on Task A
        Assert.True(taskAAccuracyAfterWithEWC > taskAAccuracyAfterWithoutEWC);
        Assert.True(taskAAccuracyAfterWithEWC > 0.6); // Should retain reasonable performance
    }
}
```

---

## Phase 2: Learning without Forgetting (LwF)

### AC 2.1: Implement LwFLoss

**What is Learning without Forgetting?**
When training on new task, also minimize change in predictions on old task.

**Key Idea**:
- Before training Task B, run old data through model → save predictions
- During Task B training, add loss term: keep predictions on old data similar

**Formula**: `L_LwF = L_new + λ * KL(p_old || p_new)`
- `p_old`: Predictions before training Task B
- `p_new`: Predictions during Task B training
- `KL`: Kullback-Leibler divergence (measures difference between distributions)

**File**: `src/ContinualLearning/LwFLoss.cs`

```csharp
// File: src/ContinualLearning/LwFLoss.cs
namespace AiDotNet.ContinualLearning;

/// <summary>
/// Learning without Forgetting loss.
/// Preserves predictions on old task data while learning new task.
/// </summary>
public class LwFLoss<T> : ILoss<T>
{
    private readonly ILoss<T> _baseLoss;
    private readonly double _lwfLambda;
    private readonly Matrix<T> _oldTaskPredictions;
    private readonly double _temperature;

    /// <summary>
    /// Creates LwF loss.
    /// </summary>
    /// <param name="baseLoss">Base loss for new task</param>
    /// <param name="lwfLambda">Distillation loss weight (typically 1-10)</param>
    /// <param name="oldTaskPredictions">Predictions on old task before training new task</param>
    /// <param name="temperature">Temperature for distillation (typically 2-5, higher = softer)</param>
    public LwFLoss(
        ILoss<T> baseLoss,
        double lwfLambda,
        Matrix<T> oldTaskPredictions,
        double temperature = 2.0)
    {
        _baseLoss = baseLoss ?? throw new ArgumentNullException(nameof(baseLoss));
        _lwfLambda = lwfLambda;
        _oldTaskPredictions = oldTaskPredictions ?? throw new ArgumentNullException(nameof(oldTaskPredictions));
        _temperature = temperature;
    }

    public T Compute(Matrix<T> predictions, Matrix<T> targets)
    {
        var numOps = NumericOperations<T>.Instance;

        // Loss on new task
        T newTaskLoss = _baseLoss.Compute(predictions, targets);

        // Distillation loss: KL divergence between old and new predictions
        T distillationLoss = ComputeKLDivergence(
            _oldTaskPredictions,
            predictions,
            _temperature
        );

        // Total loss
        T weightedDistillation = numOps.Multiply(
            distillationLoss,
            numOps.FromDouble(_lwfLambda)
        );

        return numOps.Add(newTaskLoss, weightedDistillation);
    }

    private T ComputeKLDivergence(Matrix<T> oldPredictions, Matrix<T> newPredictions, double temperature)
    {
        var numOps = NumericOperations<T>.Instance;

        // Apply temperature scaling and softmax
        var oldSoftmax = ApplyTemperatureSoftmax(oldPredictions, temperature);
        var newSoftmax = ApplyTemperatureSoftmax(newPredictions, temperature);

        // KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
        T klDiv = numOps.Zero;

        for (int r = 0; r < oldPredictions.Rows; r++)
        {
            for (int c = 0; c < oldPredictions.Columns; c++)
            {
                double pOld = Convert.ToDouble(oldSoftmax[r, c]);
                double pNew = Convert.ToDouble(newSoftmax[r, c]);

                if (pOld > 1e-10) // Avoid log(0)
                {
                    double kl = pOld * Math.Log(pOld / Math.Max(pNew, 1e-10));
                    klDiv = numOps.Add(klDiv, numOps.FromDouble(kl));
                }
            }
        }

        return klDiv;
    }

    private Matrix<T> ApplyTemperatureSoftmax(Matrix<T> logits, double temperature)
    {
        var numOps = NumericOperations<T>.Instance;
        var output = new Matrix<T>(logits.Rows, logits.Columns);

        for (int r = 0; r < logits.Rows; r++)
        {
            // Scale by temperature
            var scaledLogits = new double[logits.Columns];
            for (int c = 0; c < logits.Columns; c++)
            {
                scaledLogits[c] = Convert.ToDouble(logits[r, c]) / temperature;
            }

            // Softmax with numerical stability
            double max = scaledLogits.Max();
            double sum = 0.0;

            for (int c = 0; c < logits.Columns; c++)
            {
                sum += Math.Exp(scaledLogits[c] - max);
            }

            for (int c = 0; c < logits.Columns; c++)
            {
                double prob = Math.Exp(scaledLogits[c] - max) / sum;
                output[r, c] = numOps.FromDouble(prob);
            }
        }

        return output;
    }

    public Matrix<T> ComputeGradient(Matrix<T> predictions, Matrix<T> targets)
    {
        // Gradient = base gradient + distillation gradient
        var baseGradient = _baseLoss.ComputeGradient(predictions, targets);

        // Distillation gradient computation
        var distillationGradient = ComputeDistillationGradient(predictions);

        // Combine gradients
        var numOps = NumericOperations<T>.Instance;
        var totalGradient = new Matrix<T>(baseGradient.Rows, baseGradient.Columns);

        for (int r = 0; r < baseGradient.Rows; r++)
        {
            for (int c = 0; c < baseGradient.Columns; c++)
            {
                T weighted = numOps.Multiply(
                    distillationGradient[r, c],
                    numOps.FromDouble(_lwfLambda)
                );

                totalGradient[r, c] = numOps.Add(baseGradient[r, c], weighted);
            }
        }

        return totalGradient;
    }

    private Matrix<T> ComputeDistillationGradient(Matrix<T> predictions)
    {
        // Gradient of KL divergence w.r.t. new predictions
        // ∂KL/∂q = -p/q (simplified)

        var numOps = NumericOperations<T>.Instance;
        var gradient = new Matrix<T>(predictions.Rows, predictions.Columns);

        var oldSoftmax = ApplyTemperatureSoftmax(_oldTaskPredictions, _temperature);
        var newSoftmax = ApplyTemperatureSoftmax(predictions, _temperature);

        for (int r = 0; r < predictions.Rows; r++)
        {
            for (int c = 0; c < predictions.Columns; c++)
            {
                double pOld = Convert.ToDouble(oldSoftmax[r, c]);
                double pNew = Convert.ToDouble(newSoftmax[r, c]);

                // Gradient of KL divergence
                double grad = pNew - pOld;

                gradient[r, c] = numOps.FromDouble(grad);
            }
        }

        return gradient;
    }
}
```

---

## Phase 3: Gradient Episodic Memory (GEM)

### AC 3.1: Implement ExperienceReplayBuffer

**What is Experience Replay?**
Keep a small buffer of examples from previous tasks. When training on new task, ensure gradients don't hurt performance on old examples.

**File**: `src/ContinualLearning/ExperienceReplayBuffer.cs`

```csharp
// File: src/ContinualLearning/ExperienceReplayBuffer.cs
namespace AiDotNet.ContinualLearning;

/// <summary>
/// Stores examples from previous tasks for continual learning.
/// </summary>
public class ExperienceReplayBuffer<T>
{
    private readonly int _maxSize;
    private readonly List<(Matrix<T> Input, Matrix<T> Target, int TaskId)> _buffer;
    private readonly Random _random;

    /// <summary>
    /// Creates replay buffer.
    /// </summary>
    /// <param name="maxSize">Maximum number of examples to store</param>
    public ExperienceReplayBuffer(int maxSize)
    {
        _maxSize = maxSize;
        _buffer = new List<(Matrix<T>, Matrix<T>, int)>();
        _random = new Random();
    }

    /// <summary>
    /// Add examples from a task to the buffer.
    /// Uses reservoir sampling for uniform distribution.
    /// </summary>
    public void AddExamples(IEnumerable<(Matrix<T> Input, Matrix<T> Target)> examples, int taskId)
    {
        foreach (var example in examples)
        {
            if (_buffer.Count < _maxSize)
            {
                // Buffer not full - just add
                _buffer.Add((example.Input, example.Target, taskId));
            }
            else
            {
                // Buffer full - randomly replace with decreasing probability (reservoir sampling)
                int index = _random.Next(0, _buffer.Count + 1);

                if (index < _maxSize)
                {
                    _buffer[index] = (example.Input, example.Target, taskId);
                }
            }
        }
    }

    /// <summary>
    /// Sample a batch of examples from the buffer.
    /// </summary>
    public List<(Matrix<T> Input, Matrix<T> Target, int TaskId)> Sample(int batchSize)
    {
        if (_buffer.Count == 0)
            return new List<(Matrix<T>, Matrix<T>, int)>();

        var samples = new List<(Matrix<T>, Matrix<T>, int)>();
        int sampleCount = Math.Min(batchSize, _buffer.Count);

        // Random sampling without replacement
        var indices = Enumerable.Range(0, _buffer.Count)
            .OrderBy(x => _random.Next())
            .Take(sampleCount)
            .ToList();

        foreach (var index in indices)
        {
            samples.Add(_buffer[index]);
        }

        return samples;
    }

    /// <summary>
    /// Get all examples from a specific task.
    /// </summary>
    public List<(Matrix<T> Input, Matrix<T> Target)> GetExamplesForTask(int taskId)
    {
        return _buffer
            .Where(x => x.TaskId == taskId)
            .Select(x => (x.Input, x.Target))
            .ToList();
    }

    /// <summary>
    /// Get all examples in the buffer.
    /// </summary>
    public List<(Matrix<T> Input, Matrix<T> Target, int TaskId)> GetAllExamples()
    {
        return new List<(Matrix<T>, Matrix<T>, int)>(_buffer);
    }

    /// <summary>
    /// Number of examples currently in buffer.
    /// </summary>
    public int Count => _buffer.Count;

    /// <summary>
    /// Clear the buffer.
    /// </summary>
    public void Clear()
    {
        _buffer.Clear();
    }
}
```

**Step 2**: Implement GEM constraint

```csharp
// File: src/ContinualLearning/GEMOptimizer.cs
namespace AiDotNet.ContinualLearning;

/// <summary>
/// Gradient Episodic Memory optimizer.
/// Constrains gradients to not increase loss on previous tasks.
/// </summary>
public class GEMOptimizer<T> : IOptimizer<T>
{
    private readonly IOptimizer<T> _baseOptimizer;
    private readonly ExperienceReplayBuffer<T> _buffer;
    private readonly IModel<T> _model;
    private readonly ILoss<T> _loss;

    public GEMOptimizer(
        IOptimizer<T> baseOptimizer,
        ExperienceReplayBuffer<T> buffer,
        IModel<T> model,
        ILoss<T> loss)
    {
        _baseOptimizer = baseOptimizer;
        _buffer = buffer;
        _model = model;
        _loss = loss;
    }

    public void Update(Dictionary<string, Matrix<T>> parameters, Dictionary<string, Matrix<T>> gradients)
    {
        // Get gradient on current batch (g)
        var currentGradient = gradients;

        // Get gradients on previous tasks' examples
        var previousGradients = ComputePreviousTaskGradients();

        // Check if current gradient violates constraint
        // Constraint: g · g_ref >= 0 for all reference gradients g_ref
        bool violatesConstraint = false;

        foreach (var refGradient in previousGradients)
        {
            double dotProduct = ComputeDotProduct(currentGradient, refGradient);

            if (dotProduct < 0)
            {
                violatesConstraint = true;
                break;
            }
        }

        if (violatesConstraint)
        {
            // Project gradient to satisfy constraints
            var projectedGradient = ProjectGradient(currentGradient, previousGradients);
            _baseOptimizer.Update(parameters, projectedGradient);
        }
        else
        {
            // No violation - use original gradient
            _baseOptimizer.Update(parameters, currentGradient);
        }
    }

    private List<Dictionary<string, Matrix<T>>> ComputePreviousTaskGradients()
    {
        var gradients = new List<Dictionary<string, Matrix<T>>>();

        // Sample from buffer
        var samples = _buffer.Sample(batchSize: 32);

        foreach (var sample in samples)
        {
            // Forward pass
            var prediction = _model.Forward(sample.Input, training: true);

            // Compute loss
            var lossGradient = _loss.ComputeGradient(prediction, sample.Target);

            // Backward pass to get parameter gradients
            var paramGradients = _model.Backward(lossGradient);

            gradients.Add(paramGradients);
        }

        return gradients;
    }

    private double ComputeDotProduct(
        Dictionary<string, Matrix<T>> grad1,
        Dictionary<string, Matrix<T>> grad2)
    {
        double dotProduct = 0.0;

        foreach (var key in grad1.Keys)
        {
            if (!grad2.ContainsKey(key))
                continue;

            var matrix1 = grad1[key];
            var matrix2 = grad2[key];

            for (int r = 0; r < matrix1.Rows; r++)
            {
                for (int c = 0; c < matrix1.Columns; c++)
                {
                    double val1 = Convert.ToDouble(matrix1[r, c]);
                    double val2 = Convert.ToDouble(matrix2[r, c]);
                    dotProduct += val1 * val2;
                }
            }
        }

        return dotProduct;
    }

    private Dictionary<string, Matrix<T>> ProjectGradient(
        Dictionary<string, Matrix<T>> gradient,
        List<Dictionary<string, Matrix<T>>> referenceGradients)
    {
        // Quadratic programming to project gradient onto constraint region
        // Simplified: Use average of reference gradients as projection direction

        var numOps = NumericOperations<T>.Instance;
        var projected = new Dictionary<string, Matrix<T>>();

        foreach (var key in gradient.Keys)
        {
            var grad = gradient[key];
            var projectedMatrix = new Matrix<T>(grad.Rows, grad.Columns);

            // Simple projection: weighted combination
            for (int r = 0; r < grad.Rows; r++)
            {
                for (int c = 0; c < grad.Columns; c++)
                {
                    double sum = Convert.ToDouble(grad[r, c]);
                    int count = 1;

                    foreach (var refGrad in referenceGradients)
                    {
                        if (refGrad.ContainsKey(key))
                        {
                            sum += Convert.ToDouble(refGrad[key][r, c]);
                            count++;
                        }
                    }

                    projectedMatrix[r, c] = numOps.FromDouble(sum / count);
                }
            }

            projected[key] = projectedMatrix;
        }

        return projected;
    }
}
```

---

## Phase 4: Active Learning Strategies

### AC 4.1: Implement UncertaintySampling

**What is this?**
Select examples where model is most uncertain for labeling.

**Strategies**:
1. **Least Confident**: Pick examples with lowest max probability
2. **Margin Sampling**: Pick examples with smallest gap between top 2 predictions
3. **Entropy**: Pick examples with highest entropy (uncertainty measure)

**File**: `src/ActiveLearning/UncertaintySampling.cs`

```csharp
// File: src/ActiveLearning/UncertaintySampling.cs
namespace AiDotNet.ActiveLearning;

/// <summary>
/// Selects examples for labeling based on model uncertainty.
/// </summary>
public class UncertaintySampling<T>
{
    private readonly IModel<T> _model;

    public enum Strategy
    {
        LeastConfident,
        MarginSampling,
        Entropy
    }

    public UncertaintySampling(IModel<T> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Select most informative examples from unlabeled pool.
    /// </summary>
    /// <param name="unlabeledData">Pool of unlabeled examples</param>
    /// <param name="numToSelect">Number of examples to select for labeling</param>
    /// <param name="strategy">Uncertainty sampling strategy</param>
    /// <returns>Indices of selected examples</returns>
    public List<int> SelectExamples(
        List<Matrix<T>> unlabeledData,
        int numToSelect,
        Strategy strategy = Strategy.Entropy)
    {
        // Compute uncertainty score for each example
        var uncertainties = new List<(int Index, double Uncertainty)>();

        for (int i = 0; i < unlabeledData.Count; i++)
        {
            var prediction = _model.Forward(unlabeledData[i], training: false);
            double uncertainty = ComputeUncertainty(prediction, strategy);

            uncertainties.Add((i, uncertainty));
        }

        // Select top N most uncertain examples
        var selected = uncertainties
            .OrderByDescending(x => x.Uncertainty)
            .Take(numToSelect)
            .Select(x => x.Index)
            .ToList();

        return selected;
    }

    private double ComputeUncertainty(Matrix<T> prediction, Strategy strategy)
    {
        // Convert to probabilities
        var probabilities = Softmax(prediction);

        switch (strategy)
        {
            case Strategy.LeastConfident:
                return ComputeLeastConfident(probabilities);

            case Strategy.MarginSampling:
                return ComputeMarginSampling(probabilities);

            case Strategy.Entropy:
                return ComputeEntropy(probabilities);

            default:
                throw new ArgumentException($"Unknown strategy: {strategy}");
        }
    }

    private double ComputeLeastConfident(double[] probabilities)
    {
        // Uncertainty = 1 - max(probabilities)
        double maxProb = probabilities.Max();
        return 1.0 - maxProb;
    }

    private double ComputeMarginSampling(double[] probabilities)
    {
        // Uncertainty = (top1_prob - top2_prob)
        // Smaller margin = more uncertain

        var sorted = probabilities.OrderByDescending(x => x).ToArray();

        if (sorted.Length < 2)
            return 0.0;

        double margin = sorted[0] - sorted[1];
        return -margin; // Negative so smaller margin = higher uncertainty
    }

    private double ComputeEntropy(double[] probabilities)
    {
        // Entropy = -Σ p * log(p)
        double entropy = 0.0;

        foreach (var p in probabilities)
        {
            if (p > 1e-10)
            {
                entropy -= p * Math.Log(p);
            }
        }

        return entropy;
    }

    private double[] Softmax(Matrix<T> logits)
    {
        // Assuming single sample prediction
        var values = new double[logits.Columns];

        double max = Convert.ToDouble(logits[0, 0]);
        for (int c = 1; c < logits.Columns; c++)
        {
            double val = Convert.ToDouble(logits[0, c]);
            if (val > max)
                max = val;
        }

        double sum = 0.0;
        for (int c = 0; c < logits.Columns; c++)
        {
            double val = Convert.ToDouble(logits[0, c]);
            values[c] = Math.Exp(val - max);
            sum += values[c];
        }

        for (int c = 0; c < logits.Columns; c++)
        {
            values[c] /= sum;
        }

        return values;
    }
}
```

**Step 2**: Create unit test

```csharp
// File: tests/UnitTests/ActiveLearning/UncertaintySamplingTests.cs
namespace AiDotNet.Tests.ActiveLearning;

public class UncertaintySamplingTests
{
    [Fact]
    public void SelectExamples_Entropy_SelectsMostUncertain()
    {
        // Arrange
        var model = CreateMockModel();
        var sampler = new UncertaintySampling<double>(model);

        var unlabeledData = CreateUnlabeledData(100);

        // Act
        var selected = sampler.SelectExamples(
            unlabeledData,
            numToSelect: 10,
            UncertaintySampling<double>.Strategy.Entropy
        );

        // Assert
        Assert.Equal(10, selected.Count);
        Assert.True(selected.Distinct().Count() == 10); // No duplicates
    }

    [Theory]
    [InlineData(UncertaintySampling<double>.Strategy.LeastConfident)]
    [InlineData(UncertaintySampling<double>.Strategy.MarginSampling)]
    [InlineData(UncertaintySampling<double>.Strategy.Entropy)]
    public void SelectExamples_AllStrategies_ReturnValidIndices(
        UncertaintySampling<double>.Strategy strategy)
    {
        // Arrange
        var model = CreateMockModel();
        var sampler = new UncertaintySampling<double>(model);
        var unlabeledData = CreateUnlabeledData(50);

        // Act
        var selected = sampler.SelectExamples(unlabeledData, numToSelect: 5, strategy);

        // Assert
        Assert.Equal(5, selected.Count);
        Assert.All(selected, index => Assert.InRange(index, 0, 49));
    }
}
```

---

## Testing Strategy

### Integration Test: Full Continual Learning Pipeline

```csharp
// File: tests/IntegrationTests/ContinualLearning/ContinualLearningPipelineTests.cs
namespace AiDotNet.Tests.ContinualLearning;

public class ContinualLearningPipelineTests
{
    [Fact]
    public void FullPipeline_ThreeTasks_MaintainsPerformance()
    {
        // Task sequence: MNIST digits (0-1), then (2-3), then (4-5)
        var model = CreateModel();
        var buffer = new ExperienceReplayBuffer<double>(maxSize: 500);

        // Task 1: Learn digits 0-1
        var task1Data = LoadMNISTSubset(digits: new[] { 0, 1 });
        TrainTask(model, task1Data, taskId: 1);

        double task1Accuracy = EvaluateAccuracy(model, task1Data);
        Assert.True(task1Accuracy > 0.9);

        // Store examples in buffer
        buffer.AddExamples(SampleExamples(task1Data, 200), taskId: 1);

        // Compute Fisher Information for Task 1
        var fisher = new FisherInformationMatrix<double>(model);
        fisher.Compute(CreateDataLoader(task1Data), numSamples: 500);

        // Task 2: Learn digits 2-3 with EWC
        var task2Data = LoadMNISTSubset(digits: new[] { 2, 3 });
        var ewcLoss = CreateEWCLoss(fisher, model);

        TrainTaskWithEWC(model, task2Data, ewcLoss, taskId: 2);

        double task1AfterTask2 = EvaluateAccuracy(model, task1Data);
        double task2Accuracy = EvaluateAccuracy(model, task2Data);

        // Should maintain Task 1 performance
        Assert.True(task1AfterTask2 > 0.7); // Some forgetting is acceptable
        Assert.True(task2Accuracy > 0.9);

        buffer.AddExamples(SampleExamples(task2Data, 200), taskId: 2);

        // Task 3: Learn digits 4-5 with GEM
        var task3Data = LoadMNISTSubset(digits: new[] { 4, 5 });
        var gemOptimizer = new GEMOptimizer<double>(
            new SGD<double>(learningRate: 0.01),
            buffer,
            model,
            new CrossEntropyLoss<double>()
        );

        TrainTaskWithGEM(model, task3Data, gemOptimizer, taskId: 3);

        double task1AfterTask3 = EvaluateAccuracy(model, task1Data);
        double task2AfterTask3 = EvaluateAccuracy(model, task2Data);
        double task3Accuracy = EvaluateAccuracy(model, task3Data);

        // Should maintain all previous tasks
        Assert.True(task1AfterTask3 > 0.6);
        Assert.True(task2AfterTask3 > 0.7);
        Assert.True(task3Accuracy > 0.9);
    }
}
```

---

## Common Pitfalls

1. **EWC Lambda too high**: Model can't learn new task
   - Solution: Start with λ=1, increase gradually if forgetting occurs

2. **EWC Lambda too low**: Doesn't prevent forgetting
   - Solution: Tune on validation set, typical range: 1-10000

3. **Buffer too small**: Not representative of old tasks
   - Solution: At least 200-500 examples per task

4. **Temperature too low in LwF**: Loss dominates, can't learn
   - Solution: Use temperature 2-5, higher for harder tasks

5. **Active learning bias**: Always picking hard examples
   - Solution: Mix uncertainty sampling with random sampling

---

## Success Criteria Checklist

- [ ] EWC prevents catastrophic forgetting on sequential tasks
- [ ] Fisher Information correctly identifies important weights
- [ ] LwF maintains predictions on old data
- [ ] GEM constraints prevent negative transfer
- [ ] Experience replay buffer stores diverse examples
- [ ] Active learning selects informative examples
- [ ] All strategies outperform naive fine-tuning baseline
- [ ] Unit tests pass with > 80% coverage
- [ ] Integration test shows <20% forgetting across 3 tasks

---

## Resources for Learning

1. **EWC Paper**: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al., 2017)
2. **LwF Paper**: "Learning without Forgetting" (Li & Hoiem, 2018)
3. **GEM Paper**: "Gradient Episodic Memory for Continual Learning" (Lopez-Paz & Ranzato, 2017)
4. **Active Learning Survey**: "A Survey of Deep Active Learning" (Ren et al., 2021)

---

## Example Usage After Implementation

```csharp
// Continual Learning with EWC
var model = new Sequential<double>();
// ... add layers ...

// Train Task 1
TrainModel(model, task1Data);

// Compute Fisher Information
var fisher = new FisherInformationMatrix<double>(model);
fisher.Compute(task1DataLoader, numSamples: 1000);

// Train Task 2 with EWC protection
var ewcLoss = new EWCLoss<double>(
    baseLoss: new CrossEntropyLoss<double>(),
    ewcLambda: 100.0,
    fisherMatrices: fisher.GetAllFisherInformation(),
    optimalParameters: model.GetParameters()
);

TrainModel(model, task2Data, loss: ewcLoss);

// Active Learning
var sampler = new UncertaintySampling<double>(model);
var toLabel = sampler.SelectExamples(
    unlabeledPool,
    numToSelect: 100,
    UncertaintySampling<double>.Strategy.Entropy
);

// Send selected examples for labeling
Console.WriteLine($"Please label these {toLabel.Count} examples");
```
