# Issue #421: Junior Developer Implementation Guide

## Understanding Adversarial Robustness and AI Safety

**Goal**: Build defenses against adversarial attacks, implement safety monitoring, model alignment (RLHF), and comprehensive AI safety documentation for production systems.

---

## Key Concepts for Beginners

### What are Adversarial Attacks?

**The Problem**: Carefully crafted inputs that fool neural networks.

**Example - Image Classification**:
```
Original image: Cat (99% confident)
Add tiny noise (invisible to humans): "Dog" (95% confident)
```

The noise is so small humans can't see it, but the model completely changes its prediction.

**Why This Matters**:
- **Security**: Attacker could fool facial recognition, spam filters
- **Safety**: Self-driving car misclassifies stop sign as speed limit
- **Trust**: Model fails on inputs it should handle

### Types of Adversarial Attacks

1. **FGSM (Fast Gradient Sign Method)**:
   - Add noise in direction that increases loss
   - Fast but weak attack

2. **PGD (Projected Gradient Descent)**:
   - Iteratively apply FGSM multiple times
   - Stronger, slower attack

3. **C&W (Carlini & Wagner)**:
   - Optimization-based attack
   - Very strong, finds minimal perturbation

### What is RLHF (Reinforcement Learning from Human Feedback)?

**The Concept**: Train models to align with human preferences.

**Process**:
```
1. Model generates multiple responses
2. Humans rank the responses (best to worst)
3. Train reward model to predict human preferences
4. Use RL to optimize model to maximize predicted reward
```

**Why This Matters**: Prevents models from generating harmful, biased, or unhelpful content.

---

## Phase 1: Adversarial Attack Implementation

### AC 1.1: Implement FGSM Attack

**What is FGSM?**
Fast Gradient Sign Method - adds noise in direction of gradient to maximize loss.

**Formula**: `x_adv = x + ε * sign(∇_x L(x, y))`
- `x`: Original input
- `ε`: Perturbation size (epsilon)
- `∇_x L`: Gradient of loss w.r.t. input
- `sign()`: Take sign of gradient (+1 or -1)

**File**: `src/Adversarial/FGSMAttack.cs`

**Step 1**: Implement FGSM

```csharp
// File: src/Adversarial/FGSMAttack.cs
namespace AiDotNet.Adversarial;

/// <summary>
/// Fast Gradient Sign Method adversarial attack.
/// Generates adversarial examples by adding small perturbations in gradient direction.
/// </summary>
public class FGSMAttack<T>
{
    private readonly IModel<T> _model;
    private readonly ILoss<T> _loss;

    /// <summary>
    /// Creates FGSM attack.
    /// </summary>
    /// <param name="model">Target model to attack</param>
    /// <param name="loss">Loss function to maximize</param>
    public FGSMAttack(IModel<T> model, ILoss<T> loss)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _loss = loss ?? throw new ArgumentNullException(nameof(loss));
    }

    /// <summary>
    /// Generate adversarial example using FGSM.
    /// </summary>
    /// <param name="input">Original input</param>
    /// <param name="target">True label</param>
    /// <param name="epsilon">Perturbation magnitude (typically 0.01-0.3)</param>
    /// <returns>Adversarial example</returns>
    public Matrix<T> GenerateAdversarialExample(
        Matrix<T> input,
        Matrix<T> target,
        double epsilon)
    {
        var numOps = NumericOperations<T>.Instance;

        // Forward pass
        var prediction = _model.Forward(input, training: true);

        // Compute loss
        var lossValue = _loss.Compute(prediction, target);

        // Compute gradient of loss w.r.t. input
        var lossGradient = _loss.ComputeGradient(prediction, target);

        // Backpropagate through model to get gradient w.r.t. input
        var inputGradient = _model.BackwardToInput(lossGradient);

        // Create adversarial example: x_adv = x + epsilon * sign(gradient)
        var adversarial = new Matrix<T>(input.Rows, input.Columns);

        T epsilonValue = numOps.FromDouble(epsilon);

        for (int r = 0; r < input.Rows; r++)
        {
            for (int c = 0; c < input.Columns; c++)
            {
                // Get sign of gradient
                double gradValue = Convert.ToDouble(inputGradient[r, c]);
                double sign = gradValue > 0 ? 1.0 : (gradValue < 0 ? -1.0 : 0.0);

                // Add perturbation
                T perturbation = numOps.Multiply(
                    epsilonValue,
                    numOps.FromDouble(sign)
                );

                adversarial[r, c] = numOps.Add(input[r, c], perturbation);

                // Clip to valid range [0, 1] for images
                double clipped = Math.Max(0.0, Math.Min(1.0, Convert.ToDouble(adversarial[r, c])));
                adversarial[r, c] = numOps.FromDouble(clipped);
            }
        }

        return adversarial;
    }

    /// <summary>
    /// Evaluate attack success rate on a dataset.
    /// </summary>
    public AttackResult EvaluateAttack(
        List<(Matrix<T> Input, Matrix<T> Target)> testData,
        double epsilon)
    {
        int totalSamples = testData.Count;
        int successfulAttacks = 0;
        double avgPerturbation = 0.0;

        foreach (var sample in testData)
        {
            // Generate adversarial example
            var adversarial = GenerateAdversarialExample(sample.Input, sample.Target, epsilon);

            // Check if attack succeeded (model misclassified adversarial example)
            var originalPrediction = _model.Forward(sample.Input, training: false);
            var adversarialPrediction = _model.Forward(adversarial, training: false);

            int originalClass = GetPredictedClass(originalPrediction);
            int adversarialClass = GetPredictedClass(adversarialPrediction);

            if (originalClass != adversarialClass)
            {
                successfulAttacks++;
            }

            // Compute perturbation magnitude
            double perturbationNorm = ComputeL2Norm(sample.Input, adversarial);
            avgPerturbation += perturbationNorm;
        }

        avgPerturbation /= totalSamples;

        return new AttackResult
        {
            TotalSamples = totalSamples,
            SuccessfulAttacks = successfulAttacks,
            SuccessRate = (double)successfulAttacks / totalSamples,
            AveragePerturbationNorm = avgPerturbation
        };
    }

    private int GetPredictedClass(Matrix<T> prediction)
    {
        int maxIndex = 0;
        double maxValue = Convert.ToDouble(prediction[0, 0]);

        for (int c = 1; c < prediction.Columns; c++)
        {
            double value = Convert.ToDouble(prediction[0, c]);
            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = c;
            }
        }

        return maxIndex;
    }

    private double ComputeL2Norm(Matrix<T> original, Matrix<T> perturbed)
    {
        double sumSquared = 0.0;

        for (int r = 0; r < original.Rows; r++)
        {
            for (int c = 0; c < original.Columns; c++)
            {
                double diff = Convert.ToDouble(perturbed[r, c]) - Convert.ToDouble(original[r, c]);
                sumSquared += diff * diff;
            }
        }

        return Math.Sqrt(sumSquared);
    }
}

/// <summary>
/// Results from adversarial attack evaluation.
/// </summary>
public class AttackResult
{
    public int TotalSamples { get; set; }
    public int SuccessfulAttacks { get; set; }
    public double SuccessRate { get; set; }
    public double AveragePerturbationNorm { get; set; }
}
```

**Step 2**: Create unit test

```csharp
// File: tests/UnitTests/Adversarial/FGSMAttackTests.cs
namespace AiDotNet.Tests.Adversarial;

public class FGSMAttackTests
{
    [Fact]
    public void GenerateAdversarialExample_ChangesInput()
    {
        // Arrange
        var model = CreateSimpleModel();
        var loss = new CrossEntropyLoss<double>();
        var attack = new FGSMAttack<double>(model, loss);

        var input = CreateTestInput();
        var target = CreateTestTarget();

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, target, epsilon: 0.1);

        // Assert
        // Adversarial should be different from original
        bool hasDifference = false;
        for (int r = 0; r < input.Rows; r++)
        {
            for (int c = 0; c < input.Columns; c++)
            {
                if (input[r, c] != adversarial[r, c])
                {
                    hasDifference = true;
                    break;
                }
            }
        }

        Assert.True(hasDifference);

        // All values should be in [0, 1] range
        for (int r = 0; r < adversarial.Rows; r++)
        {
            for (int c = 0; c < adversarial.Columns; c++)
            {
                Assert.InRange(adversarial[r, c], 0.0, 1.0);
            }
        }
    }

    [Fact]
    public void EvaluateAttack_ReturnsSuccessRate()
    {
        // Arrange
        var model = CreateSimpleModel();
        var loss = new CrossEntropyLoss<double>();
        var attack = new FGSMAttack<double>(model, loss);

        var testData = CreateTestDataset(100);

        // Act
        var result = attack.EvaluateAttack(testData, epsilon: 0.15);

        // Assert
        Assert.InRange(result.SuccessRate, 0.0, 1.0);
        Assert.True(result.TotalSamples == 100);
        Assert.True(result.AveragePerturbationNorm >= 0.0);
    }
}
```

---

### AC 1.2: Implement PGD Attack

**What is PGD?**
Projected Gradient Descent - iteratively applies FGSM and projects back to valid region.

**Algorithm**:
```
1. Start with original input + small random noise
2. For k iterations:
   - Apply FGSM step
   - Project back to epsilon ball around original input
3. Return final perturbed input
```

**File**: `src/Adversarial/PGDAttack.cs`

```csharp
// File: src/Adversarial/PGDAttack.cs
namespace AiDotNet.Adversarial;

/// <summary>
/// Projected Gradient Descent adversarial attack.
/// Stronger iterative version of FGSM.
/// </summary>
public class PGDAttack<T>
{
    private readonly IModel<T> _model;
    private readonly ILoss<T> _loss;
    private readonly Random _random;

    public PGDAttack(IModel<T> model, ILoss<T> loss)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _loss = loss ?? throw new ArgumentNullException(nameof(loss));
        _random = new Random();
    }

    /// <summary>
    /// Generate adversarial example using PGD.
    /// </summary>
    /// <param name="input">Original input</param>
    /// <param name="target">True label</param>
    /// <param name="epsilon">Maximum perturbation (L-infinity bound)</param>
    /// <param name="alpha">Step size per iteration</param>
    /// <param name="numIterations">Number of PGD iterations</param>
    /// <param name="randomStart">Whether to start from random point in epsilon ball</param>
    public Matrix<T> GenerateAdversarialExample(
        Matrix<T> input,
        Matrix<T> target,
        double epsilon,
        double alpha = 0.01,
        int numIterations = 40,
        bool randomStart = true)
    {
        var numOps = NumericOperations<T>.Instance;

        // Initialize adversarial example
        var adversarial = new Matrix<T>(input.Rows, input.Columns);

        if (randomStart)
        {
            // Start from random point in epsilon ball
            for (int r = 0; r < input.Rows; r++)
            {
                for (int c = 0; c < input.Columns; c++)
                {
                    double randomNoise = (_random.NextDouble() - 0.5) * 2 * epsilon;
                    double value = Convert.ToDouble(input[r, c]) + randomNoise;
                    adversarial[r, c] = numOps.FromDouble(Math.Max(0.0, Math.Min(1.0, value)));
                }
            }
        }
        else
        {
            // Start from original input
            for (int r = 0; r < input.Rows; r++)
            {
                for (int c = 0; c < input.Columns; c++)
                {
                    adversarial[r, c] = input[r, c];
                }
            }
        }

        // Iterative FGSM
        for (int iter = 0; iter < numIterations; iter++)
        {
            // Forward pass
            var prediction = _model.Forward(adversarial, training: true);

            // Compute loss and gradient
            var lossGradient = _loss.ComputeGradient(prediction, target);
            var inputGradient = _model.BackwardToInput(lossGradient);

            // Apply FGSM step
            T alphaValue = numOps.FromDouble(alpha);

            for (int r = 0; r < adversarial.Rows; r++)
            {
                for (int c = 0; c < adversarial.Columns; c++)
                {
                    // Gradient step
                    double gradValue = Convert.ToDouble(inputGradient[r, c]);
                    double sign = gradValue > 0 ? 1.0 : (gradValue < 0 ? -1.0 : 0.0);

                    double currentValue = Convert.ToDouble(adversarial[r, c]);
                    double newValue = currentValue + alpha * sign;

                    // Project back to epsilon ball around original input
                    double originalValue = Convert.ToDouble(input[r, c]);
                    newValue = Math.Max(originalValue - epsilon, Math.Min(originalValue + epsilon, newValue));

                    // Clip to [0, 1]
                    newValue = Math.Max(0.0, Math.Min(1.0, newValue));

                    adversarial[r, c] = numOps.FromDouble(newValue);
                }
            }
        }

        return adversarial;
    }
}
```

---

## Phase 2: Adversarial Training Defense

### AC 2.1: Implement AdversarialTrainer

**What is Adversarial Training?**
Train model on both clean and adversarial examples to make it robust.

**Process**:
```
1. For each training batch:
   - Generate adversarial examples
   - Mix with clean examples
   - Train on combined dataset
2. Model learns to be robust to perturbations
```

**File**: `src/Adversarial/AdversarialTrainer.cs`

```csharp
// File: src/Adversarial/AdversarialTrainer.cs
namespace AiDotNet.Adversarial;

/// <summary>
/// Trains models with adversarial examples to improve robustness.
/// </summary>
public class AdversarialTrainer<T>
{
    private readonly IModel<T> _model;
    private readonly ILoss<T> _loss;
    private readonly IOptimizer<T> _optimizer;
    private readonly FGSMAttack<T> _fgsmAttack;
    private readonly PGDAttack<T> _pgdAttack;

    public enum AttackType
    {
        FGSM,
        PGD
    }

    /// <summary>
    /// Ratio of adversarial examples to clean examples (0.0 to 1.0).
    /// 0.5 means 50% adversarial, 50% clean.
    /// </summary>
    public double AdversarialRatio { get; set; } = 0.5;

    /// <summary>
    /// Perturbation magnitude for adversarial examples.
    /// </summary>
    public double Epsilon { get; set; } = 0.1;

    public AdversarialTrainer(
        IModel<T> model,
        ILoss<T> loss,
        IOptimizer<T> optimizer)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _loss = loss ?? throw new ArgumentNullException(nameof(loss));
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));

        _fgsmAttack = new FGSMAttack<T>(model, loss);
        _pgdAttack = new PGDAttack<T>(model, loss);
    }

    /// <summary>
    /// Train model with adversarial examples.
    /// </summary>
    public void Train(
        IDataLoader<T> dataLoader,
        int epochs,
        AttackType attackType = AttackType.PGD)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0.0;
            int batchCount = 0;

            foreach (var batch in dataLoader.GetBatches())
            {
                // Generate adversarial examples for this batch
                var adversarialBatch = GenerateAdversarialBatch(
                    batch.Input,
                    batch.Target,
                    attackType
                );

                // Mix adversarial and clean examples based on ratio
                var mixedInput = MixBatch(batch.Input, adversarialBatch, AdversarialRatio);
                var mixedTarget = DuplicateTarget(batch.Target, mixedInput.Rows);

                // Forward pass
                var prediction = _model.Forward(mixedInput, training: true);

                // Compute loss
                var lossValue = _loss.Compute(prediction, mixedTarget);
                totalLoss += Convert.ToDouble(lossValue);

                // Backward pass
                var gradient = _loss.ComputeGradient(prediction, mixedTarget);
                var paramGradients = _model.Backward(gradient);

                // Update parameters
                var parameters = _model.GetParameters();
                _optimizer.Update(parameters, paramGradients);

                batchCount++;
            }

            double avgLoss = totalLoss / batchCount;
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {avgLoss:F4}");
        }
    }

    private Matrix<T> GenerateAdversarialBatch(
        Matrix<T> input,
        Matrix<T> target,
        AttackType attackType)
    {
        int batchSize = input.Rows;
        var adversarialExamples = new List<Matrix<T>>();

        for (int i = 0; i < batchSize; i++)
        {
            // Extract single example
            var singleInput = ExtractRow(input, i);
            var singleTarget = ExtractRow(target, i);

            // Generate adversarial example
            Matrix<T> adversarial;

            if (attackType == AttackType.FGSM)
            {
                adversarial = _fgsmAttack.GenerateAdversarialExample(
                    singleInput, singleTarget, Epsilon);
            }
            else // PGD
            {
                adversarial = _pgdAttack.GenerateAdversarialExample(
                    singleInput, singleTarget, Epsilon);
            }

            adversarialExamples.Add(adversarial);
        }

        // Combine into batch
        return CombineRows(adversarialExamples);
    }

    private Matrix<T> MixBatch(Matrix<T> clean, Matrix<T> adversarial, double adversarialRatio)
    {
        int cleanCount = (int)((1.0 - adversarialRatio) * clean.Rows);
        int advCount = (int)(adversarialRatio * clean.Rows);

        var mixed = new List<Matrix<T>>();

        // Add clean examples
        for (int i = 0; i < cleanCount; i++)
        {
            mixed.Add(ExtractRow(clean, i));
        }

        // Add adversarial examples
        for (int i = 0; i < advCount; i++)
        {
            mixed.Add(ExtractRow(adversarial, i));
        }

        return CombineRows(mixed);
    }

    private Matrix<T> ExtractRow(Matrix<T> matrix, int rowIndex)
    {
        var row = new Matrix<T>(1, matrix.Columns);
        for (int c = 0; c < matrix.Columns; c++)
        {
            row[0, c] = matrix[rowIndex, c];
        }
        return row;
    }

    private Matrix<T> CombineRows(List<Matrix<T>> rows)
    {
        if (rows.Count == 0)
            throw new ArgumentException("Cannot combine empty list of rows");

        int cols = rows[0].Columns;
        var combined = new Matrix<T>(rows.Count, cols);

        for (int r = 0; r < rows.Count; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                combined[r, c] = rows[r][0, c];
            }
        }

        return combined;
    }

    private Matrix<T> DuplicateTarget(Matrix<T> target, int newRows)
    {
        var duplicated = new Matrix<T>(newRows, target.Columns);
        int originalRows = target.Rows;

        for (int r = 0; r < newRows; r++)
        {
            int sourceRow = r % originalRows;
            for (int c = 0; c < target.Columns; c++)
            {
                duplicated[r, c] = target[sourceRow, c];
            }
        }

        return duplicated;
    }
}
```

---

## Phase 3: RLHF Implementation

### AC 3.1: Implement RewardModel

**What is a Reward Model?**
A neural network trained to predict human preferences between two outputs.

**Training Data**:
```
Input: "Write a poem about cats"
Output A: "Cats are fluffy and cute..."
Output B: "Felines possess soft fur..."
Human preference: A is better (label = 1)
```

**File**: `src/RLHF/RewardModel.cs`

```csharp
// File: src/RLHF/RewardModel.cs
namespace AiDotNet.RLHF;

/// <summary>
/// Reward model that predicts human preferences.
/// Trained on pairs of outputs with human preference labels.
/// </summary>
public class RewardModel<T>
{
    private readonly IModel<T> _baseModel;
    private readonly ILoss<T> _loss;
    private readonly IOptimizer<T> _optimizer;

    public RewardModel(IModel<T> baseModel, IOptimizer<T> optimizer)
    {
        _baseModel = baseModel ?? throw new ArgumentNullException(nameof(baseModel));
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        _loss = new CrossEntropyLoss<T>(); // Binary classification
    }

    /// <summary>
    /// Train reward model on preference pairs.
    /// </summary>
    /// <param name="preferences">List of (outputA, outputB, preferenceLabel)</param>
    /// <param name="epochs">Number of training epochs</param>
    public void Train(
        List<(Matrix<T> OutputA, Matrix<T> OutputB, int PreferredIndex)> preferences,
        int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0.0;

            foreach (var pref in preferences)
            {
                // Get rewards for both outputs
                var rewardA = _baseModel.Forward(pref.OutputA, training: true);
                var rewardB = _baseModel.Forward(pref.OutputB, training: true);

                // Compute preference probability using Bradley-Terry model
                // P(A > B) = exp(r_A) / (exp(r_A) + exp(r_B))
                var prefProbability = ComputePreferenceProbability(rewardA, rewardB);

                // Create target (1 if A preferred, 0 if B preferred)
                var target = CreatePreferenceTarget(pref.PreferredIndex);

                // Compute loss
                var lossValue = _loss.Compute(prefProbability, target);
                totalLoss += Convert.ToDouble(lossValue);

                // Backpropagate and update
                var gradient = _loss.ComputeGradient(prefProbability, target);
                var paramGradients = _baseModel.Backward(gradient);
                var parameters = _baseModel.GetParameters();
                _optimizer.Update(parameters, paramGradients);
            }

            double avgLoss = totalLoss / preferences.Count;
            Console.WriteLine($"Reward Model Epoch {epoch + 1}/{epochs}, Loss: {avgLoss:F4}");
        }
    }

    /// <summary>
    /// Predict reward (score) for an output.
    /// Higher reward = better quality according to human preferences.
    /// </summary>
    public double PredictReward(Matrix<T> output)
    {
        var reward = _baseModel.Forward(output, training: false);
        return Convert.ToDouble(reward[0, 0]); // Scalar reward
    }

    private Matrix<T> ComputePreferenceProbability(Matrix<T> rewardA, Matrix<T> rewardB)
    {
        var numOps = NumericOperations<T>.Instance;

        // Bradley-Terry: P(A > B) = exp(r_A) / (exp(r_A) + exp(r_B))
        double rA = Convert.ToDouble(rewardA[0, 0]);
        double rB = Convert.ToDouble(rewardB[0, 0]);

        double expA = Math.Exp(rA);
        double expB = Math.Exp(rB);

        double probA = expA / (expA + expB);

        var result = new Matrix<T>(1, 2);
        result[0, 0] = numOps.FromDouble(probA);     // P(A > B)
        result[0, 1] = numOps.FromDouble(1.0 - probA); // P(B > A)

        return result;
    }

    private Matrix<T> CreatePreferenceTarget(int preferredIndex)
    {
        var numOps = NumericOperations<T>.Instance;
        var target = new Matrix<T>(1, 2);

        if (preferredIndex == 0) // A is preferred
        {
            target[0, 0] = numOps.One;
            target[0, 1] = numOps.Zero;
        }
        else // B is preferred
        {
            target[0, 0] = numOps.Zero;
            target[0, 1] = numOps.One;
        }

        return target;
    }
}
```

---

### AC 3.2: Implement PPO for RLHF

**What is PPO?**
Proximal Policy Optimization - stable RL algorithm for fine-tuning models with reward.

**Key Idea**: Update model to increase reward, but not too much at once (to maintain stability).

**File**: `src/RLHF/PPOTrainer.cs`

```csharp
// File: src/RLHF/PPOTrainer.cs
namespace AiDotNet.RLHF;

/// <summary>
/// Proximal Policy Optimization trainer for RLHF.
/// Fine-tunes model using rewards from reward model.
/// </summary>
public class PPOTrainer<T>
{
    private readonly IModel<T> _policyModel;
    private readonly RewardModel<T> _rewardModel;
    private readonly IOptimizer<T> _optimizer;

    /// <summary>Clip ratio for PPO (prevents large policy updates)</summary>
    public double ClipEpsilon { get; set; } = 0.2;

    /// <summary>KL divergence penalty coefficient</summary>
    public double KLCoefficient { get; set; } = 0.01;

    public PPOTrainer(
        IModel<T> policyModel,
        RewardModel<T> rewardModel,
        IOptimizer<T> optimizer)
    {
        _policyModel = policyModel ?? throw new ArgumentNullException(nameof(policyModel));
        _rewardModel = rewardModel ?? throw new ArgumentNullException(nameof(rewardModel));
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
    }

    /// <summary>
    /// Train policy model using PPO and reward model.
    /// </summary>
    public void Train(
        IDataLoader<T> dataLoader,
        int epochs,
        int ppoEpochs = 4)
    {
        // Store initial policy for KL penalty
        var initialPolicy = CloneModel(_policyModel);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            var experienceBuffer = new List<Experience<T>>();

            // Collect experience
            foreach (var batch in dataLoader.GetBatches())
            {
                // Generate outputs with current policy
                var output = _policyModel.Forward(batch.Input, training: false);

                // Get reward from reward model
                double reward = _rewardModel.PredictReward(output);

                // Store experience
                experienceBuffer.Add(new Experience<T>
                {
                    State = batch.Input,
                    Action = output,
                    Reward = reward
                });
            }

            // PPO update epochs
            for (int ppoEpoch = 0; ppoEpoch < ppoEpochs; ppoEpoch++)
            {
                double totalLoss = 0.0;

                foreach (var experience in experienceBuffer)
                {
                    // Current policy output
                    var newOutput = _policyModel.Forward(experience.State, training: true);

                    // Compute PPO loss
                    var loss = ComputePPOLoss(
                        experience.Action,
                        newOutput,
                        experience.Reward,
                        initialPolicy.Forward(experience.State, training: false)
                    );

                    totalLoss += Convert.ToDouble(loss);

                    // Update policy
                    // (Simplified - actual implementation needs proper gradient computation)
                    var parameters = _policyModel.GetParameters();
                    // Update using optimizer...
                }

                Console.WriteLine($"PPO Epoch {ppoEpoch + 1}/{ppoEpochs}, Loss: {totalLoss / experienceBuffer.Count:F4}");
            }
        }
    }

    private T ComputePPOLoss(
        Matrix<T> oldAction,
        Matrix<T> newAction,
        double reward,
        Matrix<T> initialAction)
    {
        var numOps = NumericOperations<T>.Instance;

        // Compute ratio: π_new(a|s) / π_old(a|s)
        // Simplified: using L2 distance as proxy
        double ratio = ComputeActionRatio(oldAction, newAction);

        // Clipped surrogate objective
        double clippedRatio = Math.Max(
            Math.Min(ratio, 1.0 + ClipEpsilon),
            1.0 - ClipEpsilon
        );

        double advantage = reward; // Simplified advantage (should be reward - baseline)

        double surrogateObjective = Math.Min(
            ratio * advantage,
            clippedRatio * advantage
        );

        // KL penalty to prevent drifting too far from initial policy
        double klPenalty = ComputeKLDivergence(initialAction, newAction);

        // Total loss (negative because we want to maximize)
        double loss = -surrogateObjective + KLCoefficient * klPenalty;

        return numOps.FromDouble(loss);
    }

    private double ComputeActionRatio(Matrix<T> oldAction, Matrix<T> newAction)
    {
        // Simplified ratio computation
        // In practice, this should be the probability ratio
        double distance = 0.0;

        for (int r = 0; r < oldAction.Rows; r++)
        {
            for (int c = 0; c < oldAction.Columns; c++)
            {
                double diff = Convert.ToDouble(newAction[r, c]) - Convert.ToDouble(oldAction[r, c]);
                distance += diff * diff;
            }
        }

        return Math.Exp(-distance); // Convert distance to ratio-like value
    }

    private double ComputeKLDivergence(Matrix<T> p, Matrix<T> q)
    {
        double kl = 0.0;

        for (int r = 0; r < p.Rows; r++)
        {
            for (int c = 0; c < p.Columns; c++)
            {
                double pVal = Math.Max(Convert.ToDouble(p[r, c]), 1e-10);
                double qVal = Math.Max(Convert.ToDouble(q[r, c]), 1e-10);

                kl += pVal * Math.Log(pVal / qVal);
            }
        }

        return kl;
    }

    private IModel<T> CloneModel(IModel<T> model)
    {
        // Deep copy of model (simplified - actual implementation needs proper cloning)
        // For now, return reference (should implement proper cloning)
        return model;
    }
}

public class Experience<T>
{
    public Matrix<T> State { get; set; } = new Matrix<T>(0, 0);
    public Matrix<T> Action { get; set; } = new Matrix<T>(0, 0);
    public double Reward { get; set; }
}
```

---

## Phase 4: Safety Monitoring and Documentation

### AC 4.1: Implement ModelCard

**What is a Model Card?**
Documentation of model's capabilities, limitations, biases, and intended use.

**File**: `src/Safety/ModelCard.cs`

```csharp
// File: src/Safety/ModelCard.cs
namespace AiDotNet.Safety;

/// <summary>
/// Model Card for transparent documentation of ML models.
/// Based on "Model Cards for Model Reporting" (Mitchell et al., 2019).
/// </summary>
public class ModelCard
{
    /// <summary>Model name and version</summary>
    public ModelDetails Details { get; set; } = new ModelDetails();

    /// <summary>Intended use cases and users</summary>
    public IntendedUse IntendedUse { get; set; } = new IntendedUse();

    /// <summary>Training and evaluation data</summary>
    public DataInformation Data { get; set; } = new DataInformation();

    /// <summary>Performance metrics on different subgroups</summary>
    public List<PerformanceMetric> PerformanceMetrics { get; set; } = new List<PerformanceMetric>();

    /// <summary>Known limitations and biases</summary>
    public List<string> Limitations { get; set; } = new List<string>();

    /// <summary>Ethical considerations</summary>
    public List<string> EthicalConsiderations { get; set; } = new List<string>();

    /// <summary>
    /// Export model card to JSON format.
    /// </summary>
    public string ToJson()
    {
        return System.Text.Json.JsonSerializer.Serialize(this, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
    }

    /// <summary>
    /// Export model card to markdown format.
    /// </summary>
    public string ToMarkdown()
    {
        var sb = new System.Text.StringBuilder();

        sb.AppendLine($"# Model Card: {Details.Name}");
        sb.AppendLine();

        sb.AppendLine("## Model Details");
        sb.AppendLine($"- **Version**: {Details.Version}");
        sb.AppendLine($"- **Type**: {Details.ModelType}");
        sb.AppendLine($"- **Architecture**: {Details.Architecture}");
        sb.AppendLine($"- **Date**: {Details.Date:yyyy-MM-dd}");
        sb.AppendLine($"- **Authors**: {string.Join(", ", Details.Authors)}");
        sb.AppendLine();

        sb.AppendLine("## Intended Use");
        sb.AppendLine($"- **Primary Use**: {IntendedUse.PrimaryUse}");
        sb.AppendLine($"- **Primary Users**: {string.Join(", ", IntendedUse.PrimaryUsers)}");
        sb.AppendLine($"- **Out-of-Scope Uses**: {string.Join(", ", IntendedUse.OutOfScopeUses)}");
        sb.AppendLine();

        sb.AppendLine("## Training Data");
        sb.AppendLine($"- **Dataset**: {Data.TrainingDataset}");
        sb.AppendLine($"- **Size**: {Data.TrainingDataSize:N0} examples");
        sb.AppendLine($"- **Preprocessing**: {Data.Preprocessing}");
        sb.AppendLine();

        sb.AppendLine("## Performance Metrics");
        foreach (var metric in PerformanceMetrics)
        {
            sb.AppendLine($"- **{metric.Name}** ({metric.Subgroup}): {metric.Value:F4}");
        }
        sb.AppendLine();

        sb.AppendLine("## Limitations");
        foreach (var limitation in Limitations)
        {
            sb.AppendLine($"- {limitation}");
        }
        sb.AppendLine();

        sb.AppendLine("## Ethical Considerations");
        foreach (var consideration in EthicalConsiderations)
        {
            sb.AppendLine($"- {consideration}");
        }

        return sb.ToString();
    }
}

public class ModelDetails
{
    public string Name { get; set; } = string.Empty;
    public string Version { get; set; } = "1.0.0";
    public string ModelType { get; set; } = string.Empty;
    public string Architecture { get; set; } = string.Empty;
    public DateTime Date { get; set; } = DateTime.Now;
    public List<string> Authors { get; set; } = new List<string>();
    public string License { get; set; } = string.Empty;
}

public class IntendedUse
{
    public string PrimaryUse { get; set; } = string.Empty;
    public List<string> PrimaryUsers { get; set; } = new List<string>();
    public List<string> OutOfScopeUses { get; set; } = new List<string>();
}

public class DataInformation
{
    public string TrainingDataset { get; set; } = string.Empty;
    public int TrainingDataSize { get; set; }
    public string EvaluationDataset { get; set; } = string.Empty;
    public int EvaluationDataSize { get; set; }
    public string Preprocessing { get; set; } = string.Empty;
}

public class PerformanceMetric
{
    public string Name { get; set; } = string.Empty;
    public string Subgroup { get; set; } = "Overall";
    public double Value { get; set; }
}
```

---

### AC 4.2: Implement SafetyMonitor

**What does this do?**
Monitors model outputs in production for safety issues (toxicity, bias, hallucinations).

**File**: `src/Safety/SafetyMonitor.cs`

```csharp
// File: src/Safety/SafetyMonitor.cs
namespace AiDotNet.Safety;

/// <summary>
/// Monitors model predictions for safety issues in production.
/// </summary>
public class SafetyMonitor<T>
{
    private readonly List<ISafetyCheck> _checks;
    private readonly List<SafetyIncident> _incidents;

    public SafetyMonitor()
    {
        _checks = new List<ISafetyCheck>();
        _incidents = new List<SafetyIncident>();
    }

    /// <summary>
    /// Add a safety check to the monitor.
    /// </summary>
    public void AddCheck(ISafetyCheck check)
    {
        _checks.Add(check);
    }

    /// <summary>
    /// Check a prediction for safety issues.
    /// </summary>
    /// <returns>Safety result with any violations found</returns>
    public SafetyResult CheckPrediction(
        Matrix<T> input,
        Matrix<T> prediction,
        Dictionary<string, object>? metadata = null)
    {
        var violations = new List<SafetyViolation>();

        foreach (var check in _checks)
        {
            var result = check.Check(input, prediction, metadata);

            if (!result.IsSafe)
            {
                violations.AddRange(result.Violations);
            }
        }

        if (violations.Any())
        {
            // Log incident
            _incidents.Add(new SafetyIncident
            {
                Timestamp = DateTime.UtcNow,
                Input = input,
                Prediction = prediction,
                Violations = violations,
                Metadata = metadata
            });
        }

        return new SafetyResult
        {
            IsSafe = violations.Count == 0,
            Violations = violations
        };
    }

    /// <summary>
    /// Get all safety incidents.
    /// </summary>
    public IReadOnlyList<SafetyIncident> GetIncidents()
    {
        return _incidents.AsReadOnly();
    }

    /// <summary>
    /// Get incident statistics.
    /// </summary>
    public SafetyStatistics GetStatistics()
    {
        return new SafetyStatistics
        {
            TotalIncidents = _incidents.Count,
            ViolationsByType = _incidents
                .SelectMany(i => i.Violations)
                .GroupBy(v => v.Type)
                .ToDictionary(g => g.Key, g => g.Count())
        };
    }
}

/// <summary>
/// Interface for safety checks.
/// </summary>
public interface ISafetyCheck
{
    SafetyResult Check(Matrix<T> input, Matrix<T> prediction, Dictionary<string, object>? metadata);
}

/// <summary>
/// Result of safety check.
/// </summary>
public class SafetyResult
{
    public bool IsSafe { get; set; }
    public List<SafetyViolation> Violations { get; set; } = new List<SafetyViolation>();
}

/// <summary>
/// A specific safety violation.
/// </summary>
public class SafetyViolation
{
    public string Type { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public double Severity { get; set; } // 0.0 to 1.0
    public Dictionary<string, object> Details { get; set; } = new Dictionary<string, object>();
}

/// <summary>
/// Record of a safety incident.
/// </summary>
public class SafetyIncident
{
    public DateTime Timestamp { get; set; }
    public Matrix<T> Input { get; set; } = new Matrix<T>(0, 0);
    public Matrix<T> Prediction { get; set; } = new Matrix<T>(0, 0);
    public List<SafetyViolation> Violations { get; set; } = new List<SafetyViolation>();
    public Dictionary<string, object>? Metadata { get; set; }
}

public class SafetyStatistics
{
    public int TotalIncidents { get; set; }
    public Dictionary<string, int> ViolationsByType { get; set; } = new Dictionary<string, int>();
}
```

**Step 2**: Implement common safety checks

```csharp
// File: src/Safety/SafetyChecks/ConfidenceThresholdCheck.cs
namespace AiDotNet.Safety.SafetyChecks;

/// <summary>
/// Flags predictions with very low confidence as potentially unsafe.
/// </summary>
public class ConfidenceThresholdCheck<T> : ISafetyCheck
{
    private readonly double _threshold;

    public ConfidenceThresholdCheck(double threshold = 0.5)
    {
        _threshold = threshold;
    }

    public SafetyResult Check(
        Matrix<T> input,
        Matrix<T> prediction,
        Dictionary<string, object>? metadata)
    {
        // Get max probability (confidence)
        double maxProb = 0.0;

        for (int c = 0; c < prediction.Columns; c++)
        {
            double prob = Convert.ToDouble(prediction[0, c]);
            if (prob > maxProb)
                maxProb = prob;
        }

        if (maxProb < _threshold)
        {
            return new SafetyResult
            {
                IsSafe = false,
                Violations = new List<SafetyViolation>
                {
                    new SafetyViolation
                    {
                        Type = "LowConfidence",
                        Description = $"Prediction confidence {maxProb:F2} below threshold {_threshold:F2}",
                        Severity = 1.0 - maxProb,
                        Details = new Dictionary<string, object>
                        {
                            { "confidence", maxProb },
                            { "threshold", _threshold }
                        }
                    }
                }
            };
        }

        return new SafetyResult { IsSafe = true };
    }
}
```

---

## Testing Strategy

### Integration Test: End-to-End Safety Pipeline

```csharp
// File: tests/IntegrationTests/Safety/SafetyPipelineTests.cs
namespace AiDotNet.Tests.Safety;

public class SafetyPipelineTests
{
    [Fact]
    public void FullPipeline_AdversarialTraining_ImprovesRobustness()
    {
        // Train baseline model
        var baselineModel = CreateModel();
        TrainModel(baselineModel, cleanData);

        // Evaluate against attacks
        var fgsmAttack = new FGSMAttack<double>(baselineModel, new CrossEntropyLoss<double>());
        var baselineResult = fgsmAttack.EvaluateAttack(testData, epsilon: 0.1);

        // Train adversarially robust model
        var robustModel = CreateModel();
        var advTrainer = new AdversarialTrainer<double>(
            robustModel,
            new CrossEntropyLoss<double>(),
            new SGD<double>(0.01)
        );

        advTrainer.Train(cleanData, epochs: 10, AttackType.PGD);

        // Evaluate robust model
        var fgsmAttackRobust = new FGSMAttack<double>(robustModel, new CrossEntropyLoss<double>());
        var robustResult = fgsmAttackRobust.EvaluateAttack(testData, epsilon: 0.1);

        // Robust model should have lower attack success rate
        Assert.True(robustResult.SuccessRate < baselineResult.SuccessRate);
    }
}
```

---

## Success Criteria Checklist

- [ ] FGSM generates adversarial examples that fool model
- [ ] PGD creates stronger attacks than FGSM
- [ ] Adversarial training reduces attack success rate by >50%
- [ ] Reward model correctly predicts human preferences
- [ ] PPO improves model reward without catastrophic performance loss
- [ ] Model card documents all required fields
- [ ] Safety monitor flags low-confidence predictions
- [ ] All tests pass with >80% coverage

---

## Example Usage After Implementation

```csharp
// Create model card
var modelCard = new ModelCard
{
    Details = new ModelDetails
    {
        Name = "Image Classifier v2",
        Version = "2.1.0",
        ModelType = "Convolutional Neural Network",
        Architecture = "ResNet-50",
        Authors = new List<string> { "AI Team" }
    },
    IntendedUse = new IntendedUse
    {
        PrimaryUse = "Medical image classification",
        PrimaryUsers = new List<string> { "Radiologists", "Medical researchers" },
        OutOfScopeUses = new List<string> { "Autonomous diagnosis", "Legal decisions" }
    },
    Limitations = new List<string>
    {
        "Performance degrades on low-quality images",
        "Not tested on pediatric patients",
        "May exhibit bias toward certain demographics"
    }
};

// Save model card
File.WriteAllText("model_card.md", modelCard.ToMarkdown());

// Setup safety monitoring
var safetyMonitor = new SafetyMonitor<double>();
safetyMonitor.AddCheck(new ConfidenceThresholdCheck<double>(0.7));

// Check prediction
var result = safetyMonitor.CheckPrediction(input, prediction);

if (!result.IsSafe)
{
    Console.WriteLine("SAFETY WARNING:");
    foreach (var violation in result.Violations)
    {
        Console.WriteLine($"- {violation.Type}: {violation.Description}");
    }
}
```
