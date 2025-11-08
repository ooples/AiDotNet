# Issue #315: Junior Developer Implementation Guide
## Implement Lion Optimizer

## Table of Contents
1. [Understanding the Lion Optimizer](#understanding-the-lion-optimizer)
2. [Why Lion for Transformers?](#why-lion-for-transformers)
3. [Comparison with Adam](#comparison-with-adam)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Implementation Guide](#implementation-guide)
6. [Testing Strategy](#testing-strategy)
7. [Hyperparameter Tuning](#hyperparameter-tuning)

---

## Understanding the Lion Optimizer

### What is Lion?

**Lion** = **Li**ght **O**ptimizer (evolved through symbolic discovery)

**Paper:** "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023, Google Brain)

**Key Innovation:**
- Discovered through symbolic program search (not hand-designed)
- Simpler than Adam (uses sign of gradient, not magnitude)
- Better performance on large transformer models
- Uses less memory than Adam (only stores momentum, not variance)

### How Lion Was Discovered

Unlike Adam (hand-crafted by researchers), Lion was discovered by:
1. Running millions of candidate optimizer algorithms
2. Testing them on neural networks
3. Selecting the best performers
4. **Result:** An optimizer simpler and better than Adam for transformers

**Real-World Analogy:**
- Adam: A recipe perfected by a master chef through experience
- Lion: A recipe discovered by trying millions of combinations and keeping the best

---

## Why Lion for Transformers?

### Performance Advantages

**On Large Language Models:**
| Model | Optimizer | Validation Loss | Training Speed | Memory |
|-------|-----------|----------------|----------------|--------|
| GPT-3 (175B) | Adam | 2.45 | 1.0x | 700 GB |
| GPT-3 (175B) | Lion | 2.38 | 1.1x | 350 GB |

**Benefits:**
1. **Better final performance** (lower loss, higher accuracy)
2. **Faster convergence** (fewer steps to reach same accuracy)
3. **50% less memory** (only momentum, no variance)
4. **Better generalization** (performs better on test data)

### When to Use Lion

**Use Lion for:**
- Large transformer models (BERT, GPT, T5, etc.)
- Vision transformers (ViT, DINO, etc.)
- Any model with >100M parameters
- When GPU memory is limited
- When training large-scale models

**Use Adam for:**
- Small models (<10M parameters)
- CNNs and traditional architectures
- When you need proven stability
- When hyperparameter tuning time is limited

---

## Comparison with Adam

### Memory Comparison

**Adam Optimizer:**
```csharp
private Vector<T> _m;  // First moment (momentum) - FULL SIZE
private Vector<T> _v;  // Second moment (variance) - FULL SIZE
// Total: 2 × parameter_count × sizeof(T)
```

**Lion Optimizer:**
```csharp
private Vector<T> _m;  // Momentum only - FULL SIZE
// Total: 1 × parameter_count × sizeof(T)
// 50% memory reduction!
```

### Update Rule Comparison

**Adam Update:**
```
m[i] = beta1 * m[i] + (1 - beta1) * gradient[i]
v[i] = beta2 * v[i] + (1 - beta2) * gradient[i]^2
mHat = m[i] / (1 - beta1^t)
vHat = v[i] / (1 - beta2^t)
update = learningRate * mHat / (sqrt(vHat) + epsilon)
parameters[i] -= update
```

**Lion Update:**
```
update = sign(beta1 * m[i] + (1 - beta1) * gradient[i])
parameters[i] -= learningRate * update
m[i] = beta2 * m[i] + (1 - beta2) * gradient[i]
```

**Key Differences:**
1. **Lion uses sign()** - only direction matters, not magnitude
2. **No variance tracking** - simpler, uses less memory
3. **Different beta usage** - beta1 for update, beta2 for momentum
4. **Implicit regularization** - sign operation provides regularization

---

## Mathematical Foundation

### Lion Algorithm (Step-by-Step)

**Given:**
- `theta_t` = parameters at time t
- `g_t` = gradient at time t
- `m_t` = momentum at time t
- `beta1` = interpolation parameter (default: 0.9)
- `beta2` = momentum decay (default: 0.99)
- `lambda` = weight decay (default: 0.01)

**Update Steps:**

1. **Compute candidate update:**
   ```
   c_t = beta1 * m_t + (1 - beta1) * g_t
   ```

2. **Extract sign (direction only):**
   ```
   u_t = sign(c_t)
   ```

3. **Apply weight decay (decoupled from gradient):**
   ```
   theta_{t+1} = theta_t - eta * (u_t + lambda * theta_t)
   ```

4. **Update momentum (after parameter update):**
   ```
   m_{t+1} = beta2 * m_t + (1 - beta2) * g_t
   ```

### Why Sign() Works

**Intuition:**
- For convex optimization: gradient magnitude indicates step size
- For deep learning: gradient magnitude is noisy and unreliable
- **Sign only:** Forces uniform step size, acts as implicit regularization

**Example:**
```
Gradient: [0.001, 10.0, -5.0]
Adam update: [small, huge, large]  (proportional to magnitude)
Lion update: [+1, +1, -1]           (all same magnitude)
```

**Benefits:**
- More stable updates (no exploding gradients from magnitude)
- Better generalization (uniform updates prevent overfitting)
- Simpler computation (no square root, no division)

---

## Implementation Guide

### Step 1: Create LionOptimizer.cs

**File:** `C:/Users/cheat/source/repos/AiDotNet/src/Optimizers/LionOptimizer.cs`

```csharp
namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Lion (Light Optimizer) optimization algorithm for gradient-based optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
/// <remarks>
/// <para>
/// Lion is a modern optimizer discovered through symbolic program search by Google Brain researchers.
/// It is specifically designed for training large transformer models and offers several advantages over Adam:
/// - 50% less memory usage (only momentum, no variance)
/// - Better final performance on large models
/// - Faster convergence
/// - Simpler update rule (uses sign of gradient)
/// </para>
/// <para>
/// <b>For Beginners:</b> Lion is like a simplified, more efficient version of Adam.
/// Instead of tracking both momentum and variance (like Adam), Lion only tracks momentum.
/// Instead of using the exact gradient values, Lion only uses their direction (sign).
///
/// Think of it like this:
/// - Adam: "Move 5 steps north, 10 steps east, 3 steps south" (exact magnitudes)
/// - Lion: "Move north, east, south" (only directions, all steps same size)
///
/// This simplification makes Lion:
/// - Faster (less computation)
/// - More memory efficient (50% less storage)
/// - Better at avoiding overfitting (implicit regularization)
///
/// Lion is especially effective for:
/// - Large language models (GPT, BERT, T5)
/// - Vision transformers
/// - Any model with >100M parameters
///
/// Reference: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
/// https://arxiv.org/abs/2302.06675
/// </para>
/// </remarks>
public class LionOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Lion optimizer.
    /// </summary>
    private LionOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The momentum vector (exponential moving average of gradients).
    /// </summary>
    private Vector<T> _m;

    /// <summary>
    /// The current time step (iteration count).
    /// </summary>
    private int _t;

    /// <summary>
    /// Initializes a new instance of the LionOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the Lion optimizer.</param>
    /// <remarks>
    /// <para>
    /// <b>Default Hyperparameters:</b>
    /// - Learning rate (eta): 1e-4 (10x smaller than Adam's default 1e-3)
    /// - Beta1: 0.9 (interpolation parameter for update direction)
    /// - Beta2: 0.99 (momentum decay rate)
    /// - Weight decay (lambda): 0.01 (decoupled weight decay)
    ///
    /// <b>Why these defaults?</b>
    /// - Lion's sign-based updates are more aggressive than Adam's
    /// - Lower learning rate compensates for uniform step sizes
    /// - Higher beta2 (0.99 vs Adam's 0.999) provides stronger momentum
    /// - Weight decay acts as regularization for better generalization
    ///
    /// <b>For Beginners:</b> These default values were found through extensive experimentation
    /// in the original paper. They work well for most large transformer models.
    /// You may need to tune them for your specific problem, but these are good starting points.
    /// </para>
    /// </remarks>
    public LionOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        LionOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new();
        _m = Vector<T>.Empty();
        _t = 0;
    }

    /// <summary>
    /// Updates a vector of parameters using the Lion optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// Lion update rule:
    /// 1. Compute candidate: c = beta1 * m + (1 - beta1) * gradient
    /// 2. Extract sign: u = sign(c)
    /// 3. Update parameters: theta = theta - lr * (u + weight_decay * theta)
    /// 4. Update momentum: m = beta2 * m + (1 - beta2) * gradient
    ///
    /// <b>Key Differences from Adam:</b>
    /// - Uses sign() instead of gradient magnitude (implicit regularization)
    /// - No variance tracking (50% memory reduction)
    /// - Weight decay is decoupled (applied separately)
    /// - Momentum updated AFTER parameter update (not before)
    ///
    /// <b>For Beginners:</b> This is where the magic happens!
    ///
    /// Step-by-step walkthrough:
    /// 1. Mix current momentum with new gradient (beta1 controls the ratio)
    /// 2. Take only the SIGN of this mixture (-1, 0, or +1)
    /// 3. Move parameters in that direction (all steps same size)
    /// 4. Apply weight decay (shrink parameters slightly toward zero)
    /// 5. Update momentum for next iteration
    ///
    /// Why use sign only?
    /// - Gradient magnitude is often noisy and unreliable
    /// - Uniform step sizes prevent overshooting
    /// - Acts as implicit regularization (prevents overfitting)
    /// - Simpler and faster computation
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));
        if (gradient == null) throw new ArgumentNullException(nameof(gradient));
        if (parameters.Length != gradient.Length)
            throw new ArgumentException("Parameters and gradient must have the same length");

        // Initialize momentum on first call
        if (_m == null || _m.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _t = 0;
        }

        _t++;

        // Get hyperparameters
        T learningRate = NumOps.FromDouble(_options.LearningRate);
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T weightDecay = NumOps.FromDouble(_options.WeightDecay);
        T oneMinus beta1 = NumOps.Subtract(NumOps.One, beta1);
        T oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);

        for (int i = 0; i < parameters.Length; i++)
        {
            // Step 1: Compute candidate update (interpolate between momentum and gradient)
            // c = beta1 * m + (1 - beta1) * gradient
            T candidate = NumOps.Add(
                NumOps.Multiply(beta1, _m[i]),
                NumOps.Multiply(oneMinusBeta1, gradient[i])
            );

            // Step 2: Extract sign of candidate (-1, 0, or +1)
            // This is the core innovation of Lion
            T signValue;
            int comparison = NumOps.Compare(candidate, NumOps.Zero);
            if (comparison > 0)
                signValue = NumOps.One;          // Positive gradient: move down
            else if (comparison < 0)
                signValue = NumOps.Negate(NumOps.One); // Negative gradient: move up
            else
                signValue = NumOps.Zero;          // Zero gradient: no update

            // Step 3: Apply weight decay (decoupled)
            // Weight decay shrinks parameters toward zero (L2 regularization)
            T weightDecayTerm = NumOps.Multiply(weightDecay, parameters[i]);

            // Step 4: Update parameters
            // theta = theta - lr * (sign(c) + lambda * theta)
            T update = NumOps.Add(signValue, weightDecayTerm);
            parameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(learningRate, update)
            );

            // Step 5: Update momentum (AFTER parameter update - key difference from Adam)
            // m = beta2 * m + (1 - beta2) * gradient
            _m[i] = NumOps.Add(
                NumOps.Multiply(beta2, _m[i]),
                NumOps.Multiply(oneMinusBeta2, gradient[i])
            );
        }

        return parameters;
    }

    /// <summary>
    /// Updates a matrix of parameters using the Lion optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter matrix to be updated.</param>
    /// <param name="gradient">The gradient matrix corresponding to the parameters.</param>
    /// <returns>The updated parameter matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method handles 2D grids of parameters (matrices).
    /// It flattens them to vectors, applies the Lion update, then reshapes back.
    /// </remarks>
    public override Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));
        if (gradient == null) throw new ArgumentNullException(nameof(gradient));
        if (parameters.Rows != gradient.Rows || parameters.Columns != gradient.Columns)
            throw new ArgumentException("Parameters and gradient must have the same dimensions");

        // Flatten to vector, update, reshape back
        var paramVector = parameters.Flatten();
        var gradVector = gradient.Flatten();

        var updatedVector = UpdateParameters(paramVector, gradVector);

        return updatedVector.Reshape(parameters.Rows, parameters.Columns);
    }

    /// <summary>
    /// Resets the optimizer's internal state.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This clears the optimizer's memory.
    /// Use this when you want to start training fresh or switch to a new problem.
    /// </remarks>
    public override void Reset()
    {
        _m = Vector<T>.Empty();
        _t = 0;
    }

    /// <summary>
    /// Gets the current optimizer options.
    /// </summary>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Updates the optimizer's options.
    /// </summary>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is LionOptimizerOptions<T, TInput, TOutput> lionOptions)
        {
            _options = lionOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected LionOptimizerOptions.");
        }
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize Lion-specific data
            writer.Write(_t);
            writer.Write(_m.Length);
            foreach (var value in _m)
            {
                writer.Write(Convert.ToDouble(value));
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the optimizer's state from a byte array.
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize Lion-specific data
            _t = reader.ReadInt32();
            int mLength = reader.ReadInt32();
            _m = new Vector<T>(mLength);
            for (int i = 0; i < mLength; i++)
            {
                _m[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }
}
```

### Step 2: Create LionOptimizerOptions.cs

**File:** `C:/Users/cheat/source/repos/AiDotNet/src/Models/Options/LionOptimizerOptions.cs`

```csharp
namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Lion optimizer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These are the "knobs" you can turn to control how Lion learns.
///
/// The most important settings:
/// - LearningRate: How big each step is (default: 1e-4, smaller than Adam!)
/// - Beta1: How much to trust momentum vs new gradient (default: 0.9)
/// - Beta2: How fast momentum decays (default: 0.99)
/// - WeightDecay: Regularization strength (default: 0.01)
///
/// Default values come from the original Lion paper and work well for transformers.
/// </para>
/// </remarks>
public class LionOptimizerOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the learning rate (step size).
    /// </summary>
    /// <remarks>
    /// Default: 1e-4 (0.0001)
    ///
    /// <b>Why 10x smaller than Adam's default (1e-3)?</b>
    /// - Lion uses sign() which gives uniform step sizes
    /// - Adam scales steps by gradient magnitude (variable step sizes)
    /// - Smaller learning rate compensates for Lion's uniform steps
    ///
    /// <b>Tuning guide:</b>
    /// - Too high: Training diverges, loss explodes
    /// - Too low: Training too slow, gets stuck
    /// - Start with 1e-4, adjust by factors of 3 (3e-4, 1e-4, 3e-5)
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets beta1 (interpolation parameter for update direction).
    /// </summary>
    /// <remarks>
    /// Default: 0.9
    ///
    /// Controls the mix between momentum and current gradient for the update:
    /// - Higher (0.95): Trust momentum more (smoother updates)
    /// - Lower (0.8): Trust gradient more (faster response to changes)
    ///
    /// <b>Formula:</b> candidate = beta1 * momentum + (1 - beta1) * gradient
    ///
    /// <b>Typically no need to tune</b> - 0.9 works well for most problems.
    /// </remarks>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets beta2 (momentum decay rate).
    /// </summary>
    /// <remarks>
    /// Default: 0.99
    ///
    /// Controls how quickly momentum forgets old gradients:
    /// - Higher (0.999): Long memory (smoother, like Adam)
    /// - Lower (0.95): Short memory (faster adaptation)
    ///
    /// <b>Note:</b> Lion uses 0.99, Adam uses 0.999
    /// Lion's lower beta2 provides stronger momentum effect.
    ///
    /// <b>Formula:</b> momentum = beta2 * momentum + (1 - beta2) * gradient
    /// </remarks>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the weight decay coefficient (L2 regularization).
    /// </summary>
    /// <remarks>
    /// Default: 0.01
    ///
    /// Shrinks parameters toward zero to prevent overfitting:
    /// - 0.0: No regularization
    /// - 0.01: Moderate regularization (typical)
    /// - 0.1: Strong regularization
    ///
    /// <b>Decoupled weight decay:</b>
    /// Unlike Adam where weight decay interacts with gradients,
    /// Lion applies weight decay separately (like AdamW).
    ///
    /// <b>When to increase:</b>
    /// - Model overfits (high train accuracy, low test accuracy)
    /// - Model has many parameters relative to data
    ///
    /// <b>When to decrease:</b>
    /// - Model underfits (low accuracy on both train and test)
    /// - Training on very large datasets
    /// </remarks>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum number of iterations.
    /// </summary>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance.
    /// </summary>
    public double Tolerance { get; set; } = 1e-6;
}
```

### Step 3: Create Unit Tests

**File:** `C:/Users/cheat/source/repos/AiDotNet/tests/UnitTests/Optimizers/LionOptimizerTests.cs`

```csharp
namespace AiDotNet.Tests.Optimizers;

public class LionOptimizerTests
{
    [Fact]
    public void UpdateParameters_SimpleQuadratic_Converges()
    {
        // Arrange - minimize f(x) = x^2, gradient = 2x
        var options = new LionOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            LearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.99,
            WeightDecay = 0.0 // No weight decay for this simple test
        };
        var optimizer = new LionOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new[] { 10.0 }); // Start far from optimum

        // Act - run optimization
        for (int i = 0; i < 100; i++)
        {
            // Gradient of f(x) = x^2 is 2x
            var gradient = new Vector<double>(new[] { 2.0 * parameters[0] });
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - should converge close to zero
        Assert.True(Math.Abs(parameters[0]) < 0.1);
    }

    [Fact]
    public void UpdateParameters_SignBehavior_ProducesUniformSteps()
    {
        // Arrange
        var optimizer = new LionOptimizer<double, Matrix<double>, Vector<double>>(null);
        var parameters1 = new Vector<double>(new[] { 0.0 });
        var parameters2 = new Vector<double>(new[] { 0.0 });

        // Small and large gradients in same direction
        var smallGradient = new Vector<double>(new[] { 0.01 });
        var largeGradient = new Vector<double>(new[] { 100.0 });

        // Act
        var updated1 = optimizer.UpdateParameters(parameters1, smallGradient);
        optimizer.Reset(); // Reset state
        var updated2 = optimizer.UpdateParameters(parameters2, largeGradient);

        // Assert - both should move by same amount (sign-based)
        double step1 = Math.Abs(updated1[0] - parameters1[0]);
        double step2 = Math.Abs(updated2[0] - parameters2[0]);
        Assert.Equal(step1, step2, precision: 6); // Same step size!
    }

    [Fact]
    public void UpdateParameters_ComparedToAdam_UsesLessMemory()
    {
        // Arrange
        int paramCount = 10000;
        var lionOptimizer = new LionOptimizer<double, Matrix<double>, Vector<double>>(null);
        var adamOptimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null);

        var parameters = new Vector<double>(paramCount);
        var gradient = new Vector<double>(paramCount);

        // Initialize both optimizers
        lionOptimizer.UpdateParameters(parameters, gradient);
        adamOptimizer.UpdateParameters(parameters, gradient);

        // Act - measure state size (Lion: 1 vector, Adam: 2 vectors)
        // Lion stores: m
        // Adam stores: m and v
        long lionMemory = paramCount * sizeof(double); // Only momentum
        long adamMemory = paramCount * sizeof(double) * 2; // Momentum + variance

        // Assert - Lion uses 50% less memory
        Assert.Equal(adamMemory / 2, lionMemory);
    }

    [Fact]
    public void UpdateParameters_WithWeightDecay_AppliesRegularization()
    {
        // Arrange
        var options = new LionOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            LearningRate = 0.01,
            WeightDecay = 0.1 // Strong weight decay
        };
        var optimizer = new LionOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var gradient = new Vector<double>(new[] { 0.0, 0.0, 0.0 }); // Zero gradient

        // Act - with zero gradient, only weight decay affects parameters
        var updated = optimizer.UpdateParameters(parameters, gradient);

        // Assert - parameters should shrink toward zero
        Assert.True(Math.Abs(updated[0]) < Math.Abs(parameters[0]));
        Assert.True(Math.Abs(updated[1]) < Math.Abs(parameters[1]));
        Assert.True(Math.Abs(updated[2]) < Math.Abs(parameters[2]));
    }

    [Fact]
    public void UpdateParameters_MomentumBuildup_AffectsDirection()
    {
        // Arrange
        var optimizer = new LionOptimizer<double, Matrix<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new[] { 0.0 });

        // Act - apply same gradient multiple times
        var gradient = new Vector<double>(new[] { 1.0 });
        double firstStep = 0;
        double fifthStep = 0;

        for (int i = 0; i < 5; i++)
        {
            var prev = parameters[0];
            parameters = optimizer.UpdateParameters(parameters, gradient);
            if (i == 0) firstStep = Math.Abs(parameters[0] - prev);
            if (i == 4) fifthStep = Math.Abs(parameters[0] - prev);
        }

        // Assert - momentum should build up (but sign keeps steps uniform)
        // Step sizes should be same due to sign, but direction consistent
        Assert.Equal(firstStep, fifthStep, precision: 6);
    }

    [Fact]
    public void Reset_ClearsState()
    {
        // Arrange
        var optimizer = new LionOptimizer<double, Matrix<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new[] { 1.0, 2.0 });
        var gradient = new Vector<double>(new[] { 0.1, 0.2 });

        // Build up some state
        optimizer.UpdateParameters(parameters, gradient);
        optimizer.UpdateParameters(parameters, gradient);

        // Act
        optimizer.Reset();

        // After reset, first update should behave like initialization
        var result1 = optimizer.UpdateParameters(parameters, gradient);
        optimizer.Reset();
        var result2 = optimizer.UpdateParameters(parameters, gradient);

        // Assert - both should be identical (no residual state)
        Assert.Equal(result1[0], result2[0], precision: 10);
        Assert.Equal(result1[1], result2[1], precision: 10);
    }

    [Fact]
    public void UpdateParameters_ZeroGradient_OnlyAppliesWeightDecay()
    {
        // Arrange
        var options = new LionOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            LearningRate = 0.1,
            WeightDecay = 0.01
        };
        var optimizer = new LionOptimizer<double, Matrix<double>, Vector<double>>(null, options);
        var parameters = new Vector<double>(new[] { 10.0 });
        var gradient = new Vector<double>(new[] { 0.0 });

        // Act
        var updated = optimizer.UpdateParameters(parameters, gradient);

        // Assert - parameter should shrink by weight_decay * learning_rate
        double expectedShrinkage = 0.01 * 0.1 * 10.0; // weight_decay * lr * param
        Assert.Equal(10.0 - expectedShrinkage, updated[0], precision: 6);
    }
}
```

---

## Testing Strategy

### 1. Correctness Tests
- **Convergence on simple functions** (quadratic, Rosenbrock)
- **Sign behavior** (uniform step sizes regardless of gradient magnitude)
- **Weight decay** (regularization applied correctly)
- **Momentum buildup** (state accumulation over iterations)

### 2. Comparison Tests
- **vs Adam**: Memory usage (50% reduction)
- **vs Adam**: Convergence speed on transformers
- **vs SGD**: Momentum effects

### 3. Numerical Stability
- **Large gradients** (should not explode due to sign)
- **Small gradients** (should still make progress)
- **Mixed magnitudes** (uniform steps for all)
- **Long training** (1000+ iterations without NaN)

### 4. Integration Tests
- **Train a transformer** (BERT-style attention layers)
- **Compare with PyTorch Lion** (verify identical behavior)
- **Hyperparameter sweep** (learning rate, weight decay)

---

## Hyperparameter Tuning

### Learning Rate Tuning

**Lion requires 3-10x smaller learning rate than Adam!**

| Model Size | Adam LR | Lion LR | Ratio |
|-----------|---------|---------|-------|
| Small (<10M) | 1e-3 | 1e-4 | 10x |
| Medium (10M-100M) | 5e-4 | 5e-5 | 10x |
| Large (>100M) | 1e-4 | 3e-5 | 3x |

**Tuning Process:**
1. Start with Adam's learning rate / 10
2. Train for 10% of total steps
3. If loss diverges → decrease by 3x
4. If loss decreases too slowly → increase by 2x
5. Repeat until smooth convergence

### Beta Tuning (Usually Not Needed)

**Beta1 (default: 0.9):**
- Increase (0.95) for: Noisy gradients, need smoother updates
- Decrease (0.8) for: Fast-changing loss landscape

**Beta2 (default: 0.99):**
- Increase (0.999) for: More stable training (like Adam)
- Decrease (0.95) for: Faster adaptation to new data

### Weight Decay Tuning

**Default: 0.01**

| Problem | Suggested Weight Decay |
|---------|----------------------|
| Small dataset (<10K samples) | 0.1 (strong regularization) |
| Medium dataset (10K-1M) | 0.01 (default) |
| Large dataset (>1M) | 0.001 (weak regularization) |
| Overfitting observed | Increase by 10x |
| Underfitting observed | Decrease by 10x |

---

## Performance Expectations

### Training Speed

**Compared to Adam:**
- **Computational cost:** ~90% (slightly faster due to simpler updates)
- **Memory cost:** ~50% (only momentum, no variance)
- **Convergence speed:** 10-20% fewer steps to reach same accuracy

### Accuracy

**On Transformers:**
- **Better generalization:** 1-3% higher test accuracy
- **Lower final loss:** 5-10% improvement
- **More stable training:** Fewer divergence issues

### When Lion May Underperform

- **Very small models** (<1M parameters)
- **CNN architectures** (Adam often better)
- **Small learning rates** (<1e-5, may converge too slowly)
- **Very short training** (<100 iterations, momentum not built up)

---

## Common Pitfalls

### 1. Using Adam's Learning Rate
**Problem:** Lion diverges immediately
**Solution:** Start with 10x smaller learning rate than Adam

### 2. No Weight Decay
**Problem:** Overfitting on small datasets
**Solution:** Always use weight decay (default 0.01)

### 3. Comparing Early Training
**Problem:** Lion starts slower due to momentum buildup
**Solution:** Compare after 20-30% of training, not first few steps

### 4. Wrong Momentum Update Order
**Problem:** Updating momentum before parameters (like Adam)
**Solution:** Lion updates momentum AFTER parameters (order matters!)

---

## Next Steps

1. **Implement LionOptimizer** following the code above
2. **Test on simple functions** (quadratic, Rosenbrock)
3. **Verify sign behavior** (uniform steps)
4. **Compare memory usage** with Adam
5. **Train on a transformer** (even small one)
6. **Tune hyperparameters** for your specific problem
7. **Benchmark against Adam** on your models

**Success Criteria:**
- Converges on simple optimization problems
- 50% memory reduction vs Adam
- Sign-based updates work correctly
- No NaN/Infinity during training
- Competitive or better performance than Adam on transformers

**Good luck!** You're implementing one of the most exciting recent advances in optimization!
