# Policy Architecture Base Classes and Additional Implementations

## Task Overview

Implement base classes and additional concrete implementations for the RL policy architecture to make this a comprehensive, top-tier AI library. This follows the modular policy design established in PR #481.

**Branch**: `claude/fix-issue-394-011CV3HkgfwwbaSAdrzrKd58` (PR #481)
**Worktree**: `/c/Users/cheat/source/repos/worktrees/pr-481-1763014665`

## Current State

**Already Implemented** (DO NOT recreate these):
- `IPolicy<T>` - Core policy interface
- `DiscretePolicy<T>` - Categorical distribution for discrete actions
- `ContinuousPolicy<T>` - Gaussian distribution for continuous actions
- `IExplorationStrategy<T>` - Core exploration interface
- `EpsilonGreedyExploration<T>` - ε-greedy exploration
- `GaussianNoiseExploration<T>` - Gaussian noise for continuous actions
- `NoExploration<T>` - Pure exploitation (greedy)
- `DiscretePolicyOptions<T>` - Configuration for discrete policies
- `ContinuousPolicyOptions<T>` - Configuration for continuous policies

## Part 1: Create Base Classes

### 1.1 PolicyBase<T>

**File**: `src/ReinforcementLearning/Policies/PolicyBase.cs`

**Purpose**: Abstract base class implementing common functionality for all policies.

**Key Features**:
- Protected NumOps field for numeric operations
- Protected Random instance management
- IDisposable pattern implementation
- Common validation logic
- Network management helpers

**Code Template**:
```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Abstract base class for policy implementations.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public abstract class PolicyBase<T> : IPolicy<T>
    {
        protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        protected readonly Random _random;
        protected bool _disposed;

        protected PolicyBase(Random? random = null)
        {
            _random = random ?? new Random();
        }

        public abstract Vector<T> SelectAction(Vector<T> state, bool training = true);
        public abstract T ComputeLogProb(Vector<T> state, Vector<T> action);
        public abstract IReadOnlyList<INeuralNetwork<T>> GetNetworks();

        public virtual void Reset()
        {
            // Base implementation - derived classes can override
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected void ValidateActionSize(int expected, int actual, string paramName)
        {
            if (actual != expected)
            {
                throw new ArgumentException(
                    $"Action size mismatch. Expected {expected}, got {actual}.",
                    paramName);
            }
        }
    }
}
```

### 1.2 ExplorationStrategyBase<T>

**File**: `src/ReinforcementLearning/Policies/Exploration/ExplorationStrategyBase.cs`

**Purpose**: Abstract base class for exploration strategies.

**Code Template**:
```csharp
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using System;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Abstract base class for exploration strategy implementations.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public abstract class ExplorationStrategyBase<T> : IExplorationStrategy<T>
    {
        protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        public abstract Vector<T> GetExplorationAction(
            Vector<T> state,
            Vector<T> policyAction,
            int actionSpaceSize,
            Random random);

        public abstract void Update();

        public virtual void Reset()
        {
            // Base implementation - derived classes can override
        }

        protected T BoxMullerSample(Random random)
        {
            // Box-Muller transform for Gaussian sampling
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();
            double normalSample = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return NumOps.FromDouble(normalSample);
        }

        protected Vector<T> ClampAction(Vector<T> action, double min = -1.0, double max = 1.0)
        {
            var clampedAction = new Vector<T>(action.Length);
            for (int i = 0; i < action.Length; i++)
            {
                double value = NumOps.ToDouble(action[i]);
                // Math.Clamp not available in net462
                double clamped = Math.Max(min, Math.Min(max, value));
                clampedAction[i] = NumOps.FromDouble(clamped);
            }
            return clampedAction;
        }
    }
}
```

## Part 2: Additional Exploration Strategies

### 2.1 BoltzmannExploration<T>

**File**: `src/ReinforcementLearning/Policies/Exploration/BoltzmannExploration.cs`

**Purpose**: Temperature-based exploration (softmax action selection).

**Features**:
- Temperature parameter controls randomness
- Decaying temperature over time
- For discrete action spaces

**Key Methods**:
- Softmax with temperature: `exp(Q(a)/τ) / Σ exp(Q(a')/τ)`
- Temperature annealing

### 2.2 OrnsteinUhlenbeckNoise<T>

**File**: `src/ReinforcementLearning/Policies/Exploration/OrnsteinUhlenbeckNoise.cs`

**Purpose**: Temporally correlated noise for continuous control (from DDPG paper).

**Features**:
- Mean-reverting stochastic process
- Parameters: θ (mean reversion), σ (volatility), μ (mean)
- Formula: `dx = θ(μ - x)dt + σdW`

**Code Pattern**:
```csharp
public class OrnsteinUhlenbeckNoise<T> : ExplorationStrategyBase<T>
{
    private readonly double _theta;      // Mean reversion rate
    private readonly double _sigma;      // Volatility
    private readonly double _mu;         // Long-term mean
    private Vector<T> _state;           // Current noise state

    public Vector<T> GetExplorationAction(...)
    {
        var noisyAction = new Vector<T>(actionSpaceSize);
        for (int i = 0; i < actionSpaceSize; i++)
        {
            // dx = θ(μ - x)dt + σdW
            double x = NumOps.ToDouble(_state[i]);
            double dx = _theta * (_mu - x) + _sigma * BoxMullerSample(random);
            _state[i] = NumOps.FromDouble(x + dx);

            double actionValue = NumOps.ToDouble(policyAction[i]) + x;
            noisyAction[i] = NumOps.FromDouble(actionValue);
        }
        return ClampAction(noisyAction);
    }
}
```

### 2.3 UpperConfidenceBoundExploration<T>

**File**: `src/ReinforcementLearning/Policies/Exploration/UpperConfidenceBoundExploration.cs`

**Purpose**: UCB exploration for multi-armed bandits and discrete action spaces.

**Features**:
- Exploration bonus: `√(2 ln(t) / N(a))`
- Tracks action counts
- Balances exploration and exploitation

### 2.4 ThompsonSamplingExploration<T>

**File**: `src/ReinforcementLearning/Policies/Exploration/ThompsonSamplingExploration.cs`

**Purpose**: Bayesian exploration by sampling from posterior distributions.

**Features**:
- Beta distribution sampling for discrete actions
- Gaussian distribution sampling for continuous actions

## Part 3: Additional Policy Implementations

### 3.1 DeterministicPolicy<T>

**File**: `src/ReinforcementLearning/Policies/DeterministicPolicy.cs`

**Purpose**: Deterministic policy for DDPG, TD3 (outputs single action, no sampling).

**Features**:
- Direct action output (no distribution)
- Optional Tanh squashing for bounded actions
- Used in deterministic policy gradient methods

**Extends**: `PolicyBase<T>`

### 3.2 MixedPolicy<T>

**File**: `src/ReinforcementLearning/Policies/MixedPolicy.cs`

**Purpose**: Policy for environments with both discrete and continuous action spaces.

**Features**:
- Separate networks or heads for discrete and continuous components
- Composite action vector: `[discrete_actions, continuous_actions]`

### 3.3 MultiModalPolicy<T>

**File**: `src/ReinforcementLearning/Policies/MultiModalPolicy.cs`

**Purpose**: Policy with mixture of Gaussians for multi-modal action distributions.

**Features**:
- Multiple Gaussian components
- Categorical distribution over components
- Useful for complex behaviors with multiple modes

### 3.4 BetaPolicy<T>

**File**: `src/ReinforcementLearning/Policies/BetaPolicy.cs`

**Purpose**: Policy using Beta distribution for bounded continuous actions [0, 1].

**Features**:
- Network outputs alpha and beta parameters
- Actions naturally bounded to [0, 1]
- Optional scaling to [min, max]

## Part 4: Policy Options Classes

Create configuration classes for each new policy:

### 4.1 DeterministicPolicyOptions<T>
- Network architecture configuration
- Action bounds
- Tanh squashing flag

### 4.2 MixedPolicyOptions<T>
- Discrete action size
- Continuous action size
- Separate or shared network flag

### 4.3 MultiModalPolicyOptions<T>
- Number of mixture components
- Shared parameters flag

### 4.4 BetaPolicyOptions<T>
- Action min/max bounds
- Network architecture

## Critical Coding Standards (MUST FOLLOW)

### 1. NEVER Use Null-Forgiving Operator (!)

```csharp
// ❌ WRONG
public Vector<T> Action { get; set; } = default!;

// ✅ CORRECT
public Vector<T> Action { get; set; } = new Vector<T>(0);
```

### 2. Use NumOps (Non-Generic) via Static Field

```csharp
// ✅ CORRECT
private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

// Then use:
T value = NumOps.FromDouble(1.0);
double doubleValue = NumOps.ToDouble(value);
```

### 3. Multi-Framework Compatibility (net462, net471, net8.0)

```csharp
// ❌ WRONG (Math.Clamp not in net462)
double clamped = Math.Clamp(value, -1.0, 1.0);

// ✅ CORRECT
double clamped = Math.Max(-1.0, Math.Min(1.0, value));

// ❌ WRONG (List.TakeLast not in net462)
var last5 = myList.TakeLast(5);

// ✅ CORRECT
var last5 = myList.Skip(Math.Max(0, myList.Count - 5));
```

### 4. Proper Initialization

```csharp
// ✅ CORRECT - Initialize in constructor or with default values
private Vector<T> _state;

public OrnsteinUhlenbeckNoise(int actionSize)
{
    _state = new Vector<T>(actionSize);  // Initialize in constructor
}
```

### 5. Follow Existing Code Patterns

Look at these files for reference:
- `src/ReinforcementLearning/Agents/DoubleDQN/DoubleDQNAgent.cs` - NumOps usage
- `src/ReinforcementLearning/Agents/DecisionTransformer/DecisionTransformerAgent.cs` - Vector/Tensor conversions
- `src/ReinforcementLearning/Policies/DiscretePolicy.cs` - Policy pattern
- `src/ReinforcementLearning/Policies/Exploration/EpsilonGreedyExploration.cs` - Exploration pattern

## Implementation Order

1. **First**: Create base classes (PolicyBase, ExplorationStrategyBase)
2. **Second**: Refactor existing policies to inherit from PolicyBase
3. **Third**: Refactor existing explorations to inherit from ExplorationStrategyBase
4. **Fourth**: Implement additional exploration strategies (Boltzmann, OU, UCB, Thompson)
5. **Fifth**: Implement additional policy types (Deterministic, Mixed, MultiModal, Beta)
6. **Sixth**: Create policy options classes
7. **Finally**: Build and verify all frameworks (net462, net471, net8.0)

## Testing Checklist

After implementation:
- [ ] All files compile on net8.0
- [ ] All files compile on net462
- [ ] All files compile on net471
- [ ] No use of null-forgiving operator (!)
- [ ] No use of Math.Clamp or other net6+ features
- [ ] NumOps initialized via MathHelper.GetNumericOperations<T>()
- [ ] All classes have XML documentation
- [ ] Follow naming conventions (PascalCase for public, _camelCase for private)

## Success Criteria

**Build Status**:
- 0 errors on all frameworks
- 0 warnings (ideally)

**Architecture**:
- All policies inherit from PolicyBase<T>
- All explorations inherit from ExplorationStrategyBase<T>
- Clear separation of concerns
- Consistent API across all implementations

**Completeness**:
- 7+ policy types (including existing 2)
- 7+ exploration strategies (including existing 3)
- Options classes for all policies
- Base classes with shared functionality

## Reference: Leading RL Libraries

Our policy architecture should match or exceed:
- **Stable Baselines3**: Modular policies, multiple exploration strategies
- **RLlib**: Policy/exploration separation, deterministic policies
- **TensorFlow Agents**: Distribution-based policies, composite actions

## Questions?

If you encounter ambiguities:
1. Check existing working code in the codebase
2. Follow patterns from DoubleDQNAgent.cs and DecisionTransformerAgent.cs
3. When in doubt, prefer explicit null checks over nullable types
4. Always prioritize multi-framework compatibility

## Final Notes

- This is a comprehensive, top-tier AI library - implement with production quality
- Follow C# best practices and SOLID principles
- Use meaningful variable names and thorough documentation
- Test incrementally - don't implement everything before testing
- Commit frequently (after each file or logical unit)

Good luck! This will make the RL policy architecture truly comprehensive.
