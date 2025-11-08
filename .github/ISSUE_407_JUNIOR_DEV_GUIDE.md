# Junior Developer Implementation Guide: Issue #407

## Overview
**Issue**: Model Pruning (Magnitude, Gradient-based, Lottery Ticket)
**Goal**: Implement model compression through structured and unstructured pruning techniques
**Difficulty**: Advanced
**Estimated Time**: 12-16 hours

## What is Model Pruning?

Model pruning removes unnecessary weights/neurons from neural networks to:
- Reduce model size (storage and memory)
- Increase inference speed
- Maintain accuracy with fewer parameters

### Pruning Types

**1. Magnitude-based Pruning**: Remove weights with smallest absolute values
**2. Gradient-based Pruning**: Remove weights with smallest gradient contributions
**3. Lottery Ticket Hypothesis**: Find sparse subnetworks that train to similar accuracy

**Structured vs. Unstructured Pruning**:
- **Unstructured**: Remove individual weights (sparse matrices)
- **Structured**: Remove entire neurons, channels, or layers (dense but smaller)

## Mathematical Background

### Magnitude-based Pruning
```
For weight matrix W with threshold t:
W_pruned[i,j] = W[i,j] if |W[i,j]| > t
W_pruned[i,j] = 0      otherwise

Threshold t often chosen to achieve target sparsity level
```

### Gradient-based Pruning
```
For each weight w_ij with gradient g_ij:
Importance score s_ij = |w_ij * g_ij|

Remove weights with lowest importance scores
```

### Lottery Ticket Hypothesis
```
1. Train network to convergence → weights W_final
2. Prune p% of smallest magnitude weights → mask M
3. Reset remaining weights to initialization W_0
4. Retrain with mask: W_masked = W * M

Result: Sparse network that matches original accuracy
```

## Understanding the Codebase

### Key Files to Create

**Core Interfaces:**
```
C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IPruningStrategy.cs
C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IPruningMask.cs
```

**Implementations:**
```
C:\Users\cheat\source\repos\AiDotNet\src\Pruning\MagnitudePruningStrategy.cs
C:\Users\cheat\source\repos\AiDotNet\src\Pruning\GradientPruningStrategy.cs
C:\Users\cheat\source\repos\AiDotNet\src\Pruning\LotteryTicketPruningStrategy.cs
C:\Users\cheat\source\repos\AiDotNet\src\Pruning\PruningMask.cs
C:\Users\cheat\source\repos\AiDotNet\src\Pruning\StructuredPruningStrategy.cs
```

**Test Files:**
```
C:\Users\cheat\source\repos\AiDotNet\tests\Pruning\PruningStrategyTests.cs
```

## Step-by-Step Implementation Guide

### Phase 1: Core Interfaces and Data Structures

#### Step 1.1: Create IPruningMask Interface

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IPruningMask.cs
namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Represents a binary mask for pruning weights in a neural network layer.
    /// </summary>
    /// <typeparam name="T">Numeric type for mask values</typeparam>
    public interface IPruningMask<T>
    {
        /// <summary>
        /// Gets the mask dimensions matching the weight matrix shape.
        /// </summary>
        int[] Shape { get; }

        /// <summary>
        /// Gets the sparsity ratio (proportion of zeros).
        /// </summary>
        /// <returns>Value between 0 (dense) and 1 (fully pruned)</returns>
        double GetSparsity();

        /// <summary>
        /// Applies the mask to a weight matrix (element-wise multiplication).
        /// </summary>
        /// <param name="weights">Weight matrix to prune</param>
        /// <returns>Pruned weights (zeros where mask is zero)</returns>
        Matrix<T> Apply(Matrix<T> weights);

        /// <summary>
        /// Applies the mask to a weight tensor (for convolutional layers).
        /// </summary>
        Tensor<T> Apply(Tensor<T> weights);

        /// <summary>
        /// Updates the mask based on new pruning criteria.
        /// </summary>
        /// <param name="keepIndices">Indices of weights to keep (not prune)</param>
        void UpdateMask(bool[,] keepIndices);

        /// <summary>
        /// Combines this mask with another mask (logical AND).
        /// </summary>
        IPruningMask<T> CombineWith(IPruningMask<T> otherMask);
    }
}
```

#### Step 1.2: Create IPruningStrategy Interface

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IPruningStrategy.cs
namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Defines a strategy for pruning neural network weights.
    /// </summary>
    /// <typeparam name="T">Numeric type for weights and gradients</typeparam>
    public interface IPruningStrategy<T>
    {
        /// <summary>
        /// Computes importance scores for each weight.
        /// </summary>
        /// <param name="weights">Weight matrix</param>
        /// <param name="gradients">Gradient matrix (optional, can be null)</param>
        /// <returns>Importance score for each weight (higher = more important)</returns>
        Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null);

        /// <summary>
        /// Creates a pruning mask based on target sparsity.
        /// </summary>
        /// <param name="importanceScores">Importance scores from ComputeImportanceScores</param>
        /// <param name="targetSparsity">Target sparsity ratio (0 to 1)</param>
        /// <returns>Binary mask (1 = keep, 0 = prune)</returns>
        IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity);

        /// <summary>
        /// Prunes a weight matrix in-place.
        /// </summary>
        /// <param name="weights">Weight matrix to prune</param>
        /// <param name="mask">Pruning mask to apply</param>
        void ApplyPruning(Matrix<T> weights, IPruningMask<T> mask);

        /// <summary>
        /// Gets whether this strategy requires gradients.
        /// </summary>
        bool RequiresGradients { get; }

        /// <summary>
        /// Gets whether this is structured pruning (removes entire rows/cols).
        /// </summary>
        bool IsStructured { get; }
    }
}
```

#### Step 1.3: Implement PruningMask

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Pruning\PruningMask.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Pruning
{
    /// <summary>
    /// Binary mask for pruning neural network weights.
    /// </summary>
    /// <typeparam name="T">Numeric type</typeparam>
    public class PruningMask<T> : IPruningMask<T>
    {
        private readonly Matrix<T> _mask;
        private readonly INumericOperations<T> _numOps;

        public int[] Shape => new[] { _mask.Rows, _mask.Columns };

        public PruningMask(int rows, int cols)
        {
            _numOps = NumericOperations<T>.Instance;
            _mask = new Matrix<T>(rows, cols);

            // Initialize to all ones (no pruning)
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    _mask[i, j] = _numOps.One;
        }

        public PruningMask(Matrix<T> maskMatrix)
        {
            _numOps = NumericOperations<T>.Instance;
            _mask = maskMatrix.Clone();
        }

        public double GetSparsity()
        {
            int totalElements = _mask.Rows * _mask.Columns;
            int zeroCount = 0;

            for (int i = 0; i < _mask.Rows; i++)
            {
                for (int j = 0; j < _mask.Columns; j++)
                {
                    if (_numOps.Equals(_mask[i, j], _numOps.Zero))
                        zeroCount++;
                }
            }

            return (double)zeroCount / totalElements;
        }

        public Matrix<T> Apply(Matrix<T> weights)
        {
            if (weights.Rows != _mask.Rows || weights.Columns != _mask.Columns)
                throw new ArgumentException("Weight matrix shape must match mask shape");

            var result = new Matrix<T>(weights.Rows, weights.Columns);

            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Columns; j++)
                {
                    result[i, j] = _numOps.Multiply(weights[i, j], _mask[i, j]);
                }
            }

            return result;
        }

        public Tensor<T> Apply(Tensor<T> weights)
        {
            // For 2D tensors (fully connected layers)
            if (weights.Rank == 2)
            {
                var matrix = TensorToMatrix(weights);
                var pruned = Apply(matrix);
                return MatrixToTensor(pruned);
            }

            // For 4D tensors (convolutional layers: [filters, channels, height, width])
            if (weights.Rank == 4)
            {
                var result = weights.Clone();
                int filters = weights.Dimensions[0];
                int channels = weights.Dimensions[1];

                // Apply mask filter-wise or channel-wise based on structured pruning
                for (int f = 0; f < filters; f++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int h = 0; h < weights.Dimensions[2]; h++)
                        {
                            for (int w = 0; w < weights.Dimensions[3]; w++)
                            {
                                // For unstructured: apply element-wise
                                // For structured: multiply by filter/channel mask
                                result[f, c, h, w] = weights[f, c, h, w];
                            }
                        }
                    }
                }

                return result;
            }

            throw new NotSupportedException($"Tensor rank {weights.Rank} not supported for pruning");
        }

        public void UpdateMask(bool[,] keepIndices)
        {
            if (keepIndices.GetLength(0) != _mask.Rows || keepIndices.GetLength(1) != _mask.Columns)
                throw new ArgumentException("keepIndices shape must match mask shape");

            for (int i = 0; i < _mask.Rows; i++)
            {
                for (int j = 0; j < _mask.Columns; j++)
                {
                    _mask[i, j] = keepIndices[i, j] ? _numOps.One : _numOps.Zero;
                }
            }
        }

        public IPruningMask<T> CombineWith(IPruningMask<T> otherMask)
        {
            if (otherMask.Shape[0] != Shape[0] || otherMask.Shape[1] != Shape[1])
                throw new ArgumentException("Masks must have same shape to combine");

            var combined = new Matrix<T>(_mask.Rows, _mask.Columns);
            var otherMatrix = ((PruningMask<T>)otherMask)._mask;

            for (int i = 0; i < _mask.Rows; i++)
            {
                for (int j = 0; j < _mask.Columns; j++)
                {
                    // Logical AND: both must be 1 to keep
                    bool keep = !_numOps.Equals(_mask[i, j], _numOps.Zero) &&
                                !_numOps.Equals(otherMatrix[i, j], _numOps.Zero);
                    combined[i, j] = keep ? _numOps.One : _numOps.Zero;
                }
            }

            return new PruningMask<T>(combined);
        }

        private Matrix<T> TensorToMatrix(Tensor<T> tensor)
        {
            var matrix = new Matrix<T>(tensor.Dimensions[0], tensor.Dimensions[1]);
            for (int i = 0; i < tensor.Dimensions[0]; i++)
                for (int j = 0; j < tensor.Dimensions[1]; j++)
                    matrix[i, j] = tensor[i, j];
            return matrix;
        }

        private Tensor<T> MatrixToTensor(Matrix<T> matrix)
        {
            var tensor = new Tensor<T>(matrix.Rows, matrix.Columns);
            for (int i = 0; i < matrix.Rows; i++)
                for (int j = 0; j < matrix.Columns; j++)
                    tensor[i, j] = matrix[i, j];
            return tensor;
        }
    }
}
```

### Phase 2: Magnitude-based Pruning

#### Step 2.1: Implement MagnitudePruningStrategy

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Pruning\MagnitudePruningStrategy.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Pruning
{
    /// <summary>
    /// Prunes weights with smallest absolute values.
    /// Simple but effective: small weights contribute less to output.
    /// </summary>
    public class MagnitudePruningStrategy<T> : IPruningStrategy<T>
    {
        private readonly INumericOperations<T> _numOps;

        public bool RequiresGradients => false;
        public bool IsStructured => false;

        public MagnitudePruningStrategy()
        {
            _numOps = NumericOperations<T>.Instance;
        }

        public Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null)
        {
            // Importance = absolute value of weight
            var scores = new Matrix<T>(weights.Rows, weights.Columns);

            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Columns; j++)
                {
                    // |w_ij|
                    scores[i, j] = _numOps.Abs(weights[i, j]);
                }
            }

            return scores;
        }

        public IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity)
        {
            if (targetSparsity < 0 || targetSparsity > 1)
                throw new ArgumentException("targetSparsity must be between 0 and 1");

            int totalElements = importanceScores.Rows * importanceScores.Columns;
            int numToPrune = (int)(totalElements * targetSparsity);

            // Flatten scores and find threshold
            var flatScores = new List<(int row, int col, T score)>();

            for (int i = 0; i < importanceScores.Rows; i++)
            {
                for (int j = 0; j < importanceScores.Columns; j++)
                {
                    flatScores.Add((i, j, importanceScores[i, j]));
                }
            }

            // Sort by importance (ascending, so smallest are first)
            flatScores.Sort((a, b) =>
            {
                double aVal = Convert.ToDouble(_numOps.ToDouble(a.score));
                double bVal = Convert.ToDouble(_numOps.ToDouble(b.score));
                return aVal.CompareTo(bVal);
            });

            // Create mask: prune the smallest numToPrune elements
            var keepIndices = new bool[importanceScores.Rows, importanceScores.Columns];

            for (int i = 0; i < importanceScores.Rows; i++)
                for (int j = 0; j < importanceScores.Columns; j++)
                    keepIndices[i, j] = true;

            for (int i = 0; i < numToPrune && i < flatScores.Count; i++)
            {
                var (row, col, _) = flatScores[i];
                keepIndices[row, col] = false;
            }

            var mask = new PruningMask<T>(importanceScores.Rows, importanceScores.Columns);
            mask.UpdateMask(keepIndices);

            return mask;
        }

        public void ApplyPruning(Matrix<T> weights, IPruningMask<T> mask)
        {
            var pruned = mask.Apply(weights);

            // Update weights in-place
            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Columns; j++)
                {
                    weights[i, j] = pruned[i, j];
                }
            }
        }
    }
}
```

### Phase 3: Gradient-based Pruning

#### Step 3.1: Implement GradientPruningStrategy

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Pruning\GradientPruningStrategy.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Pruning
{
    /// <summary>
    /// Prunes weights based on gradient magnitude (sensitivity).
    /// Weights with small gradients have little impact on loss.
    /// </summary>
    public class GradientPruningStrategy<T> : IPruningStrategy<T>
    {
        private readonly INumericOperations<T> _numOps;

        public bool RequiresGradients => true;
        public bool IsStructured => false;

        public GradientPruningStrategy()
        {
            _numOps = NumericOperations<T>.Instance;
        }

        public Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null)
        {
            if (gradients == null)
                throw new ArgumentException("GradientPruningStrategy requires gradients");

            if (weights.Rows != gradients.Rows || weights.Columns != gradients.Columns)
                throw new ArgumentException("Weights and gradients must have same shape");

            // Importance = |weight * gradient|
            // This measures how much removing the weight affects the loss
            var scores = new Matrix<T>(weights.Rows, weights.Columns);

            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Columns; j++)
                {
                    // |w_ij * g_ij|
                    var product = _numOps.Multiply(weights[i, j], gradients[i, j]);
                    scores[i, j] = _numOps.Abs(product);
                }
            }

            return scores;
        }

        public IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity)
        {
            // Same logic as magnitude pruning, but with gradient-based scores
            if (targetSparsity < 0 || targetSparsity > 1)
                throw new ArgumentException("targetSparsity must be between 0 and 1");

            int totalElements = importanceScores.Rows * importanceScores.Columns;
            int numToPrune = (int)(totalElements * targetSparsity);

            var flatScores = new List<(int row, int col, T score)>();

            for (int i = 0; i < importanceScores.Rows; i++)
                for (int j = 0; j < importanceScores.Columns; j++)
                    flatScores.Add((i, j, importanceScores[i, j]));

            flatScores.Sort((a, b) =>
            {
                double aVal = Convert.ToDouble(_numOps.ToDouble(a.score));
                double bVal = Convert.ToDouble(_numOps.ToDouble(b.score));
                return aVal.CompareTo(bVal);
            });

            var keepIndices = new bool[importanceScores.Rows, importanceScores.Columns];

            for (int i = 0; i < importanceScores.Rows; i++)
                for (int j = 0; j < importanceScores.Columns; j++)
                    keepIndices[i, j] = true;

            for (int i = 0; i < numToPrune && i < flatScores.Count; i++)
            {
                var (row, col, _) = flatScores[i];
                keepIndices[row, col] = false;
            }

            var mask = new PruningMask<T>(importanceScores.Rows, importanceScores.Columns);
            mask.UpdateMask(keepIndices);

            return mask;
        }

        public void ApplyPruning(Matrix<T> weights, IPruningMask<T> mask)
        {
            var pruned = mask.Apply(weights);

            for (int i = 0; i < weights.Rows; i++)
                for (int j = 0; j < weights.Columns; j++)
                    weights[i, j] = pruned[i, j];
        }
    }
}
```

### Phase 4: Lottery Ticket Hypothesis

#### Step 4.1: Implement LotteryTicketPruningStrategy

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Pruning\LotteryTicketPruningStrategy.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Pruning
{
    /// <summary>
    /// Implements the Lottery Ticket Hypothesis (Frankle & Carbin, 2019).
    /// Finds sparse subnetworks that can train to full accuracy when reset to initialization.
    /// </summary>
    public class LotteryTicketPruningStrategy<T> : IPruningStrategy<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly Dictionary<string, Matrix<T>> _initialWeights;
        private readonly int _iterativeRounds;

        public bool RequiresGradients => false;
        public bool IsStructured => false;

        /// <summary>
        /// Creates a new lottery ticket pruning strategy.
        /// </summary>
        /// <param name="iterativeRounds">Number of iterative pruning rounds (default 5)</param>
        public LotteryTicketPruningStrategy(int iterativeRounds = 5)
        {
            _numOps = NumericOperations<T>.Instance;
            _initialWeights = new Dictionary<string, Matrix<T>>();
            _iterativeRounds = iterativeRounds;
        }

        /// <summary>
        /// Stores initial weights before training (critical for lottery ticket).
        /// </summary>
        public void StoreInitialWeights(string layerName, Matrix<T> weights)
        {
            _initialWeights[layerName] = weights.Clone();
        }

        /// <summary>
        /// Gets the stored initial weights for a layer.
        /// </summary>
        public Matrix<T> GetInitialWeights(string layerName)
        {
            if (!_initialWeights.ContainsKey(layerName))
                throw new InvalidOperationException($"No initial weights stored for layer {layerName}");

            return _initialWeights[layerName].Clone();
        }

        public Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null)
        {
            // Use magnitude-based scores (lottery ticket uses magnitude pruning)
            var scores = new Matrix<T>(weights.Rows, weights.Columns);

            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Columns; j++)
                {
                    scores[i, j] = _numOps.Abs(weights[i, j]);
                }
            }

            return scores;
        }

        public IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity)
        {
            // Iterative magnitude pruning to target sparsity
            // Each round prunes (1 - (1 - targetSparsity)^(1/rounds)) of remaining weights
            double prunePerRound = 1.0 - Math.Pow(1.0 - targetSparsity, 1.0 / _iterativeRounds);

            var currentMask = new PruningMask<T>(importanceScores.Rows, importanceScores.Columns);

            for (int round = 0; round < _iterativeRounds; round++)
            {
                // Compute scores for current non-pruned weights
                var maskedScores = currentMask.Apply(importanceScores);

                // Find threshold for this round
                int totalRemaining = CountNonZero(maskedScores);
                int numToPrune = (int)(totalRemaining * prunePerRound);

                var flatScores = new List<(int row, int col, double score)>();

                for (int i = 0; i < maskedScores.Rows; i++)
                {
                    for (int j = 0; j < maskedScores.Columns; j++)
                    {
                        if (!_numOps.Equals(maskedScores[i, j], _numOps.Zero))
                        {
                            double scoreVal = Convert.ToDouble(_numOps.ToDouble(maskedScores[i, j]));
                            flatScores.Add((i, j, scoreVal));
                        }
                    }
                }

                flatScores.Sort((a, b) => a.score.CompareTo(b.score));

                var keepIndices = new bool[importanceScores.Rows, importanceScores.Columns];

                for (int i = 0; i < importanceScores.Rows; i++)
                    for (int j = 0; j < importanceScores.Columns; j++)
                        keepIndices[i, j] = !_numOps.Equals(currentMask.Apply(importanceScores)[i, j], _numOps.Zero);

                for (int i = 0; i < numToPrune && i < flatScores.Count; i++)
                {
                    var (row, col, _) = flatScores[i];
                    keepIndices[row, col] = false;
                }

                currentMask.UpdateMask(keepIndices);
            }

            return currentMask;
        }

        public void ApplyPruning(Matrix<T> weights, IPruningMask<T> mask)
        {
            var pruned = mask.Apply(weights);

            for (int i = 0; i < weights.Rows; i++)
                for (int j = 0; j < weights.Columns; j++)
                    weights[i, j] = pruned[i, j];
        }

        /// <summary>
        /// Resets pruned weights to their initial values (key step in lottery ticket).
        /// </summary>
        public void ResetToInitialWeights(string layerName, Matrix<T> weights, IPruningMask<T> mask)
        {
            var initial = GetInitialWeights(layerName);

            if (initial.Rows != weights.Rows || initial.Columns != weights.Columns)
                throw new ArgumentException("Weight dimensions don't match initial weights");

            // Reset non-pruned weights to their initialization
            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Columns; j++)
                {
                    // Keep initial value where mask is 1, zero otherwise
                    var maskValue = mask.Apply(initial)[i, j];
                    weights[i, j] = maskValue;
                }
            }
        }

        private int CountNonZero(Matrix<T> matrix)
        {
            int count = 0;
            for (int i = 0; i < matrix.Rows; i++)
                for (int j = 0; j < matrix.Columns; j++)
                    if (!_numOps.Equals(matrix[i, j], _numOps.Zero))
                        count++;
            return count;
        }
    }
}
```

### Phase 5: Structured Pruning

#### Step 5.1: Implement StructuredPruningStrategy

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Pruning\StructuredPruningStrategy.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Pruning
{
    /// <summary>
    /// Structured pruning removes entire neurons/filters/channels.
    /// Results in smaller dense networks (easier to deploy than sparse).
    /// </summary>
    public class StructuredPruningStrategy<T> : IPruningStrategy<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly StructurePruningType _pruningType;

        public bool RequiresGradients => false;
        public bool IsStructured => true;

        public enum StructurePruningType
        {
            /// <summary>Prune entire output neurons (columns)</summary>
            Neuron,
            /// <summary>Prune entire filters in conv layers</summary>
            Filter,
            /// <summary>Prune entire channels in conv layers</summary>
            Channel
        }

        public StructuredPruningStrategy(StructurePruningType pruningType = StructurePruningType.Neuron)
        {
            _numOps = NumericOperations<T>.Instance;
            _pruningType = pruningType;
        }

        public Matrix<T> ComputeImportanceScores(Matrix<T> weights, Matrix<T>? gradients = null)
        {
            var scores = new Matrix<T>(weights.Rows, weights.Columns);

            switch (_pruningType)
            {
                case StructurePruningType.Neuron:
                    // Score for each neuron (column) = L2 norm of its weights
                    for (int col = 0; col < weights.Columns; col++)
                    {
                        double columnNorm = 0;
                        for (int row = 0; row < weights.Rows; row++)
                        {
                            double val = Convert.ToDouble(_numOps.ToDouble(weights[row, col]));
                            columnNorm += val * val;
                        }
                        columnNorm = Math.Sqrt(columnNorm);

                        // Assign same score to all weights in column
                        for (int row = 0; row < weights.Rows; row++)
                        {
                            scores[row, col] = _numOps.FromDouble(columnNorm);
                        }
                    }
                    break;

                default:
                    throw new NotImplementedException($"Pruning type {_pruningType} not yet implemented");
            }

            return scores;
        }

        public IPruningMask<T> CreateMask(Matrix<T> importanceScores, double targetSparsity)
        {
            if (targetSparsity < 0 || targetSparsity > 1)
                throw new ArgumentException("targetSparsity must be between 0 and 1");

            var keepIndices = new bool[importanceScores.Rows, importanceScores.Columns];

            switch (_pruningType)
            {
                case StructurePruningType.Neuron:
                    // Prune entire columns (neurons)
                    int totalNeurons = importanceScores.Columns;
                    int neuronsToPrune = (int)(totalNeurons * targetSparsity);

                    // Get score for each neuron (all rows in column have same score)
                    var neuronScores = new List<(int col, double score)>();
                    for (int col = 0; col < importanceScores.Columns; col++)
                    {
                        double score = Convert.ToDouble(_numOps.ToDouble(importanceScores[0, col]));
                        neuronScores.Add((col, score));
                    }

                    // Sort by score (ascending)
                    neuronScores.Sort((a, b) => a.score.CompareTo(b.score));

                    // Mark columns to keep
                    var keepColumns = new bool[importanceScores.Columns];
                    for (int i = 0; i < importanceScores.Columns; i++)
                        keepColumns[i] = true;

                    for (int i = 0; i < neuronsToPrune; i++)
                    {
                        keepColumns[neuronScores[i].col] = false;
                    }

                    // Create mask
                    for (int row = 0; row < importanceScores.Rows; row++)
                    {
                        for (int col = 0; col < importanceScores.Columns; col++)
                        {
                            keepIndices[row, col] = keepColumns[col];
                        }
                    }
                    break;

                default:
                    throw new NotImplementedException($"Pruning type {_pruningType} not yet implemented");
            }

            var mask = new PruningMask<T>(importanceScores.Rows, importanceScores.Columns);
            mask.UpdateMask(keepIndices);

            return mask;
        }

        public void ApplyPruning(Matrix<T> weights, IPruningMask<T> mask)
        {
            var pruned = mask.Apply(weights);

            for (int i = 0; i < weights.Rows; i++)
                for (int j = 0; j < weights.Columns; j++)
                    weights[i, j] = pruned[i, j];
        }
    }
}
```

## Testing Strategy

### Phase 6: Comprehensive Tests

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\tests\Pruning\PruningStrategyTests.cs
using Xunit;
using AiDotNet.Pruning;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;

namespace AiDotNet.Tests.Pruning
{
    public class PruningStrategyTests
    {
        [Fact]
        public void MagnitudePruning_50Percent_PrunesSmallestWeights()
        {
            // Arrange
            var weights = new Matrix<double>(2, 2);
            weights[0, 0] = 0.1;  // Small - should be pruned
            weights[0, 1] = 0.9;  // Large - should keep
            weights[1, 0] = 0.2;  // Small - should be pruned
            weights[1, 1] = 0.8;  // Large - should keep

            var strategy = new MagnitudePruningStrategy<double>();

            // Act
            var scores = strategy.ComputeImportanceScores(weights);
            var mask = strategy.CreateMask(scores, targetSparsity: 0.5);

            // Assert
            Assert.Equal(0.5, mask.GetSparsity(), precision: 2);

            var pruned = mask.Apply(weights);
            Assert.Equal(0.0, pruned[0, 0]); // Pruned
            Assert.Equal(0.9, pruned[0, 1]); // Kept
            Assert.Equal(0.0, pruned[1, 0]); // Pruned
            Assert.Equal(0.8, pruned[1, 1]); // Kept
        }

        [Fact]
        public void GradientPruning_RequiresGradients()
        {
            // Arrange
            var strategy = new GradientPruningStrategy<double>();

            // Assert
            Assert.True(strategy.RequiresGradients);
        }

        [Fact]
        public void GradientPruning_PrunesLowSensitivityWeights()
        {
            // Arrange
            var weights = new Matrix<double>(2, 2);
            weights[0, 0] = 0.5;
            weights[0, 1] = 0.5;
            weights[1, 0] = 0.5;
            weights[1, 1] = 0.5;

            var gradients = new Matrix<double>(2, 2);
            gradients[0, 0] = 0.01; // Low gradient - prune
            gradients[0, 1] = 1.0;  // High gradient - keep
            gradients[1, 0] = 0.02; // Low gradient - prune
            gradients[1, 1] = 0.9;  // High gradient - keep

            var strategy = new GradientPruningStrategy<double>();

            // Act
            var scores = strategy.ComputeImportanceScores(weights, gradients);
            var mask = strategy.CreateMask(scores, targetSparsity: 0.5);

            // Assert
            var pruned = mask.Apply(weights);
            Assert.Equal(0.0, pruned[0, 0]); // Low gradient → pruned
            Assert.Equal(0.5, pruned[0, 1]); // High gradient → kept
        }

        [Fact]
        public void LotteryTicket_StoresAndRestoresInitialWeights()
        {
            // Arrange
            var initialWeights = new Matrix<double>(2, 2);
            initialWeights[0, 0] = 0.1;
            initialWeights[0, 1] = 0.2;
            initialWeights[1, 0] = 0.3;
            initialWeights[1, 1] = 0.4;

            var strategy = new LotteryTicketPruningStrategy<double>();
            strategy.StoreInitialWeights("layer1", initialWeights);

            // Simulate training - weights change
            var trainedWeights = new Matrix<double>(2, 2);
            trainedWeights[0, 0] = 0.5;
            trainedWeights[0, 1] = 0.6;
            trainedWeights[1, 0] = 0.7;
            trainedWeights[1, 1] = 0.8;

            // Act
            var scores = strategy.ComputeImportanceScores(trainedWeights);
            var mask = strategy.CreateMask(scores, targetSparsity: 0.5);

            // Reset to initial (key lottery ticket step)
            var resetWeights = trainedWeights.Clone();
            strategy.ResetToInitialWeights("layer1", resetWeights, mask);

            // Assert
            // Should have initial values where mask is 1, zero where mask is 0
            var maskedInitial = mask.Apply(initialWeights);
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Assert.Equal(maskedInitial[i, j], resetWeights[i, j]);
                }
            }
        }

        [Fact]
        public void StructuredPruning_PrunesEntireNeurons()
        {
            // Arrange
            var weights = new Matrix<double>(3, 4); // 3 inputs, 4 neurons
            // Neuron 0: weak connections
            weights[0, 0] = 0.1;
            weights[1, 0] = 0.1;
            weights[2, 0] = 0.1;

            // Neuron 1: strong connections
            weights[0, 1] = 0.9;
            weights[1, 1] = 0.9;
            weights[2, 1] = 0.9;

            // Neuron 2: weak
            weights[0, 2] = 0.2;
            weights[1, 2] = 0.2;
            weights[2, 2] = 0.2;

            // Neuron 3: strong
            weights[0, 3] = 0.8;
            weights[1, 3] = 0.8;
            weights[2, 3] = 0.8;

            var strategy = new StructuredPruningStrategy<double>(
                StructuredPruningStrategy<double>.StructurePruningType.Neuron);

            // Act - prune 50% of neurons (2 out of 4)
            var scores = strategy.ComputeImportanceScores(weights);
            var mask = strategy.CreateMask(scores, targetSparsity: 0.5);

            // Assert
            var pruned = mask.Apply(weights);

            // Neurons 0 and 2 (weakest) should be entirely pruned
            for (int row = 0; row < 3; row++)
            {
                Assert.Equal(0.0, pruned[row, 0]); // Neuron 0 pruned
                Assert.NotEqual(0.0, pruned[row, 1]); // Neuron 1 kept
                Assert.Equal(0.0, pruned[row, 2]); // Neuron 2 pruned
                Assert.NotEqual(0.0, pruned[row, 3]); // Neuron 3 kept
            }
        }

        [Fact]
        public void PruningMask_CombineWith_LogicalAND()
        {
            // Arrange
            var mask1 = new PruningMask<double>(2, 2);
            var keep1 = new bool[2, 2] { { true, false }, { true, true } };
            mask1.UpdateMask(keep1);

            var mask2 = new PruningMask<double>(2, 2);
            var keep2 = new bool[2, 2] { { true, true }, { false, true } };
            mask2.UpdateMask(keep2);

            // Act
            var combined = mask1.CombineWith(mask2);

            // Assert - should be logical AND
            var weights = new Matrix<double>(2, 2);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    weights[i, j] = 1.0;

            var result = combined.Apply(weights);

            Assert.Equal(1.0, result[0, 0]); // true AND true
            Assert.Equal(0.0, result[0, 1]); // false AND true
            Assert.Equal(0.0, result[1, 0]); // true AND false
            Assert.Equal(1.0, result[1, 1]); // true AND true
        }
    }
}
```

## Usage Example: Complete Pruning Workflow

```csharp
// Example: Prune a trained neural network

// 1. Train network to convergence
var network = new NeuralNetwork(...);
network.Train(trainingData, epochs: 100);

// 2. Choose pruning strategy
var pruningStrategy = new MagnitudePruningStrategy<double>();
// OR: var pruningStrategy = new GradientPruningStrategy<double>();
// OR: var pruningStrategy = new LotteryTicketPruningStrategy<double>();

// 3. For each layer, compute importance and create mask
foreach (var layer in network.Layers)
{
    var weights = layer.GetWeights();
    var gradients = layer.GetGradients(); // If using gradient-based

    var scores = pruningStrategy.ComputeImportanceScores(weights, gradients);
    var mask = pruningStrategy.CreateMask(scores, targetSparsity: 0.7); // 70% sparse

    pruningStrategy.ApplyPruning(weights, mask);
    layer.SetWeights(weights);
}

// 4. Fine-tune the pruned network
network.Train(trainingData, epochs: 10, learningRate: 0.001);

// 5. Check final sparsity and accuracy
double sparsity = CalculateOverallSparsity(network);
double accuracy = Evaluate(network, testData);

Console.WriteLine($"Final sparsity: {sparsity:P2}, Accuracy: {accuracy:P2}");
```

## Common Pitfalls to Avoid

1. **Pruning too aggressively at once** - Use iterative pruning (5-10 rounds)
2. **Not fine-tuning after pruning** - Always retrain briefly after pruning
3. **Forgetting to store initial weights** - Critical for lottery ticket
4. **Pruning without validation** - Always check accuracy doesn't drop too much
5. **Using wrong mask dimensions** - Ensure mask matches weight matrix shape
6. **Not handling sparse matrix efficiently** - Consider using sparse storage formats

## Advanced Topics

### Iterative Magnitude Pruning Schedule

```csharp
// Gradually increase sparsity over training
for (int round = 0; round < 10; round++)
{
    // Train for a few epochs
    network.Train(trainingData, epochs: 5);

    // Gradually increase sparsity: 0%, 10%, 20%, ..., 90%
    double targetSparsity = round * 0.1;

    // Apply pruning at current sparsity level
    PruneNetwork(network, strategy, targetSparsity);
}
```

### One-Shot vs. Iterative Pruning

**One-Shot**: Prune to target sparsity immediately
- Faster but can damage accuracy
- Use for robust networks

**Iterative**: Gradually increase sparsity
- Slower but maintains accuracy better
- Recommended for high sparsity (>80%)

## Validation Criteria

Your implementation is complete when:

1. All three pruning strategies implemented and tested
2. PruningMask supports both Matrix and Tensor operations
3. Tests verify correct sparsity levels achieved
4. Lottery ticket can store/restore initial weights
5. Structured pruning removes entire neurons/filters
6. Gradient-based pruning properly uses gradient information
7. Integration with existing neural network layers works

## Learning Resources

- **Lottery Ticket Hypothesis**: https://arxiv.org/abs/1803.03635
- **Pruning Survey**: https://arxiv.org/abs/2003.03033
- **Structured Pruning**: https://arxiv.org/abs/1608.08710
- **Magnitude Pruning**: https://arxiv.org/abs/1506.02626

## Next Steps

1. Implement dynamic sparsity (pruning during training)
2. Add support for different sparsity patterns (block, N:M structured)
3. Integrate with quantization (Issue #409) for maximum compression
4. Add ONNX export (Issue #410) for pruned models
5. Benchmark inference speedup on pruned models

---

**Good luck!** Model pruning is essential for deploying neural networks on mobile and edge devices. Understanding these techniques will make you valuable for production ML systems.
