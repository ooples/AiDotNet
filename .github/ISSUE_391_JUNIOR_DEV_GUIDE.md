# Issue #391: Junior Developer Implementation Guide - Imbalanced Learning

## Understanding Imbalanced Learning

### What is Class Imbalance?
Class imbalance occurs when the distribution of classes in your dataset is heavily skewed. For example:
- **Fraud Detection**: 99.9% legitimate, 0.1% fraud
- **Medical Diagnosis**: 95% healthy, 5% disease
- **Manufacturing Defects**: 99% good products, 1% defective

### Why is This a Problem?

**The Naive Model Problem**:
```csharp
// With 99% non-fraud transactions:
// A model that always predicts "not fraud" is 99% accurate!
// But it's completely useless - it never catches any fraud.
```

Models trained on imbalanced data tend to:
1. **Ignore Minority Class**: Learn to predict only the majority class
2. **High Accuracy, Low Utility**: 99% accuracy but 0% recall on fraud
3. **Biased Decision Boundaries**: Don't learn patterns in rare class

### Solutions

1. **Oversampling**: Create more minority class examples
   - SMOTE: Synthetic Minority Oversampling Technique
   - ADASYN: Adaptive Synthetic Sampling

2. **Undersampling**: Remove majority class examples
   - Random Undersampling
   - Tomek Links
   - ENN (Edited Nearest Neighbors)

3. **Hybrid**: Combine both
   - SMOTE + ENN
   - SMOTE + Tomek

---

## Phase 1: SMOTE (Synthetic Minority Oversampling Technique)

### AC 1.1: Implement SMOTE Algorithm

**File**: `src/Data/ImbalancedLearning/SMOTE.cs`

```csharp
namespace AiDotNet.Data.ImbalancedLearning;

/// <summary>
/// Synthetic Minority Oversampling Technique (SMOTE).
/// Creates synthetic samples by interpolating between minority class examples.
/// </summary>
/// <remarks>
/// <para>
/// SMOTE Algorithm:
/// 1. For each minority sample, find K nearest minority neighbors
/// 2. Randomly select one neighbor
/// 3. Create synthetic sample along the line connecting them
/// 4. Repeat until desired balance achieved
/// </para>
/// <para><b>For Beginners:</b> SMOTE creates "fake" examples of the rare class.
///
/// Imagine you have 1000 photos of dogs but only 10 photos of cats.
/// SMOTE doesn't just copy the cat photos (which wouldn't help).
/// Instead, it creates NEW cat photos by blending existing ones:
///
/// - Take Cat Photo A and Cat Photo B
/// - Create a new photo that's 70% Cat A + 30% Cat B
/// - This new photo looks like a cat, but is slightly different
/// - Repeat to create 990 synthetic cat photos
///
/// Now you have 1000 dogs and 1000 cats - balanced!
///
/// Why this works:
/// - Synthetic samples are realistic (blend of real samples)
/// - Model learns the "space" where minority class exists
/// - Prevents overfitting (not just copying existing samples)
///
/// Original Paper: Chawla et al. (2002)
/// "SMOTE: Synthetic Minority Over-sampling Technique"
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
public class SMOTE<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _k;
    private readonly Random _random;

    /// <summary>
    /// Initializes SMOTE with specified parameters.
    /// </summary>
    /// <param name="k">Number of nearest neighbors to use (default: 5).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public SMOTE(int k = 5, int? seed = null)
    {
        if (k < 1)
            throw new ArgumentException("K must be at least 1", nameof(k));

        _numOps = NumericOperations<T>.Instance;
        _k = k;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Generates synthetic samples for the minority class.
    /// </summary>
    /// <param name="minorityData">Minority class samples [samples, features].</param>
    /// <param name="syntheticCount">Number of synthetic samples to generate.</param>
    /// <returns>Matrix of synthetic samples.</returns>
    public Matrix<T> GenerateSamples(Matrix<T> minorityData, int syntheticCount)
    {
        if (minorityData.Rows < _k + 1)
        {
            throw new InvalidOperationException(
                $"Need at least {_k + 1} minority samples for K={_k}. " +
                $"Got {minorityData.Rows} samples.");
        }

        var syntheticSamples = new List<Vector<T>>();

        for (int i = 0; i < syntheticCount; i++)
        {
            // Randomly select a minority sample
            int sampleIdx = _random.Next(minorityData.Rows);
            var sample = minorityData.GetRow(sampleIdx);

            // Find K nearest neighbors
            var neighbors = FindKNearestNeighbors(minorityData, sampleIdx);

            // Randomly select one of the K neighbors
            int neighborIdx = neighbors[_random.Next(neighbors.Length)];
            var neighbor = minorityData.GetRow(neighborIdx);

            // Generate synthetic sample
            var syntheticSample = InterpolateSamples(sample, neighbor);
            syntheticSamples.Add(syntheticSample);
        }

        return Matrix<T>.FromRowVectors(syntheticSamples);
    }

    /// <summary>
    /// Finds K nearest neighbors for a given sample.
    /// </summary>
    private int[] FindKNearestNeighbors(Matrix<T> data, int sampleIdx)
    {
        var sample = data.GetRow(sampleIdx);
        var distances = new (double distance, int index)[data.Rows - 1];
        int distIdx = 0;

        // Calculate distances to all other samples
        for (int i = 0; i < data.Rows; i++)
        {
            if (i == sampleIdx) continue; // Skip self

            var other = data.GetRow(i);
            double distance = CalculateEuclideanDistance(sample, other);
            distances[distIdx++] = (distance, i);
        }

        // Sort by distance and take K nearest
        Array.Sort(distances, (a, b) => a.distance.CompareTo(b.distance));

        return distances.Take(_k).Select(d => d.index).ToArray();
    }

    /// <summary>
    /// Calculates Euclidean distance between two samples.
    /// </summary>
    private double CalculateEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have same length");

        T sumSquares = _numOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            T diff = _numOps.Subtract(a[i], b[i]);
            sumSquares = _numOps.Add(sumSquares, _numOps.Multiply(diff, diff));
        }

        return Convert.ToDouble(_numOps.Sqrt(sumSquares));
    }

    /// <summary>
    /// Creates synthetic sample by interpolating between two samples.
    /// Formula: synthetic = sample + lambda * (neighbor - sample)
    /// where lambda is random value in [0, 1]
    /// </summary>
    private Vector<T> InterpolateSamples(Vector<T> sample, Vector<T> neighbor)
    {
        double lambda = _random.NextDouble(); // Random value in [0, 1]
        T lambdaT = _numOps.FromDouble(lambda);

        var synthetic = new Vector<T>(sample.Length);

        for (int i = 0; i < sample.Length; i++)
        {
            // synthetic[i] = sample[i] + lambda * (neighbor[i] - sample[i])
            T diff = _numOps.Subtract(neighbor[i], sample[i]);
            T offset = _numOps.Multiply(lambdaT, diff);
            synthetic[i] = _numOps.Add(sample[i], offset);
        }

        return synthetic;
    }

    /// <summary>
    /// Fits and resamples a dataset to balance classes.
    /// </summary>
    /// <param name="X">Feature matrix [samples, features].</param>
    /// <param name="y">Labels vector [samples].</param>
    /// <param name="minorityLabel">Label of the minority class to oversample.</param>
    /// <param name="samplingStrategy">Target ratio (minority/majority) or "auto" for 1:1.</param>
    /// <returns>Tuple of (resampled X, resampled y).</returns>
    public (Matrix<T>, Vector<T>) FitResample(
        Matrix<T> X,
        Vector<T> y,
        T minorityLabel,
        string samplingStrategy = "auto")
    {
        // Separate minority and majority samples
        var minorityIndices = new List<int>();
        var majorityIndices = new List<int>();

        for (int i = 0; i < y.Length; i++)
        {
            if (_numOps.Equals(y[i], minorityLabel))
                minorityIndices.Add(i);
            else
                majorityIndices.Add(i);
        }

        if (minorityIndices.Count == 0)
            throw new ArgumentException("No minority samples found");

        if (majorityIndices.Count == 0)
            throw new ArgumentException("No majority samples found");

        // Extract minority data
        var minorityData = ExtractRows(X, minorityIndices);

        // Calculate how many synthetic samples to generate
        int syntheticCount;
        if (samplingStrategy == "auto")
        {
            // Generate enough to match majority class
            syntheticCount = majorityIndices.Count - minorityIndices.Count;
        }
        else if (double.TryParse(samplingStrategy, out double ratio))
        {
            // Generate to achieve specific ratio
            int targetMinorityCount = (int)(majorityIndices.Count * ratio);
            syntheticCount = targetMinorityCount - minorityIndices.Count;
        }
        else
        {
            throw new ArgumentException($"Invalid sampling strategy: {samplingStrategy}");
        }

        if (syntheticCount < 0)
            syntheticCount = 0; // Already balanced or majority is actually minority

        // Generate synthetic samples
        Matrix<T> syntheticSamples = null;
        if (syntheticCount > 0)
        {
            syntheticSamples = GenerateSamples(minorityData, syntheticCount);
        }

        // Combine original data with synthetic data
        return CombineData(X, y, syntheticSamples, minorityLabel);
    }

    /// <summary>
    /// Extracts specific rows from a matrix.
    /// </summary>
    private Matrix<T> ExtractRows(Matrix<T> matrix, List<int> rowIndices)
    {
        var result = new Matrix<T>(rowIndices.Count, matrix.Columns);

        for (int i = 0; i < rowIndices.Count; i++)
        {
            int sourceRow = rowIndices[i];
            for (int col = 0; col < matrix.Columns; col++)
            {
                result[i, col] = matrix[sourceRow, col];
            }
        }

        return result;
    }

    /// <summary>
    /// Combines original data with synthetic samples.
    /// </summary>
    private (Matrix<T>, Vector<T>) CombineData(
        Matrix<T> originalX,
        Vector<T> originalY,
        Matrix<T> syntheticX,
        T syntheticLabel)
    {
        int totalRows = originalX.Rows + (syntheticX?.Rows ?? 0);

        var newX = new Matrix<T>(totalRows, originalX.Columns);
        var newY = new Vector<T>(totalRows);

        // Copy original data
        for (int i = 0; i < originalX.Rows; i++)
        {
            for (int col = 0; col < originalX.Columns; col++)
            {
                newX[i, col] = originalX[i, col];
            }
            newY[i] = originalY[i];
        }

        // Add synthetic data
        if (syntheticX != null)
        {
            for (int i = 0; i < syntheticX.Rows; i++)
            {
                int targetRow = originalX.Rows + i;
                for (int col = 0; col < syntheticX.Columns; col++)
                {
                    newX[targetRow, col] = syntheticX[i, col];
                }
                newY[targetRow] = syntheticLabel;
            }
        }

        return (newX, newY);
    }
}
```

---

## Phase 2: ADASYN (Adaptive Synthetic Sampling)

### AC 2.1: Implement ADASYN

**File**: `src/Data/ImbalancedLearning/ADASYN.cs`

```csharp
namespace AiDotNet.Data.ImbalancedLearning;

/// <summary>
/// Adaptive Synthetic Sampling (ADASYN).
/// Generates more synthetic samples for minority examples that are harder to learn.
/// </summary>
/// <remarks>
/// <para>
/// ADASYN improves on SMOTE by focusing on "difficult" minority samples:
/// - Samples near the decision boundary get more synthetic examples
/// - Samples in dense minority regions get fewer synthetic examples
/// - Adaptively adjusts generation based on local difficulty
/// </para>
/// <para><b>For Beginners:</b> ADASYN is like SMOTE, but smarter about where to add samples.
///
/// Think of it like studying for an exam:
/// - SMOTE: Spend equal time on all topics
/// - ADASYN: Spend more time on topics you struggle with
///
/// ADASYN identifies "difficult" minority examples:
/// - Examples surrounded by majority class (hard to classify)
/// - Examples at the boundary between classes
/// - Examples in confused regions
///
/// Then it generates MORE synthetic samples for these difficult cases.
///
/// Why this helps:
/// - Strengthens weak areas of the minority class
/// - Improves decision boundary in confused regions
/// - Better performance than vanilla SMOTE
///
/// Original Paper: He et al. (2008)
/// "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning"
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
public class ADASYN<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _k;
    private readonly Random _random;
    private readonly double _beta;

    /// <summary>
    /// Initializes ADASYN.
    /// </summary>
    /// <param name="k">Number of nearest neighbors (default: 5).</param>
    /// <param name="beta">Desired balance ratio after generation (default: 1.0 for full balance).</param>
    /// <param name="seed">Random seed.</param>
    public ADASYN(int k = 5, double beta = 1.0, int? seed = null)
    {
        _numOps = NumericOperations<T>.Instance;
        _k = k;
        _beta = beta;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Fits and resamples dataset using adaptive sampling.
    /// </summary>
    public (Matrix<T>, Vector<T>) FitResample(
        Matrix<T> X,
        Vector<T> y,
        T minorityLabel)
    {
        // Separate classes
        var minorityIndices = new List<int>();
        var majorityIndices = new List<int>();

        for (int i = 0; i < y.Length; i++)
        {
            if (_numOps.Equals(y[i], minorityLabel))
                minorityIndices.Add(i);
            else
                majorityIndices.Add(i);
        }

        int minorityCount = minorityIndices.Count;
        int majorityCount = majorityIndices.Count;

        if (minorityCount == 0 || majorityCount == 0)
            throw new ArgumentException("Need samples from both classes");

        // Calculate number of synthetic samples needed
        double d = majorityCount - minorityCount;
        int totalSynthetic = (int)(_beta * d);

        if (totalSynthetic <= 0)
            return (X, y); // Already balanced

        // Calculate difficulty ratio for each minority sample
        var minorityData = ExtractRows(X, minorityIndices);
        var difficultyRatios = CalculateDifficultyRatios(X, y, minorityIndices, minorityLabel);

        // Normalize ratios to sum to 1
        double ratioSum = difficultyRatios.Sum();
        if (ratioSum == 0)
            ratioSum = 1.0; // Avoid division by zero

        var normalizedRatios = difficultyRatios.Select(r => r / ratioSum).ToArray();

        // Generate synthetic samples based on difficulty
        var syntheticSamples = new List<Vector<T>>();

        for (int i = 0; i < minorityData.Rows; i++)
        {
            int samplesForThis = (int)(normalizedRatios[i] * totalSynthetic);

            for (int s = 0; s < samplesForThis; s++)
            {
                var sample = minorityData.GetRow(i);
                var neighbors = FindKNearestNeighbors(minorityData, i);

                if (neighbors.Length > 0)
                {
                    int neighborIdx = neighbors[_random.Next(neighbors.Length)];
                    var neighbor = minorityData.GetRow(neighborIdx);
                    var synthetic = InterpolateSamples(sample, neighbor);
                    syntheticSamples.Add(synthetic);
                }
            }
        }

        // Combine with original data
        var syntheticMatrix = Matrix<T>.FromRowVectors(syntheticSamples);
        return CombineData(X, y, syntheticMatrix, minorityLabel);
    }

    /// <summary>
    /// Calculates difficulty ratio for each minority sample.
    /// Ratio = (number of majority neighbors) / K
    /// </summary>
    private double[] CalculateDifficultyRatios(
        Matrix<T> X,
        Vector<T> y,
        List<int> minorityIndices,
        T minorityLabel)
    {
        var ratios = new double[minorityIndices.Count];

        for (int i = 0; i < minorityIndices.Count; i++)
        {
            int sampleIdx = minorityIndices[i];
            var sample = X.GetRow(sampleIdx);

            // Find K nearest neighbors in entire dataset
            var neighbors = FindKNearestNeighborsInDataset(X, sampleIdx);

            // Count how many are majority class
            int majorityCount = 0;
            foreach (var neighborIdx in neighbors)
            {
                if (!_numOps.Equals(y[neighborIdx], minorityLabel))
                {
                    majorityCount++;
                }
            }

            // Difficulty ratio: more majority neighbors = higher difficulty
            ratios[i] = (double)majorityCount / _k;
        }

        return ratios;
    }

    /// <summary>
    /// Finds K nearest neighbors in the entire dataset.
    /// </summary>
    private int[] FindKNearestNeighborsInDataset(Matrix<T> data, int sampleIdx)
    {
        var sample = data.GetRow(sampleIdx);
        var distances = new (double distance, int index)[data.Rows - 1];
        int distIdx = 0;

        for (int i = 0; i < data.Rows; i++)
        {
            if (i == sampleIdx) continue;

            var other = data.GetRow(i);
            double distance = CalculateEuclideanDistance(sample, other);
            distances[distIdx++] = (distance, i);
        }

        Array.Sort(distances, (a, b) => a.distance.CompareTo(b.distance));
        return distances.Take(_k).Select(d => d.index).ToArray();
    }

    /// <summary>
    /// Finds K nearest neighbors within minority class.
    /// </summary>
    private int[] FindKNearestNeighbors(Matrix<T> minorityData, int sampleIdx)
    {
        var sample = minorityData.GetRow(sampleIdx);
        var distances = new List<(double distance, int index)>();

        for (int i = 0; i < minorityData.Rows; i++)
        {
            if (i == sampleIdx) continue;

            var other = minorityData.GetRow(i);
            double distance = CalculateEuclideanDistance(sample, other);
            distances.Add((distance, i));
        }

        distances.Sort((a, b) => a.distance.CompareTo(b.distance));
        int count = Math.Min(_k, distances.Count);
        return distances.Take(count).Select(d => d.index).ToArray();
    }

    private double CalculateEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sumSquares = _numOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = _numOps.Subtract(a[i], b[i]);
            sumSquares = _numOps.Add(sumSquares, _numOps.Multiply(diff, diff));
        }
        return Convert.ToDouble(_numOps.Sqrt(sumSquares));
    }

    private Vector<T> InterpolateSamples(Vector<T> sample, Vector<T> neighbor)
    {
        double lambda = _random.NextDouble();
        T lambdaT = _numOps.FromDouble(lambda);

        var synthetic = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
        {
            T diff = _numOps.Subtract(neighbor[i], sample[i]);
            T offset = _numOps.Multiply(lambdaT, diff);
            synthetic[i] = _numOps.Add(sample[i], offset);
        }
        return synthetic;
    }

    private Matrix<T> ExtractRows(Matrix<T> matrix, List<int> rowIndices)
    {
        var result = new Matrix<T>(rowIndices.Count, matrix.Columns);
        for (int i = 0; i < rowIndices.Count; i++)
        {
            for (int col = 0; col < matrix.Columns; col++)
            {
                result[i, col] = matrix[rowIndices[i], col];
            }
        }
        return result;
    }

    private (Matrix<T>, Vector<T>) CombineData(
        Matrix<T> originalX,
        Vector<T> originalY,
        Matrix<T> syntheticX,
        T syntheticLabel)
    {
        int totalRows = originalX.Rows + syntheticX.Rows;
        var newX = new Matrix<T>(totalRows, originalX.Columns);
        var newY = new Vector<T>(totalRows);

        // Copy original
        for (int i = 0; i < originalX.Rows; i++)
        {
            for (int col = 0; col < originalX.Columns; col++)
                newX[i, col] = originalX[i, col];
            newY[i] = originalY[i];
        }

        // Add synthetic
        for (int i = 0; i < syntheticX.Rows; i++)
        {
            int targetRow = originalX.Rows + i;
            for (int col = 0; col < syntheticX.Columns; col++)
                newX[targetRow, col] = syntheticX[i, col];
            newY[targetRow] = syntheticLabel;
        }

        return (newX, newY);
    }
}
```

---

## Phase 3: Undersampling Techniques

### AC 3.1: Random Undersampler

**File**: `src/Data/ImbalancedLearning/RandomUndersampler.cs`

```csharp
namespace AiDotNet.Data.ImbalancedLearning;

/// <summary>
/// Randomly removes samples from the majority class to balance the dataset.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of adding minority samples, remove majority samples.
///
/// Like a company downsizing:
/// - Problem: 1000 employees, but only need 100
/// - Solution: Randomly select 100 to keep
///
/// Undersampling:
/// - Fast and simple
/// - Reduces training time (smaller dataset)
/// - Risk: May lose important information
///
/// When to use:
/// - Very large datasets (millions of samples)
/// - Computational constraints
/// - Combined with oversampling (hybrid approach)
///
/// When NOT to use:
/// - Small datasets (losing data is expensive)
/// - Complex minority class (need all majority context)
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
public class RandomUndersampler<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;

    public RandomUndersampler(int? seed = null)
    {
        _numOps = NumericOperations<T>.Instance;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Undersamples the majority class to match minority class count.
    /// </summary>
    /// <param name="X">Feature matrix.</param>
    /// <param name="y">Labels.</param>
    /// <param name="minorityLabel">Label of minority class.</param>
    /// <param name="samplingStrategy">Target ratio or "auto" for 1:1.</param>
    /// <returns>Resampled (X, y).</returns>
    public (Matrix<T>, Vector<T>) FitResample(
        Matrix<T> X,
        Vector<T> y,
        T minorityLabel,
        string samplingStrategy = "auto")
    {
        // Separate classes
        var minorityIndices = new List<int>();
        var majorityIndices = new List<int>();

        for (int i = 0; i < y.Length; i++)
        {
            if (_numOps.Equals(y[i], minorityLabel))
                minorityIndices.Add(i);
            else
                majorityIndices.Add(i);
        }

        // Calculate how many majority samples to keep
        int targetMajorityCount;
        if (samplingStrategy == "auto")
        {
            targetMajorityCount = minorityIndices.Count; // 1:1 ratio
        }
        else if (double.TryParse(samplingStrategy, out double ratio))
        {
            targetMajorityCount = (int)(minorityIndices.Count / ratio);
        }
        else
        {
            throw new ArgumentException($"Invalid sampling strategy: {samplingStrategy}");
        }

        // Randomly sample from majority class
        var sampledMajorityIndices = SampleIndices(majorityIndices, targetMajorityCount);

        // Combine minority with sampled majority
        var selectedIndices = new List<int>();
        selectedIndices.AddRange(minorityIndices);
        selectedIndices.AddRange(sampledMajorityIndices);

        // Shuffle for good measure
        Shuffle(selectedIndices);

        // Extract selected samples
        return ExtractSamples(X, y, selectedIndices);
    }

    private List<int> SampleIndices(List<int> indices, int count)
    {
        if (count >= indices.Count)
            return new List<int>(indices); // Keep all

        var sampled = new List<int>();
        var available = new List<int>(indices);

        for (int i = 0; i < count; i++)
        {
            int idx = _random.Next(available.Count);
            sampled.Add(available[idx]);
            available.RemoveAt(idx);
        }

        return sampled;
    }

    private void Shuffle(List<int> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            int temp = list[i];
            list[i] = list[j];
            list[j] = temp;
        }
    }

    private (Matrix<T>, Vector<T>) ExtractSamples(
        Matrix<T> X,
        Vector<T> y,
        List<int> indices)
    {
        var newX = new Matrix<T>(indices.Count, X.Columns);
        var newY = new Vector<T>(indices.Count);

        for (int i = 0; i < indices.Count; i++)
        {
            int sourceIdx = indices[i];
            for (int col = 0; col < X.Columns; col++)
            {
                newX[i, col] = X[sourceIdx, col];
            }
            newY[i] = y[sourceIdx];
        }

        return (newX, newY);
    }
}
```

---

## Phase 4: Usage Examples and Best Practices

### AC 4.1: Complete Example

```csharp
// Example: Fraud Detection with Imbalanced Data
public class ImbalancedLearningExample
{
    public async Task RunExample()
    {
        // Simulate imbalanced fraud detection data
        // 990 legitimate transactions, 10 fraudulent
        var (X, y) = GenerateImbalancedData(
            legitimateCount: 990,
            fraudCount: 10,
            features: 20
        );

        Console.WriteLine($"Original data: {y.Count(label => label == 1)} fraud, " +
                          $"{y.Count(label => label == 0)} legitimate");

        // --- Approach 1: SMOTE Oversampling ---
        var smote = new SMOTE<double>(k: 5, seed: 42);
        var (X_smote, y_smote) = smote.FitResample(
            X, y,
            minorityLabel: 1.0,
            samplingStrategy: "auto"
        );

        Console.WriteLine($"After SMOTE: {y_smote.Count(label => label == 1)} fraud, " +
                          $"{y_smote.Count(label => label == 0)} legitimate");

        // --- Approach 2: ADASYN (Adaptive Sampling) ---
        var adasyn = new ADASYN<double>(k: 5, beta: 1.0, seed: 42);
        var (X_adasyn, y_adasyn) = adasyn.FitResample(X, y, minorityLabel: 1.0);

        Console.WriteLine($"After ADASYN: {y_adasyn.Count(label => label == 1)} fraud, " +
                          $"{y_adasyn.Count(label => label == 0)} legitimate");

        // --- Approach 3: Random Undersampling ---
        var undersampler = new RandomUndersampler<double>(seed: 42);
        var (X_under, y_under) = undersampler.FitResample(
            X, y,
            minorityLabel: 1.0,
            samplingStrategy: "auto"
        );

        Console.WriteLine($"After Undersampling: {y_under.Count(label => label == 1)} fraud, " +
                          $"{y_under.Count(label => label == 0)} legitimate");

        // --- Approach 4: Hybrid (SMOTE + Undersampling) ---
        // First oversample minority (less aggressive)
        var (X_hybrid1, y_hybrid1) = smote.FitResample(
            X, y,
            minorityLabel: 1.0,
            samplingStrategy: "0.5" // Minority = 50% of majority
        );

        // Then undersample majority
        var (X_hybrid, y_hybrid) = undersampler.FitResample(
            X_hybrid1, y_hybrid1,
            minorityLabel: 1.0,
            samplingStrategy: "auto"
        );

        Console.WriteLine($"After Hybrid: {y_hybrid.Count(label => label == 1)} fraud, " +
                          $"{y_hybrid.Count(label => label == 0)} legitimate");

        // Train models and compare
        await CompareApproaches(X, y, X_smote, y_smote, X_adasyn, y_adasyn,
                                X_under, y_under, X_hybrid, y_hybrid);
    }

    private (Matrix<double>, Vector<double>) GenerateImbalancedData(
        int legitimateCount,
        int fraudCount,
        int features)
    {
        var random = new Random(42);
        int totalSamples = legitimateCount + fraudCount;

        var X = new Matrix<double>(totalSamples, features);
        var y = new Vector<double>(totalSamples);

        // Generate legitimate transactions (label = 0)
        for (int i = 0; i < legitimateCount; i++)
        {
            for (int j = 0; j < features; j++)
            {
                X[i, j] = random.NextDouble() * 10; // Random features
            }
            y[i] = 0.0;
        }

        // Generate fraudulent transactions (label = 1)
        // Make them slightly different to simulate real fraud
        for (int i = legitimateCount; i < totalSamples; i++)
        {
            for (int j = 0; j < features; j++)
            {
                X[i, j] = 5 + random.NextDouble() * 10; // Shifted distribution
            }
            y[i] = 1.0;
        }

        return (X, y);
    }
}
```

---

## Common Pitfalls to Avoid

1. **Applying to Test Data**: NEVER resample test data - only training
   ```csharp
   // WRONG
   (X_test, y_test) = smote.FitResample(X_test, y_test);

   // CORRECT
   (X_train, y_train) = smote.FitResample(X_train, y_train);
   // Use original X_test, y_test for evaluation
   ```

2. **Wrong Metric**: Accuracy is misleading for imbalanced data
   ```csharp
   // WRONG
   double accuracy = correct / total;

   // CORRECT - Use:
   // - Precision: Of predicted frauds, how many are real?
   // - Recall: Of actual frauds, how many did we catch?
   // - F1-Score: Harmonic mean of precision and recall
   // - AUC-ROC: Area under ROC curve
   ```

3. **Over-Resampling**: Don't blindly balance to 50:50
   ```csharp
   // Sometimes 1:10 or 1:5 ratio is better than 1:1
   // Experiment with different ratios
   ```

4. **Ignoring Domain Knowledge**: Some samples are more valuable
   ```csharp
   // In medical diagnosis, false negatives are costly
   // Oversample heavily to catch rare diseases
   ```

---

## Testing Strategy

```csharp
[Fact]
public void SMOTE_GeneratesSyntheticSamples()
{
    var X = new Matrix<double>(new[,] {
        { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 },
        { 10, 10 }, { 10, 11 }, { 11, 10 }, { 11, 11 }
    });
    var y = Vector<double>.FromArray(new[] {
        0, 0, 0, 0, // Minority class
        1, 1, 1, 1  // Majority class
    });

    var smote = new SMOTE<double>(k: 3, seed: 42);
    var (X_resampled, y_resampled) = smote.FitResample(X, y, minorityLabel: 0.0);

    // Should have equal classes
    int minority = y_resampled.Count(label => label == 0.0);
    int majority = y_resampled.Count(label => label == 1.0);

    Assert.Equal(majority, minority);
    Assert.True(X_resampled.Rows > X.Rows); // Added samples
}

[Fact]
public void ADASYN_PrioritizesDifficultSamples()
{
    // Create dataset where some minority samples are isolated
    // (surrounded by majority) - these should get more synthetic samples
    // Test by checking distribution of generated samples
}

[Fact]
public void RandomUndersampler_BalancesClasses()
{
    // Test that majority class is reduced to match minority
    // Verify randomness with different seeds
}
```

---

## Next Steps

1. Implement SMOTE algorithm
2. Implement ADASYN algorithm
3. Implement RandomUndersampler
4. Implement TomekLinks and ENN (advanced undersampling)
5. Create comprehensive tests
6. Add performance benchmarks
7. Create usage examples and documentation

**Estimated Effort**: 6-7 days for a junior developer

**Files to Create**:
- `src/Data/ImbalancedLearning/SMOTE.cs`
- `src/Data/ImbalancedLearning/ADASYN.cs`
- `src/Data/ImbalancedLearning/RandomUndersampler.cs`
- `src/Data/ImbalancedLearning/TomekLinks.cs` (optional)
- `src/Data/ImbalancedLearning/ENN.cs` (optional)
- `tests/UnitTests/Data/ImbalancedLearning/SMOTETests.cs`
- `tests/UnitTests/Data/ImbalancedLearning/ADASYNTests.cs`
- `tests/UnitTests/Data/ImbalancedLearning/UndersamplerTests.cs`
