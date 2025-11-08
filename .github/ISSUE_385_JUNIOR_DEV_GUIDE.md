# Junior Developer Implementation Guide: Issue #385

## Overview
**Issue**: Naive Bayes Classifiers
**Goal**: Implement Naive Bayes classifiers (Gaussian, Multinomial, Bernoulli variants)
**Difficulty**: Intermediate
**Estimated Time**: 8-12 hours

## What You'll Be Building

You'll implement **Naive Bayes classifiers** with three variants:

1. **INaiveBayesClassifier Interface** - Defines Naive Bayes-specific methods
2. **NaiveBayesClassifierBase** - Base class with shared logic
3. **GaussianNaiveBayes** - For continuous features (assumes Gaussian distribution)
4. **MultinomialNaiveBayes** - For count data (word counts, frequencies)
5. **BernoulliNaiveBayes** - For binary features (presence/absence)
6. **Comprehensive Unit Tests** - 80%+ coverage

## Understanding Naive Bayes

### What is Naive Bayes?

**Naive Bayes** is a probabilistic classifier based on Bayes' Theorem with the "naive" assumption that features are independent.

**Key Concepts:**

1. **Bayes' Theorem**: Calculate probability of a class given features
   ```
   P(Class | Features) = P(Features | Class) * P(Class) / P(Features)
   ```

2. **Naive Assumption**: Features are independent given the class
   - This simplifies calculations dramatically
   - "Naive" because features are rarely truly independent in real life
   - Surprisingly effective despite this simplification

3. **Maximum A Posteriori (MAP)**: Predict the class with highest probability
   ```
   Predicted Class = argmax P(Class | Features)
   ```

**Real-World Analogy:**

Imagine diagnosing if someone has the flu:
- **P(Flu)**: Base rate of flu in population (prior)
- **P(Fever, Cough, Headache | Flu)**: Likelihood of symptoms given flu
- **P(Flu | Fever, Cough, Headache)**: Probability of flu given symptoms (what we want)

Naive Bayes assumes fever, cough, and headache are independent (naive), which isn't true (they're correlated), but it still works well in practice.

### Mathematical Formulas

**Bayes' Theorem:**
```
P(Ck | x) = P(x | Ck) * P(Ck) / P(x)

where:
    Ck = class k
    x = feature vector
    P(Ck) = prior probability of class k
    P(x | Ck) = likelihood of features given class k
    P(x) = evidence (normalizing constant)
```

**Naive Independence Assumption:**
```
P(x | Ck) = P(x1 | Ck) * P(x2 | Ck) * ... * P(xn | Ck)

So:
P(Ck | x) ∝ P(Ck) * ∏ P(xi | Ck)
```

**Classification Rule:**
```
y = argmax P(Ck) * ∏ P(xi | Ck)
    k              i
```

**Log-Probability (to avoid underflow):**
```
log P(Ck | x) = log P(Ck) + ∑ log P(xi | Ck)
                             i
```

### Variant-Specific Formulas

**1. Gaussian Naive Bayes (Continuous Features):**

Assumes features follow a normal distribution within each class.

```
P(xi | Ck) = (1 / sqrt(2π * σ²k)) * exp(-(xi - μk)² / (2 * σ²k))

where:
    μk = mean of feature i in class k
    σ²k = variance of feature i in class k

Estimation:
    μk = (1/nk) * ∑ xi  (for samples in class k)
    σ²k = (1/nk) * ∑ (xi - μk)²
```

**Use Cases:**
- Features are continuous (real-valued)
- Features approximately normally distributed
- Medical diagnosis (blood pressure, temperature, etc.)
- Sensor readings, measurements

**2. Multinomial Naive Bayes (Count Data):**

Models feature counts (frequencies), common in text classification.

```
P(xi | Ck) = (Nki + α) / (Nk + α * d)

where:
    Nki = count of feature i in class k
    Nk = total count of all features in class k
    α = Laplace smoothing parameter (default: 1)
    d = total number of features (vocabulary size)

Smoothing prevents zero probabilities:
    α = 0: No smoothing (can give P = 0)
    α = 1: Laplace smoothing (add-one smoothing)
    α < 1: Less aggressive smoothing
```

**Use Cases:**
- Text classification (spam detection, sentiment analysis)
- Document categorization
- Word count features
- Frequency-based data

**3. Bernoulli Naive Bayes (Binary Features):**

Models binary (presence/absence) features.

```
P(xi | Ck) = pki^xi * (1 - pki)^(1-xi)

where:
    pki = P(feature i present in class k)
    xi ∈ {0, 1}

If xi = 1: P(xi | Ck) = pki
If xi = 0: P(xi | Ck) = 1 - pki

Estimation:
    pki = (Nki + α) / (Nk + 2α)

where:
    Nki = number of documents in class k where feature i appears
    Nk = total number of documents in class k
    α = smoothing parameter
```

**Use Cases:**
- Binary text features (word present/absent)
- Boolean features
- Sparse binary data
- Document classification with presence indicators

## Understanding the Codebase

### Three-Tier Architecture Pattern

```
INaiveBayesClassifier<T>        (Interface - defines contract)
    ↓
NaiveBayesClassifierBase<T>     (Base class - shared logic)
    ↓
┌─────────────┬──────────────┬────────────────┐
GaussianNB    MultinomialNB  BernoulliNB
(Continuous)  (Counts)       (Binary)
```

## Step-by-Step Implementation Guide

### Step 1: Create INaiveBayesClassifier Interface

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\INaiveBayesClassifier.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for Naive Bayes classification models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Naive Bayes classifiers are probabilistic models based on Bayes' Theorem with the
/// "naive" assumption that features are independent given the class label.
/// </para>
/// <para><b>For Beginners:</b> Naive Bayes predicts the class by calculating probabilities.
///
/// Think of it like a weather forecaster:
/// - They have historical data about weather conditions and outcomes
/// - Given today's conditions, they calculate probability of rain
/// - They predict "rain" if P(Rain | Conditions) > P(No Rain | Conditions)
///
/// Naive Bayes does the same for any classification problem:
/// 1. Learn probability distributions from training data
/// 2. For new data, calculate probability of each class
/// 3. Predict the class with highest probability
///
/// It's "naive" because it assumes features are independent (e.g., temperature and
/// humidity don't affect each other), which is rarely true but works surprisingly well.
///
/// Real-world uses:
/// - Spam detection (most famous application)
/// - Sentiment analysis (positive/negative reviews)
/// - Document classification (news categories)
/// - Medical diagnosis
/// - Real-time prediction (very fast at prediction time)
/// </para>
/// </remarks>
public interface INaiveBayesClassifier<T>
{
    /// <summary>
    /// Trains the Naive Bayes classifier on the provided data.
    /// </summary>
    /// <param name="X">The feature matrix where each row is a sample.</param>
    /// <param name="y">The class labels for each sample.</param>
    void Fit(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Predicts class labels for new data.
    /// </summary>
    /// <param name="X">The feature matrix to predict labels for.</param>
    /// <returns>A vector of predicted class labels.</returns>
    Vector<T> Predict(Matrix<T> X);

    /// <summary>
    /// Predicts class probabilities for each sample.
    /// </summary>
    /// <param name="X">The feature matrix to predict probabilities for.</param>
    /// <returns>A matrix where each row contains probabilities for all classes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of just predicting "spam" or "not spam",
    /// this tells you "80% probability spam, 20% probability not spam".
    ///
    /// Useful when:
    /// - You need confidence measures
    /// - You want to set custom thresholds (e.g., only flag as spam if > 95% confident)
    /// - You're combining multiple classifiers
    /// </para>
    /// </remarks>
    Matrix<T> PredictProbabilities(Matrix<T> X);

    /// <summary>
    /// Predicts log probabilities for each class (more numerically stable).
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <returns>A matrix of log probabilities.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Log probabilities avoid numerical underflow.
    ///
    /// When multiplying many small probabilities (e.g., 0.001 * 0.001 * ...),
    /// you can get values too small for computers to represent accurately.
    ///
    /// Using log probabilities:
    /// - log(a * b) = log(a) + log(b)
    /// - Addition is more stable than multiplication
    /// - Can handle very small probabilities
    /// </para>
    /// </remarks>
    Matrix<T> PredictLogProbabilities(Matrix<T> X);

    /// <summary>
    /// Gets the prior probabilities for each class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prior probabilities are the base rates of each class
    /// in the training data.
    ///
    /// Example: If 60% of training emails are spam and 40% are not spam:
    /// - P(Spam) = 0.6
    /// - P(Not Spam) = 0.4
    ///
    /// These are the starting point before considering features.
    /// </para>
    /// </remarks>
    Vector<T>? ClassPriors { get; }

    /// <summary>
    /// Gets the unique class labels.
    /// </summary>
    Vector<T>? Classes { get; }
}
```

### Step 2: Create NaiveBayesClassifierBase

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Classification\NaiveBayesClassifierBase.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Classification;

/// <summary>
/// Base class for Naive Bayes classifiers implementing shared logic.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class NaiveBayesClassifierBase<T> : INaiveBayesClassifier<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The unique class labels found in training data.
    /// </summary>
    public Vector<T>? Classes { get; protected set; }

    /// <summary>
    /// The prior probabilities for each class.
    /// </summary>
    public Vector<T>? ClassPriors { get; protected set; }

    /// <summary>
    /// The number of training samples per class.
    /// </summary>
    protected Vector<T>? ClassCounts { get; set; }

    /// <summary>
    /// Trains the Naive Bayes classifier.
    /// </summary>
    public virtual void Fit(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
            throw new ArgumentException("Number of samples in X must match length of y");

        // Find unique classes
        ExtractUniqueClasses(y);

        // Calculate class priors
        CalculateClassPriors(y);

        // Let derived classes compute class-specific parameters
        FitClassParameters(X, y);
    }

    /// <summary>
    /// Extracts unique class labels from training labels.
    /// </summary>
    protected virtual void ExtractUniqueClasses(Vector<T> y)
    {
        var uniqueClasses = new HashSet<T>();
        for (int i = 0; i < y.Length; i++)
        {
            uniqueClasses.Add(y[i]);
        }

        Classes = new Vector<T>(uniqueClasses.Count);
        int index = 0;
        foreach (var c in uniqueClasses.OrderBy(x => Convert.ToDouble(x)))
        {
            Classes[index++] = c;
        }
    }

    /// <summary>
    /// Calculates prior probabilities for each class.
    /// </summary>
    protected virtual void CalculateClassPriors(Vector<T> y)
    {
        int numClasses = Classes!.Length;
        ClassPriors = new Vector<T>(numClasses);
        ClassCounts = new Vector<T>(numClasses);

        // Count samples per class
        for (int i = 0; i < y.Length; i++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                if (NumOps.Equals(y[i], Classes[c]))
                {
                    ClassCounts[c] = NumOps.Add(ClassCounts[c], NumOps.One);
                    break;
                }
            }
        }

        // Calculate priors: P(class) = count(class) / total_count
        T totalCount = NumOps.FromInt(y.Length);
        for (int c = 0; c < numClasses; c++)
        {
            ClassPriors[c] = NumOps.Divide(ClassCounts[c], totalCount);
        }
    }

    /// <summary>
    /// Fit class-specific parameters (implemented by derived classes).
    /// </summary>
    protected abstract void FitClassParameters(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Predicts class labels for new data.
    /// </summary>
    public virtual Vector<T> Predict(Matrix<T> X)
    {
        var logProbs = PredictLogProbabilities(X);
        var predictions = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            // Find class with maximum log probability
            int maxClass = 0;
            T maxLogProb = logProbs[i, 0];

            for (int c = 1; c < Classes!.Length; c++)
            {
                if (NumOps.GreaterThan(logProbs[i, c], maxLogProb))
                {
                    maxLogProb = logProbs[i, c];
                    maxClass = c;
                }
            }

            predictions[i] = Classes[maxClass];
        }

        return predictions;
    }

    /// <summary>
    /// Predicts class probabilities.
    /// </summary>
    public virtual Matrix<T> PredictProbabilities(Matrix<T> X)
    {
        var logProbs = PredictLogProbabilities(X);
        var probs = new Matrix<T>(X.Rows, Classes!.Length);

        // Convert log probabilities to probabilities using exp
        // and normalize with log-sum-exp trick for numerical stability
        for (int i = 0; i < X.Rows; i++)
        {
            // Find max log prob for this sample (for numerical stability)
            T maxLogProb = logProbs[i, 0];
            for (int c = 1; c < Classes.Length; c++)
            {
                if (NumOps.GreaterThan(logProbs[i, c], maxLogProb))
                {
                    maxLogProb = logProbs[i, c];
                }
            }

            // Compute exp(log_prob - max_log_prob) for stability
            T sum = NumOps.Zero;
            for (int c = 0; c < Classes.Length; c++)
            {
                T expVal = NumOps.Exp(NumOps.Subtract(logProbs[i, c], maxLogProb));
                probs[i, c] = expVal;
                sum = NumOps.Add(sum, expVal);
            }

            // Normalize
            for (int c = 0; c < Classes.Length; c++)
            {
                probs[i, c] = NumOps.Divide(probs[i, c], sum);
            }
        }

        return probs;
    }

    /// <summary>
    /// Predicts log probabilities (implemented by derived classes).
    /// </summary>
    public abstract Matrix<T> PredictLogProbabilities(Matrix<T> X);
}
```

### Step 3: Create GaussianNaiveBayes

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Classification\GaussianNaiveBayes.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Classification;

/// <summary>
/// Implements Gaussian Naive Bayes classifier for continuous features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Gaussian Naive Bayes assumes features follow a normal (bell curve) distribution.
///
/// Use when:
/// - Features are continuous (real numbers, not counts or binary)
/// - Features are approximately normally distributed
/// - Examples: height, weight, temperature, sensor readings
///
/// How it works:
/// 1. For each class and each feature, calculate mean and variance
/// 2. For new data, calculate probability using Gaussian formula
/// 3. Combine probabilities to predict class
///
/// Example: Classifying iris flowers based on petal length and width
/// - Calculate mean and std dev of petal measurements for each species
/// - For new flower, see which species' distribution it fits best
/// </para>
/// </remarks>
public class GaussianNaiveBayes<T> : NaiveBayesClassifierBase<T>
{
    /// <summary>
    /// Mean values for each feature in each class.
    /// Matrix of shape [num_classes, num_features]
    /// </summary>
    protected Matrix<T>? Means { get; set; }

    /// <summary>
    /// Variance values for each feature in each class.
    /// Matrix of shape [num_classes, num_features]
    /// </summary>
    protected Matrix<T>? Variances { get; set; }

    /// <summary>
    /// Small value added to variance for numerical stability.
    /// </summary>
    protected T Epsilon { get; set; }

    /// <summary>
    /// Creates a new Gaussian Naive Bayes classifier.
    /// </summary>
    /// <param name="epsilon">Small value to add to variance for numerical stability (default: 1e-9).</param>
    public GaussianNaiveBayes(T? epsilon = null)
    {
        Epsilon = epsilon ?? NumOps.FromDouble(1e-9);
    }

    /// <summary>
    /// Fits class-specific parameters (means and variances).
    /// </summary>
    protected override void FitClassParameters(Matrix<T> X, Vector<T> y)
    {
        int numClasses = Classes!.Length;
        int numFeatures = X.Columns;

        Means = new Matrix<T>(numClasses, numFeatures);
        Variances = new Matrix<T>(numClasses, numFeatures);

        // Calculate mean and variance for each class and feature
        for (int c = 0; c < numClasses; c++)
        {
            // Extract samples belonging to class c
            var classSamples = new List<Vector<T>>();
            for (int i = 0; i < y.Length; i++)
            {
                if (NumOps.Equals(y[i], Classes[c]))
                {
                    classSamples.Add(X.GetRow(i));
                }
            }

            int numSamples = classSamples.Count;

            for (int f = 0; f < numFeatures; f++)
            {
                // Calculate mean
                T sum = NumOps.Zero;
                for (int i = 0; i < numSamples; i++)
                {
                    sum = NumOps.Add(sum, classSamples[i][f]);
                }
                T mean = NumOps.Divide(sum, NumOps.FromInt(numSamples));
                Means[c, f] = mean;

                // Calculate variance
                T sumSquares = NumOps.Zero;
                for (int i = 0; i < numSamples; i++)
                {
                    T diff = NumOps.Subtract(classSamples[i][f], mean);
                    sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(diff, diff));
                }
                T variance = NumOps.Divide(sumSquares, NumOps.FromInt(numSamples));
                Variances[c, f] = NumOps.Add(variance, Epsilon); // Add epsilon for stability
            }
        }
    }

    /// <summary>
    /// Predicts log probabilities using Gaussian distribution.
    /// </summary>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> X)
    {
        if (Means == null || Variances == null)
            throw new InvalidOperationException("Model must be trained before prediction");

        int numSamples = X.Rows;
        int numClasses = Classes!.Length;
        int numFeatures = X.Columns;

        var logProbs = new Matrix<T>(numSamples, numClasses);

        T logTwoPi = NumOps.Log(NumOps.Multiply(NumOps.FromInt(2), NumOps.FromDouble(Math.PI)));

        for (int i = 0; i < numSamples; i++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                // Start with log prior
                T logProb = NumOps.Log(ClassPriors![c]);

                // Add log likelihood for each feature
                for (int f = 0; f < numFeatures; f++)
                {
                    T x = X[i, f];
                    T mean = Means[c, f];
                    T variance = Variances[c, f];

                    // log P(x | class) = -0.5 * (log(2π*σ²) + ((x-μ)²/σ²))
                    T diff = NumOps.Subtract(x, mean);
                    T numerator = NumOps.Multiply(diff, diff);
                    T exponent = NumOps.Divide(numerator, variance);

                    T logLikelihood = NumOps.Multiply(
                        NumOps.FromDouble(-0.5),
                        NumOps.Add(
                            NumOps.Add(logTwoPi, NumOps.Log(variance)),
                            exponent
                        )
                    );

                    logProb = NumOps.Add(logProb, logLikelihood);
                }

                logProbs[i, c] = logProb;
            }
        }

        return logProbs;
    }
}
```

### Step 4: Create MultinomialNaiveBayes

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Classification\MultinomialNaiveBayes.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Classification;

/// <summary>
/// Implements Multinomial Naive Bayes classifier for count data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Multinomial Naive Bayes works with count data (frequencies).
///
/// Use when:
/// - Features are counts (word frequencies, event occurrences)
/// - Text classification (most common use case)
/// - Document categorization
///
/// How it works:
/// 1. Count how often each feature appears in each class
/// 2. Calculate probabilities based on these counts
/// 3. Use smoothing to handle unseen features
///
/// Example: Spam detection
/// - Features: word counts in email ("free": 5, "money": 3, etc.)
/// - Training: Count word frequencies in spam vs. non-spam emails
/// - Prediction: Calculate which class is more likely given word counts
///
/// Why "Multinomial"? Each document is like rolling a multi-sided die
/// (with one side per word in vocabulary) multiple times.
/// </para>
/// </remarks>
public class MultinomialNaiveBayes<T> : NaiveBayesClassifierBase<T>
{
    /// <summary>
    /// Feature log probabilities for each class.
    /// Matrix of shape [num_classes, num_features]
    /// </summary>
    protected Matrix<T>? FeatureLogProbs { get; set; }

    /// <summary>
    /// Smoothing parameter (Laplace smoothing).
    /// </summary>
    protected T Alpha { get; set; }

    /// <summary>
    /// Creates a new Multinomial Naive Bayes classifier.
    /// </summary>
    /// <param name="alpha">Smoothing parameter (default: 1.0 for Laplace smoothing).</param>
    public MultinomialNaiveBayes(T? alpha = null)
    {
        Alpha = alpha ?? NumOps.One;
    }

    /// <summary>
    /// Fits class-specific parameters (feature probabilities).
    /// </summary>
    protected override void FitClassParameters(Matrix<T> X, Vector<T> y)
    {
        int numClasses = Classes!.Length;
        int numFeatures = X.Columns;

        FeatureLogProbs = new Matrix<T>(numClasses, numFeatures);

        // Calculate feature probabilities for each class
        for (int c = 0; c < numClasses; c++)
        {
            // Sum feature counts for this class
            var featureCounts = new Vector<T>(numFeatures);
            T totalCount = NumOps.Zero;

            for (int i = 0; i < y.Length; i++)
            {
                if (NumOps.Equals(y[i], Classes[c]))
                {
                    for (int f = 0; f < numFeatures; f++)
                    {
                        featureCounts[f] = NumOps.Add(featureCounts[f], X[i, f]);
                        totalCount = NumOps.Add(totalCount, X[i, f]);
                    }
                }
            }

            // Calculate log probabilities with Laplace smoothing
            // P(feature | class) = (count + alpha) / (total_count + alpha * num_features)
            T denominator = NumOps.Add(totalCount,
                NumOps.Multiply(Alpha, NumOps.FromInt(numFeatures)));

            for (int f = 0; f < numFeatures; f++)
            {
                T numerator = NumOps.Add(featureCounts[f], Alpha);
                FeatureLogProbs[c, f] = NumOps.Log(NumOps.Divide(numerator, denominator));
            }
        }
    }

    /// <summary>
    /// Predicts log probabilities for count data.
    /// </summary>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> X)
    {
        if (FeatureLogProbs == null)
            throw new InvalidOperationException("Model must be trained before prediction");

        int numSamples = X.Rows;
        int numClasses = Classes!.Length;
        int numFeatures = X.Columns;

        var logProbs = new Matrix<T>(numSamples, numClasses);

        for (int i = 0; i < numSamples; i++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                // Start with log prior
                T logProb = NumOps.Log(ClassPriors![c]);

                // Add weighted log probabilities
                // log P(x | class) = ∑ x_i * log P(feature_i | class)
                for (int f = 0; f < numFeatures; f++)
                {
                    if (NumOps.GreaterThan(X[i, f], NumOps.Zero))
                    {
                        logProb = NumOps.Add(logProb,
                            NumOps.Multiply(X[i, f], FeatureLogProbs[c, f]));
                    }
                }

                logProbs[i, c] = logProb;
            }
        }

        return logProbs;
    }
}
```

### Step 5: Create BernoulliNaiveBayes

Create file: `C:\Users\cheat\source\repos\AiDotNet\src\Classification\BernoulliNaiveBayes.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Classification;

/// <summary>
/// Implements Bernoulli Naive Bayes classifier for binary features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Bernoulli Naive Bayes works with binary (0/1) features.
///
/// Use when:
/// - Features are binary (present/absent, yes/no, true/false)
/// - Text classification with binary word indicators
/// - Each feature can only be 0 or 1
///
/// Difference from Multinomial:
/// - Multinomial: "How many times does word appear?" (0, 1, 2, 3, ...)
/// - Bernoulli: "Does word appear at all?" (0 or 1)
///
/// How it works:
/// 1. For each feature and class, calculate probability of feature being 1
/// 2. For prediction, use both P(feature=1) and P(feature=0)
/// 3. Explicitly accounts for absence of features (unlike Multinomial)
///
/// Example: Email spam detection with binary word presence
/// - Features: ["contains_free": 1, "contains_money": 0, "contains_hello": 1]
/// - For spam: P("contains_free"=1 | spam) = 0.8, P("contains_money"=0 | spam) = 0.3
/// - Combines both presence and absence probabilities
/// </para>
/// </remarks>
public class BernoulliNaiveBayes<T> : NaiveBayesClassifierBase<T>
{
    /// <summary>
    /// Log probability of each feature being 1 in each class.
    /// Matrix of shape [num_classes, num_features]
    /// </summary>
    protected Matrix<T>? FeatureLogProb1 { get; set; }

    /// <summary>
    /// Log probability of each feature being 0 in each class.
    /// Matrix of shape [num_classes, num_features]
    /// </summary>
    protected Matrix<T>? FeatureLogProb0 { get; set; }

    /// <summary>
    /// Smoothing parameter.
    /// </summary>
    protected T Alpha { get; set; }

    /// <summary>
    /// Creates a new Bernoulli Naive Bayes classifier.
    /// </summary>
    /// <param name="alpha">Smoothing parameter (default: 1.0).</param>
    public BernoulliNaiveBayes(T? alpha = null)
    {
        Alpha = alpha ?? NumOps.One;
    }

    /// <summary>
    /// Fits class-specific parameters (feature probabilities for 0 and 1).
    /// </summary>
    protected override void FitClassParameters(Matrix<T> X, Vector<T> y)
    {
        int numClasses = Classes!.Length;
        int numFeatures = X.Columns;

        FeatureLogProb1 = new Matrix<T>(numClasses, numFeatures);
        FeatureLogProb0 = new Matrix<T>(numClasses, numFeatures);

        // Calculate feature probabilities for each class
        for (int c = 0; c < numClasses; c++)
        {
            // Count samples in this class
            int classCount = Convert.ToInt32(ClassCounts![c]);

            for (int f = 0; f < numFeatures; f++)
            {
                // Count how many times feature f is 1 in class c
                T count1 = NumOps.Zero;

                for (int i = 0; i < y.Length; i++)
                {
                    if (NumOps.Equals(y[i], Classes[c]))
                    {
                        // Check if feature is present (> 0, treating as binary)
                        if (NumOps.GreaterThan(X[i, f], NumOps.Zero))
                        {
                            count1 = NumOps.Add(count1, NumOps.One);
                        }
                    }
                }

                // Calculate probabilities with smoothing
                // P(feature=1 | class) = (count_1 + alpha) / (class_count + 2*alpha)
                // P(feature=0 | class) = 1 - P(feature=1 | class)
                T denominator = NumOps.Add(NumOps.FromInt(classCount),
                    NumOps.Multiply(NumOps.FromInt(2), Alpha));

                T prob1 = NumOps.Divide(NumOps.Add(count1, Alpha), denominator);
                T prob0 = NumOps.Subtract(NumOps.One, prob1);

                FeatureLogProb1[c, f] = NumOps.Log(prob1);
                FeatureLogProb0[c, f] = NumOps.Log(prob0);
            }
        }
    }

    /// <summary>
    /// Predicts log probabilities for binary features.
    /// </summary>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> X)
    {
        if (FeatureLogProb1 == null || FeatureLogProb0 == null)
            throw new InvalidOperationException("Model must be trained before prediction");

        int numSamples = X.Rows;
        int numClasses = Classes!.Length;
        int numFeatures = X.Columns;

        var logProbs = new Matrix<T>(numSamples, numClasses);

        for (int i = 0; i < numSamples; i++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                // Start with log prior
                T logProb = NumOps.Log(ClassPriors![c]);

                // Add log probability for each feature
                for (int f = 0; f < numFeatures; f++)
                {
                    // If feature is present (> 0), use P(feature=1 | class)
                    // If feature is absent (= 0), use P(feature=0 | class)
                    if (NumOps.GreaterThan(X[i, f], NumOps.Zero))
                    {
                        logProb = NumOps.Add(logProb, FeatureLogProb1[c, f]);
                    }
                    else
                    {
                        logProb = NumOps.Add(logProb, FeatureLogProb0[c, f]);
                    }
                }

                logProbs[i, c] = logProb;
            }
        }

        return logProbs;
    }
}
```

## Test Coverage Checklist

For each Naive Bayes variant, ensure you have tests for:

**Gaussian Naive Bayes:**
- [ ] Perfect separation (100% accuracy on separable data)
- [ ] Overlapping distributions
- [ ] Multiple classes (3+)
- [ ] Probability predictions sum to 1
- [ ] Continuous features
- [ ] Edge case: Zero variance

**Multinomial Naive Bayes:**
- [ ] Text classification example
- [ ] Count data
- [ ] Smoothing prevents zero probabilities
- [ ] Sparse data handling
- [ ] Multiple documents/samples

**Bernoulli Naive Bayes:**
- [ ] Binary feature handling
- [ ] Presence/absence detection
- [ ] Difference from Multinomial on same data
- [ ] Both 0 and 1 probabilities used

## Common Mistakes to Avoid

1. **Forgetting log probabilities**: Multiplying many small probabilities causes underflow
2. **Not applying smoothing**: Can get zero probabilities for unseen features
3. **Wrong variant for data type**: Gaussian for counts, Multinomial for continuous, etc.
4. **Assuming true independence**: Naive Bayes works despite violated independence assumption
5. **Not normalizing probabilities**: Probabilities should sum to 1

## Learning Resources

- **Naive Bayes Tutorial**: https://scikit-learn.org/stable/modules/naive_bayes.html
- **Bayes' Theorem**: https://en.wikipedia.org/wiki/Bayes%27_theorem
- **Text Classification**: https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

## Validation Criteria

1. All three variants implemented with proper inheritance
2. Test coverage 80%+ for all variants
3. Handles edge cases (zero variance, unseen features, etc.)
4. Probabilities sum to 1.0 (within tolerance)
5. Log probabilities used for numerical stability

---

**Good luck!** Naive Bayes is one of the most practical and widely-used ML algorithms, especially for text classification.
