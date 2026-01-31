# Exhaustive Data Splitting Methods for AiDotNet

## Overview

Data splitting is a critical preprocessing step that divides datasets into subsets for training, validation, and testing. The choice of splitting method significantly impacts model evaluation reliability and generalization.

---

## 1. Basic Random Splitting

### 1.1 Train-Test Split
- **Description**: Randomly divides data into training and test sets
- **Use Case**: Simple evaluation, large datasets
- **Parameters**: `testSize` (default: 0.2)
- **Industry Standard**: 80/20 or 70/30 split

### 1.2 Train-Validation-Test Split (Three-Way)
- **Description**: Randomly divides data into three sets
- **Use Case**: Hyperparameter tuning + final evaluation
- **Parameters**: `trainSize` (0.7), `validationSize` (0.15), `testSize` (0.15)
- **Industry Standard**: 70/15/15 or 60/20/20

### 1.3 Holdout with Multiple Test Sets
- **Description**: Creates multiple holdout test sets for robust evaluation
- **Use Case**: When you need multiple independent test evaluations

---

## 2. Cross-Validation Methods

### 2.1 K-Fold Cross-Validation
- **Description**: Divides data into k equal folds, each fold serves as test set once
- **Use Case**: Limited data, more reliable performance estimates
- **Parameters**: `k` (default: 5 or 10)
- **Industry Standard**: k=5 for large datasets, k=10 for smaller datasets

### 2.2 Stratified K-Fold
- **Description**: K-fold that preserves class distribution in each fold
- **Use Case**: Imbalanced classification problems
- **Parameters**: `k`, preserves target distribution
- **Industry Standard**: Preferred for classification tasks

### 2.3 Repeated K-Fold
- **Description**: Runs k-fold multiple times with different random seeds
- **Use Case**: More stable performance estimates
- **Parameters**: `k`, `n_repeats` (default: 10)

### 2.4 Stratified Repeated K-Fold
- **Description**: Combines stratification with repeated k-fold
- **Use Case**: Imbalanced data with need for stable estimates

### 2.5 Leave-One-Out (LOO)
- **Description**: Each sample is test set once (k = n_samples)
- **Use Case**: Very small datasets
- **Drawback**: Computationally expensive, high variance

### 2.6 Leave-P-Out (LPO)
- **Description**: All combinations of p samples as test set
- **Use Case**: Exhaustive evaluation on tiny datasets
- **Parameters**: `p` (number of samples to leave out)

### 2.7 Monte Carlo Cross-Validation (Shuffle-Split)
- **Description**: Random train/test splits repeated multiple times
- **Use Case**: When k-fold structure isn't needed
- **Parameters**: `n_splits`, `testSize`, `trainSize`

### 2.8 Stratified Shuffle-Split
- **Description**: Monte Carlo CV with stratification
- **Use Case**: Imbalanced data with random splitting preference

---

## 3. Time Series Splitting

### 3.1 Time Series Split (Expanding Window)
- **Description**: Training set grows, test set moves forward in time
- **Use Case**: Time series forecasting, respects temporal order
- **Parameters**: `n_splits`, `maxTrainSize` (optional cap)
- **Example**: Train[1:100]->Test[101:120], Train[1:120]->Test[121:140]

### 3.2 Sliding Window Split
- **Description**: Fixed-size training window slides through time
- **Use Case**: When old data becomes less relevant
- **Parameters**: `n_splits`, `trainSize`, `testSize`
- **Example**: Train[1:100]->Test[101:120], Train[21:120]->Test[121:140]

### 3.3 Blocked Time Series Split
- **Description**: Adds gap between train and test to prevent leakage
- **Use Case**: When features have temporal dependencies
- **Parameters**: `n_splits`, `gap` (number of samples to skip)

### 3.4 Purged K-Fold (Financial)
- **Description**: Removes samples around test period to prevent leakage
- **Use Case**: Financial time series with overlapping labels
- **Parameters**: `k`, `purgeSize`, `embargoSize`

### 3.5 Combinatorial Purged Cross-Validation
- **Description**: Tests all combinations of time periods
- **Use Case**: Backtesting trading strategies
- **Reference**: Marcos López de Prado's methodology

### 3.6 Walk-Forward Validation
- **Description**: Sequential train-test splits moving through time
- **Use Case**: Production deployment simulation
- **Parameters**: `initialTrainSize`, `stepSize`, `testSize`

### 3.7 Anchored Walk-Forward
- **Description**: Walk-forward with fixed starting point
- **Use Case**: When historical data remains relevant

### 3.8 Rolling Origin Evaluation
- **Description**: Similar to walk-forward with forecast horizon
- **Use Case**: Multi-step forecasting evaluation
- **Parameters**: `origin`, `horizon`, `step`

---

## 4. Group-Based Splitting

### 4.1 Group K-Fold
- **Description**: Ensures samples from same group stay together
- **Use Case**: Multiple samples per subject/entity
- **Example**: All patient visits in same fold

### 4.2 Stratified Group K-Fold
- **Description**: Group k-fold with class distribution preservation
- **Use Case**: Grouped data with imbalanced classes

### 4.3 Leave-One-Group-Out (LOGO)
- **Description**: Each group serves as test set once
- **Use Case**: Cross-subject validation
- **Example**: Leave one patient out for testing

### 4.4 Leave-P-Groups-Out
- **Description**: All combinations of p groups as test set
- **Use Case**: Small number of groups

### 4.5 Group Shuffle Split
- **Description**: Random group-based train/test splits
- **Use Case**: When group k-fold structure isn't needed

### 4.6 Predefined Split
- **Description**: User specifies which samples go to which set
- **Use Case**: Domain-specific splitting requirements

---

## 5. Stratified Splitting (Class-Balanced)

### 5.1 Stratified Train-Test Split
- **Description**: Maintains class proportions in both sets
- **Use Case**: Imbalanced binary/multiclass classification
- **Industry Standard**: Default for classification tasks

### 5.2 Stratified Three-Way Split
- **Description**: Three-way split preserving class distribution
- **Use Case**: Imbalanced classification with validation

### 5.3 Iterative Stratification (Multi-Label)
- **Description**: Stratification for multi-label classification
- **Use Case**: When samples have multiple labels
- **Reference**: Sechidis et al. algorithm

### 5.4 Distribution-Preserving Split (Regression)
- **Description**: Maintains target distribution for continuous targets
- **Use Case**: Regression with skewed target distribution
- **Method**: Bins target values, stratifies by bins

---

## 6. Cluster-Based Splitting

### 6.1 Cluster-Based Train-Test Split
- **Description**: Clusters data, assigns clusters to train/test
- **Use Case**: Ensures test data is truly "different"
- **Method**: K-means clustering, then split by cluster

### 6.2 Anti-Clustering Split
- **Description**: Maximizes diversity within each split
- **Use Case**: Ensures both sets are representative

### 6.3 Similarity-Based Split
- **Description**: Keeps similar samples together or apart
- **Use Case**: Domain-specific similarity requirements

---

## 7. Adversarial/Robust Splitting

### 7.1 Adversarial Validation Split
- **Description**: Trains classifier to distinguish train/test, removes easy cases
- **Use Case**: When train and test distributions differ
- **Method**: If classifier can't distinguish, split is good

### 7.2 Covariate Shift Split
- **Description**: Intentionally creates distribution shift
- **Use Case**: Testing model robustness to distribution shift

### 7.3 Out-of-Distribution Split
- **Description**: Test set contains samples outside training distribution
- **Use Case**: Evaluating generalization to new domains

---

## 8. Nested/Hierarchical Splitting

### 8.1 Nested Cross-Validation
- **Description**: Inner CV for hyperparameter tuning, outer for evaluation
- **Use Case**: Unbiased performance estimation with tuning
- **Parameters**: `outerFolds`, `innerFolds`

### 8.2 Double Cross-Validation
- **Description**: Two-level CV for model selection and evaluation
- **Use Case**: Comparing multiple models fairly

### 8.3 Hierarchical Split
- **Description**: Splits respect hierarchical data structure
- **Use Case**: Multi-level data (schools > classrooms > students)

---

## 9. Bootstrap-Based Methods

### 9.1 Bootstrap Sampling
- **Description**: Sample with replacement for training, out-of-bag for test
- **Use Case**: Variance estimation, ensemble methods
- **OOB Rate**: ~36.8% samples not selected (test set)

### 9.2 .632 Bootstrap
- **Description**: Weighted combination of training and OOB error
- **Use Case**: Bias-corrected error estimation

### 9.3 .632+ Bootstrap
- **Description**: Further correction for overfitting
- **Use Case**: When .632 is too optimistic

### 9.4 Stratified Bootstrap
- **Description**: Bootstrap with class proportion preservation
- **Use Case**: Imbalanced classification

---

## 10. Domain-Specific Splitting

### 10.1 Spatial Split (Geographic)
- **Description**: Splits based on geographic coordinates
- **Use Case**: Geospatial models, avoids spatial autocorrelation
- **Methods**: Block-based, buffer zones, clustering by location

### 10.2 Temporal-Spatial Split
- **Description**: Considers both time and location
- **Use Case**: Spatio-temporal forecasting

### 10.3 Graph-Based Split
- **Description**: Splits graph nodes/edges appropriately
- **Use Case**: Graph neural networks
- **Methods**: Transductive vs inductive splits

### 10.4 Sequence Split
- **Description**: For sequential data (text, DNA, etc.)
- **Use Case**: NLP, bioinformatics
- **Consideration**: Avoid splitting within sequences

### 10.5 Image Patch Split
- **Description**: For image segmentation tasks
- **Use Case**: Medical imaging, satellite imagery
- **Consideration**: No spatial overlap between train/test

### 10.6 Multi-Task Split
- **Description**: Consistent splits across multiple related tasks
- **Use Case**: Multi-task learning evaluation

---

## 11. Active Learning Splits

### 11.1 Pool-Based Split
- **Description**: Large unlabeled pool, small labeled set
- **Use Case**: Active learning scenarios

### 11.2 Query-by-Committee Split
- **Description**: Initial split for committee disagreement
- **Use Case**: Uncertainty-based active learning

---

## 12. Federated Learning Splits

### 12.1 IID Client Split
- **Description**: Data randomly distributed across clients
- **Use Case**: Federated learning baseline

### 12.2 Non-IID Client Split
- **Description**: Each client has biased data distribution
- **Use Case**: Realistic federated scenarios
- **Methods**: Label skew, feature skew, quantity skew

### 12.3 Dirichlet Split
- **Description**: Controls non-IID-ness via Dirichlet distribution
- **Use Case**: Parameterized federated experiments
- **Parameters**: `alpha` (lower = more non-IID)

---

## 13. Incremental/Online Learning Splits

### 13.1 Prequential (Interleaved Test-Then-Train)
- **Description**: Each sample tests then trains
- **Use Case**: Online learning evaluation

### 13.2 Landmark Window
- **Description**: Fixed starting point, growing window
- **Use Case**: Concept drift detection

### 13.3 Sliding Window (Online)
- **Description**: Fixed-size window moves through stream
- **Use Case**: Data stream mining

---

## Industry Standard Defaults

| Scenario | Recommended Split | Ratios |
|----------|------------------|--------|
| Large dataset (>100k) | Train-Test | 80/20 |
| Medium dataset (10k-100k) | Train-Val-Test | 70/15/15 |
| Small dataset (<10k) | K-Fold CV | k=5 or k=10 |
| Very small (<1k) | Leave-One-Out or Nested CV | - |
| Imbalanced classification | Stratified variant | Same ratios |
| Time series | Time Series Split | No shuffle |
| Grouped data | Group K-Fold | Varies |

---

## Proposed Interface Design

```csharp
public interface IDataSplitter<T>
{
    /// <summary>
    /// Splits data into training and test sets.
    /// </summary>
    DataSplitResult<T> Split(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Generates multiple train/test splits (for CV methods).
    /// </summary>
    IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Number of splits this splitter generates.
    /// </summary>
    int NumSplits { get; }

    /// <summary>
    /// Description of the splitting strategy.
    /// </summary>
    string Description { get; }
}

public class DataSplitResult<T>
{
    public Matrix<T> XTrain { get; init; }
    public Matrix<T> XTest { get; init; }
    public Vector<T> yTrain { get; init; }
    public Vector<T> yTest { get; init; }

    // Optional validation set
    public Matrix<T>? XValidation { get; init; }
    public Vector<T>? yValidation { get; init; }

    // Indices for tracking
    public int[] TrainIndices { get; init; }
    public int[] TestIndices { get; init; }
    public int[]? ValidationIndices { get; init; }
}
```

---

## Priority Implementation Order

### Phase 1: Core (Must Have)
1. RandomSplitter (train-test, train-val-test)
2. StratifiedSplitter
3. KFoldSplitter
4. StratifiedKFoldSplitter
5. TimeSeriesSplitter

### Phase 2: Common
6. GroupKFoldSplitter
7. LeaveOneOutSplitter
8. ShuffleSplitter (Monte Carlo CV)
9. RepeatedKFoldSplitter

### Phase 3: Advanced
10. NestedCVSplitter
11. PurgedKFoldSplitter
12. BootstrapSplitter
13. SpatialSplitter

### Phase 4: Specialized
14. AdversarialValidationSplitter
15. MultiLabelStratifiedSplitter
16. FederatedSplitters
17. OnlineLearning splitters
