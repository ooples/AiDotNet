namespace AiDotNet.Evaluation.Enums;

/// <summary>
/// Specifies the cross-validation strategy to use for model evaluation.
/// </summary>
/// <remarks>
/// <para>
/// Cross-validation is a technique for assessing how well a model generalizes to independent data.
/// Different strategies are appropriate for different data types and problem domains.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cross-validation is like giving your model multiple "practice tests" on
/// different portions of your data. Each strategy splits the data differently:
/// <list type="bullet">
/// <item><b>KFold</b>: Divides data into K equal parts, trains on K-1, tests on 1, repeats K times</item>
/// <item><b>Stratified</b>: Same as KFold but preserves class proportions (important for imbalanced data)</item>
/// <item><b>TimeSeries</b>: Respects time order - always trains on past, tests on future</item>
/// <item><b>Group</b>: Keeps related samples together (e.g., all data from one patient stays together)</item>
/// </list>
/// Choose based on your data type and whether you have class imbalance, time dependencies, or grouped samples.
/// </para>
/// </remarks>
public enum CrossValidationStrategy
{
    /// <summary>
    /// Standard K-Fold cross-validation. Data is divided into K equal folds.
    /// Each fold is used once as validation while the remaining K-1 folds form the training set.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The most common CV method. With K=5, your data is split into
    /// 5 equal parts. The model is trained and tested 5 times, each time using a different
    /// part as the test set. Good general-purpose choice when you don't have special requirements.</para>
    /// <para><b>When to use:</b> General-purpose, balanced datasets, no time dependency.</para>
    /// <para><b>Research standard:</b> K=5 or K=10 (Kohavi, 1995)</para>
    /// </remarks>
    KFold = 0,

    /// <summary>
    /// Stratified K-Fold preserves the percentage of samples for each class in each fold.
    /// Essential for classification with imbalanced classes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If your data has 90% class A and 10% class B, regular KFold
    /// might create folds with very few class B samples. Stratified ensures each fold has
    /// roughly 90/10 split, giving more reliable performance estimates.</para>
    /// <para><b>When to use:</b> Classification with imbalanced classes.</para>
    /// <para><b>Industry standard:</b> Default for classification in scikit-learn.</para>
    /// </remarks>
    StratifiedKFold = 1,

    /// <summary>
    /// Repeated K-Fold runs standard K-Fold multiple times with different random shuffles.
    /// Provides more robust performance estimates at the cost of computation time.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Running KFold once can give results that depend on how
    /// the data happened to be split. Repeating it (e.g., 10 times) with different shuffles
    /// and averaging gives a more stable estimate of true performance.</para>
    /// <para><b>When to use:</b> When you need high-confidence performance estimates.</para>
    /// <para><b>Research standard:</b> 10x10 CV (10 repeats of 10-fold) is common in ML papers.</para>
    /// </remarks>
    RepeatedKFold = 2,

    /// <summary>
    /// Repeated Stratified K-Fold combines stratification with repetition.
    /// The gold standard for classification problems requiring robust estimates.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Combines the benefits of stratification (preserving class balance)
    /// with repetition (reducing variance from random splits). Best choice when you need
    /// reliable performance estimates for imbalanced classification.</para>
    /// <para><b>When to use:</b> Imbalanced classification requiring robust estimates.</para>
    /// </remarks>
    RepeatedStratifiedKFold = 3,

    /// <summary>
    /// Leave-One-Out (LOO) uses each sample as a single test case while training on all others.
    /// Provides nearly unbiased estimates but is computationally expensive for large datasets.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you have 100 samples, the model is trained 100 times,
    /// each time leaving out just one sample for testing. This uses almost all data for training
    /// but is very slow for large datasets.</para>
    /// <para><b>When to use:</b> Small datasets (N &lt; 100) where every sample matters.</para>
    /// <para><b>Note:</b> Can have high variance; consider repeated K-fold for stability.</para>
    /// </remarks>
    LeaveOneOut = 4,

    /// <summary>
    /// Leave-P-Out uses all possible combinations of P samples as test sets.
    /// More exhaustive than LOO but exponentially more expensive.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Similar to Leave-One-Out but leaves out P samples at a time.
    /// With P=2 and 10 samples, you'd have 45 different train/test splits. Becomes impractical
    /// quickly as P and N increase.</para>
    /// <para><b>When to use:</b> Very small datasets, theoretical analysis.</para>
    /// <para><b>Warning:</b> Number of splits = C(N,P), grows very fast!</para>
    /// </remarks>
    LeavePOut = 5,

    /// <summary>
    /// Shuffle-Split creates random train/test splits with configurable sizes.
    /// Useful for large datasets where K-Fold is too slow.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of systematic folds, randomly picks samples for
    /// training and testing. You control the train/test ratio and number of splits.
    /// Faster than K-Fold for large datasets.</para>
    /// <para><b>When to use:</b> Large datasets, quick experimentation.</para>
    /// </remarks>
    ShuffleSplit = 6,

    /// <summary>
    /// Stratified Shuffle-Split maintains class proportions in random splits.
    /// Combines benefits of stratification with flexibility of shuffle splitting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Random splits like ShuffleSplit, but ensures each split
    /// has the same class balance as the original data. Good for large imbalanced datasets.</para>
    /// <para><b>When to use:</b> Large imbalanced datasets.</para>
    /// </remarks>
    StratifiedShuffleSplit = 7,

    /// <summary>
    /// Group K-Fold ensures samples from the same group never appear in both train and test.
    /// Essential when samples are not independent (e.g., multiple measurements per subject).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you have data from 10 patients with multiple samples each,
    /// regular K-Fold might put some samples from patient A in training and others in testing.
    /// This can cause data leakage! Group K-Fold keeps all of patient A's data together.</para>
    /// <para><b>When to use:</b> Medical data (per-patient), customer data (per-user), etc.</para>
    /// <para><b>Critical:</b> Using regular K-Fold on grouped data causes optimistic bias!</para>
    /// </remarks>
    GroupKFold = 8,

    /// <summary>
    /// Group Shuffle-Split randomly samples groups for train/test while respecting group boundaries.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like ShuffleSplit but operates on groups instead of samples.
    /// Randomly assigns entire groups to train or test sets.</para>
    /// <para><b>When to use:</b> Large grouped datasets, quick experimentation with groups.</para>
    /// </remarks>
    GroupShuffleSplit = 9,

    /// <summary>
    /// Time Series Split uses expanding training windows. Each fold adds more historical data.
    /// Respects temporal ordering - never trains on future data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For time-ordered data (stock prices, weather), you can't
    /// use future data to predict the past. This strategy trains on data up to time T and
    /// tests on data after T. The training window grows with each fold.</para>
    /// <para><b>When to use:</b> Time series forecasting, temporal data.</para>
    /// <para><b>Critical:</b> Using regular K-Fold on time series causes future data leakage!</para>
    /// </remarks>
    TimeSeriesSplit = 10,

    /// <summary>
    /// Sliding Window Split uses fixed-size training windows that slide through time.
    /// Useful when older data becomes less relevant.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Similar to TimeSeriesSplit but keeps the training window
    /// a fixed size by dropping old data as new data is added. Useful when patterns change
    /// over time and old data might hurt performance.</para>
    /// <para><b>When to use:</b> Time series with concept drift, regime changes.</para>
    /// </remarks>
    SlidingWindowSplit = 11,

    /// <summary>
    /// Blocked Time Series Split adds gaps between train and test to prevent information leakage.
    /// Standard in financial ML to avoid look-ahead bias.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In financial data, today's features often contain information
    /// about tomorrow (lagged variables). Adding a gap (embargo) between training and test
    /// periods prevents this subtle form of leakage.</para>
    /// <para><b>When to use:</b> Financial time series, economic data.</para>
    /// <para><b>Research reference:</b> de Prado, "Advances in Financial Machine Learning"</para>
    /// </remarks>
    BlockedTimeSeriesSplit = 12,

    /// <summary>
    /// Purged K-Fold removes samples from training that are temporally close to test samples.
    /// Prevents leakage from overlapping labels or features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When labels or features span multiple time periods (e.g.,
    /// returns calculated over 5 days), samples near the train/test boundary can leak
    /// information. Purging removes these problematic samples.</para>
    /// <para><b>When to use:</b> Financial ML with overlapping labels.</para>
    /// <para><b>Research reference:</b> de Prado, "Advances in Financial Machine Learning"</para>
    /// </remarks>
    PurgedKFold = 13,

    /// <summary>
    /// Combinatorial Purged CV generates all possible train/test combinations with purging.
    /// The most rigorous approach for financial backtesting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Combines purging with combinatorial splitting to generate
    /// many more test scenarios than traditional CV. Provides the most robust performance
    /// estimates for financial applications but is computationally expensive.</para>
    /// <para><b>When to use:</b> Financial ML requiring maximum rigor.</para>
    /// <para><b>Research reference:</b> de Prado, "Advances in Financial Machine Learning"</para>
    /// </remarks>
    CombinatorialPurgedCV = 14,

    /// <summary>
    /// Nested CV uses an outer loop for performance estimation and inner loop for hyperparameter tuning.
    /// Provides unbiased estimates when hyperparameters are optimized.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you tune hyperparameters using CV and then report that
    /// same CV score, you're being optimistic! Nested CV uses separate loops: the inner loop
    /// tunes parameters, the outer loop estimates true performance.</para>
    /// <para><b>When to use:</b> Hyperparameter optimization with unbiased evaluation.</para>
    /// <para><b>Research reference:</b> Cawley & Talbot, "On Over-fitting in Model Selection"</para>
    /// </remarks>
    NestedCV = 15,

    /// <summary>
    /// Monte Carlo CV (repeated random sub-sampling) creates random train/test splits.
    /// Good for variance estimation with configurable split ratio.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like ShuffleSplit but typically with more repetitions
    /// to estimate the variance of performance metrics. Each iteration randomly assigns
    /// samples to train or test.</para>
    /// <para><b>When to use:</b> Variance estimation, confidence intervals.</para>
    /// </remarks>
    MonteCarloCV = 16,

    /// <summary>
    /// Bootstrap CV uses bootstrap resampling (sampling with replacement).
    /// Provides out-of-bag samples for testing without explicit splits.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bootstrap creates training sets by randomly sampling from
    /// your data with replacement (same sample can appear multiple times). Samples not
    /// selected (~37%) form the out-of-bag test set. Good for uncertainty estimation.</para>
    /// <para><b>When to use:</b> Small datasets, uncertainty quantification.</para>
    /// <para><b>Note:</b> On average, 63.2% of samples appear in each bootstrap.</para>
    /// </remarks>
    BootstrapCV = 17,

    /// <summary>
    /// Adversarial Split deliberately creates challenging test sets to evaluate robustness.
    /// Selects test samples that are most different from training data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of random splits, this deliberately puts the
    /// "hardest" or most different samples in the test set. Useful for stress-testing
    /// models and detecting distribution shift vulnerabilities.</para>
    /// <para><b>When to use:</b> Robustness evaluation, stress testing, domain shift analysis.</para>
    /// </remarks>
    AdversarialSplit = 18
}
