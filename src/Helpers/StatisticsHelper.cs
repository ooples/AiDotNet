global using AiDotNet.Models.Results;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Helpers;


/// <summary>
/// Provides statistical calculation methods for various data analysis tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float, decimal).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class contains methods to calculate common statistical measures 
/// like averages, variations, and statistical tests. These help you understand your data's 
/// patterns and make decisions based on statistical evidence.
/// </para>
/// </remarks>
public static class StatisticsHelper<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    private static T GetOrDefaultOptionalParameter(T? value, T defaultValue, bool treatDefaultAsMissing = true)
    {
        if (value is null)
        {
            return defaultValue;
        }

        // With unconstrained generics, optional parameters of type `T?` that are omitted at the callsite
        // are passed as `default(T)` (e.g. 0 for numeric structs), not `null`. Treat `default(T)` as
        // "not provided" so documented defaults are applied consistently.
        return treatDefaultAsMissing && EqualityComparer<T>.Default.Equals(value, default!) ? defaultValue : value;
    }

    /// <summary>
    /// Calculates the median value from a collection of numeric values.
    /// </summary>
    /// <param name="values">The collection of values to calculate the median from.</param>
    /// <returns>The median value of the collection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The median is the middle value when all values are arranged in order.
    /// If there's an even number of values, it's the average of the two middle values.
    /// </para>
    /// <para>
    /// For example, the median of [1, 3, 5, 7, 9] is 5, and the median of [1, 3, 5, 7] is 4 (average of 3 and 5).
    /// </para>
    /// </remarks>
    public static T CalculateMedian(IEnumerable<T> values)
    {
        var sortedValues = values.ToArray();
        Array.Sort(sortedValues);
        int n = sortedValues.Length;
        if (n % 2 == 0)
        {
            return _numOps.Divide(_numOps.Add(sortedValues[n / 2 - 1], sortedValues[n / 2]), _numOps.FromDouble(2));
        }

        return sortedValues[n / 2];
    }

    /// <summary>
    /// Calculates the Mean Absolute Deviation (MAD) of a vector of values from a given median.
    /// </summary>
    /// <param name="values">The vector of values to calculate MAD for.</param>
    /// <param name="median">The median value to calculate deviations from.</param>
    /// <returns>The Mean Absolute Deviation of the values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Absolute Deviation measures how spread out your data is from a central value (median).
    /// It calculates the average of the absolute differences between each value and the median.
    /// </para>
    /// <para>
    /// For example, for values [2, 4, 6, 8] with median 5:
    /// 1. Calculate absolute differences: |2-5|=3, |4-5|=1, |6-5|=1, |8-5|=3
    /// 2. Calculate average: (3+1+1+3)/4 = 2
    /// </para>
    /// </remarks>
    public static T CalculateMeanAbsoluteDeviation(Vector<T> values, T median)
    {
        return _numOps.Divide(values.Select(x => _numOps.Abs(_numOps.Subtract(x, median))).Aggregate(_numOps.Zero, _numOps.Add), _numOps.FromDouble(values.Length));
    }

    /// <summary>
    /// Calculates the variance of a vector of values from a given mean.
    /// </summary>
    /// <param name="values">The vector of values to calculate variance for.</param>
    /// <param name="mean">The mean value to calculate deviations from.</param>
    /// <returns>The variance of the values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Variance measures how spread out your data is from the average (mean).
    /// Higher variance means data points are more scattered; lower variance means they're closer together.
    /// </para>
    /// <para>
    /// It's calculated by:
    /// 1. Finding the difference between each value and the mean
    /// 2. Squaring each difference
    /// 3. Calculating the average of these squared differences
    /// </para>
    /// </remarks>
    public static T CalculateVariance(Vector<T> values, T mean)
    {
        T sumOfSquares = values.Select(x => _numOps.Square(_numOps.Subtract(x, mean))).Aggregate(_numOps.Zero, _numOps.Add);
        return _numOps.Divide(sumOfSquares, _numOps.FromDouble(values.Length - 1));
    }

    /// <summary>
    /// Calculates the variance of a collection of values.
    /// </summary>
    /// <param name="values">The collection of values to calculate variance for.</param>
    /// <returns>The variance of the values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates variance without requiring you to provide the mean.
    /// It first calculates the mean internally, then computes the variance.
    /// </para>
    /// <para>
    /// Variance is zero when all values are identical, and increases as values become more spread out.
    /// </para>
    /// </remarks>
    public static T CalculateVariance(IEnumerable<T> values)
    {
        var enumerable = values.ToList();
        int count = enumerable.Count;

        if (count < 2)
        {
            return _numOps.Zero;
        }

        T sum = enumerable.Aggregate(_numOps.Zero, (acc, val) => _numOps.Add(acc, val));
        T mean = _numOps.Divide(sum, _numOps.FromDouble(count));

        T sumOfSquaredDifferences = enumerable
            .Select(x => _numOps.Square(_numOps.Subtract(x, mean)))
            .Aggregate(_numOps.Zero, (acc, val) => _numOps.Add(acc, val));

        return _numOps.Divide(sumOfSquaredDifferences, _numOps.FromDouble(count - 1));
    }

    /// <summary>
    /// Calculates the standard deviation of a vector.
    /// </summary>
    /// <param name="values">The collection of values to calculate standard deviation for.</param>
    /// <returns>The standard deviation of the values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Standard deviation is the square root of variance. It measures how spread out 
    /// your data is, in the same units as your original data (unlike variance, which is in squared units).
    /// </para>
    /// <para>
    /// A low standard deviation means data points tend to be close to the mean, while a high standard 
    /// deviation means data points are spread out over a wider range.
    /// </para>
    /// </remarks>
    public static T CalculateStandardDeviation(IEnumerable<T> values)
    {
        return _numOps.Sqrt(CalculateVariance(values));
    }

    /// <summary>
    /// Calculates the Mean Squared Error (MSE) between actual and predicted values.
    /// </summary>
    /// <param name="actualValues">The actual observed values.</param>
    /// <param name="predictedValues">The predicted or estimated values.</param>
    /// <returns>The Mean Squared Error between the actual and predicted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Squared Error measures how accurate your predictions are compared to actual values.
    /// Lower MSE means better predictions.
    /// </para>
    /// <para>
    /// It's calculated by:
    /// 1. Finding the difference between each actual and predicted value
    /// 2. Squaring each difference (to make all values positive and emphasize larger errors)
    /// 3. Calculating the average of these squared differences
    /// </para>
    /// <para>
    /// MSE is commonly used to evaluate machine learning models, especially regression models.
    /// </para>
    /// </remarks>
    public static T CalculateMeanSquaredError(IEnumerable<T> actualValues, IEnumerable<T> predictedValues)
    {
        T sumOfSquaredErrors = actualValues.Zip(predictedValues, (a, p) => _numOps.Square(_numOps.Subtract(a, p)))
                                           .Aggregate(_numOps.Zero, _numOps.Add);
        return _numOps.Divide(sumOfSquaredErrors, _numOps.FromDouble(actualValues.Count()));
    }

    /// <summary>
    /// Calculates the variance reduction achieved by splitting data into left and right groups.
    /// </summary>
    /// <param name="y">The target values vector.</param>
    /// <param name="leftIndices">Indices of values in the left group.</param>
    /// <param name="rightIndices">Indices of values in the right group.</param>
    /// <returns>The variance reduction achieved by the split.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Variance reduction measures how much a split improves the "purity" of data.
    /// It's used in decision trees to find the best way to split data into groups.
    /// </para>
    /// <para>
    /// Higher variance reduction means the split creates more homogeneous (similar) groups,
    /// which is desirable when building decision trees.
    /// </para>
    /// </remarks>
    public static T CalculateVarianceReduction(Vector<T> y, List<int> leftIndices, List<int> rightIndices)
    {
        T totalVariance = StatisticsHelper<T>.CalculateVariance(y);
        T leftVariance = StatisticsHelper<T>.CalculateVariance(leftIndices.Select(i => y[i]));
        T rightVariance = StatisticsHelper<T>.CalculateVariance(rightIndices.Select(i => y[i]));

        T leftWeight = _numOps.Divide(_numOps.FromDouble(leftIndices.Count), _numOps.FromDouble(y.Length));
        T rightWeight = _numOps.Divide(_numOps.FromDouble(rightIndices.Count), _numOps.FromDouble(y.Length));

        return _numOps.Subtract(totalVariance, _numOps.Add(_numOps.Multiply(leftWeight, leftVariance), _numOps.Multiply(rightWeight, rightVariance)));
    }

    /// <summary>
    /// Calculates a score for a data split based on the specified criterion.
    /// </summary>
    /// <param name="y">The target values vector.</param>
    /// <param name="leftIndices">Indices of values in the left group after splitting.</param>
    /// <param name="rightIndices">Indices of values in the right group after splitting.</param>
    /// <param name="splitCriterion">The criterion to use for evaluating the split quality.</param>
    /// <returns>A score indicating the quality of the split (higher is better).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method helps determine how good a split is when dividing data into two groups.
    /// Different criteria measure quality differently - some look at how similar items are within each group,
    /// others at how well the split helps with predictions.
    /// </para>
    /// <para>
    /// Think of it like sorting fruits: if you split apples and oranges perfectly, you'd get a high score
    /// because each group is very "pure" (contains only one type of fruit).
    /// </para>
    /// </remarks>
    public static T CalculateSplitScore(Vector<T> y, List<int> leftIndices, List<int> rightIndices, SplitCriterion splitCriterion)
    {
        return splitCriterion switch
        {
            SplitCriterion.VarianceReduction => CalculateVarianceReduction(y, leftIndices, rightIndices),
            SplitCriterion.MeanSquaredError => CalculateMeanSquaredError(y, leftIndices, rightIndices),
            SplitCriterion.MeanAbsoluteError => CalculateMeanAbsoluteError(y, leftIndices, rightIndices),
            SplitCriterion.FriedmanMSE => CalculateFriedmanMSE(y, leftIndices, rightIndices),
            _ => throw new ArgumentException("Invalid split criterion"),
        };
    }

    /// <summary>
    /// Calculates the p-value for a statistical test comparing two groups.
    /// </summary>
    /// <param name="leftY">The first group of values.</param>
    /// <param name="rightY">The second group of values.</param>
    /// <param name="testType">The type of statistical test to perform.</param>
    /// <returns>The p-value from the statistical test.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A p-value tells you how likely your results could have happened by random chance.
    /// Smaller p-values (typically &lt; 0.05) suggest that the differences between groups are statistically significant
    /// and not just due to random variation.
    /// </para>
    /// <para>
    /// Different test types are appropriate for different kinds of data:
    /// - T-Test: For comparing means when data is normally distributed
    /// - Mann-Whitney U: For comparing distributions when data might not be normal
    /// - Permutation Test: A flexible test that works by randomly shuffling data
    /// - Chi-Square: For comparing categorical data
    /// - F-Test: For comparing variances
    /// </para>
    /// </remarks>
    public static T CalculatePValue(Vector<T> leftY, Vector<T> rightY, TestStatisticType testType)
    {
        return testType switch
        {
            TestStatisticType.TTest => TTest(leftY, rightY).PValue,
            TestStatisticType.MannWhitneyU => MannWhitneyUTest(leftY, rightY).PValue,
            TestStatisticType.PermutationTest => PermutationTest(leftY, rightY).PValue,
            TestStatisticType.ChiSquare => ChiSquareTest(leftY, rightY).PValue,
            TestStatisticType.FTest => FTest(leftY, rightY).PValue,
            _ => throw new ArgumentException("Invalid statistical test type", nameof(testType))
        };
    }

    /// <summary>
    /// Performs a Student's t-test to compare means between two groups.
    /// </summary>
    /// <param name="leftY">The first group of values.</param>
    /// <param name="rightY">The second group of values.</param>
    /// <param name="significanceLevel">The significance level for hypothesis testing (default: 0.05).</param>
    /// <returns>A result object containing the t-statistic, degrees of freedom, p-value, and significance level.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The t-test helps determine if there's a significant difference between the averages 
    /// of two groups. It's commonly used when you want to know if one group's average is truly different from another's.
    /// </para>
    /// <para>
    /// For example, you might use a t-test to compare the average test scores of students who studied using two different methods.
    /// </para>
    /// <para>
    /// The t-statistic measures how many standard errors the two means are apart.
    /// A larger absolute t-value suggests a greater difference between groups.
    /// </para>
    /// <para>
    /// The p-value tells you the probability of seeing such a difference by random chance.
    /// If p &lt; significanceLevel (typically 0.05), the difference is considered statistically significant.
    /// </para>
    /// </remarks>
    public static TTestResult<T> TTest(Vector<T> leftY, Vector<T> rightY, T? significanceLevel = default)
    {
        significanceLevel = GetOrDefaultOptionalParameter(significanceLevel, _numOps.FromDouble(0.05));

        T leftMean = CalculateMean(leftY);
        T rightMean = CalculateMean(rightY);
        T leftVariance = CalculateVariance(leftY, leftMean);
        T rightVariance = CalculateVariance(rightY, rightMean);

        T pooledStandardError = _numOps.Sqrt(_numOps.Add(
            _numOps.Divide(leftVariance, _numOps.FromDouble(leftY.Length)),
            _numOps.Divide(rightVariance, _numOps.FromDouble(rightY.Length))
        ));

        T tStatistic = _numOps.Divide(_numOps.Subtract(leftMean, rightMean), pooledStandardError);
        int degreesOfFreedom = leftY.Length + rightY.Length - 2;

        T pValue = CalculatePValueFromTDistribution(tStatistic, degreesOfFreedom);

        return new TTestResult<T>(tStatistic, degreesOfFreedom, pValue, significanceLevel);
    }

    /// <summary>
    /// Performs a Mann-Whitney U test to compare distributions between two groups.
    /// </summary>
    /// <param name="leftY">The first group of values.</param>
    /// <param name="rightY">The second group of values.</param>
    /// <param name="significanceLevel">The significance level for hypothesis testing (default: 0.05).</param>
    /// <returns>A result object containing the U statistic, Z-score, p-value, and significance level.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Mann-Whitney U test (also called Wilcoxon rank-sum test) compares two groups 
    /// without assuming they follow a normal distribution. It works by ranking all values and analyzing 
    /// the distribution of ranks between groups.
    /// </para>
    /// <para>
    /// This test is useful when:
    /// - Your data might not be normally distributed
    /// - You have outliers that could skew results
    /// - You're working with ordinal data (like ratings from 1-5)
    /// </para>
    /// <para>
    /// The U statistic represents the number of times values in one group precede values in the other group.
    /// The Z-score standardizes this value, and the p-value tells you if the difference is statistically significant.
    /// </para>
    /// </remarks>
    public static MannWhitneyUTestResult<T> MannWhitneyUTest(Vector<T> leftY, Vector<T> rightY, T? significanceLevel = default)
    {
        significanceLevel = GetOrDefaultOptionalParameter(significanceLevel, _numOps.FromDouble(0.05));

        var allValues = leftY.Concat(rightY).ToList();
        var ranks = CalculateRanks(allValues);

        T leftRankSum = _numOps.Zero;
        for (int i = 0; i < leftY.Length; i++)
        {
            leftRankSum = _numOps.Add(leftRankSum, ranks[i]);
        }

        T u1 = _numOps.Subtract(
            _numOps.Multiply(_numOps.FromDouble(leftY.Length), _numOps.FromDouble(rightY.Length)),
            _numOps.Subtract(leftRankSum, _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(leftY.Length), _numOps.FromDouble(leftY.Length + 1)), _numOps.FromDouble(2)))
        );

        T u2 = _numOps.Subtract(
            _numOps.Multiply(_numOps.FromDouble(leftY.Length), _numOps.FromDouble(rightY.Length)),
            u1
        );

        T u = _numOps.LessThan(u1, u2) ? u1 : u2;

        // Calculate p-value using normal approximation
        T mean = _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(leftY.Length), _numOps.FromDouble(rightY.Length)), _numOps.FromDouble(2));
        T standardDeviation = _numOps.Sqrt(_numOps.Divide(
            _numOps.Multiply(_numOps.FromDouble(leftY.Length), _numOps.FromDouble(rightY.Length)),
            _numOps.FromDouble(12)
        ));

        T zScore = _numOps.Divide(_numOps.Subtract(u, mean), standardDeviation);
        T pValue = CalculatePValueFromZScore(zScore);

        return new MannWhitneyUTestResult<T>(u, zScore, pValue, significanceLevel);
    }

    /// <summary>
    /// Performs a permutation test to determine if there is a significant difference between two groups.
    /// </summary>
    /// <param name="leftY">The first group of values.</param>
    /// <param name="rightY">The second group of values.</param>
    /// <param name="significanceLevel">The threshold p-value to determine statistical significance (default is 0.05).</param>
    /// <returns>A result object containing the test statistics and conclusion.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A permutation test is a statistical technique that helps determine if the difference 
    /// between two groups is due to chance or represents a real difference. It works by repeatedly shuffling all 
    /// the data and recalculating the difference between groups to see how often a difference as large as the 
    /// observed one occurs by random chance.
    /// </para>
    /// <para>
    /// Think of it like shuffling a deck of cards many times to see how often a particular arrangement happens 
    /// naturally. If your observed arrangement rarely occurs by chance, it's likely significant.
    /// </para>
    /// </remarks>
    public static PermutationTestResult<T> PermutationTest(Vector<T> leftY, Vector<T> rightY, T? significanceLevel = default)
    {
        significanceLevel = GetOrDefaultOptionalParameter(significanceLevel, _numOps.FromDouble(0.05));
        int _permutations = 1000; // Number of permutations
        T _observedDifference = _numOps.Subtract(CalculateMean(leftY), CalculateMean(rightY));
        int _countExtremeValues = 0;

        var _allValues = leftY.Concat(rightY).ToList();
        int _totalSize = _allValues.Count;

        for (int i = 0; i < _permutations; i++)
        {
            Shuffle(_allValues);
            var _permutedLeft = _allValues.Take(leftY.Length).ToList();
            var _permutedRight = _allValues.Skip(leftY.Length).ToList();

            T _permutedDifference = _numOps.Subtract(CalculateMean(_permutedLeft), CalculateMean(_permutedRight));
            if (_numOps.GreaterThanOrEquals(_numOps.Abs(_permutedDifference), _numOps.Abs(_observedDifference)))
            {
                _countExtremeValues++;
            }
        }

        T _pValue = _numOps.Divide(_numOps.FromDouble(_countExtremeValues), _numOps.FromDouble(_permutations));

        return new PermutationTestResult<T>(_observedDifference, _pValue, _permutations, _countExtremeValues, significanceLevel);
    }

    /// <summary>
    /// Performs a Chi-Square test to determine if there is a significant association between two categorical variables.
    /// </summary>
    /// <param name="leftY">The first group of categorical values.</param>
    /// <param name="rightY">The second group of categorical values.</param>
    /// <param name="significanceLevel">The threshold p-value to determine statistical significance (default is 0.05).</param>
    /// <returns>A result object containing the test statistics and conclusion.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Chi-Square test helps determine if there's a relationship between two categorical 
    /// variables (variables that have distinct categories rather than continuous values). It compares the observed 
    /// frequencies in your data with what would be expected if there was no relationship.
    /// </para>
    /// <para>
    /// For example, if you want to know if preference for ice cream flavors differs between children and adults, 
    /// the Chi-Square test can tell you if any observed differences are statistically significant or just due to chance.
    /// </para>
    /// </remarks>
    public static ChiSquareTestResult<T> ChiSquareTest(Vector<T> leftY, Vector<T> rightY, T? significanceLevel = default)
    {
        significanceLevel = GetOrDefaultOptionalParameter(significanceLevel, _numOps.FromDouble(0.05));

        // Combine both vectors and get unique categories
        var _allValues = leftY.Concat(rightY).ToList();
        var _categories = _allValues.Distinct().OrderBy(x => x).ToList();
        int _categoryCount = _categories.Count;

        // Calculate observed frequencies
        var _leftObserved = new Vector<T>(_categoryCount);
        var _rightObserved = new Vector<T>(_categoryCount);

        for (int i = 0; i < _categoryCount; i++)
        {
            _leftObserved[i] = _numOps.FromDouble(leftY.Count(y => _numOps.Equals(y, _categories[i])));
            _rightObserved[i] = _numOps.FromDouble(rightY.Count(y => _numOps.Equals(y, _categories[i])));
        }

        // Calculate expected frequencies
        T _leftTotal = _numOps.FromDouble(leftY.Length);
        T _rightTotal = _numOps.FromDouble(rightY.Length);
        T _totalObservations = _numOps.Add(_leftTotal, _rightTotal);

        var _leftExpected = new Vector<T>(_categoryCount);
        var _rightExpected = new Vector<T>(_categoryCount);

        for (int i = 0; i < _categoryCount; i++)
        {
            T _categoryTotal = _numOps.Add(_leftObserved[i], _rightObserved[i]);
            _leftExpected[i] = _numOps.Divide(_numOps.Multiply(_categoryTotal, _leftTotal), _totalObservations);
            _rightExpected[i] = _numOps.Divide(_numOps.Multiply(_categoryTotal, _rightTotal), _totalObservations);
        }

        // Calculate chi-square statistic
        T _chiSquare = _numOps.Zero;
        for (int i = 0; i < _categoryCount; i++)
        {
            if (_numOps.GreaterThan(_leftExpected[i], _numOps.Zero))
            {
                _chiSquare = _numOps.Add(_chiSquare,
                    _numOps.Divide(_numOps.Square(_numOps.Subtract(_leftObserved[i], _leftExpected[i])), _leftExpected[i]));
            }
            if (_numOps.GreaterThan(_rightExpected[i], _numOps.Zero))
            {
                _chiSquare = _numOps.Add(_chiSquare,
                    _numOps.Divide(_numOps.Square(_numOps.Subtract(_rightObserved[i], _rightExpected[i])), _rightExpected[i]));
            }
        }

        // Calculate degrees of freedom
        int _degreesOfFreedom = _categoryCount - 1;

        // Calculate p-value using chi-square distribution (upper tail probability)
        T _pValue = _numOps.Subtract(_numOps.One, ChiSquareCDF(_chiSquare, _degreesOfFreedom));

        // Calculate critical value
        T _criticalValue = InverseChiSquareCDF(_numOps.Subtract(_numOps.FromDouble(1), significanceLevel), _degreesOfFreedom);

        // Determine if the result is significant
        bool _isSignificant = _numOps.LessThan(_pValue, significanceLevel);

        return new ChiSquareTestResult<T>
        {
            ChiSquareStatistic = _chiSquare,
            PValue = _pValue,
            DegreesOfFreedom = _degreesOfFreedom,
            LeftObserved = _leftObserved,
            RightObserved = _rightObserved,
            LeftExpected = _leftExpected,
            RightExpected = _rightExpected,
            CriticalValue = _criticalValue,
            IsSignificant = _isSignificant
        };
    }

    /// <summary>
    /// Calculates the inverse of the Chi-Square cumulative distribution function.
    /// </summary>
    /// <param name="probability">The probability value (between 0 and 1).</param>
    /// <param name="degreesOfFreedom">The degrees of freedom for the Chi-Square distribution.</param>
    /// <returns>The Chi-Square value corresponding to the given probability.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This function helps find the critical value in a Chi-Square distribution. 
    /// Given a probability (like 0.95 for a 95% confidence level) and degrees of freedom, it returns 
    /// the threshold value that would be exceeded with that probability.
    /// </para>
    /// <para>
    /// Think of it as finding the score needed to be in the top X% of test takers, where X is determined 
    /// by the probability you provide.
    /// </para>
    /// </remarks>
    private static T InverseChiSquareCDF(T probability, int degreesOfFreedom)
    {
        if (_numOps.LessThanOrEquals(probability, _numOps.Zero) || _numOps.GreaterThanOrEquals(probability, _numOps.FromDouble(1)))
        {
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1.");
        }

        if (degreesOfFreedom <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(degreesOfFreedom), "Degrees of freedom must be positive.");
        }

        // Initial guess
        T _x = _numOps.Multiply(_numOps.FromDouble(2), _numOps.FromDouble(degreesOfFreedom));
        T _delta = _numOps.FromDouble(1);
        int _maxIterations = 100;
        T _epsilon = _numOps.FromDouble(1e-8);

        for (int i = 0; i < _maxIterations && _numOps.GreaterThan(_numOps.Abs(_delta), _epsilon); i++)
        {
            T _fx = ChiSquareCDF(_x, degreesOfFreedom);
            T _dfx = ChiSquarePDF(_x, degreesOfFreedom);

            if (_numOps.Equals(_dfx, _numOps.Zero))
            {
                break;
            }

            _delta = _numOps.Divide(_numOps.Subtract(_fx, probability), _dfx);
            _x = _numOps.Subtract(_x, _delta);

            if (_numOps.LessThanOrEquals(_x, _numOps.Zero))
            {
                _x = _numOps.Divide(_x, _numOps.FromDouble(2));
            }
        }

        return _x;
    }

    /// <summary>
    /// Calculates the probability density function (PDF) of the chi-square distribution.
    /// </summary>
    /// <param name="x">The value at which to evaluate the PDF.</param>
    /// <param name="degreesOfFreedom">The degrees of freedom parameter.</param>
    /// <returns>The PDF value at x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The chi-square PDF measures the likelihood of observing a specific value 
    /// in a chi-square distribution. Think of it as the height of the curve at a particular point.
    /// The degrees of freedom parameter determines the shape of the distribution.
    /// </para>
    /// </remarks>
    private static T ChiSquarePDF(T x, int degreesOfFreedom)
    {
        T _halfDf = _numOps.Divide(_numOps.FromDouble(degreesOfFreedom), _numOps.FromDouble(2));
        T _halfX = _numOps.Divide(x, _numOps.FromDouble(2));
        T _numerator = _numOps.Multiply(
            _numOps.Exp(_numOps.Subtract(_numOps.Multiply(_halfDf, _numOps.Log(_halfX)), _halfX)),
            _numOps.Power(_halfX, _numOps.Subtract(_halfDf, _numOps.FromDouble(1)))
        );
        T _denominator = _numOps.Multiply(_numOps.Sqrt(_numOps.FromDouble(2)), GammaFunction(_halfDf));
        return _numOps.Divide(_numerator, _denominator);
    }

    /// <summary>
    /// Calculates the Gamma function for a given value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The Gamma function value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gamma function is a mathematical function that extends the factorial 
    /// operation to non-integer values. For positive integers n, Gamma(n) = (n-1)!
    /// This implementation uses the Lanczos approximation, which is a numerical method to 
    /// calculate the Gamma function with high precision.
    /// </para>
    /// </remarks>
    private static T GammaFunction(T x)
    {
        // Lanczos approximation for the Gamma function
        T[] _p = { _numOps.FromDouble(676.5203681218851),
              _numOps.FromDouble(-1259.1392167224028),
              _numOps.FromDouble(771.32342877765313),
              _numOps.FromDouble(-176.61502916214059),
              _numOps.FromDouble(12.507343278686905),
              _numOps.FromDouble(-0.13857109526572012),
              _numOps.FromDouble(9.9843695780195716e-6),
              _numOps.FromDouble(1.5056327351493116e-7) };

        if (_numOps.LessThan(x, _numOps.FromDouble(0.5)))
        {
            return _numOps.Divide(
                MathHelper.Pi<T>(),
                _numOps.Multiply(
                    MathHelper.Sin(_numOps.Multiply(MathHelper.Pi<T>(), x)),
                    GammaFunction(_numOps.Subtract(_numOps.FromDouble(1), x))
                )
            );
        }

        x = _numOps.Subtract(x, _numOps.FromDouble(1));
        T _y = _numOps.Add(x, _numOps.FromDouble(7.5));
        T _sum = _numOps.FromDouble(0.99999999999980993);

        for (int i = 0; i < _p.Length; i++)
        {
            _sum = _numOps.Add(_sum, _numOps.Divide(_p[i], _numOps.Add(x, _numOps.FromDouble(i + 1))));
        }

        T _t = _numOps.Multiply(
            _numOps.Sqrt(_numOps.FromDouble(2 * Math.PI)),
            _numOps.Multiply(
                _numOps.Power(_y, _numOps.Add(x, _numOps.FromDouble(0.5))),
                _numOps.Multiply(
                    _numOps.Exp(_numOps.Negate(_y)),
                    _sum
                )
            )
        );

        return _t;
    }

    /// <summary>
    /// Calculates the cumulative distribution function (CDF) of the chi-square distribution.
    /// </summary>
    /// <param name="x">The value at which to evaluate the CDF.</param>
    /// <param name="df">The degrees of freedom parameter.</param>
    /// <returns>The probability that a chi-square random variable is less than or equal to x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The chi-square CDF gives the probability that a random value from 
    /// the chi-square distribution will be less than or equal to x. It's like asking "what's the 
    /// chance of getting a value of x or lower?" This is useful in hypothesis testing to determine 
    /// if observed data fits an expected distribution.
    /// </para>
    /// </remarks>
    private static T ChiSquareCDF(T x, int df)
    {
        // Chi-Square CDF = P(X <= x) = regularized lower incomplete gamma P(k/2, x/2)
        return GammaRegularized(_numOps.Divide(_numOps.FromDouble(df), _numOps.FromDouble(2)), _numOps.Divide(x, _numOps.FromDouble(2)));
    }

    /// <summary>
    /// Calculates the regularized gamma function P(a,x).
    /// </summary>
    /// <param name="a">The shape parameter.</param>
    /// <param name="x">The value at which to evaluate the function.</param>
    /// <returns>The regularized gamma function value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The regularized gamma function is a mathematical function used in 
    /// probability theory and statistics. It's particularly important for calculating probabilities 
    /// in distributions like chi-square, gamma, and exponential. Think of it as a specialized 
    /// calculator that helps determine probabilities for these distributions.
    /// </para>
    /// </remarks>
    private static T GammaRegularized(T a, T x)
    {
        if (_numOps.LessThanOrEquals(x, _numOps.Zero) || _numOps.LessThanOrEquals(a, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        if (_numOps.GreaterThan(x, _numOps.Add(a, _numOps.FromDouble(1))))
        {
            return _numOps.Subtract(_numOps.FromDouble(1), GammaRegularizedContinuedFraction(a, x));
        }

        return GammaRegularizedSeries(a, x);
    }

    /// <summary>
    /// Calculates the regularized gamma function using a series expansion.
    /// </summary>
    /// <param name="a">The shape parameter.</param>
    /// <param name="x">The value at which to evaluate the function.</param>
    /// <returns>The regularized gamma function value calculated using series expansion.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method uses a mathematical technique called "series expansion" 
    /// to calculate the regularized gamma function. It's like adding up many small terms to get 
    /// a precise answer. The method keeps adding terms until the result is accurate enough.
    /// </para>
    /// </remarks>
    private static T GammaRegularizedSeries(T a, T x)
    {
        T _sum = _numOps.FromDouble(1.0 / Convert.ToDouble(a));
        T _term = _sum;
        for (int n = 1; n < 100; n++)
        {
            _term = _numOps.Multiply(_term, _numOps.Divide(x, _numOps.Add(a, _numOps.FromDouble(n))));
            _sum = _numOps.Add(_sum, _term);
            if (_numOps.LessThan(_numOps.Abs(_term), _numOps.Multiply(_numOps.Abs(_sum), _numOps.FromDouble(1e-15))))
            {
                break;
            }
        }

        return _numOps.Multiply(_numOps.Exp(_numOps.Add(_numOps.Multiply(a, _numOps.Log(x)), _numOps.Negate(x))), _sum);
    }

    /// <summary>
    /// Calculates the regularized gamma function using a continued fraction expansion.
    /// </summary>
    /// <param name="a">The shape parameter.</param>
    /// <param name="x">The value at which to evaluate the function.</param>
    /// <returns>The regularized gamma function value calculated using continued fraction.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method uses a mathematical technique called "continued fraction" 
    /// to calculate the regularized gamma function. It's an alternative approach to series expansion 
    /// that works better for certain ranges of input values. The method iteratively refines the 
    /// calculation until it reaches the desired precision.
    /// </para>
    /// </remarks>
    private static T GammaRegularizedContinuedFraction(T a, T x)
    {
        T _b = _numOps.Add(x, _numOps.Subtract(_numOps.FromDouble(1), a));
        T _c = _numOps.Divide(_numOps.FromDouble(1), _numOps.FromDouble(1e-30));
        T _d = _numOps.Divide(_numOps.FromDouble(1), _b);
        T _h = _d;
        for (int i = 1; i <= 100; i++)
        {
            T _an = _numOps.Negate(_numOps.Multiply(_numOps.FromDouble(i), _numOps.Add(_numOps.FromDouble(i), _numOps.Negate(a))));
            _b = _numOps.Add(_b, _numOps.FromDouble(2));
            _d = _numOps.Divide(_numOps.FromDouble(1), _numOps.Add(_numOps.Multiply(_an, _d), _b));
            _c = _numOps.Add(_b, _numOps.Divide(_an, _c));
            T _del = _numOps.Multiply(_c, _d);
            _h = _numOps.Multiply(_h, _del);
            if (_numOps.LessThan(_numOps.Abs(_numOps.Subtract(_del, _numOps.FromDouble(1))), _numOps.FromDouble(1e-15)))
            {
                break;
            }
        }

        return _numOps.Multiply(_numOps.Exp(_numOps.Add(_numOps.Multiply(a, _numOps.Log(x)), _numOps.Negate(x))), _h);
    }

    /// <summary>
    /// Performs an F-test to compare the variances of two samples.
    /// </summary>
    /// <param name="leftY">The first sample vector.</param>
    /// <param name="rightY">The second sample vector.</param>
    /// <param name="significanceLevel">The significance level for hypothesis testing (default is 0.05).</param>
    /// <returns>An FTestResult object containing the test statistics and results.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The F-test compares the variability (spread) between two groups of data.
    /// It helps determine if one group is significantly more variable than the other.
    /// The significance level (typically 0.05 or 5%) represents how willing you are to be wrong
    /// when rejecting the null hypothesis that both groups have equal variance.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when either input vector is null.</exception>
    /// <exception cref="ArgumentException">Thrown when either input vector has fewer than two elements.</exception>
    /// <exception cref="InvalidOperationException">Thrown when both groups have zero variance.</exception>
    public static FTestResult<T> FTest(Vector<T> leftY, Vector<T> rightY, T? significanceLevel = default)
    {
        // Use default significance level if not provided
        significanceLevel = GetOrDefaultOptionalParameter(significanceLevel, _numOps.FromDouble(0.05));

        if (leftY == null || rightY == null)
            throw new ArgumentNullException("Input vectors cannot be null.");

        if (leftY.Length < 2 || rightY.Length < 2)
            throw new ArgumentException("Both input vectors must have at least two elements.");

        T leftVariance = CalculateVariance(leftY);
        T rightVariance = CalculateVariance(rightY);

        if (_numOps.Equals(leftVariance, _numOps.Zero) && _numOps.Equals(rightVariance, _numOps.Zero))
            throw new InvalidOperationException("Both groups have zero variance. F-test cannot be performed.");

        T fStatistic = _numOps.Divide(_numOps.GreaterThan(leftVariance, rightVariance) ? leftVariance : rightVariance,
                                     _numOps.LessThan(leftVariance, rightVariance) ? leftVariance : rightVariance);

        int numeratorDf = leftY.Length - 1;
        int denominatorDf = rightY.Length - 1;

        // Calculate p-value using F-distribution
        T pValue = CalculatePValueFromFDistribution(fStatistic, numeratorDf, denominatorDf);

        // Calculate confidence intervals (95% by default)
        T confidenceLevel = _numOps.FromDouble(0.95);
        T lowerCI = CalculateFDistributionQuantile(_numOps.Divide(_numOps.Subtract(_numOps.FromDouble(1), confidenceLevel), _numOps.FromDouble(2)), numeratorDf, denominatorDf);
        T upperCI = CalculateFDistributionQuantile(_numOps.Add(_numOps.Divide(_numOps.Subtract(_numOps.FromDouble(1), confidenceLevel), _numOps.FromDouble(2)), confidenceLevel), numeratorDf, denominatorDf);

        return new FTestResult<T>(
            fStatistic,
            pValue,
            numeratorDf,
            denominatorDf,
            leftVariance,
            rightVariance,
            lowerCI,
            upperCI,
            significanceLevel
        );
    }

    /// <summary>
    /// Calculates the p-value from an F-statistic using the F-distribution.
    /// </summary>
    /// <param name="fStatistic">The F-statistic value.</param>
    /// <param name="numeratorDf">Degrees of freedom for the numerator.</param>
    /// <param name="denominatorDf">Degrees of freedom for the denominator.</param>
    /// <returns>The p-value corresponding to the F-statistic.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The p-value tells you how likely you would observe the given F-statistic
    /// (or a more extreme value) if the null hypothesis were true. A small p-value (typically &lt; 0.05)
    /// suggests that the observed difference in variances is statistically significant.
    /// </para>
    /// </remarks>
    private static T CalculatePValueFromFDistribution(T fStatistic, int numeratorDf, int denominatorDf)
    {
        // Use the regularized incomplete beta function to calculate the cumulative F-distribution
        T _x = _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(numeratorDf), fStatistic), _numOps.Add(_numOps.Multiply(_numOps.FromDouble(numeratorDf), fStatistic), _numOps.FromDouble(denominatorDf)));
        T _a = _numOps.Divide(_numOps.FromDouble(numeratorDf), _numOps.FromDouble(2));
        T _b = _numOps.Divide(_numOps.FromDouble(denominatorDf), _numOps.FromDouble(2));

        return RegularizedIncompleteBetaFunction(_x, _a, _b);
    }

    /// <summary>
    /// Calculates a quantile (inverse cumulative distribution function) of the F-distribution.
    /// </summary>
    /// <param name="probability">The probability value (between 0 and 1).</param>
    /// <param name="numeratorDf">Degrees of freedom for the numerator.</param>
    /// <param name="denominatorDf">Degrees of freedom for the denominator.</param>
    /// <returns>The F-value corresponding to the given probability.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This function finds the F-value that corresponds to a specific probability
    /// in the F-distribution. It's used to calculate critical values and confidence intervals for the F-test.
    /// Think of it as answering the question: "What F-value has exactly this much probability to its left?"
    /// </para>
    /// </remarks>
    private static T CalculateFDistributionQuantile(T probability, int numeratorDf, int denominatorDf)
    {
        T _epsilon = _numOps.FromDouble(1e-10);
        int _maxIterations = 100;

        // Initial guess using Wilson-Hilferty approximation
        T _v1 = _numOps.FromDouble(numeratorDf);
        T _v2 = _numOps.FromDouble(denominatorDf);
        T _p = _numOps.Subtract(_numOps.FromDouble(1), probability); // We need the upper tail probability
        T _x = _numOps.Power(
            _numOps.Divide(
                _numOps.Subtract(
                    _numOps.FromDouble(1),
                    _numOps.Divide(_numOps.FromDouble(2), _numOps.Multiply(_numOps.FromDouble(9), _v2))
                ),
                _numOps.Subtract(
                    _numOps.FromDouble(1),
                    _numOps.Divide(_numOps.FromDouble(2), _numOps.Multiply(_numOps.FromDouble(9), _v1))
                )
            ),
            _numOps.FromDouble(3)
        );
        _x = _numOps.Multiply(_x, _numOps.Exp(_numOps.Multiply(_numOps.FromDouble(-2.0 / 9.0), _numOps.Add(_numOps.Divide(_numOps.FromDouble(1), _v1), _numOps.Divide(_numOps.FromDouble(1), _v2)))));
        _x = _numOps.Multiply(_x, _numOps.Power(_numOps.Divide(_v2, _v1), _numOps.Divide(_numOps.FromDouble(1), _numOps.FromDouble(3))));
        _x = _numOps.Multiply(_x, _numOps.Power(_numOps.Divide(_numOps.FromDouble(1), _p), _numOps.Divide(_numOps.FromDouble(2), _numOps.FromDouble(numeratorDf))));

        // Newton-Raphson method to refine the estimate
        for (int i = 0; i < _maxIterations; i++)
        {
            T _fx = RegularizedIncompleteBetaFunction(
                _numOps.Divide(_numOps.Multiply(_v1, _x), _numOps.Add(_v1, _numOps.Multiply(_v2, _x))),
                _numOps.Divide(_v1, _numOps.FromDouble(2)),
                _numOps.Divide(_v2, _numOps.FromDouble(2))
            );
            T _dfx = BetaPDF(
                _numOps.Divide(_numOps.Multiply(_v1, _x), _numOps.Add(_v1, _numOps.Multiply(_v2, _x))),
                _numOps.Divide(_v1, _numOps.FromDouble(2)),
                _numOps.Divide(_v2, _numOps.FromDouble(2))
            );
            _dfx = _numOps.Multiply(_dfx, _numOps.Divide(_numOps.Multiply(_v1, _v2), _numOps.Square(_numOps.Add(_v1, _numOps.Multiply(_v2, _x)))));

            T _delta = _numOps.Divide(_numOps.Subtract(_fx, _p), _dfx);
            _x = _numOps.Subtract(_x, _delta);

            if (_numOps.LessThan(_numOps.Abs(_delta), _epsilon))
                break;
        }

        return _x;
    }

    /// <summary>
    /// Calculates the probability density function (PDF) of the Beta distribution.
    /// </summary>
    /// <param name="x">The value at which to evaluate the PDF (between 0 and 1).</param>
    /// <param name="a">The first shape parameter (alpha) of the Beta distribution.</param>
    /// <param name="b">The second shape parameter (beta) of the Beta distribution.</param>
    /// <returns>The PDF value at x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Beta PDF measures the likelihood of observing a specific value
    /// in a Beta distribution. Think of it as the height of the curve at a particular point.
    /// The shape parameters alpha (a) and beta (b) determine the shape of the distribution.
    /// </para>
    /// </remarks>
    private static T BetaPDF(T x, T a, T b)
    {
        return _numOps.Exp(
            _numOps.Add(
                _numOps.Add(
                    _numOps.Multiply(_numOps.Subtract(a, _numOps.FromDouble(1)), _numOps.Log(x)),
                    _numOps.Multiply(_numOps.Subtract(b, _numOps.FromDouble(1)), _numOps.Log(_numOps.Subtract(_numOps.FromDouble(1), x)))
                ),
                _numOps.Negate(LogBeta(a, b))
            )
        );
    }

    /// <summary>
    /// Calculates the logarithm of the Beta function for parameters a and b.
    /// </summary>
    /// <param name="a">First parameter of the Beta function.</param>
    /// <param name="b">Second parameter of the Beta function.</param>
    /// <returns>The logarithm of the Beta function value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Beta function is a special mathematical function used in probability 
    /// and statistics. Taking the logarithm helps with numerical stability when dealing with very 
    /// large or small numbers. This function is primarily used as a helper for other statistical calculations.
    /// </para>
    /// </remarks>
    private static T LogBeta(T a, T b)
    {
        return _numOps.Add(
            _numOps.Add(
                LogGamma(a),
                LogGamma(b)
            ),
            _numOps.Negate(LogGamma(_numOps.Add(a, b)))
        );
    }

    /// <summary>
    /// Calculates the logarithm of the Gamma function for parameter x using the Lanczos approximation.
    /// </summary>
    /// <param name="x">The input parameter for the Gamma function.</param>
    /// <returns>The logarithm of the Gamma function value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gamma function extends the factorial function to real numbers.
    /// This implementation uses the Lanczos approximation, which is a numerical method to calculate
    /// the Gamma function efficiently. Taking the logarithm helps with numerical stability.
    /// </para>
    /// </remarks>
    private static T LogGamma(T x)
    {
        // Lanczos approximation for log(Gamma(x))
        // Based on Numerical Recipes implementation with g=5
        // For x > 0, Gamma(x) ≈ sqrt(2*pi) * (x + g + 0.5)^(x + 0.5) * exp(-(x + g + 0.5)) * Ag(x)
        // where Ag(x) = c0 + c1/(x+1) + c2/(x+2) + ...

        double[] coefficients = {
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.001208650973866179,
            -0.000005395239384953
        };

        double xd = _numOps.ToDouble(x);

        // Compute the series sum: c0 + c1/(x+1) + c2/(x+2) + ...
        double sum = 1.000000000190015;
        for (int i = 0; i < 6; i++)
        {
            sum += coefficients[i] / (xd + i + 1);
        }

        // log(Gamma(x)) = (x + 0.5) * log(x + 5.5) - (x + 5.5) + log(sqrt(2*pi) * sum / x)
        // Note: The Lanczos approximation gives Gamma(x+1), so we divide by x to get Gamma(x)
        double tmp = xd + 5.5;
        double logGammaValue = (xd + 0.5) * Math.Log(tmp) - tmp + Math.Log(2.5066282746310005 * sum / xd);

        return _numOps.FromDouble(logGammaValue);
    }

    /// <summary>
    /// Calculates the regularized incomplete Beta function for parameters x, a, and b.
    /// </summary>
    /// <param name="x">The upper limit of integration, between 0 and 1.</param>
    /// <param name="a">First shape parameter, must be positive.</param>
    /// <param name="b">Second shape parameter, must be positive.</param>
    /// <returns>The value of the regularized incomplete Beta function.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The regularized incomplete Beta function is used in probability theory
    /// to calculate cumulative probabilities for many distributions. It's particularly important
    /// for calculating p-values in statistical tests. This function uses a continued fraction
    /// representation for numerical stability and accuracy.
    /// </para>
    /// </remarks>
    private static T RegularizedIncompleteBetaFunction(T x, T a, T b)
    {
        // Handle boundary cases
        if (_numOps.LessThanOrEquals(x, _numOps.Zero))
            return _numOps.Zero;
        if (_numOps.GreaterThanOrEquals(x, _numOps.One))
            return _numOps.One;

        // Use the symmetry relation for faster convergence
        // I_x(a,b) = 1 - I_{1-x}(b,a)
        // Use the relation when x > (a+1)/(a+b+2) for better continued fraction convergence
        double xVal = _numOps.ToDouble(x);
        double aVal = _numOps.ToDouble(a);
        double bVal = _numOps.ToDouble(b);
        double switchPoint = (aVal + 1.0) / (aVal + bVal + 2.0);

        if (xVal > switchPoint)
        {
            return _numOps.Subtract(_numOps.One,
                RegularizedIncompleteBetaFunction(_numOps.Subtract(_numOps.One, x), b, a));
        }

        // Use the continued fraction representation (Lentz's algorithm)
        // I_x(a,b) = (x^a * (1-x)^b / (a * B(a,b))) * CF
        return BetaIncompleteContinuedFraction(x, a, b);
    }

    /// <summary>
    /// Computes the regularized incomplete beta function using continued fraction (Lentz's algorithm).
    /// Based on DLMF 8.17.22 and Numerical Recipes implementation.
    /// </summary>
    private static T BetaIncompleteContinuedFraction(T x, T a, T b)
    {
        const int maxIterations = 200;
        const double epsilon = 1e-14;
        const double fpmin = 1e-30;

        double xd = _numOps.ToDouble(x);
        double ad = _numOps.ToDouble(a);
        double bd = _numOps.ToDouble(b);
        double ab = ad + bd;

        // Compute the front factor: x^a * (1-x)^b / (a * B(a,b))
        // = exp(a*ln(x) + b*ln(1-x) - ln(a) - lnBeta(a,b))
        double logBeta = _numOps.ToDouble(LogBeta(a, b));
        double frontFactor = Math.Exp(ad * Math.Log(xd) + bd * Math.Log(1.0 - xd) - Math.Log(ad) - logBeta);

        // Modified Lentz's algorithm for the continued fraction
        // CF = 1/(1 + d1/(1 + d2/(1 + ...)))
        // where d_{2m+1} = -(a+m)(a+b+m)x / ((a+2m)(a+2m+1))
        //       d_{2m}   = m(b-m)x / ((a+2m-1)(a+2m))

        double c = 1.0;
        double d = 1.0 - ab * xd / (ad + 1.0);
        if (Math.Abs(d) < fpmin) d = fpmin;
        d = 1.0 / d;
        double h = d;

        for (int m = 1; m <= maxIterations; m++)
        {
            int m2 = 2 * m;

            // Even term coefficient: d_{2m} = m(b-m)x / ((a+2m-1)(a+2m))
            double numEven = m * (bd - m) * xd;
            double denEven = (ad + m2 - 1.0) * (ad + m2);
            double aEven = numEven / denEven;

            // Update using even term
            d = 1.0 + aEven * d;
            if (Math.Abs(d) < fpmin) d = fpmin;
            c = 1.0 + aEven / c;
            if (Math.Abs(c) < fpmin) c = fpmin;
            d = 1.0 / d;
            h *= d * c;

            // Odd term coefficient: d_{2m+1} = -(a+m)(a+b+m)x / ((a+2m)(a+2m+1))
            double numOdd = -(ad + m) * (ab + m) * xd;
            double denOdd = (ad + m2) * (ad + m2 + 1.0);
            double aOdd = numOdd / denOdd;

            // Update using odd term
            d = 1.0 + aOdd * d;
            if (Math.Abs(d) < fpmin) d = fpmin;
            c = 1.0 + aOdd / c;
            if (Math.Abs(c) < fpmin) c = fpmin;
            d = 1.0 / d;
            double del = d * c;
            h *= del;

            // Check for convergence
            if (Math.Abs(del - 1.0) < epsilon)
            {
                break;
            }
        }

        double result = frontFactor * h;

        // Clamp to valid range
        if (result < 0.0) result = 0.0;
        if (result > 1.0) result = 1.0;

        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Calculates the cumulative distribution function (CDF) of the Beta distribution.
    /// </summary>
    /// <param name="x">The value at which to evaluate the CDF, must be between 0 and 1.</param>
    /// <param name="alpha">The first shape parameter (α), must be positive.</param>
    /// <param name="beta">The second shape parameter (β), must be positive.</param>
    /// <returns>The probability that a Beta(α, β) random variable is less than or equal to x.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are out of valid range.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Beta distribution CDF gives the probability that a random value
    /// from a Beta distribution falls below a certain point. This is essential for calculating
    /// confidence intervals for proportions, such as in Clopper-Pearson intervals.
    /// </para>
    /// </remarks>
    public static T CalculateBetaCDF(T x, T alpha, T beta)
    {
        if (_numOps.LessThanOrEquals(alpha, _numOps.Zero))
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be positive");
        if (_numOps.LessThanOrEquals(beta, _numOps.Zero))
            throw new ArgumentOutOfRangeException(nameof(beta), "Beta must be positive");
        if (_numOps.LessThan(x, _numOps.Zero) || _numOps.GreaterThan(x, _numOps.One))
            throw new ArgumentOutOfRangeException(nameof(x), "x must be between 0 and 1");

        // Handle boundary cases
        if (_numOps.Equals(x, _numOps.Zero)) return _numOps.Zero;
        if (_numOps.Equals(x, _numOps.One)) return _numOps.One;

        return RegularizedIncompleteBetaFunction(x, alpha, beta);
    }

    /// <summary>
    /// Calculates the inverse cumulative distribution function (quantile function) of the Beta distribution.
    /// </summary>
    /// <param name="probability">The probability for which to find the quantile, must be between 0 and 1.</param>
    /// <param name="alpha">The first shape parameter (α), must be positive.</param>
    /// <param name="beta">The second shape parameter (β), must be positive.</param>
    /// <returns>The value x such that P(X ≤ x) = probability for X ~ Beta(α, β).</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are out of valid range.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This function finds the value below which a certain percentage of
    /// the Beta distribution falls. For example, if probability = 0.95, it finds the value
    /// below which 95% of the distribution lies. This is used in calculating exact confidence
    /// intervals for proportions (Clopper-Pearson intervals).
    /// </para>
    /// <para>
    /// The implementation uses Newton-Raphson iteration for numerical stability and accuracy.
    /// </para>
    /// </remarks>
    public static T CalculateInverseBetaCDF(T probability, T alpha, T beta)
    {
        if (_numOps.LessThanOrEquals(alpha, _numOps.Zero))
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be positive");
        if (_numOps.LessThanOrEquals(beta, _numOps.Zero))
            throw new ArgumentOutOfRangeException(nameof(beta), "Beta must be positive");
        if (_numOps.LessThan(probability, _numOps.Zero) || _numOps.GreaterThan(probability, _numOps.One))
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        // Handle boundary cases
        if (_numOps.Equals(probability, _numOps.Zero)) return _numOps.Zero;
        if (_numOps.Equals(probability, _numOps.One)) return _numOps.One;

        // Use bisection method for robustness
        // Bisection is guaranteed to converge for monotonic functions like CDF
        double p = _numOps.ToDouble(probability);

        double lo = 0.0;
        double hi = 1.0;
        double tolerance = 1e-14;
        int maxIterations = 100;

        for (int i = 0; i < maxIterations; i++)
        {
            double mid = (lo + hi) / 2.0;
            double cdf = _numOps.ToDouble(CalculateBetaCDF(_numOps.FromDouble(mid), alpha, beta));

            if (Math.Abs(cdf - p) < tolerance)
            {
                return _numOps.FromDouble(mid);
            }

            if (cdf < p)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }

            // Check for convergence based on interval size
            if (hi - lo < tolerance)
            {
                return _numOps.FromDouble((lo + hi) / 2.0);
            }
        }

        return _numOps.FromDouble((lo + hi) / 2.0);
    }

    /// <summary>
    /// Calculates the Clopper-Pearson (exact) confidence interval for a binomial proportion.
    /// </summary>
    /// <param name="successes">The number of observed successes.</param>
    /// <param name="trials">The total number of trials.</param>
    /// <param name="confidence">The confidence level, between 0 and 1 (e.g., 0.95 for 95% confidence).</param>
    /// <returns>A tuple containing the lower and upper bounds of the confidence interval.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are out of valid range.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Clopper-Pearson interval provides exact confidence bounds for
    /// the true probability of success in a binomial experiment. Unlike the normal approximation,
    /// it guarantees the specified coverage probability, making it suitable for small samples
    /// and extreme proportions.
    /// </para>
    /// <para>
    /// This is the mathematically correct way to compute confidence intervals for proportions,
    /// especially important in certified robustness where we need guaranteed coverage.
    /// </para>
    /// </remarks>
    public static (T Lower, T Upper) CalculateClopperPearsonInterval(int successes, int trials, T confidence)
    {
        if (successes < 0)
            throw new ArgumentOutOfRangeException(nameof(successes), "Successes must be non-negative");
        if (trials <= 0)
            throw new ArgumentOutOfRangeException(nameof(trials), "Trials must be positive");
        if (successes > trials)
            throw new ArgumentOutOfRangeException(nameof(successes), "Successes cannot exceed trials");
        if (_numOps.LessThanOrEquals(confidence, _numOps.Zero) || _numOps.GreaterThanOrEquals(confidence, _numOps.One))
            throw new ArgumentOutOfRangeException(nameof(confidence), "Confidence must be between 0 and 1");

        T alpha = _numOps.Subtract(_numOps.One, confidence);
        T alphaHalf = _numOps.Divide(alpha, _numOps.FromDouble(2));

        // Lower bound: use Beta distribution with parameters (successes, trials - successes + 1)
        // When successes is 0, lower bound is 0; otherwise use inverse Beta CDF
        T lower = successes == 0
            ? _numOps.Zero
            : CalculateInverseBetaCDF(alphaHalf,
                _numOps.FromDouble(successes),
                _numOps.FromDouble(trials - successes + 1));

        // Upper bound: use Beta distribution with parameters (successes + 1, trials - successes)
        // When successes equals trials, upper bound is 1; otherwise use inverse Beta CDF
        T upper = successes == trials
            ? _numOps.One
            : CalculateInverseBetaCDF(_numOps.Subtract(_numOps.One, alphaHalf),
                _numOps.FromDouble(successes + 1),
                _numOps.FromDouble(trials - successes));

        return (lower, upper);
    }

    /// <summary>
    /// Calculates the Clopper-Pearson lower confidence bound for a binomial proportion.
    /// </summary>
    /// <param name="successes">The number of observed successes.</param>
    /// <param name="trials">The total number of trials.</param>
    /// <param name="confidence">The confidence level for a one-sided bound.</param>
    /// <returns>The lower bound of the one-sided confidence interval.</returns>
    /// <remarks>
    /// <para>
    /// This is particularly useful for randomized smoothing certification where we need
    /// a guaranteed lower bound on the top class probability.
    /// </para>
    /// </remarks>
    public static T CalculateClopperPearsonLowerBound(int successes, int trials, T confidence)
    {
        if (successes < 0)
            throw new ArgumentOutOfRangeException(nameof(successes), "Successes must be non-negative");
        if (trials <= 0)
            throw new ArgumentOutOfRangeException(nameof(trials), "Trials must be positive");
        if (successes > trials)
            throw new ArgumentOutOfRangeException(nameof(successes), "Successes cannot exceed trials");
        if (_numOps.LessThanOrEquals(confidence, _numOps.Zero) || _numOps.GreaterThanOrEquals(confidence, _numOps.One))
            throw new ArgumentOutOfRangeException(nameof(confidence), "Confidence must be between 0 and 1");

        if (successes == 0)
        {
            return _numOps.Zero;
        }

        // For one-sided bound, alpha = 1 - confidence
        T alpha = _numOps.Subtract(_numOps.One, confidence);

        // Lower bound is the alpha quantile of Beta(successes, trials - successes + 1)
        return CalculateInverseBetaCDF(alpha,
            _numOps.FromDouble(successes),
            _numOps.FromDouble(trials - successes + 1));
    }

    /// <summary>
    /// Calculates the p-value from a t-statistic and degrees of freedom using the t-distribution.
    /// </summary>
    /// <param name="tStatistic">The t-statistic value.</param>
    /// <param name="degreesOfFreedom">The degrees of freedom.</param>
    /// <returns>The two-tailed p-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The p-value tells you how likely your results occurred by random chance.
    /// A small p-value (typically = 0.05) indicates strong evidence against the null hypothesis.
    /// The t-distribution is used when sample sizes are small or when the population standard
    /// deviation is unknown. Degrees of freedom represent the number of values that are free to vary
    /// in the final calculation of a statistic.
    /// </para>
    /// </remarks>
    private static T CalculatePValueFromTDistribution(T tStatistic, int degreesOfFreedom)
    {
        T _x = _numOps.Divide(_numOps.FromDouble(degreesOfFreedom), _numOps.Add(_numOps.Square(tStatistic), _numOps.FromDouble(degreesOfFreedom)));
        T _a = _numOps.Divide(_numOps.FromDouble(degreesOfFreedom), _numOps.FromDouble(2));
        T _b = _numOps.FromDouble(0.5);

        // Calculate the cumulative distribution function (CDF) of the t-distribution
        T _cdf = RegularizedIncompleteBetaFunction(_x, _a, _b);

        // The p-value is twice the area in the tail
        return _numOps.Multiply(_numOps.FromDouble(2), _numOps.LessThan(_cdf, _numOps.FromDouble(0.5)) ? _cdf : _numOps.Subtract(_numOps.FromDouble(1), _cdf));
    }

    /// <summary>
    /// Calculates the p-value from a given z-score in a normal distribution.
    /// </summary>
    /// <param name="zScore">The z-score (standard score) to convert to a p-value.</param>
    /// <returns>The two-tailed p-value corresponding to the given z-score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A p-value tells you how likely your observed data would be if there was no real effect.
    /// Smaller p-values (typically &lt; 0.05) suggest that your observed data is unlikely to occur by chance alone.
    /// The z-score measures how many standard deviations a data point is from the mean.
    /// </para>
    /// </remarks>
    private static T CalculatePValueFromZScore(T zScore)
    {
        // Use the error function (erf) to calculate the cumulative distribution function (CDF)
        T cdf = _numOps.Divide(
            _numOps.Add(
                _numOps.FromDouble(1),
                MathHelper.Erf(_numOps.Divide(zScore, _numOps.Sqrt(_numOps.FromDouble(2))))
            ),
            _numOps.FromDouble(2)
        );

        // The p-value is twice the area in the tail
        return _numOps.Multiply(_numOps.FromDouble(2), _numOps.LessThan(cdf, _numOps.FromDouble(0.5)) ? cdf : _numOps.Subtract(_numOps.FromDouble(1), cdf));
    }

    /// <summary>
    /// Randomly reorders the elements in a list using the Fisher-Yates shuffle algorithm.
    /// </summary>
    /// <param name="list">The list to be shuffled.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Shuffling is like randomly mixing up a deck of cards. This method
    /// rearranges the elements in the list in a completely random order, which is useful for
    /// randomizing data in machine learning applications like cross-validation.
    /// </para>
    /// </remarks>
    private static void Shuffle(List<T> list)
    {
        var rng = RandomHelper.CreateSecureRandom();
        int n = list.Count;
        while (n > 1)
        {
            n--;
            int k = rng.Next(n + 1);
            T value = list[k];
            list[k] = list[n];
            list[n] = value;
        }
    }

    /// <summary>
    /// Calculates the Mean Absolute Percentage Error (MAPE) between actual and predicted values.
    /// </summary>
    /// <param name="actual">Vector of actual observed values.</param>
    /// <param name="predicted">Vector of predicted values.</param>
    /// <returns>The MAPE as a percentage.</returns>
    /// <exception cref="ArgumentException">Thrown when actual and predicted vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MAPE measures how accurate a prediction is as a percentage of the actual value.
    /// For example, if MAPE = 5%, it means predictions are off by 5% on average. Lower values indicate better predictions.
    /// The method skips pairs where the actual value is very close to zero to avoid division by zero issues.
    /// </para>
    /// </remarks>
    public static T CalculateMeanAbsolutePercentageError(Vector<T> actual, Vector<T> predicted)
    {
        if (actual.Length != predicted.Length)
            throw new ArgumentException("Actual and predicted vectors must have the same length.");

        var _epsilon = _numOps.FromDouble(1e-10); // Small constant to avoid division by zero
        var _validPairs = 0;
        var _sumAbsPercentageErrors = _numOps.Zero;

        for (int i = 0; i < actual.Length; i++)
        {
            if (_numOps.LessThanOrEquals(_numOps.Abs(actual[i]), _epsilon))
            {
                // Skip this pair if actual value is too close to zero
                continue;
            }

            var _absolutePercentageError = _numOps.Abs(_numOps.Divide(_numOps.Subtract(actual[i], predicted[i]), actual[i]));
            _sumAbsPercentageErrors = _numOps.Add(_sumAbsPercentageErrors, _absolutePercentageError);
            _validPairs++;
        }

        if (_validPairs == 0)
            return _numOps.Zero;

        return _numOps.Multiply(_numOps.Divide(_sumAbsPercentageErrors, _numOps.FromDouble(_validPairs)), _numOps.FromDouble(100));
    }

    /// <summary>
    /// Calculates the Mean Absolute Error (MAE) for a split in decision tree algorithms.
    /// </summary>
    /// <param name="y">The target values vector.</param>
    /// <param name="leftIndices">Indices of data points in the left branch.</param>
    /// <param name="rightIndices">Indices of data points in the right branch.</param>
    /// <returns>The negative weighted MAE (negative because lower MAE is better).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method helps decision trees decide where to split data.
    /// It calculates how far, on average, each data point is from the median value in its group.
    /// The result is negated because in optimization, we typically minimize a cost function,
    /// but for MAE, lower values are better (indicating less error).
    /// </para>
    /// </remarks>
    public static T CalculateMeanAbsoluteError(Vector<T> y, List<int> leftIndices, List<int> rightIndices)
    {
        T _leftMedian = StatisticsHelper<T>.CalculateMedian(leftIndices.Select(i => y[i]));
        T _rightMedian = StatisticsHelper<T>.CalculateMedian(rightIndices.Select(i => y[i]));

        T _leftMAE = StatisticsHelper<T>.CalculateMean(leftIndices.Select(i => _numOps.Abs(_numOps.Subtract(y[i], _leftMedian))));
        T _rightMAE = StatisticsHelper<T>.CalculateMean(rightIndices.Select(i => _numOps.Abs(_numOps.Subtract(y[i], _rightMedian))));

        T _leftWeight = _numOps.Divide(_numOps.FromDouble(leftIndices.Count), _numOps.FromDouble(y.Count()));
        T _rightWeight = _numOps.Divide(_numOps.FromDouble(rightIndices.Count), _numOps.FromDouble(y.Count()));

        return _numOps.Negate(_numOps.Add(_numOps.Multiply(_leftWeight, _leftMAE), _numOps.Multiply(_rightWeight, _rightMAE)));
    }

    /// <summary>
    /// Calculates the Friedman Mean Squared Error for a potential split in a decision tree.
    /// </summary>
    /// <param name="y">The target values vector.</param>
    /// <param name="leftIndices">Indices of data points in the left branch.</param>
    /// <param name="rightIndices">Indices of data points in the right branch.</param>
    /// <returns>The Friedman MSE value for the split.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method evaluates how good a split is in a decision tree by measuring
    /// the weighted squared difference between the means of the two resulting groups. A higher value
    /// indicates a better split that creates more distinct groups. This is one of several criteria
    /// used to determine where to split data in decision tree algorithms.
    /// </para>
    /// </remarks>
    public static T CalculateFriedmanMSE(Vector<T> y, List<int> leftIndices, List<int> rightIndices)
    {
        T _leftMean = StatisticsHelper<T>.CalculateMean(leftIndices.Select(i => y[i]));
        T _rightMean = StatisticsHelper<T>.CalculateMean(rightIndices.Select(i => y[i]));

        T _totalMean = StatisticsHelper<T>.CalculateMean(y);

        T _leftWeight = _numOps.Divide(_numOps.FromDouble(leftIndices.Count), _numOps.FromDouble(y.Count()));
        T _rightWeight = _numOps.Divide(_numOps.FromDouble(rightIndices.Count), _numOps.FromDouble(y.Count()));

        T _meanDifference = _numOps.Subtract(_leftMean, _rightMean);
        return _numOps.Multiply(_numOps.Multiply(_leftWeight, _rightWeight), _numOps.Square(_meanDifference));
    }

    /// <summary>
    /// Calculates the Mean Squared Error (MSE) for a potential split in a decision tree.
    /// </summary>
    /// <param name="y">The target values vector.</param>
    /// <param name="leftIndices">Indices of data points in the left branch.</param>
    /// <param name="rightIndices">Indices of data points in the right branch.</param>
    /// <returns>The weighted MSE for the split.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method measures how much the values in each group (after splitting)
    /// vary from their group's average. Lower MSE values indicate that the data points in each group
    /// are closer to their group's average, suggesting a better split. The MSE is weighted by the
    /// size of each group to account for uneven splits.
    /// </para>
    /// </remarks>
    public static T CalculateMeanSquaredError(Vector<T> y, List<int> leftIndices, List<int> rightIndices)
    {
        T _leftMean = StatisticsHelper<T>.CalculateMean(leftIndices.Select(i => y[i]));
        T _rightMean = StatisticsHelper<T>.CalculateMean(rightIndices.Select(i => y[i]));

        T _leftMSE = StatisticsHelper<T>.CalculateMeanSquaredError(leftIndices.Select(i => y[i]), _leftMean);
        T _rightMSE = StatisticsHelper<T>.CalculateMeanSquaredError(rightIndices.Select(i => y[i]), _rightMean);

        T _leftWeight = _numOps.Divide(_numOps.FromDouble(leftIndices.Count), _numOps.FromDouble(y.Count()));
        T _rightWeight = _numOps.Divide(_numOps.FromDouble(rightIndices.Count), _numOps.FromDouble(y.Count()));

        return _numOps.Add(_numOps.Multiply(_leftWeight, _leftMSE), _numOps.Multiply(_rightWeight, _rightMSE));
    }

    /// <summary>
    /// Calculates the Mean Squared Error (MSE) between a set of values and their mean.
    /// </summary>
    /// <param name="values">The collection of values to analyze.</param>
    /// <param name="mean">The mean value to compare against.</param>
    /// <returns>The mean squared error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Squared Error measures how far, on average, values are from their mean.
    /// It squares the differences before averaging them, which gives more weight to larger errors.
    /// This is useful for understanding how spread out your data is from its average value.
    /// </para>
    /// </remarks>
    public static T CalculateMeanSquaredError(IEnumerable<T> values, T mean)
    {
        return _numOps.Divide(values.Aggregate(_numOps.Zero, (acc, x) => _numOps.Add(acc, _numOps.Square(_numOps.Subtract(x, mean)))), _numOps.FromDouble(values.Count()));
    }

    /// <summary>
    /// Calculates the Root Mean Squared Error (RMSE) between actual and predicted values.
    /// </summary>
    /// <param name="actualValues">The actual observed values.</param>
    /// <param name="predictedValues">The predicted values from a model.</param>
    /// <returns>The root mean squared error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Root Mean Squared Error is a way to measure how well a model's predictions match the actual values.
    /// It's the square root of the average of squared differences between predictions and actual values.
    /// Lower RMSE values indicate better model performance. The units of RMSE match the units of your original data,
    /// making it easier to interpret than MSE.
    /// </para>
    /// </remarks>
    public static T CalculateRootMeanSquaredError(Vector<T> actualValues, Vector<T> predictedValues)
    {
        return _numOps.Sqrt(CalculateMeanSquaredError(actualValues, predictedValues));
    }

    /// <summary>
    /// Calculates the Mean Absolute Error (MAE) between actual and predicted values.
    /// </summary>
    /// <param name="actualValues">The actual observed values.</param>
    /// <param name="predictedValues">The predicted values from a model.</param>
    /// <returns>The mean absolute error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Absolute Error measures the average magnitude of errors between predicted and actual values,
    /// without considering their direction (positive or negative). Unlike RMSE, MAE gives equal weight to all errors,
    /// making it less sensitive to outliers. Lower MAE values indicate better model performance.
    /// </para>
    /// </remarks>
    public static T CalculateMeanAbsoluteError(Vector<T> actualValues, Vector<T> predictedValues)
    {
        T sumOfAbsoluteErrors = actualValues.Zip(predictedValues, (a, p) => _numOps.Abs(_numOps.Subtract(a, p))).Aggregate(_numOps.Zero, _numOps.Add);

        return _numOps.Divide(sumOfAbsoluteErrors, _numOps.FromDouble(actualValues.Length));
    }

    /// <summary>
    /// Calculates the coefficient of determination (R²) between actual and predicted values.
    /// </summary>
    /// <param name="actualValues">The actual observed values.</param>
    /// <param name="predictedValues">The predicted values from a model.</param>
    /// <returns>The R² value, ranging from 0 to 1 (or negative in case of poor fit).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> R² (R-squared) tells you how well your model explains the variation in your data.
    /// It ranges from 0 to 1, where:
    /// - 1 means your model perfectly predicts the data
    /// - 0 means your model is no better than just using the average value
    /// - Negative values can occur when the model performs worse than using the average
    /// 
    /// For example, an R² of 0.75 means your model explains 75% of the variation in the data.
    /// </para>
    /// </remarks>
    public static T CalculateR2(Vector<T> actualValues, Vector<T> predictedValues)
    {
        T actualMean = CalculateMean(actualValues);
        T totalSumSquares = actualValues.Select(a => _numOps.Square(_numOps.Subtract(a, actualMean)))
                                        .Aggregate(_numOps.Zero, _numOps.Add);
        T residualSumSquares = actualValues.Zip(predictedValues, (a, p) => _numOps.Square(_numOps.Subtract(a, p)))
                                           .Aggregate(_numOps.Zero, _numOps.Add);

        // Handle edge case where all actual values are constant (variance = 0)
        if (_numOps.Equals(totalSumSquares, _numOps.Zero))
        {
            // If predictions also match (residuals = 0), return 1.0 (perfect prediction)
            // If predictions don't match, return 0.0 (model explains nothing)
            return _numOps.Equals(residualSumSquares, _numOps.Zero) ? _numOps.One : _numOps.Zero;
        }

        return _numOps.Subtract(_numOps.One, _numOps.Divide(residualSumSquares, totalSumSquares));
    }

    /// <summary>
    /// Calculates the arithmetic mean (average) of a collection of values.
    /// </summary>
    /// <param name="values">The collection of values to average.</param>
    /// <returns>The arithmetic mean of the values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The mean is simply the average of all values in a dataset.
    /// It's calculated by adding up all values and dividing by the number of values.
    /// </para>
    /// </remarks>
    public static T CalculateMean(IEnumerable<T> values)
    {
        return _numOps.Divide(values.Aggregate(_numOps.Zero, _numOps.Add), _numOps.FromDouble(values.Count()));
    }

    /// <summary>
    /// Calculates the adjusted R² value, which accounts for the number of predictors in the model.
    /// </summary>
    /// <param name="r2">The standard R² value.</param>
    /// <param name="n">The number of observations (sample size).</param>
    /// <param name="p">The number of predictors (independent variables) in the model.</param>
    /// <returns>The adjusted R² value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adjusted R² is a modified version of R² that accounts for the number of predictors in your model.
    /// Regular R² always increases when you add more variables to your model, even if those variables don't actually improve predictions.
    /// Adjusted R² penalizes you for adding variables that don't help, making it more useful when comparing models with different numbers of variables.
    /// Like regular R², higher values indicate better model fit.
    /// </para>
    /// </remarks>
    public static T CalculateAdjustedR2(T r2, int n, int p)
    {
        T nMinusOne = _numOps.FromDouble(n - 1);
        T nMinusPMinusOne = _numOps.FromDouble(n - p - 1);
        T oneMinusR2 = _numOps.Subtract(_numOps.One, r2);

        return _numOps.Subtract(_numOps.One, _numOps.Multiply(oneMinusR2, _numOps.Divide(nMinusOne, nMinusPMinusOne)));
    }

    /// <summary>
    /// Calculates the explained variance score between actual and predicted values.
    /// </summary>
    /// <param name="actual">The actual observed values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The explained variance score, typically between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The explained variance score measures how much of the variance in the actual data is captured by your model.
    /// It's similar to R², but focuses specifically on variance.
    /// - A score of 1 means your model perfectly captures the variance in the data
    /// - A score of 0 means your model doesn't explain any of the variance
    /// - Negative scores can occur when the model is worse than just predicting the mean
    /// 
    /// This metric helps you understand how well your model accounts for the spread in your data.
    /// </para>
    /// </remarks>
    public static T CalculateExplainedVarianceScore(Vector<T> actual, Vector<T> predicted)
    {
        T varActual = actual.Variance();
        T varError = actual.Subtract(predicted).Variance();

        return _numOps.Subtract(_numOps.One, _numOps.Divide(varError, varActual));
    }

    /// <summary>
    /// Calculates the Cumulative Distribution Function (CDF) for a normal distribution.
    /// </summary>
    /// <param name="mean">The mean of the normal distribution.</param>
    /// <param name="stdDev">The standard deviation of the normal distribution.</param>
    /// <param name="x">The value at which to evaluate the CDF.</param>
    /// <returns>The probability that a random variable from the normal distribution is less than or equal to x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The normal CDF tells you the probability that a random value from a normal distribution
    /// will be less than or equal to a given value (x). For example, if the CDF equals 0.95 at x=10, 
    /// it means there's a 95% chance that a random value from this distribution will be 10 or less.
    /// 
    /// The normal distribution is the familiar "bell curve" shape, defined by its mean (center) and 
    /// standard deviation (width/spread).
    /// </para>
    /// </remarks>
    public static T CalculateNormalCDF(T mean, T stdDev, T x)
    {
        if (_numOps.LessThan(stdDev, _numOps.Zero) || _numOps.Equals(stdDev, _numOps.Zero)) return _numOps.Zero;
        T sqrt2 = _numOps.Sqrt(_numOps.FromDouble(2.0));
        T argument = _numOps.Divide(_numOps.Subtract(x, mean), _numOps.Multiply(stdDev, sqrt2));

        return _numOps.Divide(_numOps.Add(_numOps.One, Erf(argument)), _numOps.FromDouble(2.0));
    }

    /// <summary>
    /// Calculates the error function (erf) for a given value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The error function value at x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The error function (erf) is a special mathematical function that appears in probability, 
    /// statistics, and partial differential equations. It's commonly used when working with normal distributions.
    /// Think of it as a way to measure the probability that a random variable falls within a certain range.
    /// </para>
    /// </remarks>
    private static T Erf(T x)
    {
        // Abramowitz & Stegun approximation 7.1.26
        // Uses the complementary error function: erfc(x) = 1 - erf(x)
        T absX = _numOps.Abs(x);
        T t = _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Multiply(_numOps.FromDouble(0.3275911), absX)));

        // Polynomial coefficients for the approximation
        T poly = _numOps.Add(
            _numOps.FromDouble(0.254829592),
            _numOps.Multiply(t, _numOps.Add(
                _numOps.FromDouble(-0.284496736),
                _numOps.Multiply(t, _numOps.Add(
                    _numOps.FromDouble(1.421413741),
                    _numOps.Multiply(t, _numOps.Add(
                        _numOps.FromDouble(-1.453152027),
                        _numOps.Multiply(t, _numOps.FromDouble(1.061405429))
                    ))
                ))
            ))
        );

        // erfc(x) = t * poly * exp(-x²)
        T erfc = _numOps.Multiply(_numOps.Multiply(t, poly), _numOps.Exp(_numOps.Negate(_numOps.Square(absX))));

        // erf(x) = 1 - erfc(x) for x >= 0, erf(x) = erfc(-x) - 1 for x < 0
        T erfPositive = _numOps.Subtract(_numOps.One, erfc);

        return _numOps.LessThan(x, _numOps.Zero) ? _numOps.Negate(erfPositive) : erfPositive;
    }

    /// <summary>
    /// Calculates the probability density function (PDF) for a normal (Gaussian) distribution.
    /// </summary>
    /// <param name="mean">The mean (average) of the distribution.</param>
    /// <param name="stdDev">The standard deviation of the distribution.</param>
    /// <param name="x">The point at which to evaluate the PDF.</param>
    /// <returns>The probability density at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The normal PDF tells you the relative likelihood of observing a specific value in a normal distribution.
    /// It's the familiar bell curve shape. The mean determines where the peak of the curve is located, and the standard deviation
    /// determines how wide or narrow the bell curve is. Larger standard deviations create wider, flatter curves, while smaller
    /// standard deviations create narrower, taller curves.
    /// </para>
    /// </remarks>
    public static T CalculateNormalPDF(T mean, T stdDev, T x)
    {
        if (_numOps.LessThan(stdDev, _numOps.Zero) || _numOps.Equals(stdDev, _numOps.Zero)) return _numOps.Zero;

        var num = _numOps.Divide(_numOps.Subtract(x, mean), stdDev);
        var numSquared = _numOps.Multiply(num, num);
        var exponent = _numOps.Multiply(_numOps.FromDouble(-0.5), numSquared);

        var numerator = _numOps.Exp(exponent);
        var denominator = _numOps.Multiply(_numOps.Sqrt(_numOps.FromDouble(2 * Math.PI)), stdDev);

        return _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Analyzes a dataset and determines which statistical distribution best fits the data.
    /// </summary>
    /// <param name="values">The vector of data values to analyze.</param>
    /// <returns>A result object containing the best-fitting distribution type and its parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method helps you find which statistical pattern (distribution) best describes your data.
    /// It tests several common distributions (Normal, Laplace, Student's t, Log-Normal, Exponential, and Weibull) and
    /// returns the one that most closely matches your data's pattern. This is useful for understanding the underlying
    /// structure of your data and making predictions based on that structure.
    /// </para>
    /// </remarks>
    public static DistributionFitResult<T> DetermineBestFitDistribution(Vector<T> values)
    {
        // Test Normal Distribution
        var result = FitNormalDistribution(values);
        T bestGoodnessOfFit = result.GoodnessOfFit;
        result.DistributionType = DistributionType.Normal;

        // Test Laplace Distribution
        var laplaceFit = FitLaplaceDistribution(values);
        if (_numOps.LessThan(laplaceFit.GoodnessOfFit, bestGoodnessOfFit))
        {
            bestGoodnessOfFit = laplaceFit.GoodnessOfFit;
            result = laplaceFit;
            result.DistributionType = DistributionType.Laplace;
        }

        // Test Student's t-Distribution
        var studentFit = FitStudentDistribution(values);
        if (_numOps.LessThan(studentFit.GoodnessOfFit, bestGoodnessOfFit))
        {
            bestGoodnessOfFit = studentFit.GoodnessOfFit;
            result = studentFit;
            result.DistributionType = DistributionType.Student;
        }

        // Test Log-Normal Distribution
        var logNormalFit = FitLogNormalDistribution(values);
        if (_numOps.LessThan(logNormalFit.GoodnessOfFit, bestGoodnessOfFit))
        {
            bestGoodnessOfFit = logNormalFit.GoodnessOfFit;
            result = logNormalFit;
            result.DistributionType = DistributionType.LogNormal;
        }

        // Test Exponential Distribution
        var exponentialFit = FitExponentialDistribution(values);
        if (_numOps.LessThan(exponentialFit.GoodnessOfFit, bestGoodnessOfFit))
        {
            bestGoodnessOfFit = exponentialFit.GoodnessOfFit;
            result = exponentialFit;
            result.DistributionType = DistributionType.Exponential;
        }

        // Test Weibull Distribution
        var weibullFit = FitWeibullDistribution(values);
        if (_numOps.LessThan(weibullFit.GoodnessOfFit, bestGoodnessOfFit))
        {
            result = weibullFit;
            result.DistributionType = DistributionType.Weibull;
        }

        return result;
    }

    /// <summary>
    /// Fits a Laplace distribution to the provided data values.
    /// </summary>
    /// <param name="values">The vector of data values to fit.</param>
    /// <returns>A result object containing the fitted distribution parameters and goodness of fit.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Laplace distribution (also called the double exponential distribution) is similar to 
    /// the normal distribution but has sharper peaks and heavier tails. This means it's better at modeling data that 
    /// has more extreme values than a normal distribution would predict. This method finds the parameters that make 
    /// the Laplace distribution best match your data.
    /// </para>
    /// </remarks>
    private static DistributionFitResult<T> FitLaplaceDistribution(Vector<T> values)
    {
        T median = CalculateMedian(values);
        T mad = CalculateMeanAbsoluteDeviation(values, median);
        T goodnessOfFit = CalculateGoodnessOfFit(values, x => CalculateLaplacePDF(median, mad, x));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
        {
            { "Median", median },
            { "MAD", mad }
        }
        };
    }

    /// <summary>
    /// Fits a Log-Normal distribution to the provided data values.
    /// </summary>
    /// <param name="values">The vector of data values to fit.</param>
    /// <returns>A result object containing the fitted distribution parameters and goodness of fit.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Log-Normal distribution occurs when the logarithm of a variable follows a normal distribution.
    /// This type of distribution is common for data that can't be negative and is skewed to the right (has a long tail on the right side).
    /// Examples include income distributions, stock prices, and many natural phenomena. This method transforms your data using
    /// logarithms and then finds the normal distribution parameters that best fit the transformed data.
    /// </para>
    /// </remarks>
    private static DistributionFitResult<T> FitLogNormalDistribution(Vector<T> values)
    {
        var logSample = values.Transform(x => _numOps.Log(x));
        T mu = CalculateMean(logSample);
        T sigma = CalculateStandardDeviation(logSample);
        T goodnessOfFit = CalculateGoodnessOfFit(values, x => CalculateLogNormalPDF(mu, sigma, x));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
        {
            { "Mu", mu },
            { "Sigma", sigma }
        }
        };
    }

    /// <summary>
    /// Calculates the probability density function (PDF) of the chi-square distribution.
    /// </summary>
    /// <param name="degreesOfFreedom">The degrees of freedom parameter for the chi-square distribution.</param>
    /// <param name="x">The value at which to evaluate the PDF.</param>
    /// <returns>The probability density at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The chi-square PDF tells you the relative likelihood of observing a particular value
    /// in a chi-square distribution. Think of it as measuring how common a specific value is within this distribution.
    /// The degrees of freedom parameter determines the shape of the distribution - higher values create a more
    /// symmetric bell curve.
    /// </para>
    /// </remarks>
    public static T CalculateChiSquarePDF(int degreesOfFreedom, T x)
    {
        T halfDf = _numOps.Divide(_numOps.FromDouble(degreesOfFreedom), _numOps.FromDouble(2.0));
        T term1 = _numOps.Power(x, _numOps.Subtract(halfDf, _numOps.One));
        T term2 = _numOps.Exp(_numOps.Negate(_numOps.Divide(x, _numOps.FromDouble(2.0))));
        T denominator = _numOps.Multiply(_numOps.Power(_numOps.FromDouble(2.0), halfDf), Gamma(halfDf));

        return _numOps.Divide(_numOps.Multiply(term1, term2), denominator);
    }

    /// <summary>
    /// Calculates the digamma function, which is the logarithmic derivative of the gamma function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The digamma function value at x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The digamma function is a special mathematical function that appears in various
    /// statistical calculations. It's the derivative of the natural logarithm of the gamma function.
    /// In simpler terms, it helps us understand how quickly the gamma function changes at different points,
    /// which is useful for parameter estimation in certain probability distributions.
    /// </para>
    /// </remarks>
    public static T Digamma(T x)
    {
        // Approximation of the digamma function
        T result = _numOps.Zero;
        T eight = _numOps.FromDouble(8);

        for (int i = 0; i < 100 && _numOps.LessThan(x, eight) || _numOps.Equals(x, eight); i++)
        {
            result = _numOps.Subtract(result, _numOps.Divide(_numOps.One, x));
            x = _numOps.Add(x, _numOps.One);
        }

        if (_numOps.LessThan(x, eight) || _numOps.Equals(x, eight)) return result;

        T invX = _numOps.Divide(_numOps.One, x);
        T invX2 = _numOps.Square(invX);

        return _numOps.Subtract(
            _numOps.Subtract(
                _numOps.Log(x),
                _numOps.Multiply(_numOps.FromDouble(0.5), invX)
            ),
            _numOps.Multiply(
                invX2,
                _numOps.Subtract(
                    _numOps.FromDouble(1.0 / 12),
                    _numOps.Multiply(
                        invX2,
                        _numOps.Subtract(
                            _numOps.FromDouble(1.0 / 120),
                            _numOps.Multiply(invX2, _numOps.FromDouble(1.0 / 252))
                        )
                    )
                )
            )
        );
    }

    /// <summary>
    /// Calculates the gamma function for a given value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The gamma function value at x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gamma function is an extension of the factorial function to real numbers.
    /// While factorial (n!) is only defined for positive integers, the gamma function works for any positive
    /// real number. For a positive integer n, Gamma(n) = (n-1)!. This function is widely used in probability
    /// distributions and statistical calculations.
    /// </para>
    /// </remarks>
    public static T Gamma(T x)
    {
        return _numOps.Exp(LogGamma(x));
    }

    /// <summary>
    /// Estimates the shape and scale parameters of a Weibull distribution from sample data.
    /// </summary>
    /// <param name="values">The sample data.</param>
    /// <returns>A tuple containing the estimated shape and scale parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Weibull distribution is commonly used to model things like failure rates
    /// and lifetimes of components. The shape parameter determines the overall shape of the distribution,
    /// while the scale parameter stretches or compresses it. This method analyzes your data to find the
    /// Weibull distribution that best describes it, using a technique called the "method of moments"
    /// followed by refinement with the Newton-Raphson method.
    /// </para>
    /// </remarks>
    public static (T Shape, T Scale) EstimateWeibullParameters(Vector<T> values)
    {
        // Method of moments estimation for Weibull parameters
        T mean = values.Average();
        T variance = CalculateVariance(values, mean);

        // Initial guess for shape parameter
        T shape = _numOps.Sqrt(_numOps.Divide(_numOps.Multiply(_numOps.FromDouble(Math.PI), _numOps.FromDouble(Math.PI)),
                                      _numOps.Multiply(_numOps.FromDouble(6), variance)));

        // Newton-Raphson method to refine shape estimate
        for (int i = 0; i < 10; i++)
        {
            T g = Gamma(_numOps.Add(_numOps.One, _numOps.Divide(_numOps.One, shape)));
            T g2 = Gamma(_numOps.Add(_numOps.One, _numOps.Divide(_numOps.FromDouble(2), shape)));
            T f = _numOps.Subtract(_numOps.Subtract(_numOps.Divide(g2, _numOps.Multiply(g, g)), _numOps.One),
                               _numOps.Divide(variance, _numOps.Multiply(mean, mean)));
            T fPrime = _numOps.Subtract(
                _numOps.Multiply(_numOps.FromDouble(2),
                    _numOps.Divide(_numOps.Subtract(
                        _numOps.Divide(Digamma(_numOps.Add(_numOps.One, _numOps.Divide(_numOps.FromDouble(2), shape))), shape),
                        _numOps.Divide(Digamma(_numOps.Add(_numOps.One, _numOps.Divide(_numOps.One, shape))), shape)),
                    _numOps.Divide(g2, _numOps.Multiply(g, g)))),
                _numOps.Multiply(_numOps.FromDouble(2),
                    _numOps.Multiply(_numOps.Subtract(_numOps.Divide(g2, _numOps.Multiply(g, g)), _numOps.One),
                                 _numOps.Divide(Digamma(_numOps.Add(_numOps.One, _numOps.Divide(_numOps.One, shape))), shape))));

            shape = _numOps.Subtract(shape, _numOps.Divide(f, fPrime));

            if (_numOps.LessThan(_numOps.Abs(f), _numOps.FromDouble(1e-6)))
                break;
        }

        T scale = _numOps.Divide(mean, Gamma(_numOps.Add(_numOps.One, _numOps.Divide(_numOps.One, shape))));

        return (shape, scale);
    }

    /// <summary>
    /// Calculates the inverse cumulative distribution function (CDF) of the exponential distribution.
    /// </summary>
    /// <param name="lambda">The rate parameter of the exponential distribution.</param>
    /// <param name="probability">The probability value (between 0 and 1).</param>
    /// <returns>The value x such that P(X = x) = probability for an exponential random variable X.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse CDF helps you find a value in your distribution given a probability.
    /// For example, if you want to know what value represents the 90th percentile in an exponential distribution,
    /// you would use this function with probability = 0.9. The exponential distribution is often used to model
    /// the time between events, like customer arrivals or equipment failures.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when probability is not between 0 and 1.</exception>
    public static T CalculateInverseExponentialCDF(T lambda, T probability)
    {
        if (_numOps.LessThanOrEquals(probability, _numOps.Zero) || _numOps.GreaterThanOrEquals(probability, _numOps.One))
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        return _numOps.Divide(_numOps.Negate(_numOps.Log(_numOps.Subtract(_numOps.One, probability))), lambda);
    }

    /// <summary>
    /// Calculates the credible intervals for a Weibull distribution.
    /// </summary>
    /// <param name="sample">The sample data.</param>
    /// <param name="lowerProbability">The lower probability bound (e.g., 0.025 for a 95% interval).</param>
    /// <param name="upperProbability">The upper probability bound (e.g., 0.975 for a 95% interval).</param>
    /// <returns>A tuple containing the lower and upper bounds of the credible interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Credible intervals are the Bayesian equivalent of confidence intervals.
    /// They give you a range of values where you can be reasonably confident the true parameter lies.
    /// For example, a 95% credible interval means there's a 95% probability that the true value falls
    /// within that range, based on your observed data and the Weibull distribution assumption.
    /// </para>
    /// </remarks>
    public static (T LowerBound, T UpperBound) CalculateWeibullCredibleIntervals(Vector<T> sample, T lowerProbability, T upperProbability)
    {
        // Estimate Weibull parameters
        var (shape, scale) = EstimateWeibullParameters(sample);

        // Calculate credible intervals
        T lowerBound = _numOps.Multiply(scale, _numOps.Power(_numOps.Negate(_numOps.Log(_numOps.Subtract(_numOps.One, lowerProbability))),
                                                     _numOps.Divide(_numOps.One, shape)));
        T upperBound = _numOps.Multiply(scale, _numOps.Power(_numOps.Negate(_numOps.Log(_numOps.Subtract(_numOps.One, upperProbability))),
                                                     _numOps.Divide(_numOps.One, shape)));

        return (lowerBound, upperBound);
    }


    /// <summary>
    /// Calculates the inverse of the standard normal cumulative distribution function (CDF).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="probability">A value between 0 and 1 representing the probability.</param>
    /// <returns>The z-score corresponding to the given probability.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when probability is not between 0 and 1.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse normal CDF function helps you find a specific value (called a z-score) 
    /// when you know the probability. Think of it like this: if you know that 95% of values in a normal 
    /// distribution are below a certain point, this function tells you what that point is. The normal 
    /// distribution is the familiar bell-shaped curve used in statistics. This method uses a mathematical 
    /// approximation to calculate the result quickly and accurately.
    /// </para>
    /// </remarks>
    public static T CalculateInverseNormalCDF(T probability)
    {
        // Approximation of inverse normal CDF
        if (_numOps.LessThanOrEquals(probability, _numOps.Zero) || _numOps.GreaterThanOrEquals(probability, _numOps.One))
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        T t = _numOps.Sqrt(_numOps.Multiply(_numOps.FromDouble(-2), _numOps.Log(_numOps.LessThan(probability, _numOps.FromDouble(0.5)) ? probability :
            _numOps.Subtract(_numOps.One, probability))));
        T c0 = _numOps.FromDouble(2.515517);
        T c1 = _numOps.FromDouble(0.802853);
        T c2 = _numOps.FromDouble(0.010328);
        T d1 = _numOps.FromDouble(1.432788);
        T d2 = _numOps.FromDouble(0.189269);
        T d3 = _numOps.FromDouble(0.001308);

        T x = _numOps.Subtract(t, _numOps.Divide(
            _numOps.Add(c0, _numOps.Add(_numOps.Multiply(c1, t), _numOps.Multiply(c2, _numOps.Multiply(t, t)))),
            _numOps.Add(_numOps.One, _numOps.Add(_numOps.Multiply(d1, t),
                _numOps.Add(_numOps.Multiply(d2, _numOps.Multiply(t, t)),
                    _numOps.Multiply(d3, _numOps.Multiply(t, _numOps.Multiply(t, t))))))));

        return _numOps.LessThan(probability, _numOps.FromDouble(0.5)) ? _numOps.Negate(x) : x;
    }

    /// <summary>
    /// Calculates the inverse of the chi-square cumulative distribution function (CDF).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="degreesOfFreedom">The degrees of freedom parameter for the chi-square distribution.</param>
    /// <param name="probability">A value between 0 and 1 representing the probability.</param>
    /// <returns>The chi-square value corresponding to the given probability and degrees of freedom.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when probability is not between 0 and 1 or when degrees of freedom is not positive.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse chi-square function helps you find a specific chi-square value 
    /// when you know the probability and degrees of freedom. The chi-square distribution is commonly 
    /// used in statistical tests to determine if observed data matches expected data. "Degrees of freedom" 
    /// refers to the number of values that are free to vary in a calculation, which affects the shape 
    /// of the distribution. This method uses an initial approximation followed by a refinement technique 
    /// called the Newton-Raphson method to find an accurate result.
    /// </para>
    /// </remarks>
    public static T CalculateInverseChiSquareCDF(int degreesOfFreedom, T probability)
    {
        if (_numOps.LessThanOrEquals(probability, _numOps.Zero) || _numOps.GreaterThanOrEquals(probability, _numOps.One))
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");
        if (degreesOfFreedom <= 0)
            throw new ArgumentOutOfRangeException(nameof(degreesOfFreedom), "Degrees of freedom must be positive");

        // Initial guess
        T x = _numOps.Multiply(
            _numOps.FromDouble(degreesOfFreedom),
            _numOps.Power(
                _numOps.Add(
                    _numOps.Subtract(
                        _numOps.One,
                        _numOps.Divide(_numOps.FromDouble(2), _numOps.FromDouble(9 * degreesOfFreedom))
                    ),
                    _numOps.Multiply(
                        _numOps.Sqrt(_numOps.Divide(_numOps.FromDouble(2), _numOps.FromDouble(9 * degreesOfFreedom))),
                        CalculateInverseNormalCDF(probability)
                    )
                ),
                _numOps.FromDouble(3)
            )
        );

        // Newton-Raphson method for refinement
        const int maxIterations = 20;
        T epsilon = _numOps.FromDouble(1e-8);
        for (int i = 0; i < maxIterations; i++)
        {
            T fx = _numOps.Subtract(CalculateChiSquareCDF(degreesOfFreedom, x), probability);
            T dfx = CalculateChiSquarePDF(degreesOfFreedom, x);

            T delta = _numOps.Divide(fx, dfx);
            x = _numOps.Subtract(x, delta);

            if (_numOps.LessThan(_numOps.Abs(delta), epsilon))
                break;
        }

        return x;
    }

    /// <summary>
    /// Calculates the chi-square cumulative distribution function (CDF) value for a given chi-square value and degrees of freedom.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="degreesOfFreedom">The degrees of freedom parameter for the chi-square distribution.</param>
    /// <param name="x">The chi-square value to evaluate.</param>
    /// <returns>The probability that a chi-square random variable with the specified degrees of freedom is less than or equal to x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The chi-square CDF function calculates the probability that a chi-square 
    /// random variable is less than or equal to a specific value. This is useful in hypothesis testing 
    /// to determine if your observed results could have happened by chance. This method uses the incomplete 
    /// gamma function to calculate the result, which is a standard approach for computing chi-square probabilities.
    /// </para>
    /// </remarks>
    public static T CalculateChiSquareCDF(int degreesOfFreedom, T x)
    {
        return IncompleteGamma(_numOps.Divide(_numOps.FromDouble(degreesOfFreedom), _numOps.FromDouble(2)), _numOps.Divide(x, _numOps.FromDouble(2)));
    }

    /// <summary>
    /// Calculates the incomplete gamma function, which is used in various statistical distributions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="a">The shape parameter.</param>
    /// <param name="x">The value at which to evaluate the incomplete gamma function.</param>
    /// <returns>The value of the incomplete gamma function at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The incomplete gamma function is a mathematical function used in many 
    /// statistical calculations. It's a building block for other statistical functions like the 
    /// chi-square distribution. This implementation uses a numerical approximation to calculate its 
    /// value by summing a series of terms until the desired precision is reached. Don't worry too 
    /// much about the details - this is an advanced mathematical function that works behind the scenes 
    /// to make other statistical calculations possible.
    /// </para>
    /// </remarks>
    public static T IncompleteGamma(T a, T x)
    {
        const int maxIterations = 100;
        T epsilon = _numOps.FromDouble(1e-8);

        T sum = _numOps.Zero;
        T term = _numOps.Divide(_numOps.One, a);
        for (int i = 0; i < maxIterations; i++)
        {
            sum = _numOps.Add(sum, term);
            term = _numOps.Multiply(term, _numOps.Divide(x, _numOps.Add(a, _numOps.FromDouble(i + 1))));
            if (_numOps.LessThan(term, epsilon))
                break;
        }

        return _numOps.Multiply(
            sum,
            _numOps.Exp(_numOps.Subtract(
                _numOps.Subtract(_numOps.Multiply(a, _numOps.Log(x)), x),
                LogGamma(a)
            ))
        );
    }

    /// <summary>
    /// Calculates the inverse of the normal cumulative distribution function (CDF) with specified mean and standard deviation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="mean">The mean (average) of the normal distribution.</param>
    /// <param name="stdDev">The standard deviation of the normal distribution.</param>
    /// <param name="probability">A value between 0 and 1 representing the probability.</param>
    /// <returns>The value from the normal distribution with the given mean and standard deviation corresponding to the specified probability.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This version of the inverse normal CDF function allows you to specify 
    /// the mean (average) and standard deviation (a measure of how spread out the values are) of your 
    /// normal distribution. It converts the result from the standard normal distribution (which has 
    /// mean 0 and standard deviation 1) to your specific distribution. For example, if you want to 
    /// find the value that 90% of the data falls below in a normal distribution with mean 100 and 
    /// standard deviation 15, you would use this function with those parameters and a probability of 0.9.
    /// </para>
    /// </remarks>
    public static T CalculateInverseNormalCDF(T mean, T stdDev, T probability)
    {
        return _numOps.Add(mean, _numOps.Multiply(stdDev, CalculateInverseNormalCDF(probability)));
    }

    /// <summary>
    /// Calculates the inverse of the Student's t cumulative distribution function (CDF).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="degreesOfFreedom">The degrees of freedom parameter for the t-distribution.</param>
    /// <param name="probability">A value between 0 and 1 representing the probability.</param>
    /// <returns>The t-value corresponding to the given probability and degrees of freedom.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when probability is not between 0 and 1.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse Student's t-distribution function helps you find a specific 
    /// t-value when you know the probability and degrees of freedom. The Student's t-distribution is 
    /// similar to the normal distribution but has heavier tails, making it useful when working with 
    /// small sample sizes. It's commonly used in hypothesis testing when the population standard 
    /// deviation is unknown. This method uses a series of approximations to calculate the result, with 
    /// special handling for low degrees of freedom where the approximation needs to be more careful.
    /// </para>
    /// </remarks>
    public static T CalculateInverseStudentTCDF(int degreesOfFreedom, T probability)
    {
        if (_numOps.LessThanOrEquals(probability, _numOps.Zero) || _numOps.GreaterThanOrEquals(probability, _numOps.One))
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        T x = CalculateInverseNormalCDF(probability);
        T y = _numOps.Square(x);

        T a = _numOps.Divide(_numOps.Add(y, _numOps.One), _numOps.FromDouble(4));
        T b = _numOps.Divide(_numOps.Add(_numOps.Add(_numOps.Multiply(_numOps.FromDouble(5), y), _numOps.FromDouble(16)), _numOps.Multiply(y, _numOps.FromDouble(3))), _numOps.FromDouble(96));
        T c = _numOps.Divide(_numOps.Subtract(_numOps.Add(_numOps.Multiply(_numOps.Add(_numOps.Multiply(_numOps.FromDouble(3), y), _numOps.FromDouble(19)), y), _numOps.FromDouble(17)), _numOps.FromDouble(15)), _numOps.FromDouble(384));
        T d = _numOps.Divide(_numOps.Subtract(_numOps.Subtract(_numOps.Add(_numOps.Multiply(_numOps.Add(_numOps.Multiply(_numOps.FromDouble(79), y), _numOps.FromDouble(776)), y), _numOps.FromDouble(1482)), _numOps.Multiply(y, _numOps.FromDouble(1920))), _numOps.FromDouble(945)), _numOps.FromDouble(92160));

        T t = _numOps.Multiply(x, _numOps.Add(_numOps.One, _numOps.Divide(_numOps.Add(a, _numOps.Divide(_numOps.Add(b, _numOps.Divide(_numOps.Add(c, _numOps.Divide(d, _numOps.FromDouble(degreesOfFreedom))), _numOps.FromDouble(degreesOfFreedom))), _numOps.FromDouble(degreesOfFreedom))), _numOps.FromDouble(degreesOfFreedom))));

        if (degreesOfFreedom <= 2)
        {
            // Additional refinement for low degrees of freedom
            T sign = _numOps.GreaterThan(x, _numOps.Zero) ? _numOps.One : _numOps.Negate(_numOps.One);
            T innerTerm = _numOps.Subtract(_numOps.Power(probability, _numOps.Divide(_numOps.FromDouble(-2.0), _numOps.FromDouble(degreesOfFreedom))), _numOps.One);
            t = _numOps.Multiply(sign, _numOps.Sqrt(_numOps.Multiply(_numOps.FromDouble(degreesOfFreedom), innerTerm)));
        }

        return t;
    }

    /// <summary>
    /// Calculates the inverse of the Laplace cumulative distribution function (CDF).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="median">The median (location parameter) of the Laplace distribution.</param>
    /// <param name="mad">The mean absolute deviation (scale parameter) of the Laplace distribution.</param>
    /// <param name="probability">A value between 0 and 1 representing the probability.</param>
    /// <returns>The value from the Laplace distribution corresponding to the given probability.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse Laplace CDF function helps you find a specific value when you 
    /// know the probability for a Laplace distribution. The Laplace distribution (also called the double 
    /// exponential distribution) has a peak at the median and falls off exponentially on both sides. 
    /// It's often used to model data that has heavier tails than a normal distribution. The median 
    /// parameter tells you where the center of the distribution is, while the mean absolute deviation (mad) 
    /// tells you how spread out the values are.
    /// </para>
    /// </remarks>
    public static T CalculateInverseLaplaceCDF(T median, T mad, T probability)
    {
        T half = _numOps.FromDouble(0.5);
        T sign = _numOps.GreaterThan(probability, half) ? _numOps.One : _numOps.Negate(_numOps.One);
        return _numOps.Subtract(median, _numOps.Multiply(_numOps.Multiply(mad, sign), _numOps.Log(_numOps.Subtract(_numOps.One,
            _numOps.Multiply(_numOps.FromDouble(2), _numOps.Abs(_numOps.Subtract(probability, half)))))));
    }

    /// <summary>
    /// Calculates the credible intervals for a given set of values based on the specified distribution type and confidence level.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The sample data.</param>
    /// <param name="confidenceLevel">The confidence level (e.g., 0.95 for 95% confidence).</param>
    /// <param name="distributionType">The type of distribution to use for the calculation.</param>
    /// <returns>A tuple containing the lower and upper bounds of the credible interval.</returns>
    /// <exception cref="ArgumentException">Thrown when an invalid distribution type is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Credible intervals are the Bayesian equivalent of confidence intervals. 
    /// They give you a range of values that likely contains the true parameter value with a certain 
    /// probability (the confidence level). For example, a 95% credible interval means there's a 95% 
    /// probability that the true value falls within that range. This method calculates these intervals 
    /// for different types of distributions (Normal, Laplace, Student's t, LogNormal, Exponential, or Weibull) 
    /// based on your data. The method automatically calculates the necessary parameters from your data 
    /// and then uses the appropriate inverse CDF function to find the interval bounds.
    /// </para>
    /// </remarks>
    public static (T LowerBound, T UpperBound) CalculateCredibleIntervals(Vector<T> values, T confidenceLevel, DistributionType distributionType)
    {
        T lowerProbability = _numOps.Divide(_numOps.Subtract(_numOps.One, confidenceLevel), _numOps.FromDouble(2)); // 0.025 for 95% CI
        T upperProbability = _numOps.Subtract(_numOps.One, lowerProbability); // 0.975 for 95% CI
        (var mean, var stdDev) = CalculateMeanAndStandardDeviation(values);
        var median = CalculateMedian(values);
        var mad = CalculateMeanAbsoluteDeviation(values, median);

        return distributionType switch
        {
            DistributionType.Normal => (
                CalculateInverseNormalCDF(mean, stdDev, lowerProbability),
                CalculateInverseNormalCDF(mean, stdDev, upperProbability)
            ),
            DistributionType.Laplace => (
                CalculateInverseLaplaceCDF(median, mad, lowerProbability),
                CalculateInverseLaplaceCDF(median, mad, upperProbability)
            ),
            DistributionType.Student => (
                CalculateInverseStudentTCDF(values.Length - 1, lowerProbability),
                CalculateInverseStudentTCDF(values.Length - 1, upperProbability)
            ),
            DistributionType.LogNormal => (
                _numOps.Exp(CalculateInverseNormalCDF(
                    _numOps.Subtract(_numOps.Log(mean), _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Log(_numOps.Add(_numOps.One, _numOps.Divide(_numOps.Square(stdDev), _numOps.Square(mean)))))),
                    _numOps.Sqrt(_numOps.Log(_numOps.Add(_numOps.One, _numOps.Divide(_numOps.Square(stdDev), _numOps.Square(mean))))),
                    lowerProbability
                )),
                _numOps.Exp(CalculateInverseNormalCDF(
                    _numOps.Subtract(_numOps.Log(mean), _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Log(_numOps.Add(_numOps.One, _numOps.Divide(_numOps.Square(stdDev), _numOps.Square(mean)))))),
                    _numOps.Sqrt(_numOps.Log(_numOps.Add(_numOps.One, _numOps.Divide(_numOps.Square(stdDev), _numOps.Square(mean))))),
                    upperProbability
                ))
            ),
            DistributionType.Exponential => (
                CalculateInverseExponentialCDF(_numOps.Divide(_numOps.One, mean), lowerProbability),
                CalculateInverseExponentialCDF(_numOps.Divide(_numOps.One, mean), upperProbability)
            ),
            DistributionType.Weibull => CalculateWeibullCredibleIntervals(values, lowerProbability, upperProbability),
            _ => throw new ArgumentException("Invalid distribution type"),
        };
    }

    /// <summary>
    /// Calculates confidence intervals for Weibull distribution parameters using bootstrap resampling.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The sample data.</param>
    /// <param name="confidenceLevel">The confidence level (e.g., 0.95 for 95% confidence).</param>
    /// <returns>A tuple containing the lower and upper bounds of the confidence interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method estimates confidence intervals for Weibull distribution parameters 
    /// using a technique called bootstrap resampling. Bootstrapping works by creating many new samples 
    /// by randomly selecting values from your original data (with replacement), then calculating the 
    /// parameters for each of these samples. This gives you a distribution of possible parameter values, 
    /// from which you can determine confidence intervals. The Weibull distribution is commonly used to 
    /// model things like failure rates and lifetimes of components.
    /// </para>
    /// </remarks>
    public static (T LowerBound, T UpperBound) CalculateWeibullConfidenceIntervals(Vector<T> values, T confidenceLevel)
    {
        const int bootstrapSamples = 1000;
        var rng = RandomHelper.CreateSecureRandom();
        var estimates = new List<(T Shape, T Scale)>();

        for (int i = 0; i < bootstrapSamples; i++)
        {
            var bootstrapSample = new Vector<T>(values.Length);
            for (int j = 0; j < values.Length; j++)
            {
                bootstrapSample[j] = values[rng.Next(values.Length)];
            }

            estimates.Add(EstimateWeibullParameters(bootstrapSample));
        }

        var sortedShapes = estimates.Select(e => e.Shape).OrderBy(s => s).ToList();
        var sortedScales = estimates.Select(e => e.Scale).OrderBy(s => s).ToList();

        // For a confidence interval, we need the alpha/2 and 1 - alpha/2 percentiles
        // where alpha = 1 - confidenceLevel
        // e.g., for 95% confidence: alpha = 0.05, so we want 2.5th and 97.5th percentiles
        T alpha = _numOps.Subtract(_numOps.One, confidenceLevel);
        T halfAlpha = _numOps.Divide(alpha, _numOps.FromDouble(2));
        T oneMinusHalfAlpha = _numOps.Subtract(_numOps.One, halfAlpha);

        T lowerIndexT = _numOps.Multiply(_numOps.FromDouble(bootstrapSamples - 1), halfAlpha);
        int lowerIndex = Math.Max(0, Convert.ToInt32(_numOps.Floor(lowerIndexT)));

        T upperIndexT = _numOps.Multiply(_numOps.FromDouble(bootstrapSamples - 1), oneMinusHalfAlpha);
        int upperIndex = Math.Min(bootstrapSamples - 1, Convert.ToInt32(_numOps.Ceiling(upperIndexT)));

        return (_numOps.Multiply(sortedShapes[lowerIndex], sortedScales[lowerIndex]),
                _numOps.Multiply(sortedShapes[upperIndex], sortedScales[upperIndex]));
    }

    /// <summary>
    /// Calculates the probability density function (PDF) value for an exponential distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="lambda">The rate parameter of the exponential distribution.</param>
    /// <param name="x">The value at which to evaluate the PDF.</param>
    /// <returns>The PDF value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The exponential PDF function calculates the height of the probability 
    /// curve at a specific point for an exponential distribution. The exponential distribution is 
    /// commonly used to model the time between events in a process where events occur continuously 
    /// and independently at a constant average rate. The lambda parameter represents this rate - 
    /// higher values of lambda mean events happen more frequently on average. The PDF is zero for 
    /// negative values of x, reflecting that you can't have negative time between events.
    /// </para>
    /// </remarks>
    public static T CalculateExponentialPDF(T lambda, T x)
    {
        if (_numOps.LessThanOrEquals(lambda, _numOps.Zero) || _numOps.LessThan(x, _numOps.Zero)) return _numOps.Zero;
        return _numOps.Multiply(lambda, _numOps.Exp(_numOps.Negate(_numOps.Multiply(lambda, x))));
    }

    /// <summary>
    /// Calculates the probability density function (PDF) value for a Weibull distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="k">The shape parameter of the Weibull distribution.</param>
    /// <param name="lambda">The scale parameter of the Weibull distribution.</param>
    /// <param name="x">The value at which to evaluate the PDF.</param>
    /// <returns>The PDF value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Weibull PDF function calculates the height of the probability curve 
    /// at a specific point for a Weibull distribution. The Weibull distribution is versatile and can 
    /// model many different shapes depending on its parameters. It's commonly used in reliability 
    /// engineering to model failure rates. The shape parameter (k) determines the overall shape of 
    /// the distribution - values less than 1 give a decreasing failure rate, equal to 1 gives a constant 
    /// failure rate (equivalent to an exponential distribution), and greater than 1 gives an increasing 
    /// failure rate. The scale parameter (lambda) stretches or compresses the distribution.
    /// </para>
    /// </remarks>
    public static T CalculateWeibullPDF(T k, T lambda, T x)
    {
        if (_numOps.LessThanOrEquals(k, _numOps.Zero) || _numOps.LessThanOrEquals(lambda, _numOps.Zero) || _numOps.LessThan(x, _numOps.Zero)) return _numOps.Zero;
        return _numOps.Multiply(
            _numOps.Divide(k, lambda),
            _numOps.Multiply(
                _numOps.Power(_numOps.Divide(x, lambda), _numOps.Subtract(k, _numOps.One)),
                _numOps.Exp(_numOps.Negate(_numOps.Power(_numOps.Divide(x, lambda), k)))
            )
        );
    }

    /// <summary>
    /// Calculates the probability density function (PDF) value for a log-normal distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="mu">The mean of the natural logarithm of the distribution.</param>
    /// <param name="sigma">The standard deviation of the natural logarithm of the distribution.</param>
    /// <param name="x">The value at which to evaluate the PDF.</param>
    /// <returns>The PDF value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The log-normal PDF function calculates the height of the probability curve 
    /// at a specific point for a log-normal distribution. A log-normal distribution occurs when the 
    /// logarithm of a variable follows a normal distribution. This distribution is useful for modeling 
    /// quantities that can't be negative and are positively skewed, such as income, house prices, or 
    /// certain biological measurements. The parameters mu and sigma are the mean and standard deviation 
    /// of the variable's natural logarithm, not of the variable itself.
    /// </para>
    /// </remarks>
    public static T CalculateLogNormalPDF(T mu, T sigma, T x)
    {
        if (_numOps.LessThanOrEquals(x, _numOps.Zero) || _numOps.LessThanOrEquals(sigma, _numOps.Zero)) return _numOps.Zero;
        var logX = _numOps.Log(x);
        var twoSigmaSquared = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Square(sigma));
        return _numOps.Divide(
            _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Square(_numOps.Subtract(logX, mu)), twoSigmaSquared))),
            _numOps.Multiply(x, _numOps.Multiply(sigma, _numOps.Sqrt(_numOps.FromDouble(2 * Math.PI))))
        );
    }

    /// <summary>
    /// Calculates the probability density function (PDF) value for a Laplace distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="median">The median (location parameter) of the Laplace distribution.</param>
    /// <param name="mad">The mean absolute deviation (scale parameter) of the Laplace distribution.</param>
    /// <param name="x">The value at which to evaluate the PDF.</param>
    /// <returns>The PDF value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Laplace PDF function calculates the height of the probability curve 
    /// at a specific point for a Laplace distribution. The Laplace distribution (also called the double 
    /// exponential distribution) has a peak at the median and falls off exponentially on both sides. 
    /// Unlike the normal distribution which has a bell shape, the Laplace distribution has a sharper peak 
    /// and heavier tails. The median parameter tells you where the center of the distribution is, while 
    /// the mean absolute deviation (mad) tells you how spread out the values are.
    /// </para>
    /// </remarks>
    public static T CalculateLaplacePDF(T median, T mad, T x)
    {
        return _numOps.Divide(
            _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Abs(_numOps.Subtract(x, median)), mad))),
            _numOps.Multiply(_numOps.FromDouble(2), mad)
        );
    }

    /// <summary>
    /// Calculates the goodness of fit for a probability distribution against sample data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The sample data.</param>
    /// <param name="pdfFunction">A function that calculates the PDF value for a given data point.</param>
    /// <returns>The negative log-likelihood, which measures how well the distribution fits the data (smaller values indicate better fit).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method measures how well a particular probability distribution fits your 
    /// data. It works by calculating the probability density function (PDF) value for each data point, 
    /// taking the logarithm of these values, summing them up, and then negating the result. This gives 
    /// what's called the "negative log-likelihood" - a common measure in statistics. The smaller this 
    /// value, the better the distribution fits your data. This is useful when you want to determine which 
    /// type of distribution (normal, exponential, Weibull, etc.) best describes your data.
    /// </para>
    /// </remarks>
    public static T CalculateGoodnessOfFit(Vector<T> values, Func<T, T> pdfFunction)
    {
        T logLikelihood = _numOps.Zero;
        foreach (var value in values)
        {
            logLikelihood = _numOps.Add(logLikelihood, _numOps.Log(pdfFunction(value)));
        }

        return _numOps.Negate(logLikelihood);
    }

    /// <summary>
    /// Fits a normal distribution to the provided sample data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The sample data.</param>
    /// <returns>A result object containing the fitted parameters and goodness of fit measure.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your data to find the normal distribution (bell curve) 
    /// that best describes it. It calculates the mean (average) and standard deviation (a measure of spread) 
    /// from your data, then uses these parameters to define a normal distribution. It also calculates how 
    /// well this distribution fits your data using the goodness of fit measure. The result includes both 
    /// the parameters (mean and standard deviation) and the goodness of fit value, which you can use to 
    /// compare with other distribution types.
    /// </para>
    /// </remarks>
    private static DistributionFitResult<T> FitNormalDistribution(Vector<T> values)
    {
        (var mean, var stdDev) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(values);
        T goodnessOfFit = CalculateGoodnessOfFit(values, x => CalculateNormalPDF(mean, stdDev, x));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
        {
            { "Mean", mean },
            { "StandardDeviation", stdDev }
        }
        };
    }

    /// <summary>
    /// Calculates the mean (average) and standard deviation of a set of values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The sample data.</param>
    /// <returns>A tuple containing the mean and standard deviation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates two fundamental statistical measures from your data. 
    /// The mean is simply the average of all values - add them up and divide by how many there are. The 
    /// standard deviation measures how spread out the values are from the mean. A small standard deviation 
    /// means the values tend to be close to the mean, while a large standard deviation means they're more 
    /// spread out. This method uses a computationally efficient approach that only requires a single pass 
    /// through the data.
    /// </para>
    /// </remarks>
    public static (T Mean, T StandardDeviation) CalculateMeanAndStandardDeviation(Vector<T> values)
    {
        if (values.Length == 0)
        {
            return (_numOps.Zero, _numOps.Zero);
        }

        T sum = _numOps.Zero;
        T sumOfSquares = _numOps.Zero;
        int count = values.Length;

        for (int i = 0; i < count; i++)
        {
            sum = _numOps.Add(sum, values[i]);
            sumOfSquares = _numOps.Add(sumOfSquares, _numOps.Square(values[i]));
        }

        T mean = _numOps.Divide(sum, _numOps.FromDouble(count));
        T variance = _numOps.Subtract(
            _numOps.Divide(sumOfSquares, _numOps.FromDouble(count)),
            _numOps.Square(mean)
        );

        T standardDeviation = _numOps.Sqrt(variance);

        return (mean, standardDeviation);
    }

    /// <summary>
    /// Calculates the probability density function (PDF) value for a Student's t-distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The value at which to evaluate the PDF.</param>
    /// <param name="mean">The mean (location parameter) of the distribution.</param>
    /// <param name="stdDev">The standard deviation (scale parameter) of the distribution.</param>
    /// <param name="df">The degrees of freedom parameter.</param>
    /// <returns>The PDF value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Student's t-distribution PDF function calculates the height of the 
    /// probability curve at a specific point. The Student's t-distribution looks similar to the normal 
    /// distribution but has heavier tails, making it more appropriate when working with small sample sizes. 
    /// The mean parameter specifies the center of the distribution, the standard deviation controls how 
    /// spread out it is, and the degrees of freedom parameter affects the shape - smaller values give 
    /// heavier tails. This distribution is commonly used in hypothesis testing when the population standard 
    /// deviation is unknown.
    /// </para>
    /// </remarks>
    public static T CalculateStudentPDF(T x, T mean, T stdDev, int df)
    {
        T t = _numOps.Divide(_numOps.Subtract(x, mean), stdDev);

        T numerator = Gamma(_numOps.Divide(_numOps.Add(_numOps.FromDouble(df), _numOps.One), _numOps.FromDouble(2.0)));
        T denominator = _numOps.Multiply(
            _numOps.Sqrt(_numOps.Multiply(_numOps.FromDouble(df), _numOps.FromDouble(Math.PI))),
            _numOps.Multiply(
                Gamma(_numOps.Divide(_numOps.FromDouble(df), _numOps.FromDouble(2.0))),
                _numOps.Power(
                    _numOps.Add(_numOps.One, _numOps.Divide(_numOps.Multiply(t, t), _numOps.FromDouble(df))),
                    _numOps.Divide(_numOps.Add(_numOps.FromDouble(df), _numOps.One), _numOps.FromDouble(2.0))
                )
            )
        );

        return _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Fits a Weibull distribution to the provided sample data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The sample data.</param>
    /// <returns>A result object containing the fitted parameters and goodness of fit measure.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your data to find the Weibull distribution that best 
    /// describes it. The Weibull distribution is commonly used to model things like failure rates and 
    /// lifetimes of components. It has two parameters: k (shape) and lambda (scale). This method uses 
    /// an iterative approach called the Newton-Raphson method to find the optimal values for these 
    /// parameters. It starts with initial guesses and refines them until they converge to the best fit. 
    /// The result includes both the parameters and a goodness of fit value, which you can use to compare 
    /// with other distribution types.
    /// </para>
    /// </remarks>
    private static DistributionFitResult<T> FitWeibullDistribution(Vector<T> values)
    {
        // Initial guess for k and lambda
        T k = _numOps.One;
        T lambda = StatisticsHelper<T>.CalculateMean(values);

        const int maxIterations = 100;
        T epsilon = _numOps.FromDouble(1e-6);

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            T sumXk = _numOps.Zero;
            T sumXkLnX = _numOps.Zero;
            T sumLnX = _numOps.Zero;

            foreach (var x in values)
            {
                T xk = _numOps.Power(x, k);
                sumXk = _numOps.Add(sumXk, xk);
                sumXkLnX = _numOps.Add(sumXkLnX, _numOps.Multiply(xk, _numOps.Log(x)));
                sumLnX = _numOps.Add(sumLnX, _numOps.Log(x));
            }

            T n = _numOps.FromDouble(values.Length);
            T kInverse = _numOps.Divide(_numOps.One, k);

            // Update lambda
            lambda = _numOps.Power(_numOps.Divide(sumXk, n), kInverse);

            // Calculate the gradient and Hessian for k
            T gradient = _numOps.Add(
                _numOps.Divide(_numOps.One, k),
                _numOps.Divide(sumLnX, n)
            );
            gradient = _numOps.Subtract(gradient, _numOps.Divide(sumXkLnX, sumXk));

            T hessian = _numOps.Negate(_numOps.Divide(_numOps.One, _numOps.Square(k)));

            // Update k using Newton-Raphson method
            T kNew = _numOps.Subtract(k, _numOps.Divide(gradient, hessian));

            // Check for convergence
            if (_numOps.LessThan(_numOps.Abs(_numOps.Subtract(kNew, k)), epsilon))
            {
                k = kNew;
                break;
            }

            k = kNew;
        }

        T goodnessOfFit = CalculateGoodnessOfFit(values, x => CalculateWeibullPDF(k, lambda, x));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
        {
            { "K", k },
            { "Lambda", lambda }
        }
        };
    }

    /// <summary>
    /// Fits an exponential distribution to the provided sample data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="sample">The sample data.</param>
    /// <returns>A result object containing the fitted parameter and goodness of fit measure.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your data to find the exponential distribution that 
    /// best describes it. The exponential distribution is commonly used to model the time between events 
    /// in a process where events occur continuously and independently at a constant average rate. It has 
    /// one parameter, lambda, which is the rate parameter. This method calculates lambda as the reciprocal 
    /// (1 divided by) of the mean of your data. It also calculates how well this distribution fits your 
    /// data using the goodness of fit measure. The result includes both the parameter (lambda) and the 
    /// goodness of fit value, which you can use to compare with other distribution types.
    /// </para>
    /// </remarks>
    private static DistributionFitResult<T> FitExponentialDistribution(Vector<T> sample)
    {
        T lambda = _numOps.Divide(_numOps.One, StatisticsHelper<T>.CalculateMean(sample));

        T goodnessOfFit = CalculateGoodnessOfFit(sample, x => CalculateExponentialPDF(lambda, x));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
        {
            { "Lambda", lambda }
        }
        };
    }

    /// <summary>
    /// Fits a Student's t-distribution to the provided sample data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The sample data.</param>
    /// <returns>A result object containing the fitted parameters and goodness of fit measure.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your data to find the Student's t-distribution that 
    /// best describes it. The Student's t-distribution is similar to the normal distribution but has 
    /// heavier tails, making it useful when working with small sample sizes. This method calculates 
    /// the degrees of freedom (which affects the shape of the distribution), the mean (center), and 
    /// the standard deviation (spread) from your data. It then evaluates how well this distribution 
    /// fits your data using a goodness of fit measure. The result includes both the parameters and 
    /// the goodness of fit value, which you can use to compare with other distribution types.
    /// </para>
    /// </remarks>
    private static DistributionFitResult<T> FitStudentDistribution(Vector<T> values)
    {
        var df = values.Length - 1;
        (var mean, var stdDev) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(values);
        T goodnessOfFit = CalculateGoodnessOfFit(values, x => CalculateStudentPDF(x, mean, stdDev, df));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
        {
            { "DegreesOfFreedom", _numOps.FromDouble(df) },
            { "Mean", mean },
            { "StandardDeviation", stdDev }
        }
        };
    }

    /// <summary>
    /// Calculates confidence intervals for the mean of a set of values based on the specified distribution type and confidence level.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The sample data.</param>
    /// <param name="confidenceLevel">The confidence level (e.g., 0.95 for 95% confidence).</param>
    /// <param name="distributionType">The type of distribution to use for the calculation.</param>
    /// <returns>A tuple containing the lower and upper bounds of the confidence interval.</returns>
    /// <exception cref="ArgumentException">Thrown when an invalid distribution type is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Confidence intervals tell you the range where the true population mean 
    /// is likely to be, based on your sample data. For example, a 95% confidence interval means that 
    /// if you were to take many samples and calculate the confidence interval for each, about 95% of 
    /// these intervals would contain the true population mean. This method calculates these intervals 
    /// for different types of distributions (Normal, Laplace, Student's t, LogNormal, Exponential, or Weibull). 
    /// The calculation approach varies by distribution type, but all involve finding the appropriate 
    /// critical values based on the confidence level and using them to calculate the margin of error 
    /// around the sample mean or median.
    /// </para>
    /// </remarks>
    public static (T LowerBound, T UpperBound) CalculateConfidenceIntervals(Vector<T> values, T confidenceLevel, DistributionType distributionType)
    {
        (var mean, var stdDev) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(values);
        var median = StatisticsHelper<T>.CalculateMedian(values);
        var mad = StatisticsHelper<T>.CalculateMeanAbsoluteDeviation(values, median);
        T lowerBound, upperBound;

        switch (distributionType)
        {
            case DistributionType.Normal:
                var zScore = CalculateInverseNormalCDF(_numOps.Subtract(_numOps.One, _numOps.Divide(_numOps.Subtract(_numOps.One, confidenceLevel), _numOps.FromDouble(2))));
                var marginOfError = _numOps.Multiply(zScore, _numOps.Divide(stdDev, _numOps.Sqrt(_numOps.FromDouble(values.Length))));
                lowerBound = _numOps.Subtract(mean, marginOfError);
                upperBound = _numOps.Add(mean, marginOfError);
                break;
            case DistributionType.Laplace:
                var laplaceValue = CalculateInverseLaplaceCDF(median, mad, confidenceLevel);
                lowerBound = _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(2), median), laplaceValue);
                upperBound = laplaceValue;
                break;
            case DistributionType.Student:
                var tValue = CalculateInverseStudentTCDF(values.Length - 1, _numOps.Subtract(_numOps.One, _numOps.Divide(_numOps.Subtract(_numOps.One, confidenceLevel), _numOps.FromDouble(2))));
                var tMarginOfError = _numOps.Multiply(tValue, _numOps.Divide(stdDev, _numOps.Sqrt(_numOps.FromDouble(values.Length))));
                lowerBound = _numOps.Subtract(mean, tMarginOfError);
                upperBound = _numOps.Add(mean, tMarginOfError);
                break;
            case DistributionType.LogNormal:
                var logSample = values.Transform(x => _numOps.Log(x));
                (var logMean, var logStdDev) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(logSample);
                var logZScore = CalculateInverseNormalCDF(_numOps.Subtract(_numOps.One, _numOps.Divide(_numOps.Subtract(_numOps.One, confidenceLevel), _numOps.FromDouble(2))));
                lowerBound = _numOps.Exp(_numOps.Subtract(logMean, _numOps.Multiply(logZScore, _numOps.Divide(logStdDev, _numOps.Sqrt(_numOps.FromDouble(values.Length))))));
                upperBound = _numOps.Exp(_numOps.Add(logMean, _numOps.Multiply(logZScore, _numOps.Divide(logStdDev, _numOps.Sqrt(_numOps.FromDouble(values.Length))))));
                break;
            case DistributionType.Exponential:
                var lambda = _numOps.Divide(_numOps.One, mean);
                var chiSquareLower = CalculateInverseChiSquareCDF(
                    Convert.ToInt32(_numOps.Multiply(_numOps.FromDouble(2), _numOps.FromDouble(values.Length))),
                    _numOps.Divide(_numOps.Subtract(_numOps.One, confidenceLevel), _numOps.FromDouble(2)));

                var chiSquareUpper = CalculateInverseChiSquareCDF(
                    Convert.ToInt32(_numOps.Multiply(_numOps.FromDouble(2), _numOps.FromDouble(values.Length))),
                    _numOps.Subtract(_numOps.One, _numOps.Divide(_numOps.Subtract(_numOps.One, confidenceLevel), _numOps.FromDouble(2))));
                lowerBound = _numOps.Multiply(_numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), _numOps.FromDouble(values.Length)), chiSquareUpper),
                    _numOps.Divide(_numOps.One, lambda));
                upperBound = _numOps.Multiply(_numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), _numOps.FromDouble(values.Length)), chiSquareLower),
                    _numOps.Divide(_numOps.One, lambda));
                break;
            case DistributionType.Weibull:
                // For Weibull, we'll use a bootstrap method to estimate confidence intervals
                (lowerBound, upperBound) = CalculateWeibullConfidenceIntervals(values, confidenceLevel);
                break;
            default:
                throw new ArgumentException("Invalid distribution type");
        }

        return (lowerBound, upperBound);
    }

    /// <summary>
    /// Calculates the Pearson correlation coefficient between two sets of values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The first set of values.</param>
    /// <param name="y">The second set of values.</param>
    /// <returns>The Pearson correlation coefficient, a value between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Pearson correlation coefficient measures the linear relationship 
    /// between two variables. It ranges from -1 to 1, where 1 means a perfect positive linear 
    /// relationship (as one variable increases, the other increases proportionally), -1 means a 
    /// perfect negative linear relationship (as one variable increases, the other decreases 
    /// proportionally), and 0 means no linear relationship. This method calculates this coefficient 
    /// using the standard formula that involves the covariance of the variables divided by the 
    /// product of their standard deviations. It's useful for understanding how strongly two variables 
    /// are related to each other.
    /// </para>
    /// </remarks>
    public static T CalculatePearsonCorrelation(Vector<T> x, Vector<T> y)
    {
        var n = _numOps.FromDouble(x.Length);

        var sumX = x.Aggregate(_numOps.Zero, _numOps.Add);
        var sumY = y.Aggregate(_numOps.Zero, _numOps.Add);
        var sumXY = x.Zip(y, _numOps.Multiply).Aggregate(_numOps.Zero, _numOps.Add);
        var sumXSquare = x.Select(_numOps.Square).Aggregate(_numOps.Zero, _numOps.Add);
        var sumYSquare = y.Select(_numOps.Square).Aggregate(_numOps.Zero, _numOps.Add);

        var numerator = _numOps.Subtract(_numOps.Multiply(n, sumXY), _numOps.Multiply(sumX, sumY));
        var denominatorX = _numOps.Sqrt(_numOps.Subtract(_numOps.Multiply(n, sumXSquare), _numOps.Square(sumX)));
        var denominatorY = _numOps.Sqrt(_numOps.Subtract(_numOps.Multiply(n, sumYSquare), _numOps.Square(sumY)));

        return _numOps.Divide(numerator, _numOps.Multiply(denominatorX, denominatorY));
    }

    /// <summary>
    /// Calculates a learning curve by evaluating model performance on increasingly larger subsets of data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="yActual">The actual values.</param>
    /// <param name="yPredicted">The predicted values from a model.</param>
    /// <param name="steps">The number of points to calculate in the learning curve.</param>
    /// <returns>A list of R-squared values representing the model performance at each step.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A learning curve helps you understand how a model's performance improves 
    /// as it sees more training data. This method creates a learning curve by calculating the R-squared 
    /// (a measure of how well the model fits the data) for increasingly larger subsets of your data. 
    /// For example, it might calculate R-squared for the first 10% of the data, then the first 20%, 
    /// and so on. The resulting curve can help you determine if your model would benefit from more 
    /// training data or if it's already reached its potential. A curve that's still rising at the end 
    /// suggests that more data could improve performance, while a plateau indicates that additional 
    /// data might not help much.
    /// </para>
    /// </remarks>
    public static List<T> CalculateLearningCurve(Vector<T> yActual, Vector<T> yPredicted, int steps)
    {
        var learningCurveList = new List<T>();

        // Guard against divide by zero: ensure we have enough data points
        if (yActual.Length < 2 || steps < 1)
        {
            // Return single R2 value for all available data if we can't create a curve
            if (yActual.Length >= 2)
            {
                learningCurveList.Add(CalculateR2(yActual, yPredicted));
            }
            return learningCurveList;
        }

        // Ensure step size is at least 1 to avoid empty subsets
        var stepSize = Math.Max(1, yActual.Length / steps);

        for (int i = 1; i <= steps; i++)
        {
            var subsetSize = Math.Min(i * stepSize, yActual.Length);
            // Ensure we have at least 2 data points for R2 calculation
            if (subsetSize < 2) continue;

            var subsetActual = new Vector<T>(yActual.Take(subsetSize));
            var subsetPredicted = new Vector<T>(yPredicted.Take(subsetSize));

            var r2 = CalculateR2(subsetActual, subsetPredicted);
            learningCurveList.Add(r2);
        }

        return learningCurveList;
    }

    /// <summary>
    /// Calculates prediction intervals for future observations based on a model's predictions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="confidenceLevel">The confidence level (e.g., 0.95 for 95% confidence).</param>
    /// <returns>A tuple containing the lower and upper bounds of the prediction interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Prediction intervals tell you the range where future individual observations 
    /// are likely to fall, based on your model's predictions. Unlike confidence intervals (which estimate 
    /// where the true mean is), prediction intervals account for both the uncertainty in estimating the 
    /// mean and the natural variability in individual observations. This method calculates these intervals 
    /// by first determining the mean squared error (MSE) between actual and predicted values, then using 
    /// this to estimate the standard error. It then applies a t-value (based on the confidence level) to 
    /// calculate the margin of error around the mean prediction. The resulting interval gives you a range 
    /// where you can expect future observations to fall with the specified level of confidence.
    /// </para>
    /// </remarks>
    public static (T LowerInterval, T UpperInterval) CalculatePredictionIntervals(Vector<T> actual, Vector<T> predicted, T confidenceLevel)
    {
        int n = actual.Length;
        T sumSquaredErrors = _numOps.Zero;
        T meanPredicted = _numOps.Zero;

        for (int i = 0; i < n; i++)
        {
            T error = _numOps.Subtract(actual[i], predicted[i]);
            sumSquaredErrors = _numOps.Add(sumSquaredErrors, _numOps.Multiply(error, error));
            meanPredicted = _numOps.Add(meanPredicted, predicted[i]);
        }

        meanPredicted = _numOps.Divide(meanPredicted, _numOps.FromDouble(n));
        T mse = _numOps.Divide(sumSquaredErrors, _numOps.FromDouble(n - 2));
        T standardError = _numOps.Sqrt(mse);

        T tValue = CalculateTValue(n - 2, confidenceLevel);
        T margin = _numOps.Multiply(tValue, standardError);

        return (_numOps.Subtract(meanPredicted, margin), _numOps.Add(meanPredicted, margin));
    }

    /// <summary>
    /// Calculates the proportion of actual values that fall within a specified prediction interval.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="lowerInterval">The lower bound of the prediction interval.</param>
    /// <param name="upperInterval">The upper bound of the prediction interval.</param>
    /// <returns>The proportion of actual values that fall within the prediction interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method helps you evaluate how well your prediction intervals are 
    /// calibrated. It calculates the percentage of actual values that fall within the specified 
    /// prediction interval. For example, if you've calculated a 95% prediction interval, ideally 
    /// about 95% of the actual values should fall within this interval. If a much lower percentage 
    /// falls within the interval, your model might be underestimating uncertainty. If a much higher 
    /// percentage falls within the interval, your model might be overestimating uncertainty (making 
    /// the intervals unnecessarily wide). This metric is useful for assessing whether your prediction 
    /// intervals are appropriately sized for your data and model.
    /// </para>
    /// </remarks>
    public static T CalculatePredictionIntervalCoverage(Vector<T> actual, Vector<T> predicted, T lowerInterval, T upperInterval)
    {
        int covered = 0;
        int n = actual.Length;

        for (int i = 0; i < n; i++)
        {
            if (_numOps.GreaterThanOrEquals(actual[i], lowerInterval) && _numOps.LessThanOrEquals(actual[i], upperInterval))
            {
                covered++;
            }
        }

        return _numOps.Divide(_numOps.FromDouble(covered), _numOps.FromDouble(n));
    }

    /// <summary>
    /// Calculates the mean absolute prediction error between actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The mean absolute prediction error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The mean prediction error (specifically, the mean absolute error or MAE) 
    /// measures how far off your predictions are from the actual values, on average. It calculates the 
    /// absolute difference between each predicted value and its corresponding actual value, then takes 
    /// the average of these differences. This gives you a single number that represents the typical 
    /// magnitude of your prediction errors. The MAE is in the same units as your original data, making 
    /// it easy to interpret. For example, if you're predicting house prices in dollars and get an MAE 
    /// of $10,000, it means your predictions are off by about $10,000 on average. Lower values indicate 
    /// better predictive accuracy.
    /// </para>
    /// </remarks>
    public static T CalculateMeanPredictionError(Vector<T> actual, Vector<T> predicted)
    {
        int n = actual.Length;
        T sumError = _numOps.Zero;

        for (int i = 0; i < n; i++)
        {
            sumError = _numOps.Add(sumError, _numOps.Abs(_numOps.Subtract(actual[i], predicted[i])));
        }

        return _numOps.Divide(sumError, _numOps.FromDouble(n));
    }

    /// <summary>
    /// Calculates the median absolute prediction error between actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The median absolute prediction error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The median prediction error (specifically, the median absolute error) is 
    /// similar to the mean absolute error, but it uses the median instead of the mean. It calculates 
    /// the absolute difference between each predicted value and its corresponding actual value, then 
    /// finds the middle value (median) of these differences. This metric is less sensitive to outliers 
    /// than the mean absolute error, making it useful when your data contains extreme values that might 
    /// skew the average. Like the mean absolute error, it's in the same units as your original data, 
    /// making it easy to interpret. Lower values indicate better predictive accuracy.
    /// </para>
    /// </remarks>
    public static T CalculateMedianPredictionError(Vector<T> actual, Vector<T> predicted)
    {
        int n = actual.Length;
        var absoluteErrors = new T[n];

        for (int i = 0; i < n; i++)
        {
            absoluteErrors[i] = _numOps.Abs(_numOps.Subtract(actual[i], predicted[i]));
        }

        Array.Sort(absoluteErrors, (a, b) => _numOps.LessThan(a, b) ? -1 : _numOps.Equals(a, b) ? 0 : 1);

        if (n % 2 == 0)
        {
            int middleIndex = n / 2;
            return _numOps.Divide(_numOps.Add(absoluteErrors[middleIndex - 1], absoluteErrors[middleIndex]), _numOps.FromDouble(2));
        }
        else
        {
            return absoluteErrors[n / 2];
        }
    }

    /// <summary>
    /// Calculates the t-value for a given degrees of freedom and confidence level.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="degreesOfFreedom">The degrees of freedom.</param>
    /// <param name="confidenceLevel">The confidence level (e.g., 0.95 for 95% confidence).</param>
    /// <returns>The t-value corresponding to the given confidence level and degrees of freedom.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The t-value is a critical value used in statistical hypothesis testing 
    /// and confidence interval calculations. It represents how many standard errors away from the mean 
    /// you need to go to capture a certain percentage of the data. This method calculates the t-value 
    /// based on the Student's t-distribution for a given confidence level and degrees of freedom. 
    /// For example, with a 95% confidence level, the t-value tells you how far from the mean you need 
    /// to go to include 95% of the data in your interval. Higher confidence levels result in larger 
    /// t-values, meaning wider intervals.
    /// </para>
    /// </remarks>
    public static T CalculateTValue(int degreesOfFreedom, T confidenceLevel)
    {
        T alpha = _numOps.Divide(_numOps.Subtract(_numOps.One, confidenceLevel), _numOps.FromDouble(2));
        return CalculateInverseStudentTCDF(degreesOfFreedom, _numOps.Subtract(_numOps.One, alpha));
    }

    /// <summary>
    /// Calculates the first and third quartiles (25th and 75th percentiles) of a data set.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data vector.</param>
    /// <returns>A tuple containing the first quartile (Q1) and third quartile (Q3).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quartiles divide your data into four equal parts after sorting it from 
    /// smallest to largest. The first quartile (Q1) is the value below which 25% of the data falls, 
    /// and the third quartile (Q3) is the value below which 75% of the data falls. These values are 
    /// useful for understanding the spread of your data and identifying potential outliers. The 
    /// difference between Q3 and Q1 is called the interquartile range (IQR) and is a robust measure 
    /// of variability. This method sorts your data and then uses linear interpolation to estimate 
    /// the quartile values.
    /// </para>
    /// </remarks>
    public static (T FirstQuantile, T ThirdQuantile) CalculateQuantiles(Vector<T> data)
    {
        var sortedData = data.OrderBy(x => x).ToArray();
        int n = sortedData.Length;

        T Q1 = CalculateQuantile(sortedData, _numOps.FromDouble(0.25));
        T Q3 = CalculateQuantile(sortedData, _numOps.FromDouble(0.75));

        return (Q1, Q3);
    }

    /// <summary>
    /// Calculates a specific quantile from sorted data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="sortedData">The data array, already sorted in ascending order.</param>
    /// <param name="quantile">The quantile to calculate (between 0 and 1).</param>
    /// <returns>The value at the specified quantile.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A quantile divides your sorted data into equal portions. For example, 
    /// the 0.25 quantile (also called the 25th percentile or first quartile) is the value below which 
    /// 25% of your data falls. This method calculates any quantile between 0 and 1 using linear 
    /// interpolation, which means it estimates values between actual data points when necessary. 
    /// The method requires that your data is already sorted in ascending order (smallest to largest). 
    /// Common quantiles include 0.25 (first quartile), 0.5 (median), and 0.75 (third quartile).
    /// </para>
    /// </remarks>
    public static T CalculateQuantile(T[] sortedData, T quantile)
    {
        int n = sortedData.Length;
        T position = _numOps.Multiply(_numOps.FromDouble(n - 1), quantile);
        // Use Floor to match NumPy's linear interpolation (method='linear')
        int index = Convert.ToInt32(_numOps.Floor(position));
        T fraction = _numOps.Subtract(position, _numOps.FromDouble(index));

        if (index + 1 < n)
            return _numOps.Add(
                _numOps.Multiply(sortedData[index], _numOps.Subtract(_numOps.One, fraction)),
                _numOps.Multiply(sortedData[index + 1], fraction)
            );
        else
            return sortedData[index];
    }

    /// <summary>
    /// Calculates the skewness and kurtosis of a sample.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="sample">The sample data.</param>
    /// <param name="mean">The pre-calculated mean of the sample.</param>
    /// <param name="stdDev">The pre-calculated standard deviation of the sample.</param>
    /// <param name="n">The sample size.</param>
    /// <returns>A tuple containing the skewness and kurtosis values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Skewness and kurtosis are measures that describe the shape of your data's 
    /// distribution. Skewness measures the asymmetry - a positive skew means the distribution has a 
    /// longer tail on the right side, while a negative skew means a longer tail on the left. A skewness 
    /// of zero indicates a symmetric distribution. Kurtosis measures how "heavy" the tails are compared 
    /// to a normal distribution. Higher kurtosis means more of the variance comes from infrequent extreme 
    /// deviations, while lower kurtosis indicates more frequent moderate deviations. This method calculates 
    /// both measures in a single pass through the data, using pre-calculated mean and standard deviation 
    /// values for efficiency.
    /// </para>
    /// </remarks>
    public static (T skewness, T kurtosis) CalculateSkewnessAndKurtosis(Vector<T> sample, T mean, T stdDev, int n)
    {
        T skewnessSum = _numOps.Zero, kurtosisSum = _numOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T diff = _numOps.Divide(_numOps.Subtract(sample[i], mean), stdDev);
            T diff3 = _numOps.Multiply(_numOps.Multiply(diff, diff), diff);
            skewnessSum = _numOps.Add(skewnessSum, diff3);
            kurtosisSum = _numOps.Add(kurtosisSum, _numOps.Multiply(diff3, diff));
        }

        T skewness = n > 2 ? _numOps.Divide(skewnessSum, _numOps.Multiply(_numOps.FromDouble(n - 1), _numOps.FromDouble(n - 2))) : _numOps.Zero;
        T kurtosis = n > 3 ?
            _numOps.Subtract(
                _numOps.Multiply(
                    _numOps.Divide(kurtosisSum, _numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(n - 1), _numOps.FromDouble(n - 2)), _numOps.FromDouble(n - 3))),
                    _numOps.Multiply(_numOps.FromDouble(n), _numOps.FromDouble(n + 1))
                ),
                _numOps.Divide(
                    _numOps.Multiply(_numOps.FromDouble(3), _numOps.Square(_numOps.FromDouble(n - 1))),
                    _numOps.Multiply(_numOps.FromDouble(n - 2), _numOps.FromDouble(n - 3))
                )
            ) : _numOps.Zero;

        return (skewness, kurtosis);
    }

    /// <summary>
    /// Calculates a tolerance interval for a set of predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="confidenceLevel">The confidence level (e.g., 0.95 for 95% confidence).</param>
    /// <returns>A tuple containing the lower and upper bounds of the tolerance interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A tolerance interval is a range that is expected to contain a specified 
    /// proportion of a population with a certain confidence level. Unlike confidence intervals (which 
    /// estimate where the true mean is) or prediction intervals (which predict where a single future 
    /// observation will fall), tolerance intervals aim to capture a specified proportion of the entire 
    /// population. This method calculates tolerance intervals for predicted values, taking into account 
    /// both the variability in the data and the sample size. The resulting interval gives you a range 
    /// where you can expect a certain percentage of all future values to fall, with the specified level 
    /// of confidence.
    /// </para>
    /// </remarks>
    public static (T Lower, T Upper) CalculateToleranceInterval(Vector<T> actual, Vector<T> predicted, T confidenceLevel)
    {
        int n = actual.Length;
        T mean = predicted.Average();
        T stdDev = CalculateStandardDeviation(predicted);
        T factor = _numOps.FromDouble(Math.Sqrt(1 + 1.0 / n));
        T tValue = CalculateTValue(n - 1, confidenceLevel);
        T margin = _numOps.Multiply(tValue, _numOps.Multiply(stdDev, factor));

        return (_numOps.Subtract(mean, margin), _numOps.Add(mean, margin));
    }

    /// <summary>
    /// Calculates a forecast interval for future predictions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="confidenceLevel">The confidence level (e.g., 0.95 for 95% confidence).</param>
    /// <returns>A tuple containing the lower and upper bounds of the forecast interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A forecast interval is a range where future observations are expected to 
    /// fall with a certain probability. It's similar to a prediction interval but specifically designed 
    /// for time series forecasting. This method calculates forecast intervals based on the mean squared 
    /// error between actual and predicted values, adjusted by a t-value corresponding to the desired 
    /// confidence level. The resulting interval gives you a range where you can expect future values to 
    /// fall, with the specified level of confidence. Wider intervals indicate greater uncertainty in 
    /// your forecasts.
    /// </para>
    /// </remarks>
    public static (T Lower, T Upper) CalculateForecastInterval(Vector<T> actual, Vector<T> predicted, T confidenceLevel)
    {
        int n = actual.Length;
        T mean = predicted.Average();
        T mse = CalculateMeanSquaredError(actual, predicted);
        T factor = _numOps.FromDouble(Math.Sqrt(1 + 1.0 / n));
        T tValue = CalculateTValue(n - 1, confidenceLevel);
        T margin = _numOps.Multiply(tValue, _numOps.Multiply(_numOps.Sqrt(mse), factor));

        return (_numOps.Subtract(mean, margin), _numOps.Add(mean, margin));
    }

    /// <summary>
    /// Calculates confidence intervals around specified quantiles of the predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="quantiles">An array of quantiles to calculate intervals for.</param>
    /// <returns>A list of tuples containing the quantile and its corresponding lower and upper interval bounds.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates confidence intervals around specific quantiles 
    /// (percentiles) of your predicted values. For example, you might want to know the range around 
    /// the median (50th percentile) or the 90th percentile of your predictions. For each quantile you 
    /// specify, this method calculates a lower and upper bound by looking at nearby quantiles (²2.5%). 
    /// This gives you an idea of the uncertainty around different parts of your prediction distribution. 
    /// These intervals are useful when you're interested in specific parts of the distribution rather 
    /// than just the mean or a single prediction.
    /// </para>
    /// </remarks>
    public static List<(T Quantile, T Lower, T Upper)> CalculateQuantileIntervals(Vector<T> actual, Vector<T> predicted, T[] quantiles)
    {
        var result = new List<(T Quantile, T Lower, T Upper)>();
        var sortedPredictions = new Vector<T>([.. predicted.OrderBy(x => x)]);

        foreach (var q in quantiles)
        {
            T lowerQuantile = CalculateQuantile(sortedPredictions, _numOps.Subtract(q, _numOps.FromDouble(0.025)));
            T upperQuantile = CalculateQuantile(sortedPredictions, _numOps.Add(q, _numOps.FromDouble(0.025)));
            result.Add((q, lowerQuantile, upperQuantile));
        }

        return result;
    }

    /// <summary>
    /// Calculates confidence intervals using bootstrap resampling.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="confidenceLevel">The confidence level (e.g., 0.95 for 95% confidence).</param>
    /// <returns>A tuple containing the lower and upper bounds of the bootstrap confidence interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bootstrap intervals are a powerful way to estimate confidence intervals 
    /// without making assumptions about the underlying distribution of your data. This method creates 
    /// many new samples by randomly selecting values from your original predictions (with replacement), 
    /// calculates the mean for each of these samples, and then determines the interval bounds based on 
    /// the distribution of these means. For example, for a 95% confidence interval, it finds the values 
    /// that contain the middle 95% of the bootstrap sample means. This approach is particularly useful 
    /// when your data doesn't follow a normal distribution or when you have a small sample size.
    /// </para>
    /// </remarks>
    public static (T Lower, T Upper) CalculateBootstrapInterval(Vector<T> actual, Vector<T> predicted, T confidenceLevel)
    {
        int n = actual.Length;
        int bootstrapSamples = 1000;
        var bootstrapMeans = new List<T>();

        var random = RandomHelper.CreateSecureRandom();
        for (int i = 0; i < bootstrapSamples; i++)
        {
            var sample = new Vector<T>(n);
            for (int j = 0; j < n; j++)
            {
                int index = random.Next(n);
                sample[j] = predicted[index];
            }
            bootstrapMeans.Add(sample.Average());
        }

        bootstrapMeans.Sort();
        int lowerIndex = Convert.ToInt32(_numOps.Divide(_numOps.Multiply(confidenceLevel, _numOps.FromDouble(bootstrapSamples)), _numOps.FromDouble(2)));
        int upperIndex = bootstrapSamples - lowerIndex - 1;

        return (bootstrapMeans[lowerIndex], bootstrapMeans[upperIndex]);
    }

    /// <summary>
    /// Calculates a simultaneous prediction interval for multiple future observations.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="confidenceLevel">The confidence level (e.g., 0.95 for 95% confidence).</param>
    /// <returns>A tuple containing the lower and upper bounds of the simultaneous prediction interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> While a regular prediction interval gives you a range where a single future 
    /// observation is likely to fall, a simultaneous prediction interval gives you a range where multiple 
    /// future observations are all likely to fall at the same time. This is important when you're making 
    /// multiple predictions and want to ensure that all of them (not just each one individually) are 
    /// within the interval with a certain probability. This method calculates such intervals by adjusting 
    /// the margin of error to account for multiple comparisons. The resulting interval is wider than a 
    /// regular prediction interval but provides stronger guarantees for multiple predictions.
    /// </para>
    /// </remarks>
    public static (T Lower, T Upper) CalculateSimultaneousPredictionInterval(Vector<T> actual, Vector<T> predicted, T confidenceLevel)
    {
        int n = actual.Length;
        T mean = predicted.Average();
        T mse = CalculateMeanSquaredError(actual, predicted);
        T factor = _numOps.Sqrt(_numOps.Multiply(_numOps.FromDouble(2), confidenceLevel));
        T margin = _numOps.Multiply(factor, _numOps.Sqrt(mse));

        return (_numOps.Subtract(mean, margin), _numOps.Add(mean, margin));
    }

    /// <summary>
    /// Calculates confidence intervals using the jackknife resampling method.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>A tuple containing the lower and upper bounds of the jackknife confidence interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The jackknife method is a resampling technique that helps estimate the 
    /// bias and variance of a statistic. Unlike bootstrap (which creates new samples by randomly selecting 
    /// with replacement), jackknife creates new samples by leaving out one observation at a time. This 
    /// method calculates jackknife confidence intervals by computing the mean of each leave-one-out sample, 
    /// then using the distribution of these means to estimate the standard error. It then applies a t-value 
    /// to calculate the confidence interval. Jackknife intervals are useful when you have a small sample 
    /// size or when you want to reduce the influence of potential outliers.
    /// </para>
    /// </remarks>
    public static (T Lower, T Upper) CalculateJackknifeInterval(Vector<T> actual, Vector<T> predicted)
    {
        int n = actual.Length;
        if (n == 0)
        {
            return (_numOps.Zero, _numOps.Zero);
        }

        if (n == 1)
        {
            return (predicted[0], predicted[0]);
        }

        var jackknifeSamples = new List<T>();

        for (int i = 0; i < n; i++)
        {
            var sample = new Vector<T>(n - 1);
            int index = 0;
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                {
                    sample[index++] = predicted[j];
                }
            }
            jackknifeSamples.Add(sample.Average());
        }

        T jackknifeEstimate = new Vector<T>([.. jackknifeSamples]).Average();
        T jackknifeStdError = CalculateStandardDeviation(new Vector<T>([.. jackknifeSamples]));
        T tValue = CalculateTValue(n - 1, _numOps.FromDouble(0.95));
        T margin = _numOps.Multiply(tValue, jackknifeStdError);

        return (_numOps.Subtract(jackknifeEstimate, margin), _numOps.Add(jackknifeEstimate, margin));
    }

    /// <summary>
    /// Calculates a percentile-based confidence interval from predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="confidenceLevel">The confidence level (e.g., 0.95 for 95% confidence).</param>
    /// <returns>A tuple containing the lower and upper bounds of the percentile interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A percentile interval is a simple way to create a confidence interval 
    /// directly from your data without assuming any particular distribution. This method sorts your 
    /// predicted values and then finds the values at the percentiles corresponding to the edges of 
    /// your desired confidence level. For example, for a 95% confidence interval, it finds the values 
    /// at the 2.5th and 97.5th percentiles. This approach is non-parametric (doesn't assume a normal 
    /// distribution) and is useful when your data doesn't follow a normal distribution or when you 
    /// want a straightforward interpretation of your interval.
    /// </para>
    /// </remarks>
    public static (T Lower, T Upper) CalculatePercentileInterval(Vector<T> predicted, T confidenceLevel)
    {
        var sortedPredictions = new Vector<T>([.. predicted.OrderBy(x => x)]);
        int n = sortedPredictions.Length;
        T alpha = _numOps.Subtract(_numOps.One, confidenceLevel);
        int lowerIndex = Convert.ToInt32(_numOps.Divide(_numOps.Multiply(alpha, _numOps.FromDouble(n)), _numOps.FromDouble(2.0)));
        int upperIndex = n - lowerIndex - 1;

        return (sortedPredictions[lowerIndex], sortedPredictions[upperIndex]);
    }

    /// <summary>
    /// Calculates the median absolute error between actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The median absolute error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The median absolute error (MedAE) is a measure of prediction accuracy 
    /// that's more robust to outliers than the mean absolute error. It calculates the absolute difference 
    /// between each actual and predicted value, then finds the median of these differences. This gives 
    /// you the "typical" error in your predictions without being overly influenced by a few very large 
    /// errors. Like the mean absolute error, it's in the same units as your original data, making it 
    /// easy to interpret. The MedAE is particularly useful when your error distribution is skewed or 
    /// contains outliers.
    /// </para>
    /// </remarks>
    public static T CalculateMedianAbsoluteError(Vector<T> actual, Vector<T> predicted)
    {
        var absoluteErrors = actual.Subtract(predicted).Select(_numOps.Abs).OrderBy(x => x).ToArray();
        int n = absoluteErrors.Length;

        return n % 2 == 0
            ? _numOps.Divide(_numOps.Add(absoluteErrors[n / 2 - 1], absoluteErrors[n / 2]), _numOps.FromDouble(2))
            : absoluteErrors[n / 2];
    }

    /// <summary>
    /// Calculates the maximum absolute error between actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The maximum absolute error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The maximum error measures the largest absolute difference between any 
    /// actual value and its corresponding prediction. This metric gives you the "worst-case scenario" 
    /// for your model's predictions. While mean or median error metrics tell you about the typical 
    /// performance, the maximum error tells you about the extreme cases. This can be important in 
    /// applications where even a single large error could have serious consequences. Like other error 
    /// metrics, it's in the same units as your original data, making it easy to interpret in the context 
    /// of your problem.
    /// </para>
    /// </remarks>
    public static T CalculateMaxError(Vector<T> actual, Vector<T> predicted)
    {
        return actual.Subtract(predicted).Select(_numOps.Abs).Max();
    }

    /// <summary>
    /// Calculates the sample standard error of the estimate, adjusted for the number of model parameters.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="numberOfParameters">The number of parameters in the model.</param>
    /// <returns>The sample standard error of the estimate.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The sample standard error of the estimate measures the accuracy of predictions 
    /// made by a regression model, adjusted for the complexity of the model. It's calculated by taking 
    /// the square root of the mean squared error divided by the degrees of freedom (sample size minus 
    /// the number of parameters). This adjustment accounts for the fact that models with more parameters 
    /// tend to fit the training data better by chance. The standard error gives you an idea of how much 
    /// your predictions typically deviate from the actual values, in the same units as your original data. 
    /// Lower values indicate better predictive accuracy.
    /// </para>
    /// </remarks>
    public static T CalculateSampleStandardError(Vector<T> actual, Vector<T> predicted, int numberOfParameters)
    {
        T mse = CalculateMeanSquaredError(actual, predicted);
        int degreesOfFreedom = actual.Length - numberOfParameters;

        return _numOps.Sqrt(_numOps.Divide(mse, _numOps.FromDouble(degreesOfFreedom)));
    }

    /// <summary>
    /// Calculates the population standard error of the estimate.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The population standard error of the estimate.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The population standard error is simply the square root of the mean squared 
    /// error between actual and predicted values. Unlike the sample standard error, it doesn't adjust 
    /// for the number of model parameters. It represents the standard deviation of the prediction errors 
    /// and gives you an idea of how much your predictions typically deviate from the actual values, in 
    /// the same units as your original data. This metric is useful when you're treating your entire dataset 
    /// as the population rather than as a sample from a larger population. Lower values indicate better 
    /// predictive accuracy.
    /// </para>
    /// </remarks>
    public static T CalculatePopulationStandardError(Vector<T> actual, Vector<T> predicted)
    {
        return _numOps.Sqrt(CalculateMeanSquaredError(actual, predicted));
    }

    /// <summary>
    /// Calculates the mean bias error between actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The mean bias error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The mean bias error (MBE) measures the average direction of errors in your
    /// predictions. It's calculated by taking the average of the differences between predicted and actual
    /// values (predicted - actual). A positive MBE indicates that your model tends to overestimate values
    /// (predictions are too high on average), while a negative MBE indicates that your model tends to
    /// underestimate values (predictions are too low on average). An MBE close to zero suggests that your
    /// model's errors are balanced in both directions. This metric is useful for detecting systematic bias
    /// in your predictions, but it can mask the magnitude of errors since positive and negative errors
    /// can cancel each other out.
    /// </para>
    /// </remarks>
    public static T CalculateMeanBiasError(Vector<T> actual, Vector<T> predicted)
    {
        // MBE = mean(predicted - actual): positive = overprediction, negative = underprediction
        return _numOps.Divide(predicted.Subtract(actual).Aggregate(_numOps.Zero, _numOps.Add), _numOps.FromDouble(actual.Length));
    }

    /// <summary>
    /// Calculates Theil's U statistic, a measure of forecast accuracy.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>Theil's U statistic.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Theil's U statistic is a measure of forecast accuracy that compares your 
    /// model's predictions to a simple no-change forecast. It's calculated by dividing the root mean 
    /// squared error of your predictions by the sum of the root mean squared values of the actual and 
    /// predicted series. A value of 0 indicates perfect forecasts, while a value of 1 indicates that 
    /// your model performs no better than a naive no-change forecast. Values less than 1 indicate that 
    /// your model outperforms the naive forecast, while values greater than 1 indicate that your model 
    /// performs worse than the naive forecast. This metric is particularly useful in time series forecasting 
    /// to evaluate whether your model adds value beyond simple forecasting methods.
    /// </para>
    /// </remarks>
    public static T CalculateTheilUStatistic(Vector<T> actual, Vector<T> predicted)
    {
        T numerator = _numOps.Sqrt(CalculateMeanSquaredError(actual, predicted));
        T denominatorActual = _numOps.Sqrt(_numOps.Divide(actual.Select(x => _numOps.Square(x)).Aggregate(_numOps.Zero, _numOps.Add), _numOps.FromDouble(actual.Length)));
        T denominatorPredicted = _numOps.Sqrt(_numOps.Divide(predicted.Select(x => _numOps.Square(x)).Aggregate(_numOps.Zero, _numOps.Add), _numOps.FromDouble(predicted.Length)));

        return _numOps.Divide(numerator, _numOps.Add(denominatorActual, denominatorPredicted));
    }

    /// <summary>
    /// Calculates the Durbin-Watson statistic to test for autocorrelation in residuals.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The Durbin-Watson statistic.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Durbin-Watson statistic tests whether there is autocorrelation in the 
    /// residuals (errors) of a regression model. Autocorrelation means that the error at one point is 
    /// correlated with the error at another point, which violates a key assumption of many regression 
    /// models. The statistic ranges from 0 to 4, with a value of 2 indicating no autocorrelation. Values 
    /// less than 2 suggest positive autocorrelation (adjacent errors tend to have the same sign), while 
    /// values greater than 2 suggest negative autocorrelation (adjacent errors tend to have opposite signs). 
    /// This method calculates the statistic by first computing the residuals (actual - predicted) and then 
    /// passing them to the other overload of this method.
    /// </para>
    /// </remarks>
    public static T CalculateDurbinWatsonStatistic(Vector<T> actual, Vector<T> predicted)
    {
        var errors = actual.Subtract(predicted);
        return CalculateDurbinWatsonStatistic([.. errors]);
    }

    /// <summary>
    /// Calculates the Durbin-Watson statistic from a list of residuals.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="residualList">The list of residuals (errors) from a model.</param>
    /// <returns>The Durbin-Watson statistic.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This version of the Durbin-Watson statistic calculation takes a list of 
    /// residuals (errors) directly. The statistic is calculated as the sum of squared differences between 
    /// consecutive residuals divided by the sum of squared residuals. It tests for autocorrelation in 
    /// the residuals, which means checking whether the error at one point is related to the error at 
    /// adjacent points. The statistic ranges from 0 to 4, with a value of 2 indicating no autocorrelation. 
    /// Values less than 2 suggest positive autocorrelation, while values greater than 2 suggest negative 
    /// autocorrelation. Detecting autocorrelation is important because it can indicate that your model 
    /// is missing important variables or structure in the data.
    /// </para>
    /// </remarks>
    public static T CalculateDurbinWatsonStatistic(List<T> residualList)
    {
        if (residualList == null)
        {
            throw new ArgumentNullException(nameof(residualList), "Residual list cannot be null.");
        }

        if (residualList.Count < 2)
        {
            throw new ArgumentException("Durbin-Watson statistic requires at least 2 residuals.", nameof(residualList));
        }

        T sumSquaredDifferences = _numOps.Zero;
        T sumSquaredErrors = _numOps.Zero;

        for (int i = 1; i < residualList.Count; i++)
        {
            sumSquaredDifferences = _numOps.Add(sumSquaredDifferences, _numOps.Square(_numOps.Subtract(residualList[i], residualList[i - 1])));
            sumSquaredErrors = _numOps.Add(sumSquaredErrors, _numOps.Square(residualList[i]));
        }
        sumSquaredErrors = _numOps.Add(sumSquaredErrors, _numOps.Square(residualList[0]));

        // Check for zero residuals (perfect fit) - DW is undefined
        if (_numOps.Equals(sumSquaredErrors, _numOps.Zero))
        {
            throw new ArgumentException(
                "Durbin-Watson statistic is undefined when all residuals are zero (perfect fit). " +
                "This typically indicates the model perfectly predicts all observations, which is unusual in practice.",
                nameof(residualList));
        }

        return _numOps.Divide(sumSquaredDifferences, sumSquaredErrors);
    }

    /// <summary>
    /// Calculates an alternative formulation of the Akaike Information Criterion (AIC).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="sampleSize">The number of observations in the sample.</param>
    /// <param name="parameterSize">The number of parameters in the model.</param>
    /// <param name="rss">The residual sum of squares (sum of squared errors).</param>
    /// <returns>The AIC value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Akaike Information Criterion (AIC) is a measure used to compare different 
    /// models for the same data. This alternative formulation calculates AIC as n*ln(RSS/n) + 2k, where 
    /// n is the sample size, RSS is the residual sum of squares, and k is the number of parameters. The 
    /// AIC balances model fit (the first term) against model complexity (the second term). Lower AIC values 
    /// indicate better models. When comparing models, differences of 2 or less are considered negligible, 
    /// differences between 4 and 7 indicate the model with the lower AIC is considerably better, and 
    /// differences greater than 10 indicate the model with the lower AIC is substantially better. This 
    /// metric helps prevent overfitting by penalizing models with too many parameters.
    /// </para>
    /// </remarks>
    public static T CalculateAICAlternative(int sampleSize, int parameterSize, T rss)
    {
        if (sampleSize <= 0 || _numOps.LessThanOrEquals(rss, _numOps.Zero)) return _numOps.Zero;
        T logData = _numOps.Divide(rss, _numOps.FromDouble(sampleSize));

        return _numOps.Add(_numOps.Multiply(_numOps.FromDouble(sampleSize), _numOps.Log(logData)), _numOps.Multiply(_numOps.FromDouble(2), _numOps.FromDouble(parameterSize)));
    }

    /// <summary>
    /// Calculates the Akaike Information Criterion (AIC) for model comparison.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="sampleSize">The number of observations in the sample.</param>
    /// <param name="parameterSize">The number of parameters in the model.</param>
    /// <param name="rss">The residual sum of squares (sum of squared errors).</param>
    /// <returns>The AIC value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Akaike Information Criterion (AIC) helps you compare different models 
    /// for the same data. It's calculated as 2k + n*[ln(2pRSS/n) + 1], where k is the number of parameters, 
    /// n is the sample size, and RSS is the residual sum of squares. The AIC balances model fit against 
    /// model complexity - a lower AIC indicates a better model. This formulation of AIC is based on the 
    /// likelihood function assuming normally distributed errors. When comparing models, the absolute AIC 
    /// value isn't important; what matters is the difference between models. A model with an AIC that's 
    /// 2 or more points lower than another is considered better. This metric helps you avoid overfitting 
    /// by penalizing models that use too many parameters.
    /// </para>
    /// </remarks>
    public static T CalculateAIC(int sampleSize, int parameterSize, T rss)
    {
        if (sampleSize <= 0 || _numOps.LessThanOrEquals(rss, _numOps.Zero)) return _numOps.Zero;
        T logData = _numOps.Multiply(_numOps.FromDouble(2 * Math.PI), _numOps.Divide(rss, _numOps.FromDouble(sampleSize)));

        return _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), _numOps.FromDouble(parameterSize)),
                            _numOps.Multiply(_numOps.FromDouble(sampleSize), _numOps.Add(_numOps.Log(logData), _numOps.One)));
    }

    /// <summary>
    /// Calculates the Bayesian Information Criterion (BIC) for model comparison.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="sampleSize">The number of observations in the sample.</param>
    /// <param name="parameterSize">The number of parameters in the model.</param>
    /// <param name="rss">The residual sum of squares (sum of squared errors).</param>
    /// <returns>The BIC value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Bayesian Information Criterion (BIC) is similar to AIC but penalizes 
    /// model complexity more strongly. It's calculated as n*ln(RSS/n) + k*ln(n), where n is the sample 
    /// size, RSS is the residual sum of squares, and k is the number of parameters. Like AIC, lower BIC 
    /// values indicate better models. The BIC tends to favor simpler models than AIC does, especially 
    /// with larger sample sizes, because its penalty for additional parameters increases with sample size. 
    /// When comparing models, differences of 2-6 points indicate positive evidence for the model with the 
    /// lower BIC, differences of 6-10 indicate strong evidence, and differences greater than 10 indicate 
    /// very strong evidence. BIC is particularly useful when you want to be more conservative about adding 
    /// parameters to your model.
    /// </para>
    /// </remarks>
    public static T CalculateBIC(int sampleSize, int parameterSize, T rss)
    {
        if (sampleSize <= 0 || _numOps.LessThanOrEquals(rss, _numOps.Zero)) return _numOps.Zero;
        T logData = _numOps.Divide(rss, _numOps.FromDouble(sampleSize));

        return _numOps.Add(_numOps.Multiply(_numOps.FromDouble(sampleSize), _numOps.Log(logData)),
                            _numOps.Multiply(_numOps.FromDouble(parameterSize), _numOps.Log(_numOps.FromDouble(sampleSize))));
    }

    /// <summary>
    /// Calculates the accuracy of predictions by comparing them to actual values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values.</param>
    /// <returns>The accuracy as a proportion between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Accuracy is the simplest measure of prediction performance - it's the 
    /// proportion of predictions that exactly match the actual values. This method calculates accuracy 
    /// by counting how many predictions are exactly equal to their corresponding actual values, then 
    /// dividing by the total number of predictions. The result ranges from 0 (no correct predictions) 
    /// to 1 (all predictions correct). While easy to understand, this strict definition of accuracy can 
    /// be limiting for many problems, especially with continuous values where exact matches are rare. 
    /// This basic version is most appropriate for classification problems with discrete categories.
    /// </para>
    /// </remarks>
    public static T CalculateAccuracy(Vector<T> actual, Vector<T> predicted)
    {
        var correctPredictions = _numOps.Zero;
        var totalPredictions = _numOps.FromDouble(actual.Length);

        for (int i = 0; i < actual.Length; i++)
        {
            if (_numOps.Equals(actual[i], predicted[i]))
            {
                correctPredictions = _numOps.Add(correctPredictions, _numOps.One);
            }
        }

        return _numOps.Divide(correctPredictions, totalPredictions);
    }

    /// <summary>
    /// Calculates the accuracy of predictions with support for different prediction types and tolerance levels.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="predictionType">The type of prediction (Binary or Regression).</param>
    /// <param name="tolerance">For regression, the acceptable error tolerance as a proportion (default is 0.05 or 5%).</param>
    /// <returns>The accuracy as a proportion between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This enhanced accuracy calculation supports both binary classification and 
    /// regression problems. For binary classification, it works like the simpler version, counting exact 
    /// matches. For regression problems, it introduces a tolerance parameter that allows predictions to 
    /// be "close enough" rather than requiring exact matches. A prediction is considered correct if it's 
    /// within a certain percentage (the tolerance) of the actual value. For example, with the default 5% 
    /// tolerance, a prediction of 95 would be considered correct for an actual value of 100. This makes 
    /// the accuracy metric much more useful for continuous values, where exact matches are unlikely.
    /// </para>
    /// </remarks>
    public static T CalculateAccuracy(Vector<T> actual, Vector<T> predicted, PredictionType predictionType, T? tolerance = default)
        => CalculateAccuracy(actual, predicted, predictionType, tolerance, treatDefaultAsMissing: true);

    public static T CalculateAccuracy(Vector<T> actual, Vector<T> predicted, PredictionType predictionType, T? tolerance, bool treatDefaultAsMissing)
    {
        var correctPredictions = _numOps.Zero;
        var totalPredictions = _numOps.FromDouble(actual.Length);
        tolerance = GetOrDefaultOptionalParameter(tolerance, _numOps.FromDouble(0.05), treatDefaultAsMissing);
        var classificationThreshold = _numOps.FromDouble(0.5);

        for (int i = 0; i < actual.Length; i++)
        {
            if (predictionType == PredictionType.BinaryClassification)
            {
                var actualLabel = _numOps.GreaterThanOrEquals(actual[i], classificationThreshold) ? _numOps.One : _numOps.Zero;
                var predictedLabel = _numOps.GreaterThanOrEquals(predicted[i], classificationThreshold) ? _numOps.One : _numOps.Zero;
                if (_numOps.Equals(actualLabel, predictedLabel))
                    correctPredictions = _numOps.Add(correctPredictions, _numOps.One);
            }
            else if (predictionType == PredictionType.MultiClass)
            {
                int actualLabel = _numOps.ToInt32(actual[i]);
                int predictedLabel = _numOps.ToInt32(predicted[i]);
                if (actualLabel == predictedLabel)
                    correctPredictions = _numOps.Add(correctPredictions, _numOps.One);
            }
            else if (predictionType == PredictionType.MultiLabel)
            {
                var actualLabel = _numOps.GreaterThanOrEquals(actual[i], classificationThreshold) ? _numOps.One : _numOps.Zero;
                var predictedLabel = _numOps.GreaterThanOrEquals(predicted[i], classificationThreshold) ? _numOps.One : _numOps.Zero;
                if (_numOps.Equals(actualLabel, predictedLabel))
                    correctPredictions = _numOps.Add(correctPredictions, _numOps.One);
            }
            else // Regression
            {
                var difference = _numOps.Abs(_numOps.Subtract(actual[i], predicted[i]));
                var threshold = _numOps.Multiply(actual[i], tolerance);
                if (_numOps.LessThanOrEquals(difference, threshold))
                {
                    correctPredictions = _numOps.Add(correctPredictions, _numOps.One);
                }
            }
        }

        return _numOps.Divide(correctPredictions, totalPredictions);
    }

    /// <summary>
    /// Calculates precision, recall, and F1 score for a set of predictions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="predictionType">The type of prediction (Binary or Regression).</param>
    /// <param name="threshold">
    /// For regression, the absolute error threshold for considering a prediction "close enough" (default is 0.1).
    /// For binary/multi-label classification, the probability threshold for converting scores to labels (default is 0.5).
    /// </param>
    /// <returns>A tuple containing the precision, recall, and F1 score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates three important metrics for evaluating prediction 
    /// performance. Precision measures how many of your positive predictions were actually correct 
    /// (true positives ² (true positives + false positives)). Recall measures how many of the actual 
    /// positives your model correctly identified (true positives ² (true positives + false negatives)). 
    /// The F1 score is the harmonic mean of precision and recall, providing a single metric that balances 
    /// both concerns. For binary classification, these metrics are calculated based on the standard 
    /// definitions. For regression problems, the method adapts these concepts by considering predictions 
    /// within a threshold of the actual value as "correct." These metrics are particularly useful when 
    /// the classes are imbalanced or when false positives and false negatives have different costs.
    /// </para>
    /// </remarks>
    public static (T Precision, T Recall, T F1Score) CalculatePrecisionRecallF1(
        Vector<T> actual,
        Vector<T> predicted,
        PredictionType predictionType,
        T? threshold = default)
        => CalculatePrecisionRecallF1(actual, predicted, predictionType, threshold, treatDefaultAsMissing: true);

    public static (T Precision, T Recall, T F1Score) CalculatePrecisionRecallF1(
        Vector<T> actual,
        Vector<T> predicted,
        PredictionType predictionType,
        T? threshold,
        bool treatDefaultAsMissing)
    {
        var truePositives = _numOps.Zero;
        var falsePositives = _numOps.Zero;
        var falseNegatives = _numOps.Zero;
        var defaultRegressionThreshold = _numOps.FromDouble(0.1); // default of 10%
        var classificationThreshold = GetOrDefaultOptionalParameter(threshold, _numOps.FromDouble(0.5), treatDefaultAsMissing);
        var regressionThreshold = GetOrDefaultOptionalParameter(threshold, defaultRegressionThreshold, treatDefaultAsMissing);

        if (predictionType == PredictionType.MultiClass)
        {
            return CalculateMultiClassMacroPrecisionRecallF1(actual, predicted);
        }

        for (int i = 0; i < actual.Length; i++)
        {
            if (predictionType == PredictionType.BinaryClassification)
            {
                var actualLabel = _numOps.GreaterThanOrEquals(actual[i], classificationThreshold) ? _numOps.One : _numOps.Zero;
                var predictedLabel = _numOps.GreaterThanOrEquals(predicted[i], classificationThreshold) ? _numOps.One : _numOps.Zero;

                if (_numOps.Equals(predictedLabel, _numOps.One))
                {
                    if (_numOps.Equals(actualLabel, _numOps.One))
                        truePositives = _numOps.Add(truePositives, _numOps.One);
                    else
                        falsePositives = _numOps.Add(falsePositives, _numOps.One);
                }
                else if (_numOps.Equals(actualLabel, _numOps.One))
                {
                    falseNegatives = _numOps.Add(falseNegatives, _numOps.One);
                }
            }
            else if (predictionType == PredictionType.MultiLabel)
            {
                var actualLabel = _numOps.GreaterThanOrEquals(actual[i], classificationThreshold) ? _numOps.One : _numOps.Zero;
                var predictedLabel = _numOps.GreaterThanOrEquals(predicted[i], classificationThreshold) ? _numOps.One : _numOps.Zero;

                if (_numOps.Equals(predictedLabel, _numOps.One))
                {
                    if (_numOps.Equals(actualLabel, _numOps.One))
                        truePositives = _numOps.Add(truePositives, _numOps.One);
                    else
                        falsePositives = _numOps.Add(falsePositives, _numOps.One);
                }
                else if (_numOps.Equals(actualLabel, _numOps.One))
                {
                    falseNegatives = _numOps.Add(falseNegatives, _numOps.One);
                }
            }
            else // Regression
            {
                var difference = _numOps.Abs(_numOps.Subtract(actual[i], predicted[i]));
                if (_numOps.LessThanOrEquals(difference, regressionThreshold))
                {
                    truePositives = _numOps.Add(truePositives, _numOps.One);
                }
                else if (_numOps.GreaterThan(predicted[i], actual[i]))
                {
                    falsePositives = _numOps.Add(falsePositives, _numOps.One);
                }
                else
                {
                    falseNegatives = _numOps.Add(falseNegatives, _numOps.One);
                }
            }
        }

        var precision = _numOps.Divide(truePositives, _numOps.Add(truePositives, falsePositives));
        var recall = _numOps.Divide(truePositives, _numOps.Add(truePositives, falseNegatives));
        var f1Score = CalculateF1Score(precision, recall);

        return (precision, recall, f1Score);
    }

    private static (T Precision, T Recall, T F1Score) CalculateMultiClassMacroPrecisionRecallF1(Vector<T> actual, Vector<T> predicted)
    {
        var classIds = new HashSet<int>();

        for (int i = 0; i < actual.Length; i++)
        {
            classIds.Add(_numOps.ToInt32(actual[i]));
            classIds.Add(_numOps.ToInt32(predicted[i]));
        }

        if (classIds.Count == 0)
        {
            return (_numOps.Zero, _numOps.Zero, _numOps.Zero);
        }

        T macroPrecision = _numOps.Zero;
        T macroRecall = _numOps.Zero;
        T macroF1 = _numOps.Zero;
        T classCount = _numOps.FromDouble(classIds.Count);

        foreach (int classId in classIds)
        {
            var tp = _numOps.Zero;
            var fp = _numOps.Zero;
            var fn = _numOps.Zero;

            for (int i = 0; i < actual.Length; i++)
            {
                int a = _numOps.ToInt32(actual[i]);
                int p = _numOps.ToInt32(predicted[i]);

                bool actualIsClass = a == classId;
                bool predictedIsClass = p == classId;

                if (predictedIsClass)
                {
                    if (actualIsClass)
                        tp = _numOps.Add(tp, _numOps.One);
                    else
                        fp = _numOps.Add(fp, _numOps.One);
                }
                else if (actualIsClass)
                {
                    fn = _numOps.Add(fn, _numOps.One);
                }
            }

            var precisionDen = _numOps.Add(tp, fp);
            var recallDen = _numOps.Add(tp, fn);

            var precision = _numOps.Equals(precisionDen, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(tp, precisionDen);
            var recall = _numOps.Equals(recallDen, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(tp, recallDen);
            var f1 = CalculateF1Score(precision, recall);

            macroPrecision = _numOps.Add(macroPrecision, precision);
            macroRecall = _numOps.Add(macroRecall, recall);
            macroF1 = _numOps.Add(macroF1, f1);
        }

        return (
            _numOps.Divide(macroPrecision, classCount),
            _numOps.Divide(macroRecall, classCount),
            _numOps.Divide(macroF1, classCount));
    }

    /// <summary>
    /// Calculates the F1 score from precision and recall values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="precision">The precision value.</param>
    /// <param name="recall">The recall value.</param>
    /// <returns>The F1 score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The F1 score is a single metric that balances precision and recall. It's 
    /// calculated as 2 * (precision * recall) / (precision + recall), which is the harmonic mean of 
    /// precision and recall. The F1 score ranges from 0 to 1, with higher values indicating better 
    /// performance. It's particularly useful when you need a single metric to evaluate your model and 
    /// when the classes in your data are imbalanced. The F1 score gives equal weight to precision and 
    /// recall, making it a good choice when both false positives and false negatives are important to 
    /// minimize. If the denominator is zero (which happens when both precision and recall are zero), 
    /// this method returns zero to avoid division by zero errors.
    /// </para>
    /// </remarks>
    public static T CalculateF1Score(T precision, T recall)
    {
        var numerator = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Multiply(precision, recall));
        var denominator = _numOps.Add(precision, recall);

        return _numOps.Equals(denominator, _numOps.Zero) ? _numOps.Zero : _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Calculates a correlation matrix for a set of features.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="features">The matrix of features, where each column represents a feature.</param>
    /// <param name="options">Options for model statistics calculations, including multicollinearity threshold.</param>
    /// <returns>A matrix of correlation coefficients between each pair of features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A correlation matrix shows how each feature in your dataset relates to every 
    /// other feature. Each cell in the matrix contains the Pearson correlation coefficient between two 
    /// features, which ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), 
    /// with 0 indicating no linear relationship. The diagonal of the matrix always contains 1s since each 
    /// feature perfectly correlates with itself. This method also checks for multicollinearity, which 
    /// occurs when features are highly correlated with each other. High multicollinearity can cause problems 
    /// in regression models because it makes it difficult to determine the individual effect of each feature. 
    /// The method logs a warning when it detects correlation above the threshold specified in the options.
    /// </para>
    /// </remarks>
    public static Matrix<T> CalculateCorrelationMatrix(Matrix<T> features, ModelStatsOptions options)
    {
        int featureCount = features.Columns;
        var correlationMatrix = new Matrix<T>(featureCount, featureCount);

        for (int i = 0; i < featureCount; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                if (i == j)
                {
                    correlationMatrix[i, j] = _numOps.One;
                }
                else
                {
                    Vector<T> vectorI = features.GetColumn(i);
                    Vector<T> vectorJ = features.GetColumn(j);

                    T correlation = CalculatePearsonCorrelation(vectorI, vectorJ);
                    correlationMatrix[i, j] = correlation;

                    // Check for multicollinearity
                    if (_numOps.GreaterThan(_numOps.Abs(correlation), _numOps.FromDouble(options.MulticollinearityThreshold)))
                    {
                        // You might want to log this or handle it in some way
                        Console.WriteLine($"High correlation detected between features {i} and {j}: {correlation}");
                    }
                }
            }
        }

        return correlationMatrix;
    }

    /// <summary>
    /// Calculates the Variance Inflation Factor (VIF) for each feature based on a correlation matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="correlationMatrix">The correlation matrix between features.</param>
    /// <param name="options">Options for model statistics calculations, including maximum VIF threshold.</param>
    /// <returns>A list of VIF values, one for each feature.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Variance Inflation Factor (VIF) measures how much the variance of a 
    /// regression coefficient is increased due to multicollinearity (correlation between features). 
    /// For each feature, the VIF is calculated by regressing that feature against all other features 
    /// and then using the formula 1/(1-R²), where R² is the coefficient of determination from that 
    /// regression. A VIF of 1 means there's no correlation between this feature and others, while higher 
    /// values indicate increasing multicollinearity. As a rule of thumb, VIF values above 5-10 are 
    /// considered problematic. This method calculates VIF for each feature and logs a warning when it 
    /// detects values above the threshold specified in the options. High VIF values suggest you might 
    /// want to remove or combine some features to reduce multicollinearity.
    /// </para>
    /// </remarks>
    public static List<T> CalculateVIF(Matrix<T> correlationMatrix, ModelStatsOptions options)
    {
        var vifValues = new List<T>();

        // The correct formula for VIF from a correlation matrix is:
        // VIF_i = (R^-1)_ii where R^-1 is the inverse of the correlation matrix
        // This gives the diagonal elements of the inverse correlation matrix
        var inverseCorrelationMatrix = correlationMatrix.Inverse();

        for (int i = 0; i < correlationMatrix.Rows; i++)
        {
            // VIF is the diagonal element of the inverse correlation matrix
            var vif = inverseCorrelationMatrix[i, i];
            vifValues.Add(vif);

            // Check if VIF exceeds the maximum allowed value
            if (_numOps.GreaterThan(vif, _numOps.FromDouble(options.MaxVIF)))
            {
                Console.WriteLine($"High VIF detected for feature {i}: {vif}");
            }
        }

        return vifValues;
    }

    /// <summary>
    /// Calculates the condition number of a matrix using the specified method.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <param name="options">Options for model statistics calculations, including the condition number calculation method.</param>
    /// <returns>The condition number of the matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported condition number calculation method is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The condition number of a matrix measures how sensitive a linear system is 
    /// to errors or changes in the input. A high condition number indicates that small changes in the 
    /// input can lead to large changes in the output, which is a sign of an ill-conditioned problem. 
    /// In the context of regression, a high condition number suggests that the model might be unstable 
    /// and sensitive to small changes in the data. This method supports several approaches to calculate 
    /// the condition number, including Singular Value Decomposition (SVD), L1 norm, infinity norm, and 
    /// power iteration. Each approach has different computational characteristics, but they all provide 
    /// a measure of the matrix's conditioning. Lower condition numbers (closer to 1) indicate better 
    /// conditioning.
    /// </para>
    /// </remarks>
    public static T CalculateConditionNumber(Matrix<T> matrix, ModelStatsOptions options)
    {
        return options.ConditionNumberMethod switch
        {
            ConditionNumberMethod.SVD => CalculateConditionNumberSVD(matrix),
            ConditionNumberMethod.L1Norm => CalculateConditionNumberL1Norm(matrix),
            ConditionNumberMethod.InfinityNorm => CalculateConditionNumberLInfNorm(matrix),
            ConditionNumberMethod.PowerIteration => CalculateConditionNumberPowerIteration(matrix),
            _ => throw new ArgumentException("Unsupported condition number calculation method", nameof(options))
        };
    }

    /// <summary>
    /// Calculates the condition number of a matrix using Singular Value Decomposition (SVD).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <returns>The condition number calculated as the ratio of the largest to smallest singular value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the condition number using Singular Value Decomposition 
    /// (SVD), which is considered one of the most reliable approaches. SVD decomposes a matrix into three 
    /// components and produces a set of singular values. The condition number is then calculated as the 
    /// ratio of the largest singular value to the smallest. A large ratio indicates that the matrix is 
    /// ill-conditioned. If the smallest singular value is zero, the matrix is singular (non-invertible), 
    /// and the condition number is considered infinite. SVD is computationally intensive but provides a 
    /// robust measure of the matrix's conditioning. This approach is particularly useful for detecting 
    /// numerical instability in regression models.
    /// </para>
    /// </remarks>
    private static T CalculateConditionNumberSVD(Matrix<T> matrix)
    {
        var svd = new SvdDecomposition<T>(matrix);
        var singularValues = svd.S;

        if (singularValues.Length == 0)
        {
            return _numOps.Zero;
        }

        T maxSingularValue = singularValues.Max();
        T minSingularValue = singularValues.Min();

        if (_numOps.Equals(minSingularValue, _numOps.Zero))
        {
            return _numOps.MaxValue;
        }

        return _numOps.Divide(maxSingularValue, minSingularValue);
    }

    /// <summary>
    /// Calculates the condition number of a matrix using the L1 norm.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <returns>The condition number calculated as the product of the L1 norm of the matrix and its inverse.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the condition number using the L1 norm, which is the 
    /// maximum absolute column sum of the matrix. The condition number is computed as the product of the 
    /// L1 norm of the matrix and the L1 norm of its inverse. This approach is generally faster than SVD 
    /// but may be less accurate in some cases. The L1 norm condition number provides an upper bound on 
    /// how much the relative error in the solution can be amplified compared to the relative error in the 
    /// input. Like other condition number measures, higher values indicate greater sensitivity to changes 
    /// in the input data, which can lead to numerical instability in regression models.
    /// </para>
    /// </remarks>
    private static T CalculateConditionNumberL1Norm(Matrix<T> matrix)
    {
        T normA = MatrixL1Norm(matrix);
        Matrix<T> inverseMatrix = matrix.Inverse();
        T normAInverse = MatrixL1Norm(inverseMatrix);

        return _numOps.Multiply(normA, normAInverse);
    }

    /// <summary>
    /// Calculates the condition number of a matrix using the infinity norm.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <returns>The condition number calculated as the product of the infinity norm of the matrix and its inverse.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the condition number using the infinity norm, which 
    /// is the maximum absolute row sum of the matrix. The condition number is computed as the product of 
    /// the infinity norm of the matrix and the infinity norm of its inverse. Like the L1 norm approach, 
    /// this method is generally faster than SVD but may provide a different estimate of the condition number. 
    /// The infinity norm condition number gives another perspective on how errors can propagate through 
    /// the system. Higher values indicate greater sensitivity to changes in the input data. This approach 
    /// is useful when you want a quick estimate of the matrix's conditioning without the computational 
    /// cost of SVD.
    /// </para>
    /// </remarks>
    private static T CalculateConditionNumberLInfNorm(Matrix<T> matrix)
    {
        T normA = MatrixInfinityNorm(matrix);
        Matrix<T> inverseMatrix = matrix.Inverse();
        T normAInverse = MatrixInfinityNorm(inverseMatrix);

        return _numOps.Multiply(normA, normAInverse);
    }

    /// <summary>
    /// Calculates the condition number of a matrix using the power iteration method.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <param name="maxIterations">The maximum number of iterations for the power method (default is 100).</param>
    /// <param name="tolerance">The convergence tolerance (default is 1e-10).</param>
    /// <returns>The condition number calculated as the ratio of the largest to smallest eigenvalue.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the condition number using the power iteration method, 
    /// which approximates the largest and smallest eigenvalues of the matrix. The condition number is then 
    /// calculated as the ratio of the largest eigenvalue to the smallest. The power iteration method is an 
    /// iterative approach that converges to the dominant eigenvalue of a matrix. To find the smallest 
    /// eigenvalue, the method applies power iteration to the inverse of the matrix. This approach can be 
    /// more efficient than SVD for large matrices, but it may require more iterations to converge for 
    /// matrices with closely spaced eigenvalues. The method stops when the eigenvalue estimate changes by 
    /// less than the specified tolerance or when it reaches the maximum number of iterations.
    /// </para>
    /// </remarks>
    private static T CalculateConditionNumberPowerIteration(Matrix<T> matrix, int maxIterations = 100, T? tolerance = default)
    {
        tolerance = GetOrDefaultOptionalParameter(tolerance, _numOps.FromDouble(1e-10));

        T largestEigenvalue = PowerIteration(matrix, maxIterations, tolerance);
        T smallestEigenvalue = PowerIteration(matrix.Inverse(), maxIterations, tolerance);

        return _numOps.Divide(largestEigenvalue, smallestEigenvalue);
    }

    /// <summary>
    /// Calculates the L1 norm (maximum absolute column sum) of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <returns>The L1 norm of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The L1 norm of a matrix is the maximum absolute column sum. To calculate it, 
    /// you sum the absolute values of each element in a column, and then find the maximum of these sums 
    /// across all columns. This norm provides a measure of the "size" of the matrix and is used in various 
    /// numerical analysis contexts, including calculating condition numbers. The L1 norm is one of several 
    /// matrix norms, each with different properties and uses. In the context of regression analysis, the 
    /// L1 norm helps assess the stability of the model by contributing to the calculation of the condition 
    /// number.
    /// </para>
    /// </remarks>
    private static T MatrixL1Norm(Matrix<T> matrix)
    {
        T maxColumnSum = _numOps.Zero;
        for (int j = 0; j < matrix.Columns; j++)
        {
            T columnSum = _numOps.Zero;
            for (int i = 0; i < matrix.Rows; i++)
            {
                columnSum = _numOps.Add(columnSum, _numOps.Abs(matrix[i, j]));
            }
            maxColumnSum = _numOps.GreaterThan(maxColumnSum, columnSum) ? maxColumnSum : columnSum;
        }

        return maxColumnSum;
    }

    /// <summary>
    /// Calculates the infinity norm (maximum absolute row sum) of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <returns>The infinity norm of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The infinity norm of a matrix is the maximum absolute row sum. To calculate 
    /// it, you sum the absolute values of each element in a row, and then find the maximum of these sums 
    /// across all rows. Like the L1 norm, the infinity norm provides a measure of the "size" of the matrix 
    /// but from a different perspective. It's used in various numerical analysis contexts, including 
    /// calculating condition numbers. The infinity norm is particularly useful when you're interested in 
    /// the maximum effect that the matrix can have on a vector in terms of its largest component. In 
    /// regression analysis, it helps assess the stability of the model through the condition number 
    /// calculation.
    /// </para>
    /// </remarks>
    private static T MatrixInfinityNorm(Matrix<T> matrix)
    {
        T maxRowSum = _numOps.Zero;
        for (int i = 0; i < matrix.Rows; i++)
        {
            T rowSum = _numOps.Zero;
            for (int j = 0; j < matrix.Columns; j++)
            {
                rowSum = _numOps.Add(rowSum, _numOps.Abs(matrix[i, j]));
            }
            maxRowSum = _numOps.GreaterThan(maxRowSum, rowSum) ? maxRowSum : rowSum;
        }

        return maxRowSum;
    }

    /// <summary>
    /// Implements the power iteration method to find the dominant eigenvalue of a matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="matrix">The matrix to analyze.</param>
    /// <param name="maxIterations">The maximum number of iterations.</param>
    /// <param name="tolerance">The convergence tolerance.</param>
    /// <returns>The dominant eigenvalue of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The power iteration method is an iterative technique to find the dominant 
    /// eigenvalue (the eigenvalue with the largest absolute value) of a matrix. It works by repeatedly 
    /// multiplying a vector by the matrix and normalizing the result. Over time, this process converges 
    /// to the eigenvector corresponding to the dominant eigenvalue, and the eigenvalue itself can be 
    /// calculated from this vector. The method starts with a random vector and continues until the 
    /// eigenvalue estimate changes by less than the specified tolerance or until it reaches the maximum 
    /// number of iterations. Power iteration is particularly useful for large, sparse matrices where more 
    /// direct methods like SVD might be computationally expensive.
    /// </para>
    /// </remarks>
    private static T PowerIteration(Matrix<T> matrix, int maxIterations, T tolerance)
    {
        Vector<T> v = Vector<T>.CreateRandom(matrix.Rows);
        T eigenvalue = _numOps.Zero;

        for (int i = 0; i < maxIterations; i++)
        {
            Vector<T> Av = matrix * v;
            T newEigenvalue = v.DotProduct(Av);

            if (_numOps.LessThan(_numOps.Abs(_numOps.Subtract(newEigenvalue, eigenvalue)), tolerance))
            {
                return newEigenvalue;
            }

            eigenvalue = newEigenvalue;
            v = Av.Normalize();
        }

        return eigenvalue;
    }

    /// <summary>
    /// Calculates the Deviance Information Criterion (DIC) for Bayesian model comparison.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="modelStats">The model statistics object containing necessary information.</param>
    /// <returns>The DIC value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Deviance Information Criterion (DIC) is a hierarchical modeling 
    /// generalization of the AIC and BIC, used for Bayesian model comparison. It's calculated as 
    /// D(?²) + 2pD, where D(?²) is the deviance at the posterior mean (a measure of how well the model 
    /// fits the data), and pD is the effective number of parameters (a measure of model complexity). 
    /// Lower DIC values indicate better models. DIC is particularly useful for comparing Bayesian models 
    /// where the posterior distributions have been obtained using Markov Chain Monte Carlo (MCMC) methods. 
    /// Like AIC and BIC, DIC balances model fit against complexity, but it's specifically designed for 
    /// Bayesian models where the effective number of parameters might not be clear due to prior information 
    /// and hierarchical structure.
    /// </para>
    /// </remarks>
    public static T CalculateDIC<TInput, TOutput>(ModelStats<T, TInput, TOutput> modelStats)
    {
        // DIC = D(?²) + 2pD
        // where D(?²) is the deviance at the posterior mean, and pD is the effective number of parameters
        var devianceAtPosteriorMean = _numOps.Multiply(_numOps.FromDouble(-2), modelStats.LogLikelihood);
        var effectiveNumberOfParameters = modelStats.EffectiveNumberOfParameters;

        return _numOps.Add(devianceAtPosteriorMean, _numOps.Multiply(_numOps.FromDouble(2), effectiveNumberOfParameters));
    }

    /// <summary>
    /// Calculates the Widely Applicable Information Criterion (WAIC) for Bayesian model comparison.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="modelStats">The model statistics object containing necessary information.</param>
    /// <returns>The WAIC value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Widely Applicable Information Criterion (WAIC) is a fully Bayesian 
    /// approach to estimating the out-of-sample expectation. It's calculated as -2 * (lppd - pWAIC), 
    /// where lppd is the log pointwise predictive density (a measure of how well the model fits the data), 
    /// and pWAIC is the effective number of parameters (a measure of model complexity). Lower WAIC values 
    /// indicate better models. WAIC is considered an improvement over DIC because it's fully Bayesian, 
    /// uses the entire posterior distribution, and is invariant to parameterization. It's particularly 
    /// useful for hierarchical models and models with many parameters. Like other information criteria, 
    /// WAIC helps you select the model that best balances fit and complexity.
    /// </para>
    /// </remarks>
    public static T CalculateWAIC<TInput, TOutput>(ModelStats<T, TInput, TOutput> modelStats)
    {
        // WAIC = -2 * (lppd - pWAIC)
        // where lppd is the log pointwise predictive density, and pWAIC is the effective number of parameters
        var lppd = modelStats.LogPointwisePredictiveDensity;
        var pWAIC = modelStats.EffectiveNumberOfParameters;

        return _numOps.Multiply(_numOps.FromDouble(-2), _numOps.Subtract(lppd, pWAIC));
    }

    /// <summary>
    /// Calculates the Leave-One-Out Cross-Validation (LOO-CV) criterion for Bayesian model comparison.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="modelStats">The model statistics object containing necessary information.</param>
    /// <returns>The LOO-CV value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Leave-One-Out Cross-Validation (LOO-CV) is a method for estimating how well 
    /// a model will perform on unseen data. It works by fitting the model multiple times, each time leaving 
    /// out one observation, and then predicting that observation with the model trained on all other data. 
    /// This method calculates the LOO-CV criterion as -2 times the sum of the logarithms of these leave-one-out 
    /// predictive densities. Lower values indicate better models. LOO-CV is particularly useful because it 
    /// directly estimates out-of-sample prediction accuracy without requiring you to hold out a separate 
    /// validation set. It's more computationally intensive than information criteria like AIC or BIC, but 
    /// it often provides a more accurate estimate of a model's predictive performance.
    /// </para>
    /// </remarks>
    public static T CalculateLOO<TInput, TOutput>(ModelStats<T, TInput, TOutput> modelStats)
    {
        // LOO = -2 * (S log(p(yi | y-i)))
        // where p(yi | y-i) is the leave-one-out predictive density for the i-th observation
        var looSum = modelStats.LeaveOneOutPredictiveDensities.Aggregate(_numOps.Zero,
            (acc, density) => _numOps.Add(acc, _numOps.Log(density))
        );

        return _numOps.Multiply(_numOps.FromDouble(-2), looSum);
    }

    /// <summary>
    /// Calculates a posterior predictive p-value for model checking.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="modelStats">The model statistics object containing necessary information.</param>
    /// <returns>The posterior predictive p-value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Posterior predictive checking is a way to assess whether a Bayesian model 
    /// fits the observed data well. This method calculates a posterior predictive p-value, which is the 
    /// proportion of simulated datasets (generated from the posterior distribution) that are more extreme 
    /// than the observed data according to some test statistic. A p-value close to 0.5 suggests the model 
    /// fits well, while values close to 0 or 1 indicate poor fit. For example, if 95% of simulated datasets 
    /// have a test statistic more extreme than the observed data (p-value = 0.95), this suggests the model 
    /// doesn't capture some important aspect of the data. Posterior predictive checks are valuable because 
    /// they directly assess the model's ability to generate data similar to what was observed.
    /// </para>
    /// </remarks>
    public static T CalculatePosteriorPredictiveCheck<TInput, TOutput>(ModelStats<T, TInput, TOutput> modelStats)
    {
        // Calculate the proportion of posterior predictive samples that are more extreme than the observed data
        var observedStatistic = modelStats.ObservedTestStatistic;
        var posteriorPredictiveSamples = modelStats.PosteriorPredictiveSamples;
        var moreExtremeSamples = posteriorPredictiveSamples.Count(sample => _numOps.GreaterThan(sample, observedStatistic));

        return _numOps.Divide(_numOps.FromDouble(moreExtremeSamples), _numOps.FromDouble(posteriorPredictiveSamples.Count));
    }

    /// <summary>
    /// Calculates the Bayes Factor for comparing two models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="modelStats">The model statistics object containing necessary information.</param>
    /// <returns>The Bayes Factor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Bayes Factor is a ratio that compares the evidence for two competing 
    /// models. It's calculated as the ratio of the marginal likelihood of one model to the marginal 
    /// likelihood of another (reference) model. A Bayes Factor greater than 1 indicates evidence in favor 
    /// of the first model, while a value less than 1 favors the reference model. The strength of evidence 
    /// is often interpreted using guidelines: 1-3 is considered weak evidence, 3-10 is substantial, 10-30 
    /// is strong, 30-100 is very strong, and >100 is decisive evidence. Unlike p-values, Bayes Factors 
    /// can provide evidence in favor of the null hypothesis and allow for direct comparison of non-nested 
    /// models. They're a fundamental tool in Bayesian model selection.
    /// </para>
    /// </remarks>
    public static T CalculateBayesFactor<TInput, TOutput>(ModelStats<T, TInput, TOutput> modelStats)
    {
        // Bayes Factor = P(D|M1) / P(D|M2)
        // where P(D|M1) is the marginal likelihood of the current model and P(D|M2) is the marginal likelihood of a reference model
        var currentModelMarginalLikelihood = modelStats.MarginalLikelihood;
        var referenceModelMarginalLikelihood = modelStats.ReferenceModelMarginalLikelihood;

        return _numOps.Divide(currentModelMarginalLikelihood, referenceModelMarginalLikelihood);
    }

    /// <summary>
    /// Calculates the likelihood of an observed value given a predicted value.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed value.</param>
    /// <param name="predicted">The predicted value from a model.</param>
    /// <returns>The likelihood value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Likelihood measures how probable the observed data is under a specific model. 
    /// This method calculates the likelihood for a single observation using a Gaussian (normal) distribution 
    /// centered at the predicted value. It computes exp(-0.5 * residual²), where residual is the difference 
    /// between the actual and predicted values. Higher likelihood values indicate that the model's prediction 
    /// is closer to the actual value. Likelihood is a fundamental concept in statistics and forms the basis 
    /// for many estimation methods, including maximum likelihood estimation. In Bayesian statistics, the 
    /// likelihood function is combined with prior distributions to obtain posterior distributions, which 
    /// are used for inference and prediction.
    /// </para>
    /// </remarks>
    public static T CalculateLikelihood(T actual, T predicted)
    {
        T residual = _numOps.Subtract(actual, predicted);
        return _numOps.Exp(_numOps.Multiply(_numOps.FromDouble(-0.5), _numOps.Multiply(residual, residual)));
    }

    /// <summary>
    /// Generates samples from the posterior predictive distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="features">The feature matrix.</param>
    /// <param name="coefficients">The model coefficients.</param>
    /// <param name="numSamples">The number of samples to generate.</param>
    /// <returns>A collection of test statistics calculated from the posterior predictive samples.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Posterior predictive sampling is a way to generate new data that might be 
    /// observed if the model is correct. This method creates samples by first calculating predicted values 
    /// using the model's coefficients, then adding random noise to simulate the natural variability in the 
    /// data. For each sample, it calculates a test statistic that compares the predicted values to the 
    /// noisy predictions. These samples form the posterior predictive distribution, which can be used for 
    /// model checking (as in posterior predictive checks) or to make predictions with uncertainty estimates. 
    /// This approach is valuable because it accounts for both the uncertainty in the model parameters and 
    /// the inherent randomness in the data generation process.
    /// </para>
    /// </remarks>
    public static IEnumerable<T> GeneratePosteriorPredictiveSamples(Matrix<T> features, Vector<T> coefficients, int numSamples)
    {
        var samples = new List<T>();

        for (int i = 0; i < numSamples; i++)
        {
            var predictedValues = features.Multiply(coefficients);
            var noise = Vector<T>.CreateRandom(predictedValues.Length);
            samples.Add(CalculateObservedTestStatistic(predictedValues, predictedValues.Add(noise)));
        }

        return samples;
    }

    /// <summary>
    /// Calculates leave-one-out predictive densities for each observation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="features">The feature matrix.</param>
    /// <param name="actualValues">The actual observed values.</param>
    /// <param name="modelFitFunction">A function that fits the model and returns coefficients.</param>
    /// <returns>A list of leave-one-out predictive densities.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method implements leave-one-out cross-validation by systematically 
    /// excluding each observation, fitting the model on the remaining data, and then calculating how well 
    /// that model predicts the excluded observation. For each observation, it removes that data point, 
    /// trains the model on the remaining data, predicts the value for the excluded point, and calculates 
    /// the likelihood of the actual value given this prediction. The result is a list of predictive 
    /// densities, one for each observation. These values are used in the LOO-CV criterion to assess the 
    /// model's predictive performance. This approach is computationally intensive but provides a robust 
    /// estimate of out-of-sample prediction accuracy.
    /// </para>
    /// </remarks>
    public static List<T> CalculateLeaveOneOutPredictiveDensities(Matrix<T> features, Vector<T> actualValues, Func<Matrix<T>, Vector<T>, Vector<T>> modelFitFunction)
    {
        var looPredictiveDensities = new List<T>();

        for (int i = 0; i < features.Rows; i++)
        {
            var trainingFeatures = features.RemoveRow(i);
            var trainingValues = actualValues.RemoveAt(i);
            var coefficients = modelFitFunction(trainingFeatures, trainingValues);
            var predictedValue = features.GetRow(i).DotProduct(coefficients);
            looPredictiveDensities.Add(CalculateLikelihood(actualValues[i], predictedValue));
        }

        return looPredictiveDensities;
    }

    /// <summary>
    /// Calculates the log pointwise predictive density (LPPD) for a model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actualValues">The actual observed values.</param>
    /// <param name="predictedValues">The predicted values from a model.</param>
    /// <returns>The log pointwise predictive density.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The log pointwise predictive density (LPPD) measures how well a model's 
    /// predictions match the observed data. It's calculated by summing the logarithms of the likelihoods 
    /// for each observation. Higher LPPD values indicate better fit. This metric is used in information 
    /// criteria like WAIC (Widely Applicable Information Criterion) to assess model performance. Unlike 
    /// simple measures like mean squared error, LPPD accounts for the uncertainty in predictions by using 
    /// the full likelihood function. It's particularly useful in Bayesian statistics because it can be 
    /// calculated directly from posterior samples. The LPPD forms the basis for more complex model 
    /// comparison metrics that balance fit against model complexity.
    /// </para>
    /// </remarks>
    public static T CalculateLogPointwisePredictiveDensity(Vector<T> actualValues, Vector<T> predictedValues)
    {
        T lppd = _numOps.Zero;

        for (int i = 0; i < actualValues.Length; i++)
        {
            T likelihood = CalculateLikelihood(actualValues[i], predictedValues[i]);
            lppd = _numOps.Add(lppd, _numOps.Log(likelihood));
        }

        return lppd;
    }

    /// <summary>
    /// Calculates the median absolute deviation (MAD) of a set of values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The values to analyze.</param>
    /// <returns>The median absolute deviation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The median absolute deviation (MAD) is a robust measure of variability in 
    /// a dataset. It's calculated by finding the median of the absolute deviations from the data's median. 
    /// First, calculate the median of the data. Then, calculate how far each value is from this median 
    /// (the absolute deviations). Finally, find the median of these absolute deviations. MAD is less 
    /// sensitive to outliers than the standard deviation, making it useful for datasets with extreme values. 
    /// In robust statistics, MAD is often used as an alternative to standard deviation. It can be scaled 
    /// (multiplied by 1.4826) to make it comparable to standard deviation for normally distributed data.
    /// </para>
    /// </remarks>
    public static T CalculateMedianAbsoluteDeviation(Vector<T> values)
    {
        T median = CalculateMedian(values);
        Vector<T> absoluteDeviations = new Vector<T>(values.Length);

        for (int i = 0; i < values.Length; i++)
        {
            absoluteDeviations[i] = _numOps.Abs(_numOps.Subtract(values[i], median));
        }

        return CalculateMedian(absoluteDeviations);
    }

    /// <summary>
    /// Calculates the difference between peak values in two distributions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x1">The x-coordinates of the first distribution.</param>
    /// <param name="y1">The y-coordinates (values) of the first distribution.</param>
    /// <param name="x2">The x-coordinates of the second distribution.</param>
    /// <param name="y2">The y-coordinates (values) of the second distribution.</param>
    /// <returns>The absolute difference between the x-coordinates of the peak values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the difference between the locations of peak values in two 
    /// distributions. For each distribution, it identifies the x-coordinate where the y-value is at its 
    /// maximum (the peak), then calculates the absolute difference between these two x-coordinates. This 
    /// is useful for comparing distributions to see how far apart their modes (most common values) are. 
    /// For example, in spectroscopy, you might want to know how much a peak has shifted between two spectra. 
    /// In statistics, this could help identify shifts in the central tendency of distributions. The method 
    /// works with any paired x-y data that represents a distribution or curve.
    /// </para>
    /// </remarks>
    public static T CalculatePeakDifference(Vector<T> x1, Vector<T> y1, Vector<T> x2, Vector<T> y2)
    {
        T peak1 = FindPeakValue(x1, y1);
        T peak2 = FindPeakValue(x2, y2);

        return _numOps.Abs(_numOps.Subtract(peak1, peak2));
    }

    /// <summary>
    /// Finds the x-coordinate of the peak value in a distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The x-coordinates of the distribution.</param>
    /// <param name="y">The y-coordinates (values) of the distribution.</param>
    /// <returns>The x-coordinate where the y-value is at its maximum.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method identifies the location of the peak (maximum value) in a 
    /// distribution or curve. It scans through all the y-values to find the maximum, then returns the 
    /// corresponding x-coordinate. This is useful for finding the mode of a distribution (the most common 
    /// value) or the location of a peak in any curve represented by x-y data. For example, in a probability 
    /// density function, the peak represents the most likely value. In a spectrum, the peak might represent 
    /// the wavelength with the strongest signal. This method assumes that the x and y vectors are properly 
    /// paired and that higher y-values represent more significant points in the distribution.
    /// </para>
    /// </remarks>
    private static T FindPeakValue(Vector<T> x, Vector<T> y)
    {
        T maxValue = _numOps.MinValue;
        int maxIndex = 0;

        for (int i = 0; i < y.Length; i++)
        {
            if (_numOps.GreaterThan(y[i], maxValue))
            {
                maxValue = y[i];
                maxIndex = i;
            }
        }

        return x[maxIndex];
    }

    /// <summary>
    /// Calculates the log-likelihood of a model given actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actualValues">The actual observed values.</param>
    /// <param name="predictedValues">The predicted values from a model.</param>
    /// <returns>The log-likelihood value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The log-likelihood measures how well a model fits the observed data. It's 
    /// calculated by summing the logarithms of the absolute residuals (differences between actual and 
    /// predicted values) and multiplying by -0.5. Higher log-likelihood values (closer to zero) indicate 
    /// better fit. Log-likelihood is used in many statistical contexts, including maximum likelihood 
    /// estimation and information criteria like AIC and BIC. Working with log-likelihood instead of 
    /// likelihood directly helps avoid numerical underflow with very small probability values. This 
    /// particular implementation assumes a Laplace distribution for the errors, which is more robust to 
    /// outliers than the normal distribution typically used in log-likelihood calculations.
    /// </para>
    /// </remarks>
    public static T CalculateLogLikelihood(Vector<T> actualValues, Vector<T> predictedValues)
    {
        T logLikelihood = _numOps.Zero;

        for (int i = 0; i < actualValues.Length; i++)
        {
            T residual = _numOps.Subtract(actualValues[i], predictedValues[i]);
            logLikelihood = _numOps.Add(logLikelihood, _numOps.Log(_numOps.Abs(residual)));
        }

        return _numOps.Multiply(_numOps.FromDouble(-0.5), logLikelihood);
    }

    /// <summary>
    /// Calculates the effective number of parameters in a model using the trace of the hat matrix.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="features">The feature matrix.</param>
    /// <param name="coefficients">The model coefficients.</param>
    /// <returns>The effective number of parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The effective number of parameters measures the complexity of a model, 
    /// accounting for regularization and prior information that might reduce the effective complexity 
    /// below the actual number of parameters. This method calculates it using the trace of the "hat matrix" 
    /// (H = X(X'X)^(-1)X'), which maps the observed values to the fitted values in linear regression. 
    /// The trace of this matrix equals the number of parameters in ordinary least squares regression, but 
    /// can be less in regularized models. This metric is important in information criteria like AIC, BIC, 
    /// and DIC, which penalize model complexity to prevent overfitting. It's particularly useful for 
    /// hierarchical and regularized models where the nominal number of parameters might overstate the 
    /// model's complexity.
    /// </para>
    /// </remarks>
    public static T CalculateEffectiveNumberOfParameters(Matrix<T> features, Vector<T> coefficients)
    {
        // Calculate the hat matrix (H = X(X'X)^(-1)X')
        var transposeFeatures = features.Transpose();
        var inverseMatrix = (transposeFeatures * features).Inverse();
        var hatMatrix = features * (inverseMatrix * transposeFeatures);

        // The effective number of parameters is the trace of the hat matrix
        return hatMatrix.Diagonal().Sum();
    }

    /// <summary>
    /// Calculates a test statistic comparing actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actualValues">The actual observed values.</param>
    /// <param name="predictedValues">The predicted values from a model.</param>
    /// <param name="testType">The type of test statistic to calculate (default is ChiSquare).</param>
    /// <returns>The calculated test statistic.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported test statistic type is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates a test statistic that measures the discrepancy between 
    /// observed and predicted values. It supports two types of statistics: Chi-square and F-test. The 
    /// Chi-square statistic sums the squared residuals divided by the predicted values, which is useful 
    /// for assessing goodness of fit, especially for count data. The F-test statistic compares the variance 
    /// explained by the model to the unexplained variance, which helps determine if the model is significantly 
    /// better than a simpler model. Both statistics can be used in hypothesis testing to determine if the 
    /// model's predictions are significantly different from what would be expected by chance. Higher values 
    /// generally indicate a greater discrepancy between the model and the data.
    /// </para>
    /// </remarks>
    public static T CalculateObservedTestStatistic(Vector<T> actualValues, Vector<T> predictedValues, TestStatisticType testType = TestStatisticType.ChiSquare)
    {
        switch (testType)
        {
            case TestStatisticType.ChiSquare:
                // Chi-square test statistic
                T chiSquare = _numOps.Zero;
                for (int i = 0; i < actualValues.Length; i++)
                {
                    T residual = _numOps.Subtract(actualValues[i], predictedValues[i]);
                    chiSquare = _numOps.Add(chiSquare, _numOps.Divide(_numOps.Multiply(residual, residual), predictedValues[i]));
                }
                return chiSquare;

            case TestStatisticType.FTest:
                // F-test statistic
                T sst = CalculateTotalSumOfSquares(actualValues);
                T sse = CalculateResidualSumOfSquares(actualValues, predictedValues);
                int dfModel = predictedValues.Length - 1;
                int dfResidual = actualValues.Length - predictedValues.Length;

                T msModel = _numOps.Divide(_numOps.Subtract(sst, sse), _numOps.FromDouble(dfModel));
                T msResidual = _numOps.Divide(sse, _numOps.FromDouble(dfResidual));

                return _numOps.Divide(msModel, msResidual);

            default:
                throw new ArgumentException("Unsupported test statistic type");
        }
    }

    /// <summary>
    /// Calculates an approximation of the marginal likelihood for a model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actualValues">The actual observed values.</param>
    /// <param name="predictedValues">The predicted values from a model.</param>
    /// <param name="numParameters">The number of parameters in the model.</param>
    /// <returns>The approximated marginal likelihood.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The marginal likelihood (also called the evidence) is a key quantity in 
    /// Bayesian statistics that measures how well a model explains the observed data, integrating over 
    /// all possible parameter values. This method approximates the marginal likelihood using the Bayesian 
    /// Information Criterion (BIC). It first calculates the log-likelihood of the model, then computes 
    /// the BIC as -2*log-likelihood + k*log(n), where k is the number of parameters and n is the sample 
    /// size. Finally, it converts this to an approximation of the marginal likelihood using the formula 
    /// exp(-0.5*BIC). The marginal likelihood is used in Bayes factors for model comparison, with higher 
    /// values indicating models that better explain the data while accounting for model complexity.
    /// </para>
    /// </remarks>
    public static T CalculateMarginalLikelihood(Vector<T> actualValues, Vector<T> predictedValues, int numParameters)
    {
        // Calculate log-likelihood
        T logLikelihood = CalculateLogLikelihood(actualValues, predictedValues);

        // Calculate BIC (Bayesian Information Criterion)
        T n = _numOps.FromDouble(actualValues.Length);
        T k = _numOps.FromDouble(numParameters);
        T bic = _numOps.Multiply(_numOps.FromDouble(-2), logLikelihood);
        bic = _numOps.Add(bic, _numOps.Multiply(k, _numOps.Log(n)));

        // Approximate marginal likelihood using BIC
        return _numOps.Exp(_numOps.Multiply(_numOps.FromDouble(-0.5), bic));
    }

    /// <summary>
    /// Calculates the total sum of squares (SST) for a set of values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The values to analyze.</param>
    /// <returns>The total sum of squares.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The total sum of squares (SST) measures the total variation in a dataset. 
    /// It's calculated by summing the squared differences between each value and the mean of all values. 
    /// SST represents how much the data points vary from their average, regardless of any model. In 
    /// regression analysis, SST is the total variance that a model attempts to explain. It can be 
    /// partitioned into the explained sum of squares (SSR, the variation explained by the model) and 
    /// the residual sum of squares (SSE, the unexplained variation). The ratio SSR/SST gives the 
    /// coefficient of determination (R²), which indicates the proportion of variance explained by the model.
    /// </para>
    /// </remarks>
    public static T CalculateTotalSumOfSquares(Vector<T> values)
    {
        T mean = values.Mean();
        var meanVector = Vector<T>.CreateDefault(values.Length, mean);
        var differences = values.Subtract(meanVector);

        return differences.DotProduct(differences);
    }

    /// <summary>
    /// Calculates the residual sum of squares (SSE) between actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The residual sum of squares.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The residual sum of squares (SSE) measures how much variation in the data 
    /// remains unexplained by a model. It's calculated by summing the squared differences between each 
    /// actual value and its corresponding predicted value. Lower SSE values indicate a better fit to the 
    /// data. SSE is used in many statistical calculations, including mean squared error (MSE = SSE/n), 
    /// R-squared (1 - SSE/SST), and F-tests for model comparison. It's also used in calculating standard 
    /// errors and confidence intervals. The SSE is particularly important because it penalizes larger 
    /// errors more heavily than smaller ones, making it sensitive to outliers and large prediction errors.
    /// </para>
    /// </remarks>
    public static T CalculateResidualSumOfSquares(Vector<T> actual, Vector<T> predicted)
    {
        var residuals = actual.Subtract(predicted);
        return residuals.DotProduct(residuals);
    }

    /// <summary>
    /// Calculates the residuals (errors) between actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>A vector of residuals.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Residuals are the differences between observed values and the values predicted 
    /// by a model. They represent the part of the data that the model doesn't explain. This method simply 
    /// subtracts each predicted value from its corresponding actual value. Analyzing residuals is a crucial 
    /// step in assessing model fit. In a good model, residuals should be randomly distributed around zero 
    /// with no obvious patterns. Patterns in residuals (like trends, curves, or changing variance) can 
    /// indicate that the model is missing important structure in the data. Residuals are used in many 
    /// diagnostic plots and tests, including residual plots, Q-Q plots, and tests for autocorrelation 
    /// like the Durbin-Watson test.
    /// </para>
    /// </remarks>
    public static Vector<T> CalculateResiduals(Vector<T> actual, Vector<T> predicted)
    {
        return actual.Subtract(predicted);
    }

    /// <summary>
    /// Generates samples from the posterior predictive distribution for a model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="featureCount">The number of features (parameters) in the model.</param>
    /// <param name="numSamples">The number of samples to generate (default is 1000).</param>
    /// <returns>A list of samples from the posterior predictive distribution.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Posterior predictive sampling generates new data that might be observed if 
    /// the model is correct, accounting for both parameter uncertainty and random variation. This method 
    /// estimates the error variance from the residuals, then generates samples by adding random noise to 
    /// the model's predictions. Each sample is an average of n simulated values, where n is the sample size. 
    /// These samples form a distribution that represents our uncertainty about future observations. Posterior 
    /// predictive samples are useful for model checking (comparing the distribution of simulated data to 
    /// actual data) and for making predictions with appropriate uncertainty intervals. This approach is 
    /// particularly valuable in Bayesian statistics but can be used with any regression model.
    /// </para>
    /// </remarks>
    public static List<T> CalculatePosteriorPredictiveSamples(Vector<T> actual, Vector<T> predicted, int featureCount, int numSamples = 1000)
    {
        var n = actual.Length;
        var rss = CalculateResidualSumOfSquares(actual, predicted);
        var sigma2 = _numOps.Divide(rss, _numOps.FromDouble(n - featureCount));
        var standardError = _numOps.Sqrt(sigma2);

        var random = RandomHelper.CreateSecureRandom();
        var samples = new List<T>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            var sample = _numOps.Zero;
            for (int j = 0; j < n; j++)
            {
                var noise = _numOps.Multiply(standardError, _numOps.FromDouble(random.NextGaussian()));
                sample = _numOps.Add(sample, _numOps.Add(predicted[j], noise));
            }
            samples.Add(_numOps.Divide(sample, _numOps.FromDouble(n)));
        }

        return samples;
    }

    /// <summary>
    /// Calculates the marginal likelihood for a reference model (intercept-only model).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed values.</param>
    /// <returns>The marginal likelihood of the reference model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the marginal likelihood for a simple reference model 
    /// that predicts the mean of the data for all observations (an intercept-only model). It first calculates 
    /// the mean and variance of the actual values, then computes the log marginal likelihood using a formula 
    /// based on the normal distribution. The result is exponentiated to get the marginal likelihood. This 
    /// reference model serves as a baseline for comparison in Bayes factor calculations. By comparing the 
    /// marginal likelihood of a more complex model to this reference model, you can assess whether the 
    /// additional complexity is justified by improved fit to the data. A Bayes factor greater than 1 
    /// indicates evidence in favor of the more complex model.
    /// </para>
    /// </remarks>
    public static T CalculateReferenceModelMarginalLikelihood(Vector<T> actual)
    {
        var n = actual.Length;
        var mean = actual.Mean();
        var variance = _numOps.Divide(CalculateTotalSumOfSquares(actual), _numOps.FromDouble(n - 1));

        // Calculate log marginal likelihood for the reference model (intercept-only model)
        var logML = _numOps.Multiply(_numOps.FromDouble(-0.5), _numOps.FromDouble(n * Math.Log(2 * Math.PI)));
        logML = _numOps.Subtract(logML, _numOps.Multiply(_numOps.FromDouble(0.5 * n), _numOps.Log(variance)));
        logML = _numOps.Subtract(logML, _numOps.Divide(_numOps.FromDouble(n - 1), _numOps.FromDouble(2)));

        return _numOps.Exp(logML);
    }

    /// <summary>
    /// Calculates the area under the precision-recall curve (PR AUC).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The area under the precision-recall curve.</returns>
    /// <exception cref="ArgumentException">Thrown when inputs have different lengths or when there are no positive or negative samples.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The precision-recall AUC measures how well a model can identify positive cases 
    /// without raising too many false alarms. It's particularly useful for imbalanced datasets where negative 
    /// cases are much more common than positive ones. This method calculates the AUC by sorting predictions 
    /// from highest to lowest, then tracking how precision and recall change as the threshold is lowered. 
    /// The area is calculated using the trapezoidal rule. PR AUC ranges from 0 to 1, with higher values 
    /// indicating better performance. Unlike ROC AUC, which can be misleadingly high with imbalanced data, 
    /// PR AUC focuses on the positive class and provides a more realistic assessment when the positive class 
    /// is rare but important to identify correctly.
    /// </para>
    /// </remarks>
    public static T CalculatePrecisionRecallAUC(Vector<T> actual, Vector<T> predicted)
    {
        if (actual.Length != predicted.Length)
            throw new ArgumentException("Actual and predicted vectors must have the same length.");

        var sortedPairs = actual.Zip(predicted, (a, p) => new { Actual = a, Predicted = p })
                                .OrderByDescending(pair => pair.Predicted)
                                .ToList();

        T totalPositives = _numOps.FromDouble(actual.Where(a => _numOps.GreaterThan(a, _numOps.Zero)).Length);
        T totalNegatives = _numOps.Subtract(_numOps.FromDouble(actual.Length), totalPositives);

        if (_numOps.Equals(totalPositives, _numOps.Zero) || _numOps.Equals(totalNegatives, _numOps.Zero))
        {
            // Return 0 for regression data or data without both classes
            // AUC is a classification metric and is not meaningful for regression
            return _numOps.Zero;
        }

        T truePositives = _numOps.Zero;
        T falsePositives = _numOps.Zero;
        T auc = _numOps.Zero;
        T prevRecall = _numOps.Zero;
        T prevPrecision = _numOps.One;

        foreach (var pair in sortedPairs)
        {
            if (_numOps.GreaterThan(pair.Actual, _numOps.Zero))
            {
                truePositives = _numOps.Add(truePositives, _numOps.One);
            }
            else
            {
                falsePositives = _numOps.Add(falsePositives, _numOps.One);
            }

            T recall = _numOps.Divide(truePositives, totalPositives);
            T precision = _numOps.Divide(truePositives, _numOps.Add(truePositives, falsePositives));

            T deltaRecall = _numOps.Subtract(recall, prevRecall);
            T avgPrecision = _numOps.Divide(_numOps.Add(precision, prevPrecision), _numOps.FromDouble(2.0));
            auc = _numOps.Add(auc, _numOps.Multiply(deltaRecall, avgPrecision));

            prevRecall = recall;
            prevPrecision = precision;
        }

        return auc;
    }

    /// <summary>
    /// Generates a set of unique threshold values from predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="predictedValues">The predicted values from a model.</param>
    /// <returns>A vector of unique threshold values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This utility method extracts all unique values from a set of predictions to 
    /// use as thresholds for classification. When evaluating a classification model that outputs continuous 
    /// scores (like probabilities), you need to choose a threshold above which predictions are considered 
    /// positive. Instead of arbitrarily selecting thresholds, this method identifies all possible thresholds 
    /// by finding the unique values in the predictions. These thresholds can then be used to calculate 
    /// performance metrics at different operating points, allowing you to create curves like ROC or 
    /// precision-recall curves. This approach ensures that you evaluate the model at all meaningful 
    /// threshold values without redundancy.
    /// </para>
    /// </remarks>
    public static Vector<T> GenerateThresholds(Vector<T> predictedValues)
    {
        var uniqueValues = new HashSet<T>(predictedValues);
        var sortedValues = uniqueValues.OrderByDescending(v => _numOps.ToDouble(v)).ToList();
        var thresholds = new Vector<T>(sortedValues.Count);
        for (int i = 0; i < sortedValues.Count; i++)
        {
            thresholds[i] = sortedValues[i];
        }

        return thresholds;
    }

    /// <summary>
    /// Calculates the Receiver Operating Characteristic (ROC) curve for a set of predictions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actualValues">The actual observed values.</param>
    /// <param name="predictedValues">The predicted values from a model.</param>
    /// <returns>A tuple containing vectors of false positive rates and true positive rates.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The ROC curve plots the true positive rate (sensitivity) against the false 
    /// positive rate (1-specificity) at various threshold settings. This method calculates these rates 
    /// for each unique threshold in the predicted values. The resulting curve shows the tradeoff between 
    /// catching true positives and avoiding false positives. A perfect classifier would reach the top-left 
    /// corner (100% sensitivity, 0% false positives), while a random classifier would follow the diagonal 
    /// line. The area under this curve (ROC AUC) is a common metric for classification performance. ROC 
    /// curves are particularly useful when you need to balance sensitivity and specificity, or when the 
    /// optimal threshold isn't known in advance.
    /// </para>
    /// </remarks>
    public static (Vector<T> fpr, Vector<T> tpr) CalculateROCCurve(Vector<T> actualValues, Vector<T> predictedValues)
    {
        var thresholds = GenerateThresholds(predictedValues);

        // Build list of (FPR, TPR) points including endpoints
        var points = new List<(double fpr, double tpr)>();

        // Add endpoint (0, 0) - threshold above all predictions (nothing predicted positive)
        points.Add((0.0, 0.0));

        for (int i = 0; i < thresholds.Length; i++)
        {
            var confusionMatrix = StatisticsHelper<T>.CalculateConfusionMatrix(actualValues, predictedValues, thresholds[i]);
            var totalNegatives = _numOps.Add(confusionMatrix.FalsePositives, confusionMatrix.TrueNegatives);
            var totalPositives = _numOps.Add(confusionMatrix.TruePositives, confusionMatrix.FalseNegatives);

            double fprVal = _numOps.Equals(totalNegatives, _numOps.Zero)
                ? 0.0
                : _numOps.ToDouble(_numOps.Divide(confusionMatrix.FalsePositives, totalNegatives));
            double tprVal = _numOps.Equals(totalPositives, _numOps.Zero)
                ? 0.0
                : _numOps.ToDouble(_numOps.Divide(confusionMatrix.TruePositives, totalPositives));

            points.Add((fprVal, tprVal));
        }

        // Add endpoint (1, 1) - threshold below all predictions (everything predicted positive)
        points.Add((1.0, 1.0));

        // For proper ROC curve, we need the upper envelope: for each FPR, take the max TPR
        var upperEnvelope = points
            .GroupBy(p => p.fpr)
            .Select(g => (fpr: g.Key, tpr: g.Max(p => p.tpr)))
            .OrderBy(p => p.fpr)
            .ToList();

        var fpr = new Vector<T>(upperEnvelope.Count);
        var tpr = new Vector<T>(upperEnvelope.Count);

        for (int i = 0; i < upperEnvelope.Count; i++)
        {
            fpr[i] = _numOps.FromDouble(upperEnvelope[i].fpr);
            tpr[i] = _numOps.FromDouble(upperEnvelope[i].tpr);
        }

        return (fpr, tpr);
    }

    /// <summary>
    /// Calculates both the AUC and F1 score for model evaluation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="evaluationData">The model evaluation data containing actual and predicted values.</param>
    /// <returns>A tuple containing the AUC and F1 score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates two important classification metrics in one go: the 
    /// Area Under the precision-recall Curve (AUC) and the F1 score. AUC measures the model's ability to 
    /// rank positive instances higher than negative ones across all possible thresholds, while the F1 score 
    /// balances precision and recall at a specific threshold. Together, these metrics provide a comprehensive 
    /// view of model performance. AUC gives a threshold-independent assessment of ranking ability, while 
    /// F1 shows performance at an operating point. This is useful because a model might have a good AUC 
    /// (ranking ability) but still perform poorly at the chosen threshold, or vice versa. Having both 
    /// metrics helps you understand different aspects of your model's performance.
    /// </para>
    /// </remarks>
    public static (T, T) CalculateAucF1Score<TInput, TOutput>(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);
        var predicted = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Predicted);
        var auc = StatisticsHelper<T>.CalculatePrecisionRecallAUC(actual, predicted);
        var (_, _, f1Score) = StatisticsHelper<T>.CalculatePrecisionRecallF1(actual, predicted, PredictionType.Regression);

        return (auc, f1Score);
    }

    /// <summary>
    /// Calculates a confusion matrix for binary classification at a specified threshold.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <param name="threshold">The threshold above which predictions are considered positive.</param>
    /// <returns>A confusion matrix containing counts of true positives, true negatives, false positives, and false negatives.</returns>
    /// <exception cref="ArgumentException">Thrown when inputs have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A confusion matrix summarizes the performance of a classification model by 
    /// counting how many instances were correctly and incorrectly classified for each class. This method 
    /// creates a confusion matrix for binary classification by comparing actual values to predictions at 
    /// a specified threshold. It counts true positives (correctly identified positives), true negatives 
    /// (correctly identified negatives), false positives (negatives incorrectly classified as positives), 
    /// and false negatives (positives incorrectly classified as negatives). The confusion matrix is the 
    /// foundation for many classification metrics, including accuracy, precision, recall, F1 score, and 
    /// specificity. It provides a more detailed view of model performance than single metrics, showing 
    /// exactly where the model makes mistakes.
    /// </para>
    /// </remarks>
    public static ConfusionMatrix<T> CalculateConfusionMatrix(Vector<T> actual, Vector<T> predicted, T threshold)
    {
        if (actual.Length != predicted.Length)
            throw new ArgumentException("Actual and predicted vectors must have the same size.");

        T truePositives = _numOps.Zero;
        T trueNegatives = _numOps.Zero;
        T falsePositives = _numOps.Zero;
        T falseNegatives = _numOps.Zero;

        for (int i = 0; i < actual.Length; i++)
        {
            bool actualPositive = _numOps.GreaterThan(actual[i], _numOps.Zero);
            bool predictedPositive = _numOps.GreaterThanOrEquals(predicted[i], threshold);

            if (actualPositive && predictedPositive)
                truePositives = _numOps.Add(truePositives, _numOps.One);
            else if (!actualPositive && !predictedPositive)
                trueNegatives = _numOps.Add(trueNegatives, _numOps.One);
            else if (!actualPositive && predictedPositive)
                falsePositives = _numOps.Add(falsePositives, _numOps.One);
            else
                falseNegatives = _numOps.Add(falseNegatives, _numOps.One);
        }

        return new ConfusionMatrix<T>(truePositives, trueNegatives, falsePositives, falseNegatives);
    }

    /// <summary>
    /// Calculates the Pearson correlation coefficient between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The first vector.</param>
    /// <param name="y">The second vector.</param>
    /// <returns>The Pearson correlation coefficient.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Pearson correlation coefficient measures the linear relationship between 
    /// two variables. It ranges from -1 to 1, where 1 indicates a perfect positive linear relationship, 
    /// -1 indicates a perfect negative linear relationship, and 0 indicates no linear relationship. This 
    /// method calculates the coefficient by first finding the mean of each vector, then computing the sum 
    /// of products of the deviations from the means, and finally dividing by the square root of the product 
    /// of the sum of squared deviations. The Pearson correlation is widely used in statistics to measure 
    /// how strongly two variables are related. It's sensitive to outliers and only captures linear 
    /// relationships, so it might miss non-linear patterns in the data.
    /// </para>
    /// </remarks>
    public static T CalculatePearsonCorrelationCoefficient(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        T meanX = CalculateMean(x);
        T meanY = CalculateMean(y);
        T sumXY = _numOps.Zero;
        T sumX2 = _numOps.Zero;
        T sumY2 = _numOps.Zero;

        for (int i = 0; i < x.Length; i++)
        {
            T xDiff = _numOps.Subtract(x[i], meanX);
            T yDiff = _numOps.Subtract(y[i], meanY);
            sumXY = _numOps.Add(sumXY, _numOps.Multiply(xDiff, yDiff));
            sumX2 = _numOps.Add(sumX2, _numOps.Multiply(xDiff, xDiff));
            sumY2 = _numOps.Add(sumY2, _numOps.Multiply(yDiff, yDiff));
        }

        return _numOps.Divide(sumXY, _numOps.Sqrt(_numOps.Multiply(sumX2, sumY2)));
    }

    /// <summary>
    /// Calculates the Spearman rank correlation coefficient between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The first vector.</param>
    /// <param name="y">The second vector.</param>
    /// <returns>The Spearman rank correlation coefficient.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Spearman rank correlation measures the monotonic relationship between 
    /// two variables, which means it detects whether one variable tends to increase or decrease as the 
    /// other increases, regardless of whether the relationship is linear. This method calculates the 
    /// coefficient by first converting the values to ranks, then applying the Pearson correlation formula 
    /// to these ranks. The result ranges from -1 to 1, with the same interpretation as Pearson correlation. 
    /// Spearman correlation is more robust to outliers and can detect non-linear relationships as long as 
    /// they're monotonic. It's particularly useful when the data doesn't meet the assumptions required for 
    /// Pearson correlation, such as normality or linear relationship.
    /// </para>
    /// </remarks>
    public static T CalculateSpearmanRankCorrelationCoefficient(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        var xRanks = CalculateRanks(x.ToList());
        var yRanks = CalculateRanks(y.ToList());

        return CalculatePearsonCorrelationCoefficient(new Vector<T>(xRanks), new Vector<T>(yRanks));
    }

    /// <summary>
    /// Calculates Kendall's tau correlation coefficient between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The first vector.</param>
    /// <param name="y">The second vector.</param>
    /// <returns>Kendall's tau correlation coefficient.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kendall's tau is another rank correlation measure that assesses the ordinal 
    /// association between two variables. It's calculated by comparing every possible pair of observations 
    /// and counting concordant pairs (where both variables change in the same direction) and discordant 
    /// pairs (where variables change in opposite directions). The coefficient is the difference between 
    /// concordant and discordant pairs, divided by the total number of possible pairs. Like other correlation 
    /// coefficients, it ranges from -1 to 1. Kendall's tau is particularly robust to outliers and doesn't 
    /// assume any particular distribution. It's often used when the data has many tied ranks or when a more 
    /// robust measure than Spearman's correlation is needed.
    /// </para>
    /// </remarks>
    public static T CalculateKendallTau(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        int concordantPairs = 0;
        int discordantPairs = 0;

        for (int i = 0; i < x.Length - 1; i++)
        {
            for (int j = i + 1; j < x.Length; j++)
            {
                T xDiff = _numOps.Subtract(x[i], x[j]);
                T yDiff = _numOps.Subtract(y[i], y[j]);
                T product = _numOps.Multiply(xDiff, yDiff);

                if (_numOps.GreaterThan(product, _numOps.Zero))
                    concordantPairs++;
                else if (_numOps.LessThan(product, _numOps.Zero))
                    discordantPairs++;
            }
        }

        int n = x.Length;
        T numerator = _numOps.Subtract(_numOps.FromDouble(concordantPairs), _numOps.FromDouble(discordantPairs));
        T denominator = _numOps.Sqrt(_numOps.Multiply(
            _numOps.FromDouble(n * (n - 1) / 2),
            _numOps.FromDouble(n * (n - 1) / 2)
        ));

        return _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Calculates the ranks of values in a collection, handling ties appropriately.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="values">The collection of values to rank.</param>
    /// <returns>A list of ranks corresponding to the original values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method converts a set of values to their ranks, which are needed 
    /// for rank-based correlation measures like Spearman's correlation. It sorts the values and assigns 
    /// ranks from 1 to n, with tied values receiving the average of the ranks they would have received if 
    /// they were slightly different. For example, if the 2nd and 3rd values are tied, they both receive a 
    /// rank of 2.5. The method preserves the original order of the values by tracking their indices during 
    /// sorting. Ranking is a fundamental operation in non-parametric statistics, which makes fewer assumptions 
    /// about the data distribution than parametric methods. This approach allows correlation measures to 
    /// focus on the relative ordering of values rather than their exact magnitudes.
    /// </para>
    /// </remarks>
    private static List<T> CalculateRanks(IEnumerable<T> values)
    {
        var sortedValues = values.Select((value, index) => new { Value = value, Index = index })
                                 .OrderBy(x => x.Value)
                                 .ToList();

        var ranks = new List<T>(new T[sortedValues.Count]);

        for (int i = 0; i < sortedValues.Count; i++)
        {
            int start = i;
            while (i < sortedValues.Count - 1 && _numOps.Equals(sortedValues[i].Value, sortedValues[i + 1].Value))
                i++;

            T rank = _numOps.Divide(_numOps.FromDouble(start + i + 1), _numOps.FromDouble(2));

            for (int j = start; j <= i; j++)
                ranks[sortedValues[j].Index] = rank;
        }

        return ranks;
    }

    /// <summary>
    /// Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The SMAPE value as a percentage.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Symmetric Mean Absolute Percentage Error (SMAPE) measures the accuracy 
    /// of predictions as a percentage, but unlike the standard MAPE, it treats over-predictions and 
    /// under-predictions symmetrically. It's calculated as 200% * average(|actual - predicted| / (|actual| + |predicted|)). 
    /// The result ranges from 0% (perfect predictions) to 200% (worst possible predictions). SMAPE is 
    /// particularly useful when the data contains zeros or very small values, which can cause standard 
    /// percentage errors to explode. It's also more balanced than MAPE, which penalizes over-predictions 
    /// more heavily than under-predictions. This method handles zero denominators by skipping those points, 
    /// ensuring the calculation remains valid even with zero values.
    /// </para>
    /// </remarks>
    public static T CalculateSymmetricMeanAbsolutePercentageError(Vector<T> actual, Vector<T> predicted)
    {
        if (actual.Length != predicted.Length)
            throw new ArgumentException("Actual and predicted vectors must have the same length.");

        T sum = _numOps.Zero;
        int count = 0;

        for (int i = 0; i < actual.Length; i++)
        {
            T denominator = _numOps.Add(_numOps.Abs(actual[i]), _numOps.Abs(predicted[i]));
            if (!_numOps.Equals(denominator, _numOps.Zero))
            {
                T diff = _numOps.Abs(_numOps.Subtract(actual[i], predicted[i]));
                sum = _numOps.Add(sum, _numOps.Divide(diff, denominator));
                count++;
            }
        }

        return _numOps.Multiply(_numOps.Divide(sum, _numOps.FromDouble(count)), _numOps.FromDouble(200));
    }

    /// <summary>
    /// Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed values.</param>
    /// <param name="predicted">The predicted values from a model.</param>
    /// <returns>The ROC AUC value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The ROC AUC (Area Under the Curve) is a performance measurement for classification 
    /// problems that tells how well a model can distinguish between classes. It ranges from 0 to 1, where 
    /// 1 means perfect classification, 0.5 means the model is no better than random guessing, and values 
    /// below 0.5 indicate worse-than-random performance. This method calculates the ROC curve (plotting 
    /// true positive rate against false positive rate at various thresholds) and then computes the area 
    /// under this curve. ROC AUC is particularly useful when you need a single metric to compare models 
    /// and when the classes are somewhat balanced. It's threshold-invariant, meaning it measures the model's 
    /// ability to rank positive instances higher than negative ones, regardless of the specific threshold used.
    /// </para>
    /// </remarks>
    public static T CalculateROCAUC(Vector<T> actual, Vector<T> predicted)
    {
        var (fpr, tpr) = CalculateROCCurve(actual, predicted);
        return CalculateAUC(fpr, tpr);
    }

    /// <summary>
    /// Calculates the Area Under a Curve (AUC) given x and y coordinates.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="fpr">The x-coordinates (typically false positive rates for ROC curves).</param>
    /// <param name="tpr">The y-coordinates (typically true positive rates for ROC curves).</param>
    /// <returns>The area under the curve.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the area under a curve defined by a series of points. 
    /// It uses the trapezoidal rule, which approximates the area by dividing it into trapezoids and summing 
    /// their areas. For each adjacent pair of points, it calculates the width (difference in x-coordinates) 
    /// and the average height (average of y-coordinates), then multiplies them to get the area of that 
    /// trapezoid. While this method is typically used for ROC curves (where x is the false positive rate 
    /// and y is the true positive rate), it can be used for any curve where you need to calculate the area 
    /// underneath. The more points you have defining your curve, the more accurate the area calculation will be.
    /// </para>
    /// </remarks>
    public static T CalculateAUC(Vector<T> fpr, Vector<T> tpr)
    {
        // Guard against empty or single-point curves
        if (fpr.Length < 2 || tpr.Length < 2)
        {
            return _numOps.FromDouble(0.5); // Return baseline AUC for invalid input
        }

        T auc = _numOps.Zero;
        for (int i = 1; i < fpr.Length; i++)
        {
            var width = _numOps.Subtract(fpr[i], fpr[i - 1]);
            var height = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Add(tpr[i], tpr[i - 1]));
            auc = _numOps.Add(auc, _numOps.Multiply(width, height));
        }

        // Clamp AUC to valid range [0, 1]
        // Negative AUC can occur with unsorted FPR values or poor data
        if (_numOps.LessThan(auc, _numOps.Zero))
        {
            auc = _numOps.Zero;
        }
        else if (_numOps.GreaterThan(auc, _numOps.One))
        {
            auc = _numOps.One;
        }

        return auc;
    }

    /// <summary>
    /// Calculates the Euclidean distance between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>The Euclidean distance.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Euclidean distance is the "ordinary" straight-line distance between two 
    /// points in Euclidean space. It's calculated as the square root of the sum of squared differences 
    /// between corresponding elements of the two vectors. This is a direct application of the Pythagorean 
    /// theorem extended to multiple dimensions. Euclidean distance is widely used in machine learning for 
    /// tasks like clustering (e.g., k-means), nearest neighbor searches, and measuring similarity between 
    /// data points. It works well when the data dimensions have similar scales and are somewhat independent. 
    /// However, it can be sensitive to outliers and may not perform well when dimensions have very different 
    /// scales or are highly correlated.
    /// </para>
    /// </remarks>
    public static T EuclideanDistance(Vector<T> v1, Vector<T> v2)
    {
        if (v1.Length != v2.Length)
            throw new ArgumentException("Vectors must have the same length");

        T sum = _numOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            T diff = _numOps.Subtract(v1[i], v2[i]);
            sum = _numOps.Add(sum, _numOps.Multiply(diff, diff));
        }

        return _numOps.Sqrt(sum);
    }

    /// <summary>
    /// Calculates the Manhattan distance (L1 norm) between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>The Manhattan distance.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Manhattan distance (also known as taxicab or city block distance) measures 
    /// the distance between two points as the sum of the absolute differences of their coordinates. It's 
    /// called Manhattan distance because it's like measuring the distance a taxi would drive in a city laid 
    /// out in a grid (like Manhattan). Unlike Euclidean distance, which measures the shortest path "as the 
    /// crow flies," Manhattan distance follows the grid. This metric is less sensitive to outliers than 
    /// Euclidean distance and can be more appropriate when the dimensions represent features that are not 
    /// directly comparable or when movement along the axes has different costs. It's commonly used in 
    /// machine learning algorithms like k-nearest neighbors when dealing with high-dimensional data.
    /// </para>
    /// </remarks>
    public static T ManhattanDistance(Vector<T> v1, Vector<T> v2)
    {
        if (v1.Length != v2.Length)
            throw new ArgumentException("Vectors must have the same length");

        T sum = _numOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Abs(_numOps.Subtract(v1[i], v2[i])));
        }

        return sum;
    }

    /// <summary>
    /// Calculates the cosine similarity between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>The cosine similarity.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cosine similarity measures the cosine of the angle between two vectors, 
    /// indicating how similar their orientations are, regardless of their magnitudes. It ranges from -1 
    /// (exactly opposite) through 0 (orthogonal or unrelated) to 1 (exactly the same direction). This 
    /// method calculates it by dividing the dot product of the vectors by the product of their magnitudes. 
    /// Cosine similarity is particularly useful in text analysis and recommendation systems, where the 
    /// absolute values (like document length or user rating scale) are less important than the pattern of 
    /// values. It's invariant to scaling, meaning that multiplying a vector by a constant doesn't change 
    /// its direction and therefore doesn't affect the cosine similarity with other vectors.
    /// </para>
    /// </remarks>
    public static T CosineSimilarity(Vector<T> v1, Vector<T> v2)
    {
        if (v1.Length != v2.Length)
            throw new ArgumentException("Vectors must have the same length");

        T dotProduct = _numOps.Zero;
        T norm1 = _numOps.Zero;
        T norm2 = _numOps.Zero;

        for (int i = 0; i < v1.Length; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(v1[i], v2[i]));
            norm1 = _numOps.Add(norm1, _numOps.Multiply(v1[i], v1[i]));
            norm2 = _numOps.Add(norm2, _numOps.Multiply(v2[i], v2[i]));
        }

        return _numOps.Divide(dotProduct, _numOps.Multiply(_numOps.Sqrt(norm1), _numOps.Sqrt(norm2)));
    }

    /// <summary>
    /// Calculates the Jaccard similarity coefficient between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>The Jaccard similarity coefficient.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Jaccard similarity coefficient measures the similarity between two sets 
    /// by comparing what they have in common with what they have in total. It's calculated as the size of 
    /// the intersection divided by the size of the union. This method adapts this concept to numeric vectors 
    /// by treating each position as a partial membership in a set, using the minimum value as the intersection 
    /// and the maximum value as the union at each position. The result ranges from 0 (no overlap) to 1 
    /// (identical). Jaccard similarity is particularly useful for sparse binary data, like presence/absence 
    /// features or one-hot encoded categorical variables. It focuses on the attributes that are present in 
    /// at least one of the vectors, ignoring positions where both vectors have zeros.
    /// </para>
    /// </remarks>
    public static T JaccardSimilarity(Vector<T> v1, Vector<T> v2)
    {
        if (v1.Length != v2.Length)
            throw new ArgumentException("Vectors must have the same length");

        T intersection = _numOps.Zero;
        T union = _numOps.Zero;

        for (int i = 0; i < v1.Length; i++)
        {
            intersection = _numOps.Add(intersection, MathHelper.Min(v1[i], v2[i]));
            union = _numOps.Add(union, MathHelper.Max(v1[i], v2[i]));
        }

        return _numOps.Divide(intersection, union);
    }

    /// <summary>
    /// Calculates the Hamming distance between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>The Hamming distance.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Hamming distance counts the number of positions at which two vectors differ. 
    /// It's a simple but powerful measure of dissimilarity that's particularly useful for categorical data, 
    /// binary vectors, or strings. This method increments a counter each time it finds a position where the 
    /// two vectors have different values. The result ranges from 0 (identical vectors) to the length of the 
    /// vectors (completely different). Hamming distance is commonly used in information theory, coding theory, 
    /// and error detection/correction. In machine learning, it's useful for comparing categorical features 
    /// or binary representations. Unlike metrics that consider the magnitude of differences, Hamming distance 
    /// only cares about whether values match exactly.
    /// </para>
    /// </remarks>
    public static T HammingDistance(Vector<T> v1, Vector<T> v2)
    {
        if (v1.Length != v2.Length)
            throw new ArgumentException("Vectors must have the same length");

        T distance = _numOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            if (!_numOps.Equals(v1[i], v2[i]))
            {
                distance = _numOps.Add(distance, _numOps.One);
            }
        }

        return distance;
    }

    /// <summary>
    /// Calculates the Mahalanobis distance between two vectors, accounting for correlations in the data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <param name="covarianceMatrix">The covariance matrix of the data.</param>
    /// <returns>The Mahalanobis distance.</returns>
    /// <exception cref="ArgumentException">Thrown when dimensions of vectors and covariance matrix don't match.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Mahalanobis distance measures how many standard deviations away a point 
    /// is from the mean of a distribution, taking into account the correlations between variables. Unlike 
    /// Euclidean distance, which treats all dimensions equally, Mahalanobis distance gives less weight to 
    /// dimensions with high variance and to dimensions that are highly correlated with others. This method 
    /// calculates it by first finding the difference between the vectors, then multiplying by the inverse 
    /// of the covariance matrix, and finally taking the square root of the dot product with the original 
    /// difference. Mahalanobis distance is particularly useful for multivariate outlier detection, 
    /// classification with correlated features, and when features have very different scales or units.
    /// </para>
    /// </remarks>
    public static T MahalanobisDistance(Vector<T> v1, Vector<T> v2, Matrix<T> covarianceMatrix)
    {
        if (v1.Length != v2.Length || v1.Length != covarianceMatrix.Rows || covarianceMatrix.Rows != covarianceMatrix.Columns)
            throw new ArgumentException("Vectors and covariance matrix dimensions must match");

        Vector<T> diff = v1.Subtract(v2);
        Vector<T> invCovDiff = covarianceMatrix.Inverse().Multiply(diff);

        return _numOps.Sqrt(diff.DotProduct(invCovDiff));
    }

    /// <summary>
    /// Calculates the distance or similarity between two vectors using the specified metric.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <param name="metric">The distance metric to use.</param>
    /// <param name="covarianceMatrix">The covariance matrix (required only for Mahalanobis distance).</param>
    /// <returns>The calculated distance or similarity value.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported distance metric is specified.</exception>
    /// <exception cref="ArgumentNullException">Thrown when covariance matrix is null for Mahalanobis distance.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides a unified interface for calculating various distance and 
    /// similarity measures between two vectors. It supports common metrics like Euclidean (straight-line) 
    /// distance, Manhattan (city block) distance, cosine similarity (angle between vectors), Jaccard 
    /// similarity (overlap between sets), Hamming distance (number of different positions), and Mahalanobis 
    /// distance (accounting for correlations). Each metric has different properties and is suitable for 
    /// different types of data and applications. For example, cosine similarity is good for text data, 
    /// Euclidean distance works well for low-dimensional continuous data, and Hamming distance is appropriate 
    /// for categorical features. This method lets you easily switch between metrics to find the one that 
    /// works best for your specific problem.
    /// </para>
    /// </remarks>
    public static T CalculateDistance(Vector<T> v1, Vector<T> v2, DistanceMetricType metric, Matrix<T>? covarianceMatrix = null)
    {
        return metric switch
        {
            DistanceMetricType.Euclidean => EuclideanDistance(v1, v2),
            DistanceMetricType.Manhattan => ManhattanDistance(v1, v2),
            DistanceMetricType.Cosine => CosineSimilarity(v1, v2),
            DistanceMetricType.Jaccard => JaccardSimilarity(v1, v2),
            DistanceMetricType.Hamming => HammingDistance(v1, v2),
            DistanceMetricType.Mahalanobis => covarianceMatrix != null
                ? MahalanobisDistance(v1, v2, covarianceMatrix)
                : throw new ArgumentNullException(nameof(covarianceMatrix), "Covariance matrix is required for Mahalanobis distance"),
            _ => throw new ArgumentException("Unsupported distance metric", nameof(metric)),
        };
    }

    /// <summary>
    /// Calculates the covariance matrix for a dataset.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="matrix">The data matrix where each row is an observation and each column is a variable.</param>
    /// <returns>The covariance matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The covariance matrix measures how variables in a dataset vary together. Each 
    /// element (i,j) in the matrix represents the covariance between the i-th and j-th variables. Diagonal 
    /// elements are variances of individual variables, while off-diagonal elements show how pairs of variables 
    /// co-vary. This method calculates the covariance matrix by first computing the mean of each variable, 
    /// then for each pair of variables, calculating the average product of their deviations from their 
    /// respective means. The resulting matrix is symmetric (covariance of X with Y equals covariance of Y 
    /// with X). The covariance matrix is essential for many statistical techniques, including principal 
    /// component analysis, Mahalanobis distance, and multivariate normal distributions. It helps understand 
    /// the structure and relationships in multivariate data.
    /// </para>
    /// </remarks>
    public static Matrix<T> CalculateCovarianceMatrix(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        int m = matrix.Columns;
        var covMatrix = new Matrix<T>(m, m);
        var means = new T[m];

        // Calculate means
        for (int j = 0; j < m; j++)
        {
            means[j] = _numOps.Zero;
            for (int i = 0; i < n; i++)
            {
                means[j] = _numOps.Add(means[j], matrix[i, j]);
            }
            means[j] = _numOps.Divide(means[j], _numOps.FromDouble(n));
        }

        // Calculate covariances
        for (int i = 0; i < m; i++)
        {
            for (int j = i; j < m; j++)
            {
                T covariance = _numOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    T diff1 = _numOps.Subtract(matrix[k, i], means[i]);
                    T diff2 = _numOps.Subtract(matrix[k, j], means[j]);
                    covariance = _numOps.Add(covariance, _numOps.Multiply(diff1, diff2));
                }
                covariance = _numOps.Divide(covariance, _numOps.FromDouble(n - 1));
                covMatrix[i, j] = covariance;
                covMatrix[j, i] = covariance; // Covariance matrix is symmetric
            }
        }

        return covMatrix;
    }

    /// <summary>
    /// Calculates the mutual information between two discrete random variables.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The first variable.</param>
    /// <param name="y">The second variable.</param>
    /// <returns>The mutual information value.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mutual information measures how much knowing one variable reduces uncertainty 
    /// about another. It's a fundamental concept in information theory that quantifies the "shared information" 
    /// between two random variables. This method calculates mutual information by estimating the joint and 
    /// marginal probability distributions of the variables, then computing the sum of p(x,y) * log(p(x,y)/(p(x)*p(y))) 
    /// over all possible value combinations. Higher mutual information indicates stronger dependency between 
    /// variables. Unlike correlation, mutual information can detect non-linear relationships. It's particularly 
    /// useful in feature selection, as it helps identify variables that provide unique information about the 
    /// target. This implementation treats the input vectors as discrete variables, counting occurrences to 
    /// estimate probabilities.
    /// </para>
    /// </remarks>
    public static T CalculateMutualInformation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        int n = x.Length;
        var jointCounts = new Dictionary<string, int>();
        var xCounts = new Dictionary<string, int>();
        var yCounts = new Dictionary<string, int>();

        // Count occurrences
        for (int i = 0; i < n; i++)
        {
            var xKey = x[i]?.ToString() ?? string.Empty;
            var yKey = y[i]?.ToString() ?? string.Empty;
            string jointKey = $"{xKey},{yKey}";

            if (!jointCounts.ContainsKey(jointKey))
                jointCounts[jointKey] = 0;
            jointCounts[jointKey]++;

            if (!xCounts.ContainsKey(xKey))
                xCounts[xKey] = 0;
            xCounts[xKey]++;

            if (!yCounts.ContainsKey(yKey))
                yCounts[yKey] = 0;
            yCounts[yKey]++;
        }

        T mi = _numOps.Zero;
        T nDouble = _numOps.FromDouble(n);

        foreach (var jointEntry in jointCounts)
        {
            string[] keys = jointEntry.Key.Split(',');
            string xKey = keys[0];
            string yKey = keys[1];

            T pxy = _numOps.Divide(_numOps.FromDouble(jointEntry.Value), nDouble);
            T px = _numOps.Divide(_numOps.FromDouble(xCounts[xKey]), nDouble);
            T py = _numOps.Divide(_numOps.FromDouble(yCounts[yKey]), nDouble);

            if (!_numOps.Equals(pxy, _numOps.Zero) && !_numOps.Equals(px, _numOps.Zero) && !_numOps.Equals(py, _numOps.Zero))
            {
                T term = _numOps.Multiply(pxy, _numOps.Log(_numOps.Divide(pxy, _numOps.Multiply(px, py))));
                mi = _numOps.Add(mi, term);
            }
        }

        return mi;
    }

    /// <summary>
    /// Calculates the normalized mutual information between two variables.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The first variable.</param>
    /// <param name="y">The second variable.</param>
    /// <returns>The normalized mutual information value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Normalized mutual information (NMI) scales the mutual information to a range 
    /// between 0 and 1, making it easier to interpret and compare across different variable pairs. It's 
    /// calculated by dividing the mutual information by the square root of the product of the entropies 
    /// of the individual variables. A value of 0 means the variables are independent, while 1 indicates 
    /// perfect dependency (one variable completely determines the other). NMI is particularly useful in 
    /// clustering evaluation, where it measures how well cluster assignments match ground truth labels, 
    /// and in feature selection, where it helps identify informative but non-redundant features. Unlike 
    /// raw mutual information, NMI accounts for the different entropy levels of the variables, providing 
    /// a more balanced measure of their relationship.
    /// </para>
    /// </remarks>
    public static T CalculateNormalizedMutualInformation(Vector<T> x, Vector<T> y)
    {
        var mi = CalculateMutualInformation(x, y);
        var hx = CalculateEntropy(x);
        var hy = CalculateEntropy(y);

        return _numOps.Divide(mi, _numOps.Sqrt(_numOps.Multiply(hx, hy)));
    }

    /// <summary>
    /// Calculates the variation of information (also known as shared information distance) between two variables.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The first variable.</param>
    /// <param name="y">The second variable.</param>
    /// <returns>The variation of information value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The variation of information (VI) is a measure of the distance between two 
    /// clusterings or partitions of the same data. It's calculated as the sum of the entropies of the 
    /// variables minus twice their mutual information: VI(X,Y) = H(X) + H(Y) - 2*MI(X,Y). Lower values 
    /// indicate more similar distributions, with 0 meaning identical distributions. VI satisfies the 
    /// properties of a true metric (non-negativity, symmetry, and triangle inequality), making it useful 
    /// for comparing different clusterings of the same data. It's particularly valuable in clustering 
    /// evaluation and consensus clustering, where you need to measure the distance between different 
    /// partitions of the data. Unlike some other measures, VI penalizes both splitting and merging of 
    /// clusters equally.
    /// </para>
    /// </remarks>
    public static T CalculateVariationOfInformation(Vector<T> x, Vector<T> y)
    {
        var mi = CalculateMutualInformation(x, y);
        var hx = CalculateEntropy(x);
        var hy = CalculateEntropy(y);

        return _numOps.Subtract(_numOps.Add(hx, hy), _numOps.Multiply(_numOps.FromDouble(2), mi));
    }

    /// <summary>
    /// Calculates the Shannon entropy of a discrete random variable.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="x">The variable values.</param>
    /// <returns>The entropy value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Entropy measures the uncertainty or randomness in a variable. It's calculated 
    /// as the negative sum of p(x) * log(p(x)) over all possible values of x, where p(x) is the probability 
    /// of each value. This method estimates these probabilities by counting occurrences in the input vector. 
    /// Higher entropy indicates more uncertainty or information content. For example, a uniform distribution 
    /// has maximum entropy, while a constant (all values the same) has zero entropy. Entropy is a fundamental 
    /// concept in information theory with applications in machine learning, compression, and feature selection. 
    /// It helps quantify how much information is contained in a variable and is a building block for other 
    /// information-theoretic measures like mutual information and variation of information.
    /// </para>
    /// </remarks>
    private static T CalculateEntropy(Vector<T> x)
    {
        var prob = new Dictionary<string, T>();
        int n = x.Length;
        T nInverse = _numOps.Divide(_numOps.One, _numOps.FromDouble(n));

        foreach (var xi in x)
        {
            string key = xi?.ToString() ?? string.Empty;
            if (prob.TryGetValue(key, out T? value))
            {
                prob[key] = _numOps.Add(value, nInverse);
            }
            else
            {
                prob[key] = nInverse;
            }
        }

        var entropy = _numOps.Zero;
        foreach (var p in prob.Values)
        {
            entropy = _numOps.Subtract(entropy, _numOps.Multiply(p, _numOps.Log(p)));
        }

        return entropy;
    }

    /// <summary>
    /// Calculates the silhouette score for a clustering result.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data matrix where each row is an observation.</param>
    /// <param name="labels">The cluster labels for each observation.</param>
    /// <returns>The average silhouette score across all observations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The silhouette score measures how well each object fits within its assigned 
    /// cluster compared to other clusters. For each point, it calculates (b-a)/max(a,b), where a is the 
    /// average distance to other points in the same cluster, and b is the average distance to points in 
    /// the nearest different cluster. The score ranges from -1 to 1, where higher values indicate better 
    /// clustering. A score near 1 means points are well-matched to their clusters and far from neighboring 
    /// clusters. A score near 0 indicates overlapping clusters, while negative scores suggest points might 
    /// be assigned to the wrong clusters. This method calculates the silhouette for each point and returns 
    /// the average across all points. It's a valuable tool for evaluating clustering quality and comparing 
    /// different clustering algorithms or parameter settings.
    /// </para>
    /// </remarks>
    public static T CalculateSilhouetteScore(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        var uniqueLabels = labels.Distinct().ToList();
        var silhouetteScores = new List<T>();

        for (int i = 0; i < n; i++)
        {
            T a = CalculateAverageIntraClusterDistance(data, labels, i);
            T b = CalculateMinAverageInterClusterDistance(data, labels, i);
            T silhouette = _numOps.Divide(_numOps.Subtract(b, a), MathHelper.Max(a, b));
            silhouetteScores.Add(silhouette);
        }

        return _numOps.Divide(silhouetteScores.Aggregate(_numOps.Zero, _numOps.Add), _numOps.FromDouble(n));
    }

    /// <summary>
    /// Calculates the average distance from a point to all other points in the same cluster.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data matrix where each row is an observation.</param>
    /// <param name="labels">The cluster labels for each observation.</param>
    /// <param name="index">The index of the point to calculate the distance for.</param>
    /// <returns>The average intra-cluster distance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method calculates the average distance from a specific point to 
    /// all other points in the same cluster. It's a key component in computing the silhouette score, where 
    /// it represents the "a" value (cohesion). A lower average intra-cluster distance indicates that the 
    /// point is similar to other points in its cluster, suggesting good cluster assignment. The method 
    /// identifies all points with the same label as the target point, calculates the Euclidean distance to 
    /// each, and returns the average. If there are no other points in the cluster (singleton cluster), it 
    /// returns zero. This measure helps assess how compact or tight a cluster is around a specific point, 
    /// which is essential for evaluating clustering quality.
    /// </para>
    /// </remarks>
    private static T CalculateAverageIntraClusterDistance(Matrix<T> data, Vector<T> labels, int index)
    {
        T sum = _numOps.Zero;
        int count = 0;
        T label = labels[index];

        for (int i = 0; i < data.Rows; i++)
        {
            if (i != index && _numOps.Equals(labels[i], label))
            {
                sum = _numOps.Add(sum, EuclideanDistance(data.GetRow(index), data.GetRow(i)));
                count++;
            }
        }

        return count > 0 ? _numOps.Divide(sum, _numOps.FromDouble(count)) : _numOps.Zero;
    }

    /// <summary>
    /// Calculates the minimum average distance from a point to points in any other cluster.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data matrix where each row is an observation.</param>
    /// <param name="labels">The cluster labels for each observation.</param>
    /// <param name="index">The index of the point to calculate the distance for.</param>
    /// <returns>The minimum average inter-cluster distance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method finds the nearest neighboring cluster to a specific point 
    /// and calculates the average distance to points in that cluster. It's a key component in computing the 
    /// silhouette score, where it represents the "b" value (separation). For each cluster different from 
    /// the point's own cluster, it calculates the average distance to all points in that cluster, then 
    /// returns the minimum of these averages. A higher minimum average inter-cluster distance indicates 
    /// better separation between clusters. This measure helps assess how well-separated a point is from 
    /// other clusters, which is essential for evaluating clustering quality. Good clustering results in 
    /// high separation (b values) relative to cohesion (a values).
    /// </para>
    /// </remarks>
    private static T CalculateMinAverageInterClusterDistance(Matrix<T> data, Vector<T> labels, int index)
    {
        var uniqueLabels = labels.Distinct().ToList();
        T minDistance = _numOps.MaxValue;
        T label = labels[index];

        foreach (T otherLabel in uniqueLabels)
        {
            if (!_numOps.Equals(otherLabel, label))
            {
                T avgDistance = CalculateAverageDistanceToCluster(data, labels, index, otherLabel);
                minDistance = MathHelper.Min(minDistance, avgDistance);
            }
        }

        return minDistance;
    }

    /// <summary>
    /// Calculates the average distance from a point to all points in a specific cluster.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data matrix where each row is an observation.</param>
    /// <param name="labels">The cluster labels for each observation.</param>
    /// <param name="index">The index of the point to calculate the distance from.</param>
    /// <param name="clusterLabel">The label of the cluster to calculate the distance to.</param>
    /// <returns>The average distance to the specified cluster.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method calculates the average distance from a specific point to 
    /// all points in a given cluster. It's used in computing the minimum average inter-cluster distance 
    /// for the silhouette score. The method identifies all points with the specified cluster label, 
    /// calculates the Euclidean distance from the target point to each, and returns the average. If the 
    /// specified cluster is empty, it returns the maximum possible value to ensure it's not selected as 
    /// the nearest cluster. This measure helps identify which neighboring cluster is closest to a point, 
    /// which is important for assessing whether the point has been assigned to the most appropriate cluster. 
    /// In a good clustering, points should be far from all clusters other than their own.
    /// </para>
    /// </remarks>
    private static T CalculateAverageDistanceToCluster(Matrix<T> data, Vector<T> labels, int index, T clusterLabel)
    {
        T sum = _numOps.Zero;
        int count = 0;

        for (int i = 0; i < data.Rows; i++)
        {
            if (_numOps.Equals(labels[i], clusterLabel))
            {
                sum = _numOps.Add(sum, EuclideanDistance(data.GetRow(index), data.GetRow(i)));
                count++;
            }
        }

        return count > 0 ? _numOps.Divide(sum, _numOps.FromDouble(count)) : _numOps.MaxValue;
    }

    /// <summary>
    /// Calculates the Calinski-Harabasz Index (Variance Ratio Criterion) for a clustering result.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data matrix where each row is an observation.</param>
    /// <param name="labels">The cluster labels for each observation.</param>
    /// <returns>The Calinski-Harabasz Index.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Calinski-Harabasz Index (CHI), also known as the Variance Ratio Criterion, 
    /// measures the ratio of between-cluster variance to within-cluster variance, adjusted for the number 
    /// of clusters and data points. Higher values indicate better clustering with dense, well-separated 
    /// clusters. It's calculated as [(n-k)/(k-1)] * [B/W], where n is the number of points, k is the number 
    /// of clusters, B is the between-cluster variance, and W is the within-cluster variance. This method 
    /// computes these components by first calculating cluster centroids and the global centroid, then 
    /// measuring the variances based on these. CHI is particularly useful for comparing clustering results 
    /// with different numbers of clusters, as it accounts for this difference in its formula. It works best 
    /// for convex, well-separated clusters.
    /// </para>
    /// </remarks>
    public static T CalculateCalinskiHarabaszIndex(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        int k = labels.Distinct().Count();
        var centroids = CalculateCentroids(data, labels);
        Vector<T> globalCentroid = CalculateGlobalCentroid(data);

        T betweenClusterVariance = CalculateBetweenClusterVariance(centroids, globalCentroid, labels);
        T withinClusterVariance = CalculateWithinClusterVariance(data, labels, centroids);

        T numerator = _numOps.Multiply(_numOps.FromDouble(n - k), betweenClusterVariance);
        T denominator = _numOps.Multiply(_numOps.FromDouble(k - 1), withinClusterVariance);

        return _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Calculates the centroids for each cluster in a clustering result.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data matrix where each row is an observation.</param>
    /// <param name="labels">The cluster labels for each observation.</param>
    /// <returns>A dictionary mapping cluster labels to their centroids.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method calculates the center point (centroid) of each cluster by 
    /// averaging all points assigned to that cluster. It first groups the data points by their cluster labels, 
    /// then for each cluster, sums all the points and divides by the count to get the average. The result 
    /// is a dictionary where keys are cluster labels and values are the corresponding centroids. Centroids 
    /// are essential in clustering analysis as they represent the "typical" or average member of each cluster. 
    /// They're used in many clustering algorithms (like k-means) and evaluation metrics (like Calinski-Harabasz 
    /// and Davies-Bouldin indices). Centroids also help interpret clustering results by showing the 
    /// characteristic values for each cluster, which can reveal patterns in the data.
    /// </para>
    /// </remarks>
    private static Dictionary<string, Vector<T>> CalculateCentroids(Matrix<T> data, Vector<T> labels)
    {
        var centroids = new Dictionary<string, Vector<T>>();
        var counts = new Dictionary<string, int>();

        for (int i = 0; i < data.Rows; i++)
        {
            string label = labels[i]?.ToString() ?? string.Empty;
            if (!centroids.ContainsKey(label))
            {
                centroids[label] = new Vector<T>(data.Columns);
                counts[label] = 0;
            }

            centroids[label] = centroids[label].Add(data.GetRow(i));
            counts[label]++;
        }

        foreach (var label in centroids.Keys.ToList())
        {
            centroids[label] = centroids[label].Divide(_numOps.FromDouble(counts[label]));
        }

        return centroids;
    }

    /// <summary>
    /// Calculates the global centroid (mean) of all data points.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data matrix where each row is an observation.</param>
    /// <returns>The global centroid vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method calculates the overall average of all data points, regardless 
    /// of their cluster assignments. It sums all the data points and divides by the total count to get the 
    /// mean vector. The global centroid represents the center of the entire dataset and serves as a reference 
    /// point for measuring between-cluster variance in clustering evaluation metrics like the Calinski-Harabasz 
    /// Index. By comparing cluster centroids to the global centroid, we can assess how distinct the clusters 
    /// are from the overall data distribution. A good clustering typically results in cluster centroids that 
    /// are far from the global centroid, indicating that the clusters capture meaningful structure in the data 
    /// rather than random groupings.
    /// </para>
    /// </remarks>
    private static Vector<T> CalculateGlobalCentroid(Matrix<T> data)
    {
        Vector<T> centroid = new Vector<T>(data.Columns);
        for (int i = 0; i < data.Rows; i++)
        {
            centroid = centroid.Add(data.GetRow(i));
        }

        return centroid.Divide(_numOps.FromDouble(data.Rows));
    }

    /// <summary>
    /// Calculates the between-cluster variance for a clustering result.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="centroids">A dictionary mapping cluster labels to their centroids.</param>
    /// <param name="globalCentroid">The global centroid of all data points.</param>
    /// <param name="labels">The cluster labels for each observation.</param>
    /// <returns>The between-cluster variance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Between-cluster variance measures how different the clusters are from each other. 
    /// It's calculated as the weighted sum of squared distances between each cluster's centroid and the global 
    /// centroid, where the weights are the number of points in each cluster. Higher between-cluster variance 
    /// indicates more distinct, well-separated clusters. This method first counts how many points are in each 
    /// cluster, then for each cluster, calculates the squared distance between its centroid and the global 
    /// centroid, multiplies by the cluster size, and adds to the total variance. Between-cluster variance is 
    /// a key component in the Calinski-Harabasz Index and other clustering evaluation metrics. It quantifies 
    /// the "separation" aspect of clustering quality²how well the algorithm has identified distinct groups 
    /// in the data.
    /// </para>
    /// </remarks>
    private static T CalculateBetweenClusterVariance(Dictionary<string, Vector<T>> centroids, Vector<T> globalCentroid, Vector<T> labels)
    {
        T variance = _numOps.Zero;
        var counts = new Dictionary<string, int>();

        foreach (T label in labels)
        {
            string key = label?.ToString() ?? string.Empty;
            if (!counts.ContainsKey(key))
                counts[key] = 0;
            counts[key]++;
        }

        foreach (var kvp in centroids)
        {
            T distance = EuclideanDistance(kvp.Value, globalCentroid);
            variance = _numOps.Add(variance, _numOps.Multiply(_numOps.FromDouble(counts[kvp.Key]), _numOps.Multiply(distance, distance)));
        }

        return variance;
    }

    /// <summary>
    /// Calculates the within-cluster variance for a clustering result.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data matrix where each row is an observation.</param>
    /// <param name="labels">The cluster labels for each observation.</param>
    /// <param name="centroids">A dictionary mapping cluster labels to their centroids.</param>
    /// <returns>The within-cluster variance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Within-cluster variance measures how compact or tight the clusters are. It's 
    /// calculated as the sum of squared distances between each point and its cluster's centroid. Lower 
    /// within-cluster variance indicates more compact, homogeneous clusters. This method iterates through 
    /// all data points, finds the centroid of the cluster each point belongs to, calculates the squared 
    /// distance between the point and its centroid, and adds to the total variance. Within-cluster variance 
    /// is a key component in the Calinski-Harabasz Index and other clustering evaluation metrics. It quantifies 
    /// the "cohesion" aspect of clustering quality²how similar points within the same cluster are to each 
    /// other. Good clustering algorithms minimize this variance while maximizing between-cluster variance.
    /// </para>
    /// </remarks>
    private static T CalculateWithinClusterVariance(Matrix<T> data, Vector<T> labels, Dictionary<string, Vector<T>> centroids)
    {
        T variance = _numOps.Zero;

        for (int i = 0; i < data.Rows; i++)
        {
            string label = labels[i]?.ToString() ?? string.Empty;
            T distance = EuclideanDistance(data.GetRow(i), centroids[label]);
            variance = _numOps.Add(variance, _numOps.Multiply(distance, distance));
        }

        return variance;
    }

    /// <summary>
    /// Calculates the Davies-Bouldin Index for a clustering result.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data matrix where each row is an observation.</param>
    /// <param name="labels">The cluster labels for each observation.</param>
    /// <returns>The Davies-Bouldin Index.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Davies-Bouldin Index (DBI) measures the average similarity between each 
    /// cluster and its most similar cluster. Lower values indicate better clustering with more distinct, 
    /// well-separated clusters. For each cluster, it calculates the ratio of the sum of within-cluster 
    /// scatter to the between-cluster separation for the most similar cluster, then averages these ratios. 
    /// This method first calculates centroids for all clusters, then for each cluster, finds the maximum 
    /// ratio with any other cluster and adds it to the total. DBI is particularly useful because it considers 
    /// both the compactness of clusters (how close points are to their centroids) and the separation between 
    /// clusters. Unlike some metrics, it doesn't improve simply by increasing the number of clusters, making 
    /// it valuable for comparing clustering results with different numbers of clusters.
    /// </para>
    /// </remarks>
    public static T CalculateDaviesBouldinIndex(Matrix<T> data, Vector<T> labels)
    {
        var centroids = CalculateCentroids(data, labels);
        var uniqueLabels = labels.Select(l => l?.ToString() ?? string.Empty).Distinct().ToList();
        int k = uniqueLabels.Count;

        T sum = _numOps.Zero;

        for (int i = 0; i < k; i++)
        {
            T maxRatio = _numOps.Zero;
            string labelI = uniqueLabels[i];

            for (int j = 0; j < k; j++)
            {
                if (i != j)
                {
                    string labelJ = uniqueLabels[j];
                    T si = CalculateAverageDistance(data, labels, labelI, centroids[labelI]);
                    T sj = CalculateAverageDistance(data, labels, labelJ, centroids[labelJ]);
                    T dij = EuclideanDistance(centroids[labelI], centroids[labelJ]);
                    T ratio = _numOps.Divide(_numOps.Add(si, sj), dij);
                    maxRatio = MathHelper.Max(maxRatio, ratio);
                }
            }

            sum = _numOps.Add(sum, maxRatio);
        }

        return _numOps.Divide(sum, _numOps.FromDouble(k));
    }

    /// <summary>
    /// Calculates the Adjusted Rand Index (ARI) between two clusterings.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="labels1">The first set of cluster labels.</param>
    /// <param name="labels2">The second set of cluster labels (e.g., ground truth).</param>
    /// <returns>The Adjusted Rand Index value, ranging from -1 to 1, where 1 indicates perfect agreement.</returns>
    /// <remarks>
    /// <para>
    /// The Adjusted Rand Index (ARI) is a measure of the similarity between two data clusterings, adjusted for chance.
    /// It computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs
    /// that are assigned in the same or different clusters in the predicted and true clusterings.
    /// </para>
    /// <para><b>For Beginners:</b> This method compares two different ways of grouping the same data and tells you
    /// how similar they are, while accounting for random chance.
    ///
    /// The score ranges from -1 to 1:
    /// - 1.0 means the two clusterings are identical
    /// - 0.0 means the agreement is what you'd expect from random clustering
    /// - Negative values mean the clusterings are worse than random
    ///
    /// This is useful for:
    /// - Comparing your clustering results to known "ground truth" labels
    /// - Evaluating how stable your clustering algorithm is across different runs
    /// - Comparing different clustering algorithms on the same data
    ///
    /// For example, if you cluster customer data and want to see how well it matches manually-defined segments,
    /// ARI tells you how similar your automated clustering is to the manual segmentation.
    /// </para>
    /// </remarks>
    public static T CalculateAdjustedRandIndex(Vector<T> labels1, Vector<T> labels2)
    {
        if (labels1.Length != labels2.Length)
        {
            throw new ArgumentException("Label vectors must have the same length");
        }

        int n = labels1.Length;

        // Handle edge case where n < 2
        if (n < 2)
        {
            return _numOps.One; // Perfect agreement for trivial cases
        }

        // Build contingency table with type-safe keys
        // Note: Suppressing CS8714 because T is constrained by INumericOperations which ensures
        // valid comparison semantics. The notnull constraint would break backwards compatibility.
#pragma warning disable CS8714
        var contingencyTable = new Dictionary<(T, T), int>(EqualityComparer<(T, T)>.Default);
        var rowSums = new Dictionary<T, int>(EqualityComparer<T>.Default);
        var colSums = new Dictionary<T, int>(EqualityComparer<T>.Default);
#pragma warning restore CS8714

        for (int i = 0; i < n; i++)
        {
            T label1 = labels1[i];
            T label2 = labels2[i];

            var key = (label1, label2);
            if (!contingencyTable.TryGetValue(key, out var cellCount))
                cellCount = 0;
            contingencyTable[key] = cellCount + 1;

            if (!rowSums.TryGetValue(label1, out var rowCount))
                rowCount = 0;
            rowSums[label1] = rowCount + 1;

            if (!colSums.TryGetValue(label2, out var colCount))
                colCount = 0;
            colSums[label2] = colCount + 1;
        }

        // Calculate sum of combinations for contingency table entries
        T sumCombinations = _numOps.Zero;
        foreach (var count in contingencyTable.Values.Where(c => c > 1))
        {
            sumCombinations = _numOps.Add(sumCombinations, _numOps.FromDouble((long)count * (count - 1) / 2.0));
        }

        // Calculate sum of combinations for row sums (a_i)
        T sumRowCombinations = _numOps.Zero;
        foreach (var count in rowSums.Values.Where(c => c > 1))
        {
            sumRowCombinations = _numOps.Add(sumRowCombinations, _numOps.FromDouble((long)count * (count - 1) / 2.0));
        }

        // Calculate sum of combinations for column sums (b_j)
        T sumColCombinations = _numOps.Zero;
        foreach (var count in colSums.Values.Where(c => c > 1))
        {
            sumColCombinations = _numOps.Add(sumColCombinations, _numOps.FromDouble((long)count * (count - 1) / 2.0));
        }

        // Total number of pairs
        T totalCombinations = _numOps.FromDouble((long)n * (n - 1) / 2.0);

        // Calculate expected index
        T expectedIndex = _numOps.Divide(
            _numOps.Multiply(sumRowCombinations, sumColCombinations),
            totalCombinations
        );

        // Calculate max index
        T maxIndex = _numOps.Divide(
            _numOps.Add(sumRowCombinations, sumColCombinations),
            _numOps.FromDouble(2.0)
        );

        // Calculate Adjusted Rand Index
        T numerator = _numOps.Subtract(sumCombinations, expectedIndex);
        T denominator = _numOps.Subtract(maxIndex, expectedIndex);

        // Handle edge case where denominator is zero
        if (_numOps.Equals(denominator, _numOps.Zero))
        {
            return _numOps.One;
        }

        return _numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Calculates the average distance from all points in a cluster to the cluster's centroid.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="data">The data matrix where each row is an observation.</param>
    /// <param name="labels">The cluster labels for each observation.</param>
    /// <param name="label">The specific cluster label to calculate for.</param>
    /// <param name="centroid">The centroid of the specified cluster.</param>
    /// <returns>The average distance from points to the centroid.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method calculates the average distance from all points in a specific
    /// cluster to that cluster's centroid. It's a measure of cluster scatter or dispersion—how spread out
    /// the points in a cluster are. The method identifies all points with the specified label, calculates
    /// the Euclidean distance from each point to the centroid, and returns the average. Lower average
    /// distances indicate more compact, homogeneous clusters. This measure is used in clustering evaluation
    /// metrics like the Davies-Bouldin Index to assess the compactness of clusters. It helps determine
    /// whether a clustering algorithm has successfully grouped similar points together, which is a key
    /// aspect of clustering quality.
    /// </para>
    /// </remarks>
    private static T CalculateAverageDistance(Matrix<T> data, Vector<T> labels, string label, Vector<T> centroid)
    {
        T sum = _numOps.Zero;
        int count = 0;

        for (int i = 0; i < data.Rows; i++)
        {
            if ((labels[i]?.ToString() ?? string.Empty) == label)
            {
                sum = _numOps.Add(sum, EuclideanDistance(data.GetRow(i), centroid));
                count++;
            }
        }

        return count > 0 ? _numOps.Divide(sum, _numOps.FromDouble(count)) : _numOps.Zero;
    }

    /// <summary>
    /// Calculates the Mean Average Precision (MAP) at k for a ranking task.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual relevance scores.</param>
    /// <param name="predicted">The predicted scores used for ranking.</param>
    /// <param name="k">The number of top items to consider.</param>
    /// <returns>The Mean Average Precision at k.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Average Precision (MAP) is a metric for evaluating ranking algorithms, 
    /// particularly in information retrieval and recommendation systems. It measures how well a system ranks 
    /// relevant items higher than irrelevant ones. This method calculates MAP by first sorting items by their 
    /// predicted scores, then for each relevant item in the top k positions, calculating the precision at 
    /// that position (the fraction of relevant items up to that point) and averaging these precision values. 
    /// MAP ranges from 0 to 1, with higher values indicating better ranking performance. It's particularly 
    /// useful because it considers both the order of relevant items and their positions in the ranking. 
    /// Unlike metrics that only count relevant items, MAP rewards algorithms that place relevant items 
    /// higher in the ranking.
    /// </para>
    /// </remarks>
    public static T CalculateMeanAveragePrecision(Vector<T> actual, Vector<T> predicted, int k)
    {
        var sortedPairs = actual.Zip(predicted, (a, p) => (Actual: a, Predicted: p))
                                .OrderByDescending(pair => pair.Predicted)
                                .Take(k)
                                .ToList();

        T sum = _numOps.Zero;
        T relevantCount = _numOps.Zero;

        for (int i = 0; i < sortedPairs.Count; i++)
        {
            if (_numOps.GreaterThan(sortedPairs[i].Actual, _numOps.Zero))
            {
                relevantCount = _numOps.Add(relevantCount, _numOps.One);
                sum = _numOps.Add(sum, _numOps.Divide(relevantCount, _numOps.FromDouble(i + 1)));
            }
        }

        return _numOps.Divide(sum, relevantCount);
    }

    /// <summary>
    /// Calculates the Normalized Discounted Cumulative Gain (NDCG) at k for a ranking task.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual relevance scores.</param>
    /// <param name="predicted">The predicted scores used for ranking.</param>
    /// <param name="k">The number of top items to consider.</param>
    /// <returns>The NDCG at k.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Normalized Discounted Cumulative Gain (NDCG) is a metric for evaluating ranking 
    /// algorithms that accounts for both the relevance of items and their positions in the ranking. It's 
    /// particularly useful when items have graded relevance (not just relevant/irrelevant). This method 
    /// calculates NDCG by first computing the Discounted Cumulative Gain (DCG), which sums the relevance 
    /// scores of items divided by the logarithm of their position (to discount items lower in the ranking). 
    /// It then normalizes this by dividing by the Ideal DCG (IDCG), which is the DCG of the perfect ranking. 
    /// NDCG ranges from 0 to 1, with 1 indicating a perfect ranking. It's widely used in search engines, 
    /// recommendation systems, and other ranking applications because it captures both the quality and 
    /// position of relevant items.
    /// </para>
    /// </remarks>
    public static T CalculateNDCG(Vector<T> actual, Vector<T> predicted, int k)
    {
        var sortedPairs = actual.Zip(predicted, (a, p) => (Actual: a, Predicted: p))
                                .OrderByDescending(pair => pair.Predicted)
                                .Take(k)
                                .ToList();

        T dcg = _numOps.Zero;
        T idcg = _numOps.Zero;

        for (int i = 0; i < sortedPairs.Count; i++)
        {
            T relevance = sortedPairs[i].Actual;
            T position = _numOps.FromDouble(i + 1);
            T logDenominator = _numOps.Log(_numOps.Add(position, _numOps.One));
            dcg = _numOps.Add(dcg, _numOps.Divide(relevance, logDenominator));
        }

        var idealOrder = sortedPairs.OrderByDescending(pair => pair.Actual).ToList();
        for (int i = 0; i < idealOrder.Count; i++)
        {
            T relevance = idealOrder[i].Actual;
            T position = _numOps.FromDouble(i + 1);
            T logDenominator = _numOps.Log(_numOps.Add(position, _numOps.One));
            idcg = _numOps.Add(idcg, _numOps.Divide(relevance, logDenominator));
        }

        return _numOps.Divide(dcg, idcg);
    }

    /// <summary>
    /// Calculates the Mean Reciprocal Rank (MRR) for a ranking task.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual relevance scores.</param>
    /// <param name="predicted">The predicted scores used for ranking.</param>
    /// <returns>The Mean Reciprocal Rank.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Reciprocal Rank (MRR) is a metric for evaluating ranking algorithms that 
    /// focuses on the position of the first relevant item. It's calculated as the reciprocal of the rank of 
    /// the first relevant item (1/rank). This method sorts items by their predicted scores and returns the 
    /// reciprocal of the position of the first item with a positive actual score. MRR ranges from 0 to 1, 
    /// with higher values indicating better performance. It's particularly useful in scenarios where the user 
    /// is likely to stop after finding the first relevant result, such as question answering or search engines. 
    /// Unlike metrics that consider all relevant items, MRR only cares about how quickly the system can 
    /// provide at least one correct answer, making it a good measure of the system's ability to quickly 
    /// satisfy the user's information need.
    /// </para>
    /// </remarks>
    public static T CalculateMeanReciprocalRank(Vector<T> actual, Vector<T> predicted)
    {
        var sortedPairs = actual.Zip(predicted, (a, p) => (Actual: a, Predicted: p))
                                .OrderByDescending(pair => pair.Predicted)
                                .ToList();

        for (int i = 0; i < sortedPairs.Count; i++)
        {
            if (_numOps.GreaterThan(sortedPairs[i].Actual, _numOps.Zero))
            {
                return _numOps.Divide(_numOps.One, _numOps.FromDouble(i + 1));
            }
        }

        return _numOps.Zero;
    }

    /// <summary>
    /// Calculates the Mean Squared Logarithmic Error (MSLE) between actual and predicted values.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual values.</param>
    /// <param name="predicted">The predicted values.</param>
    /// <returns>The Mean Squared Logarithmic Error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Squared Logarithmic Error (MSLE) is a regression metric that measures the 
    /// average squared difference between the logarithms of predicted and actual values. It's calculated by 
    /// taking the logarithm of both actual and predicted values (after adding 1 to avoid issues with zeros), 
    /// finding the squared differences, and averaging them. MSLE is particularly useful when you're more 
    /// concerned with relative errors than absolute ones, or when the target variable spans multiple orders 
    /// of magnitude. It penalizes underestimation more than overestimation, which is desirable in some 
    /// applications like demand forecasting where underestimating can be more costly. The logarithmic 
    /// transformation makes MSLE less sensitive to outliers compared to mean squared error (MSE), making 
    /// it suitable for datasets with skewed distributions.
    /// </para>
    /// </remarks>
    public static T CalculateMeanSquaredLogError(Vector<T> actual, Vector<T> predicted)
    {
        var sum = _numOps.Zero;
        var n = actual.Length;

        for (int i = 0; i < n; i++)
        {
            var logActual = _numOps.Log(_numOps.Add(actual[i], _numOps.One));
            var logPredicted = _numOps.Log(_numOps.Add(predicted[i], _numOps.One));
            var diff = _numOps.Subtract(logActual, logPredicted);
            sum = _numOps.Add(sum, _numOps.Multiply(diff, diff));
        }

        return _numOps.Divide(sum, _numOps.FromDouble(n));
    }

    /// <summary>
    /// Calculates the Continuous Ranked Probability Score (CRPS) for probabilistic forecasts.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed values.</param>
    /// <param name="predictedMean">The predicted mean values (point forecasts).</param>
    /// <param name="predictedStdDev">The predicted standard deviations (uncertainty estimates).</param>
    /// <returns>The average CRPS value across all observations. Lower values indicate better probabilistic forecasts.</returns>
    /// <remarks>
    /// <para>
    /// The Continuous Ranked Probability Score (CRPS) is a proper scoring rule that measures the accuracy
    /// of probabilistic predictions. It compares the predicted cumulative distribution function (CDF)
    /// with the observed value, rewarding forecasts that assign high probability to what actually happens.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> CRPS evaluates how well a probabilistic forecast (one that includes uncertainty)
    /// matches reality. Unlike MAE which only looks at point predictions, CRPS considers the entire
    /// predicted probability distribution.
    ///
    /// Imagine a weather forecast that says "temperature will be 20°C ± 3°C" (mean=20, std=3).
    /// If the actual temperature is 20°C, that's a good forecast.
    /// If the actual temperature is 25°C, that's outside the predicted range - not as good.
    /// CRPS quantifies this, penalizing forecasts that are both inaccurate and overconfident.
    ///
    /// Key properties:
    /// - Lower CRPS is better (like MAE)
    /// - When predictions have zero uncertainty, CRPS equals MAE
    /// - CRPS rewards well-calibrated uncertainty estimates
    /// - Units are the same as the predicted variable
    ///
    /// This implementation assumes Gaussian (normal) predictive distributions, which is appropriate
    /// for most time series forecasting models like DeepAR, TFT, and Chronos.
    /// </para>
    /// </remarks>
    public static T CalculateCRPS(Vector<T> actual, Vector<T> predictedMean, Vector<T> predictedStdDev)
    {
        if (actual.Length != predictedMean.Length || actual.Length != predictedStdDev.Length)
            throw new ArgumentException("All input vectors must have the same length.");

        var n = actual.Length;
        if (n == 0) return _numOps.Zero;

        var totalCrps = _numOps.Zero;
        // Correct constant for Gaussian CRPS: 1/sqrt(pi) (not sqrt(2/pi))
        // Reference: Gneiting & Raftery (2007), "Strictly Proper Scoring Rules, Prediction, and Estimation"
        var oneOverSqrtPi = _numOps.FromDouble(1.0 / Math.Sqrt(Math.PI));

        for (int i = 0; i < n; i++)
        {
            var y = actual[i];
            var mu = predictedMean[i];
            var sigma = predictedStdDev[i];

            // Handle case where sigma is zero or very small (deterministic prediction)
            if (_numOps.ToDouble(sigma) < 1e-10)
            {
                // CRPS reduces to absolute error for deterministic predictions
                totalCrps = _numOps.Add(totalCrps, _numOps.Abs(_numOps.Subtract(y, mu)));
            }
            else
            {
                // Standardized observation: z = (y - mu) / sigma
                var z = _numOps.Divide(_numOps.Subtract(y, mu), sigma);

                // Standard normal CDF at z: Phi(z)
                var phi = _numOps.FromDouble(NormalCDF(_numOps.ToDouble(z)));

                // Standard normal PDF at z: phi(z)
                var pdf = _numOps.FromDouble(NormalPDF(_numOps.ToDouble(z)));

                // CRPS for Gaussian: sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
                var term1 = _numOps.Multiply(z, _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(2.0), phi), _numOps.One));
                var term2 = _numOps.Multiply(_numOps.FromDouble(2.0), pdf);
                var term3 = oneOverSqrtPi;

                var crpsSingle = _numOps.Multiply(sigma, _numOps.Subtract(_numOps.Add(term1, term2), term3));
                totalCrps = _numOps.Add(totalCrps, crpsSingle);
            }
        }

        return _numOps.Divide(totalCrps, _numOps.FromDouble(n));
    }

    /// <summary>
    /// Calculates the CRPS for point predictions (assumes zero uncertainty).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">The actual observed values.</param>
    /// <param name="predicted">The predicted values (point forecasts without uncertainty).</param>
    /// <returns>The average CRPS value, which equals MAE for deterministic predictions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This overload is for deterministic predictions (no uncertainty estimates).
    /// In this case, CRPS is equivalent to Mean Absolute Error (MAE).
    /// Use the overload with predictedStdDev for probabilistic forecasts.
    /// </para>
    /// </remarks>
    public static T CalculateCRPS(Vector<T> actual, Vector<T> predicted)
    {
        // For deterministic predictions, CRPS equals MAE
        return CalculateMeanAbsoluteError(actual, predicted);
    }

    /// <summary>
    /// Standard normal cumulative distribution function.
    /// </summary>
    private static double NormalCDF(double x)
    {
        // Approximation using error function
        return 0.5 * (1 + Erf(x / Math.Sqrt(2)));
    }

    /// <summary>
    /// Standard normal probability density function.
    /// </summary>
    private static double NormalPDF(double x)
    {
        return Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
    }

    /// <summary>
    /// Error function approximation (Abramowitz and Stegun).
    /// </summary>
    private static double Erf(double x)
    {
        // Constants for approximation
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        // Save the sign of x
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);

        // Abramowitz and Stegun formula 7.1.26
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }

    /// <summary>
    /// Calculates the Dynamic Time Warping (DTW) distance between two time series.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="series1">The first time series.</param>
    /// <param name="series2">The second time series.</param>
    /// <returns>The DTW distance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dynamic Time Warping (DTW) is a technique for measuring similarity between two 
    /// temporal sequences that may vary in speed or timing. Unlike Euclidean distance, which compares points 
    /// at the same time index, DTW finds the optimal alignment between sequences by warping the time axis. 
    /// This method implements DTW using dynamic programming to build a matrix of distances and find the path 
    /// with minimal cumulative distance. Lower DTW distances indicate more similar sequences. DTW is particularly 
    /// useful for comparing patterns in time series data, such as speech recognition, gesture recognition, 
    /// and financial time series analysis. It can detect similarities even when sequences are shifted, stretched, 
    /// or compressed in time, making it more robust than point-by-point comparison methods.
    /// </para>
    /// </remarks>
    public static T CalculateDynamicTimeWarping(Vector<T> series1, Vector<T> series2)
    {
        int n = series1.Length;
        int m = series2.Length;
        var dtw = new T[n + 1, m + 1];

        for (int i = 0; i <= n; i++)
            for (int j = 0; j <= m; j++)
                dtw[i, j] = _numOps.MaxValue;

        dtw[0, 0] = _numOps.Zero;

        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                var cost = _numOps.Abs(_numOps.Subtract(series1[i - 1], series2[j - 1]));
                dtw[i, j] = _numOps.Add(cost, MathHelper.Min(dtw[i - 1, j], MathHelper.Min(dtw[i, j - 1], dtw[i - 1, j - 1])));
            }
        }

        return dtw[n, m];
    }

    /// <summary>
    /// Calculates the autocorrelation function (ACF) for a time series up to a specified maximum lag.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="series">The time series data.</param>
    /// <param name="maxLag">The maximum lag to calculate autocorrelation for.</param>
    /// <returns>A vector containing autocorrelation values for lags 0 to maxLag.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The autocorrelation function (ACF) measures the correlation between a time series 
    /// and a lagged version of itself. It helps identify patterns, seasonality, and the degree of randomness 
    /// in time series data. This method calculates autocorrelation for lags from 0 to maxLag by first computing 
    /// the mean and variance of the series, then for each lag, calculating the sum of products of deviations 
    /// from the mean for points separated by that lag, and finally normalizing by the variance. ACF values 
    /// range from -1 to 1, with values close to 1 indicating strong positive correlation, values close to -1 
    /// indicating strong negative correlation, and values close to 0 indicating little correlation. The ACF 
    /// is a fundamental tool in time series analysis, used for model identification in ARIMA modeling, detecting 
    /// seasonality, and testing for randomness.
    /// </para>
    /// </remarks>
    public static Vector<T> CalculateAutoCorrelationFunction(Vector<T> series, int maxLag)
    {
        if (series == null)
        {
            throw new ArgumentNullException(nameof(series), "Time series cannot be null.");
        }

        var n = series.Length;

        if (n < 2)
        {
            throw new ArgumentException("Time series must have at least 2 observations for autocorrelation.", nameof(series));
        }

        if (maxLag < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLag), "Maximum lag must be non-negative.");
        }

        if (maxLag >= n)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLag), 
                $"Maximum lag ({maxLag}) must be less than series length ({n}).");
        }

        var acf = new T[maxLag + 1];
        var mean = CalculateMean(series);

        // Calculate the total sum of squared deviations (denominator for normalization)
        var totalSumSquares = _numOps.Zero;
        for (int i = 0; i < n; i++)
        {
            var deviation = _numOps.Subtract(series[i], mean);
            totalSumSquares = _numOps.Add(totalSumSquares, _numOps.Multiply(deviation, deviation));
        }

        // Check for constant series (zero variance) - ACF is undefined
        if (_numOps.Equals(totalSumSquares, _numOps.Zero))
        {
            throw new ArgumentException(
                "Autocorrelation is undefined for a constant series (zero variance). " +
                "All values in the series are identical.",
                nameof(series));
        }

        for (int lag = 0; lag <= maxLag; lag++)
        {
            var sum = _numOps.Zero;
            for (int i = 0; i < n - lag; i++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(
                    _numOps.Subtract(series[i], mean),
                    _numOps.Subtract(series[i + lag], mean)));
            }
            acf[lag] = _numOps.Divide(sum, totalSumSquares);
        }

        return new Vector<T>(acf);
    }

    /// <summary>
    /// Calculates the partial autocorrelation function (PACF) for a time series up to a specified maximum lag.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="series">The time series data.</param>
    /// <param name="maxLag">The maximum lag to calculate partial autocorrelation for.</param>
    /// <returns>A vector containing partial autocorrelation values for lags 0 to maxLag.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The partial autocorrelation function (PACF) measures the correlation between a 
    /// time series and a lagged version of itself, after removing the effects of intermediate lags. While the 
    /// ACF shows all correlations (direct and indirect), the PACF isolates the direct correlation at each lag. 
    /// This method implements the Durbin-Levinson algorithm to calculate partial autocorrelations, which involves 
    /// solving a system of equations based on the regular autocorrelations. PACF values also range from -1 to 1, 
    /// with the interpretation similar to ACF. The PACF is particularly useful for identifying the order of an 
    /// autoregressive (AR) model in time series analysis. A significant spike at lag k in the PACF suggests 
    /// that an AR(k) model might be appropriate. Together with the ACF, the PACF provides crucial information 
    /// for time series model selection and specification.
    /// </para>
    /// </remarks>
    public static Vector<T> CalculatePartialAutoCorrelationFunction(Vector<T> series, int maxLag)
    {
        if (series == null)
        {
            throw new ArgumentNullException(nameof(series), "Time series cannot be null.");
        }

        var n = series.Length;

        if (n < 2)
        {
            throw new ArgumentException("Time series must have at least 2 observations for partial autocorrelation.", nameof(series));
        }

        if (maxLag < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLag), "Maximum lag must be non-negative.");
        }

        if (maxLag >= n)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLag), 
                $"Maximum lag ({maxLag}) must be less than series length ({n}).");
        }

        var pacf = new T[maxLag + 1];
        pacf[0] = _numOps.One;

        // Compute ACF on the full series once (correct Durbin-Levinson approach)
        var acf = CalculateAutoCorrelationFunction(series, maxLag);

        var phi = new T[maxLag + 1, maxLag + 1];

        for (int k = 1; k <= maxLag; k++)
        {
            // Durbin-Levinson recursion
            var numerator = acf[k];
            for (int j = 1; j < k; j++)
            {
                numerator = _numOps.Subtract(numerator, _numOps.Multiply(phi[k - 1, j], acf[k - j]));
            }

            var denominator = _numOps.One;
            for (int j = 1; j < k; j++)
            {
                denominator = _numOps.Subtract(denominator, _numOps.Multiply(phi[k - 1, j], acf[j]));
            }

            phi[k, k] = _numOps.Divide(numerator, denominator);
            pacf[k] = phi[k, k];

            // Update phi coefficients for next iteration
            if (k < maxLag)
            {
                for (int j = 1; j < k; j++)
                {
                    phi[k, j] = _numOps.Subtract(phi[k - 1, j], _numOps.Multiply(phi[k, k], phi[k - 1, k - j]));
                }
            }
        }

        return new Vector<T>(pacf);
    }
}
