namespace AiDotNet.Enums;

/// <summary>
/// Defines the types of metrics used to evaluate machine learning models.
/// </summary>
public enum MetricType
{
    /// <summary>
    /// Coefficient of determination, measuring how well the model explains the variance in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> R� (R-squared) tells you how well your model fits the data, on a scale from 0 to 1.
    /// A value of 1 means your model perfectly predicts the data, while 0 means it's no better than
    /// just guessing the average value. For example, an R� of 0.75 means your model explains 75% of
    /// the variation in the data.
    /// </para>
    /// </remarks>
    R2,
    
    /// <summary>
    /// A modified version of R� that accounts for the number of predictors in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adjusted R� is similar to R�, but it penalizes you for adding too many input variables
    /// that don't help much. This prevents "overfitting" - when your model becomes too complex and starts
    /// memorizing the training data rather than learning general patterns. Use this instead of regular R�
    /// when comparing models with different numbers of input variables.
    /// </para>
    /// </remarks>
    AdjustedR2,

    /// <summary>
    /// Mean Squared Error - measures the average squared difference between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Squared Error (MSE) tells you how far off your predictions are from the actual values.
    /// It squares the differences, which means larger errors are penalized more heavily. Lower MSE values indicate better model performance.
    /// </para>
    /// </remarks>
    MeanSquaredError,

    /// <summary>
    /// Root Mean Squared Error - the square root of the mean squared error, in the same units as the target variable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Root Mean Squared Error (RMSE) is like MSE but takes the square root at the end,
    /// which puts it back in the same units as your original data. This makes it easier to interpret.
    /// For example, if predicting house prices in dollars, RMSE will also be in dollars.
    /// </para>
    /// </remarks>
    RootMeanSquaredError,

    /// <summary>
    /// Mean Absolute Error - measures the average absolute difference between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Absolute Error (MAE) tells you the average difference between your predictions
    /// and actual values, ignoring whether the errors are positive or negative. It's easier to understand than MSE
    /// because it doesn't square the errors - a MAE of 5 means your predictions are off by 5 units on average.
    /// </para>
    /// </remarks>
    MeanAbsoluteError,

    /// <summary>
    /// R-Squared - coefficient of determination measuring the proportion of variance explained by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RSquared is another way to refer to R� (R-squared), which measures how well
    /// your model explains the variation in the data on a scale from 0 to 1.
    /// </para>
    /// </remarks>
    RSquared,

    /// <summary>
    /// Area Under the Curve - measures the area under the ROC curve for classification performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AUC (Area Under the Curve) measures how well your model can distinguish between classes.
    /// A value of 1.0 means perfect classification, 0.5 means random guessing, and values above 0.7 are generally considered good.
    /// </para>
    /// </remarks>
    AUC,

    /// <summary>
    /// The smallest value in the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The minimum is simply the smallest number in your data.
    /// 
    /// For the numbers [5, 12, 3, 8, 9], the minimum is 3.
    /// </para>
    /// </remarks>
    Min,

    /// <summary>
    /// The largest value in the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The maximum is simply the largest number in your data.
    /// 
    /// For the numbers [5, 12, 3, 8, 9], the maximum is 12.
    /// </para>
    /// </remarks>
    Max,

    /// <summary>
    /// The number of values in the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> N simply counts how many data points you have in your dataset.
    /// 
    /// For the numbers [5, 12, 3, 8, 9], N is 5.
    /// </para>
    /// </remarks>
    N,

    /// <summary>
    /// The value below which 25% of the observations may be found.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The First Quartile (Q1) is the value that separates the lowest 25% of your data from the rest.
    /// 
    /// If you line up all your data from smallest to largest, Q1 is the value one-quarter of the way through the list.
    /// It helps you understand the spread of the lower half of your data.
    /// </para>
    /// </remarks>
    FirstQuartile,

    /// <summary>
    /// The value below which 75% of the observations may be found.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Third Quartile (Q3) is the value that separates the highest 25% of your data from the rest.
    /// 
    /// If you line up all your data from smallest to largest, Q3 is the value three-quarters of the way through the list.
    /// It helps you understand the spread of the upper half of your data.
    /// </para>
    /// </remarks>
    ThirdQuartile,

    /// <summary>
    /// A measure of the randomness or unpredictability in a dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Entropy measures how unpredictable or random your data is.
    /// 
    /// High entropy means your data is very diverse and unpredictable - like a completely random sequence of numbers.
    /// Low entropy means your data is more predictable or has patterns - like a sequence that repeats.
    /// 
    /// For example, the sequence [1, 2, 3, 4, 5] has lower entropy than [7, 2, 9, 1, 4] because the first one follows
    /// a clear pattern while the second appears more random.
    /// </para>
    /// </remarks>
    Entropy,

    /// <summary>
    /// Identifies which statistical distribution best fits the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> BestDistributionType tells you which common statistical pattern your data follows.
    /// 
    /// Data can follow different patterns like:
    /// - Normal (bell curve): common for natural phenomena like heights
    /// - Exponential: common for waiting times
    /// - Poisson: common for counting rare events
    /// 
    /// Knowing which distribution your data follows helps you make better predictions and understand the underlying process.
    /// </para>
    /// </remarks>
    BestDistributionType,

    /// <summary>
    /// A statistical measure that indicates how well a distribution fits the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GoodnessOfFitPValue tells you how well a statistical distribution matches your data.
    /// 
    /// It ranges from 0 to 1:
    /// - Values closer to 1 suggest a good fit (the distribution matches your data well)
    /// - Values closer to 0 suggest a poor fit (the distribution doesn't match your data well)
    /// 
    /// For example, if testing whether your data follows a normal distribution gives a p-value of 0.8,
    /// it suggests your data is very likely to follow a normal distribution.
    /// </para>
    /// </remarks>
    GoodnessOfFitPValue,

    /// <summary>
    /// Measures the correlation between a time series and a lagged version of itself.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutocorrelationCoefficient measures how similar your time series data is to itself at different time delays.
    /// 
    /// It ranges from -1 to 1:
    /// - Values close to 1 indicate strong positive correlation (patterns repeat)
    /// - Values close to -1 indicate strong negative correlation (patterns invert)
    /// - Values close to 0 indicate little correlation (no pattern)
    /// 
    /// For example, daily temperature data might have a high autocorrelation at a lag of 24 hours,
    /// showing that temperatures at the same time each day are related.
    /// </para>
    /// </remarks>
    AutocorrelationCoefficient,

    /// <summary>
    /// Variance Inflation Factor - Measures multicollinearity among features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VIF helps identify if some features are too similar to others.
    /// High VIF values (usually above 5 or 10) suggest that a feature might be redundant,
    /// as its information is already captured by other features. This can cause problems
    /// for your model because these overlapping features don't provide unique information.
    /// 
    /// For example, if you include both "years of education" and "highest degree obtained" 
    /// in your model, they might have high VIF values because they capture similar information.
    /// </para>
    /// </remarks>
    VIF,

    /// <summary>
    /// Leave-One-Out Predictive Densities - Measures model performance when training without each data point.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This shows how well the model predicts each data point when it's trained without that point.
    /// It helps identify which data points might be harder for the model to predict accurately.
    /// 
    /// Think of it like taking a practice test where for each question, you first study everything except that topic,
    /// then try to answer it. The result helps you see which topics you can predict based on your knowledge of other topics,
    /// and which ones are more independent or difficult.
    /// </para>
    /// </remarks>
    LeaveOneOutPredictiveDensities,

    /// <summary>
    /// Correlation Matrix<double> - Shows the relationships between features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Correlation Matrix<double> shows how closely related each pair of features is to each other.
    /// Each cell in the matrix contains a correlation coefficient between two features, ranging from -1 to 1:
    /// - Values close to 1 indicate strong positive correlation (when one goes up, the other tends to go up)
    /// - Values close to -1 indicate strong negative correlation (when one goes up, the other tends to go down)
    /// - Values close to 0 indicate little to no relationship
    /// 
    /// This helps you understand which features might be providing similar information. For example,
    /// if "height" and "weight" have a correlation of 0.8, they contain somewhat redundant information.
    /// </para>
    /// </remarks>
    CorrelationMatrix,

    /// <summary>
    /// Covariance Matrix<double> - Shows how features vary together.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Covariance Matrix<double> shows how features change together, but unlike
    /// the Correlation Matrix<double>, it isn't standardized to a -1 to 1 scale. 
    /// 
    /// Each cell in the matrix contains the covariance between two features:
    /// - Positive values indicate that the features tend to increase or decrease together
    /// - Negative values indicate that as one feature increases, the other tends to decrease
    /// - The magnitude depends on the scales of the features
    /// 
    /// The diagonal elements contain the variances of each feature (how much they vary on their own).
    /// The Covariance Matrix<double> is important for many statistical techniques and is used in calculations
    /// like Mahalanobis Distance and Principal Component Analysis.
    /// </para>
    /// </remarks>
    CovarianceMatrix,

    /// <summary>
    /// Posterior Predictive Samples - Provides possible predictions based on the model's uncertainty.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are possible predictions your model might make if you ran it multiple times.
    /// They help you understand the range and uncertainty of your model's predictions.
    /// 
    /// Instead of getting a single prediction (like "house price = $300,000"), posterior predictive samples
    /// might give you multiple possible predictions (like $290,000, $305,000, $315,000). This helps you
    /// understand how confident your model is and what range of outcomes might be reasonable.
    /// </para>
    /// </remarks>
    PosteriorPredictiveSamples,

    /// <summary>
    /// Auto-Correlation Function - Measures the correlation between a time series and lagged versions of itself.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This function helps you understand patterns in time-based data.
    /// It shows how similar your data is to itself at different time delays. This can reveal:
    /// - Repeating patterns (like seasonal effects)
    /// - How long effects last in your data
    /// - If your model is missing important time-based patterns
    /// 
    /// For example, for daily temperature data, the ACF might show a strong correlation at lag 365,
    /// indicating an annual seasonal pattern.
    /// </para>
    /// </remarks>
    AutoCorrelationFunction,

    /// <summary>
    /// Partial Auto-Correlation Function - Measures the direct relationship between an observation and its lag.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This function is similar to the Auto-Correlation Function, but it focuses on the direct relationship
    /// between data points at different time delays, removing the influence of points in between.
    /// It helps you:
    /// - Identify how many past time points directly influence the current point
    /// - Decide how many past observations to use in time series models
    /// - Understand the "memory" of your time series data
    /// 
    /// For example, if PACF shows strong correlation at lag 1 but not beyond, it suggests each value
    /// is directly influenced only by the immediately preceding value.
    /// </para>
    /// </remarks>
    PartialAutoCorrelationFunction,

    /// <summary>
    /// The standard error of the mean, measuring the precision of the sample mean.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Standard Error measures how precisely we know the true population mean based on our sample data.
    /// It's calculated as the standard deviation divided by the square root of the sample size.
    /// As sample size increases, standard error decreases, indicating more precise estimates.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Standard Error tells you how accurately you know the average. Think of it as the "margin of error" 
    /// around your average.
    /// </para>
    /// <para>
    /// For example, if you calculate an average of 100 with a standard error of 5, the true population average is likely
    /// between 95 and 105. A smaller standard error means you can be more confident in your average value. As you collect 
    /// more data, your standard error typically gets smaller, indicating a more precise estimate.
    /// </para>
    /// </remarks>
    StandardError,

    /// <summary>
    /// The lower bound of the confidence interval for the mean.
    /// </summary>
    /// <remarks>
    /// <para>
    /// MeanConfidenceIntervalLower represents the lower boundary of a confidence interval for the population mean.
    /// It's calculated as the sample mean minus a critical value multiplied by the standard error.
    /// Confidence intervals provide a range of plausible values for the true population parameter.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the lower end of a range where we believe the true average likely falls.
    /// </para>
    /// <para>
    /// For example, if your MeanConfidenceIntervalLower is 95, it means you're reasonably confident that the true average
    /// is at least 95. The width of this interval depends on your confidence level (typically 95%) and how much data you have.
    /// More data usually results in narrower, more precise confidence intervals.
    /// </para>
    /// </remarks>
    MeanConfidenceIntervalLower,

    /// <summary>
    /// The upper bound of the confidence interval for the mean.
    /// </summary>
    /// <remarks>
    /// <para>
    /// MeanConfidenceIntervalUpper represents the upper boundary of a confidence interval for the population mean.
    /// It's calculated as the sample mean plus a critical value multiplied by the standard error.
    /// Together with MeanConfidenceIntervalLower, it provides a range of plausible values for the true mean.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the upper end of a range where we believe the true average likely falls.
    /// </para>
    /// <para>
    /// For example, if your MeanConfidenceIntervalUpper is 105, it means you're reasonably confident that the true average
    /// is at most 105. When combined with the lower bound, you get a range (like 95 to 105) that likely contains the true 
    /// population average. The narrower this range, the more precise your estimate is.
    /// </para>
    /// </remarks>
    MeanConfidenceIntervalUpper,

    /// <summary>
    /// The ratio of the standard deviation to the mean, measuring relative variability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Coefficient of Variation (CV) is a standardized measure of dispersion that allows comparison of variability
    /// across different scales. It's calculated as the standard deviation divided by the mean and often expressed as a percentage.
    /// Lower values indicate less relative variability.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Coefficient of Variation tells you how spread out your data is relative to its average.
    /// </para>
    /// <para>
    /// For example, if one dataset has a CV of 5% and another has a CV of 15%, the second dataset has more variability 
    /// relative to its average. This is useful when comparing datasets with different units or vastly different means.
    /// For instance, comparing the variability of heights (measured in cm) with weights (measured in kg) would be 
    /// difficult with standard deviation alone, but CV makes this comparison meaningful.
    /// </para>
    /// </remarks>
    CoefficientOfVariation,

    /// <summary>
    /// A normalized version of the median absolute deviation for improved interpretation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// StandardizedMAD adjusts the Median Absolute Deviation (MAD) to make it comparable to standard deviation
    /// when data follows a normal distribution. It's calculated by dividing MAD by 0.6745 (a constant derived from
    /// the normal distribution). This standardization allows MAD to be interpreted on the same scale as standard deviation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> StandardizedMAD is a robust way to measure spread that works well even with outliers,
    /// but is adjusted so you can compare it directly with standard deviation.
    /// </para>
    /// <para>
    /// If your data follows a bell curve, StandardizedMAD and standard deviation should be approximately equal.
    /// If StandardizedMAD is much lower than standard deviation, it suggests your data contains outliers that are
    /// inflating the standard deviation. This makes it a valuable tool for detecting when outliers might be affecting
    /// your analysis.
    /// </para>
    /// </remarks>
    StandardizedMAD,

    /// <summary>
    /// A scaling factor used for calculating robust Z-scores based on the median and MAD.
    /// </summary>
    /// <remarks>
    /// <para>
    /// RobustScale provides a factor (typically 1/MAD) that can be used to compute robust Z-scores by scaling
    /// the difference between a value and the median. Unlike traditional Z-scores that use mean and standard deviation,
    /// robust Z-scores use median and MAD, making them resistant to outliers.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> RobustScale helps you standardize your data in a way that isn't thrown off by extreme values.
    /// </para>
    /// <para>
    /// To use it, you multiply each value minus the median by this scale factor. The resulting scores tell you
    /// how many MADs away each point is from the median. This is particularly useful for outlier detection in
    /// datasets where the traditional mean and standard deviation might be heavily influenced by extreme values.
    /// </para>
    /// </remarks>
    RobustScale,

    /// <summary>
    /// A robust measure of skewness based on quartiles rather than moments.
    /// </summary>
    /// <remarks>
    /// <para>
    /// QuartileSkewness is a robust alternative to the traditional moment-based skewness. It's calculated as
    /// (Q3 + Q1 - 2�Median) / (Q3 - Q1), where Q1 and Q3 are the first and third quartiles. This measure is less
    /// sensitive to outliers than traditional skewness.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> QuartileSkewness tells you if your data is lopsided, but uses the middle values
    /// rather than every single point, so extreme values don't distort the result as much.
    /// </para>
    /// <para>
    /// Like traditional skewness, positive values indicate right skew (longer tail on right), negative values
    /// indicate left skew (longer tail on left), and values near zero suggest symmetry. This metric is especially
    /// useful when working with data that might contain outliers or when the distribution is not well-behaved.
    /// </para>
    /// </remarks>
    QuartileSkewness,

    /// <summary>
    /// A scaling factor used for calculating Z-scores.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ZScoreScale provides a factor (1/standard deviation) that can be used to compute Z-scores by scaling
    /// the difference between a value and the mean. Z-scores tell you how many standard deviations a value
    /// is from the mean, standardizing measurements across different scales.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> ZScoreScale is a factor that helps you standardize your data to see how unusual each value is.
    /// </para>
    /// <para>
    /// To use it, you multiply each value minus the mean by this scale factor. The resulting scores are in
    /// standard deviation units, where values between -2 and 2 are typically considered "normal" (covering about 95%
    /// of a normal distribution). This standardization allows for comparing values from different distributions
    /// on a common scale.
    /// </para>
    /// </remarks>
    ZScoreScale,

    /// <summary>
    /// Indicates which correlation coefficient best describes the relationship between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// BestCorrelationType identifies whether linear correlation (Pearson) or rank correlation (Spearman) 
    /// better captures the relationship between actual and predicted values. This is determined by comparing 
    /// the absolute values of different correlation coefficients and selecting the one with the strongest 
    /// relationship.
    /// </para>
    /// <para>
    /// This metric returns a numeric representation of the correlation type:
    /// - 1 represents Pearson correlation (linear relationship)
    /// - 2 represents Spearman correlation (monotonic but potentially non-linear relationship)
    /// - Additional values may represent other correlation types like Kendall's Tau
    /// </para>
    /// <para>
    /// Understanding which correlation type best fits your data provides insights into the nature of the 
    /// relationship that your model has captured:
    /// - If Pearson is best, your model has captured a linear relationship
    /// - If Spearman is best, your model has captured a monotonic but potentially non-linear relationship
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> BestCorrelationType tells you what kind of relationship exists between your 
    /// actual and predicted values.
    /// </para>
    /// <para>
    /// Think of it like this:
    /// - If Pearson correlation is strongest (value = 1), your predictions follow a straight-line relationship 
    ///   with the actual values (when one goes up, the other goes up proportionally)
    /// - If Spearman correlation is strongest (value = 2), your predictions follow the same up-and-down pattern 
    ///   as the actual values, but not necessarily in a straight line
    /// </para>
    /// <para>
    /// This helps you understand whether your model is capturing linear or more complex non-linear patterns 
    /// in your data, which can guide further model development and interpretation.
    /// </para>
    /// </remarks>
    BestCorrelationType,

    /// <summary>
    /// A measure of similarity between two temporal sequences.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Dynamic Time Warping (DTW) is a time series analysis technique that measures the similarity 
    /// between two temporal sequences, even if they vary in speed or timing. Unlike point-by-point 
    /// Euclidean distance, DTW can handle sequences of different lengths and sequences where 
    /// similar patterns occur at different times or rates.
    /// </para>
    /// <para>
    /// DTW works by finding the optimal alignment between two sequences by warping them non-linearly 
    /// in the time dimension. This allows for more precise similarity measurements between time series data 
    /// where timing differences are expected or irrelevant to the comparison.
    /// </para>
    /// <para>
    /// Smaller DTW values indicate greater similarity between sequences. A DTW value of zero would 
    /// indicate identical sequences.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Dynamic Time Warping measures how similar two sequences are, even if they're 
    /// not perfectly aligned in time. Think of it like comparing two audio recordings of the same song 
    /// where one singer performs some parts faster or slower than the other - DTW can still recognize 
    /// they're the same song.
    /// </para>
    /// <para>
    /// Unlike simple point-by-point comparisons, DTW can handle sequences that are stretched, 
    /// compressed, or shifted in time. This is particularly useful for comparing time series 
    /// where patterns might occur at different rates.
    /// </para>
    /// <para>
    /// Lower values indicate more similar sequences. This metric is especially useful for time series 
    /// data like speech recognition, gesture recognition, or any data where timing may vary but 
    /// the overall pattern is similar.
    /// </para>
    /// </remarks>
    DynamicTimeWarping,

    /// <summary>
    /// Information about the statistical distribution that best fits the prediction data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// BestDistributionFit analyzes the predictions to determine which statistical distribution they most closely follow.
    /// Unlike scalar metrics that return a single value, this returns a DistributionFitResult object containing
    /// the type of distribution (e.g., Normal, Poisson, Exponential) and its parameters.
    /// </para>
    /// <para>
    /// Understanding the underlying distribution of predictions is crucial for:
    /// - Creating accurate confidence and prediction intervals
    /// - Determining appropriate hypothesis tests
    /// - Validating assumptions in parametric models
    /// - Detecting anomalies or unexpected patterns in predictions
    /// </para>
    /// <para>
    /// The fitting process typically uses goodness-of-fit tests (like Kolmogorov-Smirnov or Anderson-Darling)
    /// to compare several candidate distributions and select the one with the best fit.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> BestDistributionFit helps you understand the shape of your prediction distribution.
    /// It tells you whether your predictions follow common patterns like the bell curve (Normal distribution),
    /// are skewed to one side, have long tails, etc.
    /// </para>
    /// <para>
    /// Knowing the distribution can help with:
    /// - Creating better intervals (by using the right distribution assumptions)
    /// - Understanding the types of predictions your model makes (are they normally distributed? skewed?)
    /// - Identifying potential issues with your predictions
    /// </para>
    /// <para>
    /// For example, if your house price predictions should follow a log-normal distribution (common for prices),
    /// but the distribution fit shows they're uniformly distributed, this might indicate a problem with your model.
    /// </para>
    /// </remarks>
    BestDistributionFit,

    /// <summary>
    /// A list of performance metrics calculated at different training set sizes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Learning curves show how model performance changes as the amount of training data increases.
    /// They are calculated by training the model on progressively larger subsets of the training data
    /// and evaluating performance at each step.
    /// </para>
    /// <para>
    /// Learning curves are valuable for diagnosing underfitting and overfitting:
    /// - If both training and validation performance quickly plateau at a low level, the model may be underfitting (high bias)
    /// - If there's a large gap between training and validation performance, the model may be overfitting (high variance)
    /// - If validation performance continues to improve as data increases, more data could help
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> LearningCurve shows how your model's performance changes as you give it more training data.
    /// </para>
    /// <para>
    /// This is helpful for diagnosing if your model is suffering from high bias (underfitting)
    /// or high variance (overfitting):
    /// - If performance quickly plateaus with small amounts of data, you might have high bias
    /// - If performance continues improving with more data, you might need even more data
    /// - If there's a large gap between training and validation performance, you might have high variance
    /// </para>
    /// <para>
    /// Unlike most metrics that return a single value, this returns a list of performance values
    /// at different training set sizes. This helps you visualize how your model learns from data.
    /// </para>
    /// </remarks>
    LearningCurve,

    /// <summary>
    /// Residual Sum of Squares - The sum of squared differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// RSS is calculated by summing the squared differences between each predicted value and its
    /// corresponding actual value. It serves as the basis for many other metrics such as MSE
    /// (which is RSS divided by the number of observations) and is used in calculating information
    /// criteria like AIC and BIC.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> RSS is the total squared error of your model. It's the basis for many other metrics like MSE
    /// (which is just RSS divided by the number of observations).
    /// Lower values indicate a better fit. It's used in calculating metrics like AIC and BIC
    /// that help you compare different models.
    /// </para>
    /// </remarks>
    RSS,

    /// <summary>
    /// Measures the proportion of variance in the dependent variable explained by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Explained Variance Score measures how much of the variation in your data is captured
    /// by your model. Like R�, it ranges from 0 to 1, with higher values being better. The main difference
    /// is that this metric focuses purely on variance explained, while R� also considers how far predictions
    /// are from the actual values.
    /// </para>
    /// </remarks>
    ExplainedVarianceScore,
    
    /// <summary>
    /// The average difference between predicted values and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Prediction Error simply calculates the average difference between what your model
    /// predicted and what the actual values were. A lower value is better. This metric helps you understand
    /// if your model tends to overestimate or underestimate the results, as positive and negative errors
    /// don't cancel each other out.
    /// </para>
    /// </remarks>
    MeanPredictionError,
    
    /// <summary>
    /// The middle value of all differences between predicted values and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Median Prediction Error finds the middle value of all the prediction errors when sorted.
    /// Unlike the mean, the median isn't affected by extreme outliers, so it gives you a more robust measure
    /// of your model's typical error when some predictions are way off.
    /// </para>
    /// </remarks>
    MedianPredictionError,
    
    /// <summary>
    /// The proportion of correct predictions among all predictions made.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Accuracy simply measures what percentage of your predictions were exactly right.
    /// For example, if your model made 100 predictions and got 85 correct, the accuracy is 85%.
    /// This metric is most useful when all types of errors are equally important and your data is balanced.
    /// </para>
    /// </remarks>
    Accuracy,
    
    /// <summary>
    /// The proportion of true positive predictions among all positive predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Precision measures how many of the items your model identified as positive were actually positive.
    /// For example, if your spam filter marked 10 emails as spam, but only 8 were actually spam, your precision is 80%.
    /// High precision means few false positives - you're not incorrectly flagging things that are actually negative.
    /// </para>
    /// </remarks>
    Precision,
    
    /// <summary>
    /// The proportion of true positive predictions among all actual positives.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Recall measures how many of the actual positive items your model correctly identified.
    /// For example, if there were 20 spam emails, and your filter caught 15 of them, your recall is 75%.
    /// High recall means few false negatives - you're not missing things that should be flagged as positive.
    /// </para>
    /// </remarks>
    Recall,
    
    /// <summary>
    /// The harmonic mean of precision and recall, providing a balance between the two metrics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> F1 Score combines precision and recall into a single number. It's useful when you need
    /// to balance between not missing positives (recall) and not incorrectly flagging negatives (precision).
    /// The score ranges from 0 to 1, with higher values being better. It's especially useful when your data
    /// has an uneven distribution of classes.
    /// </para>
    /// </remarks>
    F1Score,
    
    /// <summary>
    /// The percentage of actual values that fall within the model's prediction intervals.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Prediction Interval Coverage checks if your model's uncertainty estimates are reliable.
    /// Instead of just making a single prediction, some models provide a range (like "between 10-15 units").
    /// This metric tells you what percentage of actual values fall within these predicted ranges. Ideally,
    /// a 95% prediction interval should contain the actual value 95% of the time.
    /// </para>
    /// </remarks>
    PredictionIntervalCoverage,
    
    /// <summary>
    /// Measures the linear correlation between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pearson Correlation measures how well the relationship between your predictions and
    /// actual values can be described with a straight line. It ranges from -1 to 1, where:
    /// � 1 means perfect positive correlation (when actual values increase, predictions increase)
    /// � 0 means no correlation
    /// � -1 means perfect negative correlation (when actual values increase, predictions decrease)
    /// A high positive value indicates your model is capturing the right patterns, even if the exact values differ.
    /// </para>
    /// </remarks>
    PearsonCorrelation,
    
    /// <summary>
    /// Measures the monotonic relationship between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spearman Correlation is similar to Pearson, but it measures whether predictions and
    /// actual values increase or decrease together, without requiring a straight-line relationship.
    /// It works by ranking the values and then comparing the ranks. This makes it useful when your data
    /// has outliers or when the relationship isn't strictly linear but still follows a pattern.
    /// Like Pearson, it ranges from -1 to 1.
    /// </para>
    /// </remarks>
    SpearmanCorrelation,
    
    /// <summary>
    /// Measures the ordinal association between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kendall Tau measures how well your model preserves the correct ordering of values.
    /// It compares every possible pair of data points and checks if your model predicts the same relationship
    /// (is A greater than B, less than B, or equal to B?). This is useful when you care more about getting
    /// the ranking right than the exact values. For example, in a recommendation system, you might care more
    /// about showing the most relevant items first, rather than predicting exact relevance scores.
    /// </para>
    /// </remarks>
    KendallTau,

    /// <summary>
    /// The arithmetic average of a set of values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean is what most people call the "average". You calculate it by adding up all 
    /// the numbers and dividing by how many there are. It gives you a sense of the typical value in your data. 
    /// For example, if you have test scores of 80, 85, 90, 95, and 100, the mean is 90.
    /// </para>
    /// </remarks>
    Mean,

    /// <summary>
    /// The middle value in a sorted list of numbers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Median is the middle number when all your values are sorted. If you have an odd 
    /// number of values, it's the middle one. If you have an even number, it's the average of the two middle 
    /// numbers. It's useful when you have some extreme values that might skew the mean. For example, in the 
    /// list 1, 3, 3, 6, 7, 8, 100, the median is 6.
    /// </para>
    /// </remarks>
    Median,

    /// <summary>
    /// The most frequently occurring value in a dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mode is the value that appears most often in your data. It's especially useful 
    /// for categorical data (like favorite colors) or when you have distinct groups in your data. For example, 
    /// in the list 2, 3, 3, 4, 5, 5, 5, 6, the mode is 5.
    /// </para>
    /// </remarks>
    Mode,

    /// <summary>
    /// A measure of variability in the data, calculated as the average squared deviation from the mean.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Variance measures how spread out your data is. A low variance means the numbers 
    /// are clustered close to the average, while a high variance means they're more spread out. It's calculated 
    /// by finding the average of the squared differences from the Mean. Variance helps you understand the 
    /// distribution of your data, but it's in squared units, which can be hard to interpret.
    /// </para>
    /// </remarks>
    Variance,

    /// <summary>
    /// The square root of the variance, measuring the average deviation from the mean.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Standard Deviation is like Variance, but easier to understand because it's in 
    /// the same units as your original data. It tells you, on average, how far each value is from the mean. 
    /// About 68% of your data falls within one standard deviation of the mean, 95% within two, and 99.7% 
    /// within three. This helps you understand how spread out your data is in practical terms.
    /// </para>
    /// </remarks>
    StandardDeviation,

    /// <summary>
    /// The difference between the largest and smallest values in a dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Range is simply the difference between the highest and lowest values in your data. 
    /// It gives you a quick idea of how spread out your data is, but it can be misleading if you have outliers 
    /// (extreme values). For example, if your data is 1, 2, 3, 4, 100, the range is 99, even though most of 
    /// the values are close together.
    /// </para>
    /// </remarks>
    Range,

    /// <summary>
    /// The range of the middle 50% of the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Interquartile Range (IQR) is like range, but it focuses on the middle 50% of your 
    /// data. It's the difference between the 75th percentile (Q3) and the 25th percentile (Q1). IQR is useful 
    /// because it's not affected by outliers. It gives you an idea of where the "bulk" of your data lies. 
    /// For example, if Q1 is 20 and Q3 is 30, the IQR is 10.
    /// </para>
    /// </remarks>
    InterquartileRange,

    /// <summary>
    /// A measure of the relative quality of statistical models for a given set of data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Akaike Information Criterion (AIC) helps you compare different models and choose 
    /// the best one. It balances how well the model fits the data against how complex the model is. A lower 
    /// AIC is better. It's like choosing a car - you want one that's fast (fits the data well) but also 
    /// fuel-efficient (not too complex). AIC helps you find that balance.
    /// </para>
    /// </remarks>
    AIC,

    /// <summary>
    /// A criterion for model selection that penalizes model complexity more strongly than AIC.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bayesian Information Criterion (BIC) is similar to AIC, but it's stricter about 
    /// model complexity. It's useful when you have a lot of data and want to avoid overfitting. If AIC is like 
    /// choosing a car based on speed and fuel efficiency, BIC is like also considering the car's price more 
    /// heavily. It helps you find a model that's good enough without being unnecessarily complex.
    /// </para>
    /// </remarks>
    BIC,

    /// <summary>
    /// A measure of how probable the observed data is under the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Likelihood measures how well your model explains the observed data. A higher 
    /// likelihood means your model is more consistent with the data you've observed. It's like measuring how 
    /// well a theory explains the evidence. However, likelihood values can be very small and hard to interpret 
    /// directly, which is why we often use log-likelihood instead.
    /// </para>
    /// </remarks>
    Likelihood,

    /// <summary>
    /// The natural logarithm of the likelihood, used for numerical stability and easier interpretation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Log-Likelihood is the natural logarithm of the Likelihood. We use it because 
    /// likelihood values can be extremely small and hard to work with. Log-Likelihood turns these tiny numbers 
    /// into more manageable negative numbers. Higher (closer to zero) is better. It's like using the Richter 
    /// scale for earthquakes - it makes it easier to compare and work with a wide range of values.
    /// </para>
    /// </remarks>
    LogLikelihood,

    /// <summary>
    /// A proper score function that measures the accuracy of probabilistic predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Brier Score is used for probabilistic predictions, especially in classification 
    /// problems. It's like a "penalty" for how far off your probabilities are. If you predict a 70% chance of 
    /// rain and it rains, you get a small penalty; if you predict a 70% chance and it doesn't rain, you get a 
    /// larger penalty. The lower the Brier Score, the better your predictions.
    /// </para>
    /// </remarks>
    BrierScore,

    /// <summary>
    /// Measures the average log-loss across all classes in classification problems.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cross-Entropy Loss is like a penalty system for your model's confidence in its 
    /// predictions. If your model is very confident about a correct prediction, it gets a small penalty. 
    /// But if it's very confident about a wrong prediction, it gets a big penalty. It's useful in 
    /// classification problems, especially when you care about how certain your model is. Lower values are 
    /// better. It's like grading a test where you not only mark answers right or wrong, but also consider 
    /// how sure the student was about each answer.
    /// </para>
    /// </remarks>
    CrossEntropyLoss,

    /// <summary>
    /// Measures the maximum error between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Max Error is simply the largest mistake your model makes. It tells you the 
    /// worst-case scenario for your predictions. For example, if your model predicts house prices and the 
    /// Max Error is $50,000, you know that your model's biggest mistake was being off by $50,000. This is 
    /// useful when you really want to avoid large errors, even if they're rare. It's like focusing on the 
    /// lowest grade in a class - it doesn't tell you about typical performance, but it shows you the biggest 
    /// problem area.
    /// </para>
    /// </remarks>
    MaxError,

    /// <summary>
    /// Measures the harmonic mean of precision and recall, with more weight on recall.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> F2 Score is similar to F1 Score, but it gives more importance to recall than 
    /// precision. This is useful when the cost of false negatives is higher than false positives. For example, 
    /// in medical testing, you might prefer to have some false alarms (false positives) rather than miss any 
    /// actual cases of a disease (false negatives). F2 Score ranges from 0 to 1, with higher values being better. 
    /// It's like a grading system that cares more about not missing any important information than about 
    /// including some extra, unnecessary details.
    /// </para>
    /// </remarks>
    F2Score,

    /// <summary>
    /// Measures the geometric mean of precision and recall.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> G-Mean (Geometric Mean) balances precision and recall by multiplying them together 
    /// and then taking the square root. It's useful when you have imbalanced classes and want to give equal 
    /// importance to the performance on both the majority and minority classes. A high G-Mean indicates that 
    /// your model is performing well on both classes. It's like judging a decathlon athlete - you want someone 
    /// who's good at all events, not just exceptional in one or two.
    /// </para>
    /// </remarks>
    GMean,

    /// <summary>
    /// Measures the weighted harmonic mean of precision and recall.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> F-Beta Score is a flexible version of the F1 Score where you can adjust the balance 
    /// between precision and recall. By changing the 'beta' value, you can decide whether precision or recall is 
    /// more important for your specific problem. If beta > 1, recall is more important; if beta < 1, precision 
    /// is more important. It's like having a adjustable grading scale where you can decide whether it's more 
    /// important to get all the right answers (precision) or to not miss any important points (recall).
    /// </para>
    /// </remarks>
    FBetaScore,

    /// <summary>
    /// Measures the average precision across different recall levels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Average Precision summarizes the precision-recall curve as a single number. 
    /// It's like getting an overall grade that takes into account how well your model performs at different 
    /// levels of strictness. A higher average precision means your model is good at identifying positive 
    /// instances without raising too many false alarms, even as you adjust how strict it is. It's particularly 
    /// useful in tasks like information retrieval, where you want to know how well your model ranks relevant 
    /// items.
    /// </para>
    /// </remarks>
    AveragePrecision,

    /// <summary>
    /// Measures how well the predicted probabilities match the actual outcomes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Calibration Error measures how well your model's predicted probabilities match 
    /// the actual outcomes. A well-calibrated model with a 70% confidence should be correct about 70% of the time. 
    /// Lower calibration error is better. It's like checking if a weather forecast is reliable - if it says 
    /// there's a 30% chance of rain, it should actually rain on about 30% of such days. This is important when 
    /// you need to trust the confidence levels your model provides, not just its final decisions.
    /// </para>
    /// </remarks>
    CalibrationError,

    /// <summary>
    /// Measures the quality of clustering algorithms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Silhouette Score measures how well each item fits into its assigned cluster 
    /// compared to other clusters. It ranges from -1 to 1, where a high value indicates that the object is 
    /// well matched to its own cluster and poorly matched to neighboring clusters. It's like measuring how 
    /// well students are grouped in classes - are students in each class similar to each other and different 
    /// from students in other classes?
    /// </para>
    /// </remarks>
    SilhouetteScore,

    /// <summary>
    /// Measures the agreement between two sets of labels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cohen's Kappa measures how much better your model is at classification compared 
    /// to random guessing. It's especially useful when your classes are imbalanced. A score of 1 indicates 
    /// perfect agreement, 0 indicates no better than random chance. It's like comparing a student's test 
    /// answers to the correct answers, but also taking into account how many they might get right just by 
    /// guessing.
    /// </para>
    /// </remarks>
    CohenKappa,

    /// <summary>
    /// Measures the mutual dependence between two variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mutual Information measures how much knowing one variable reduces uncertainty 
    /// about the other. It's useful for feature selection and understanding relationships between variables. 
    /// Higher values indicate stronger relationships. It's like measuring how much knowing a student's grade 
    /// in one subject tells you about their grade in another subject.
    /// </para>
    /// </remarks>
    MutualInformation,

    /// <summary>
    /// Measures the log-likelihood of held-out data under a topic model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Perplexity is often used in natural language processing to evaluate how well a 
    /// model predicts a sample. Lower perplexity is better. It's like measuring how surprised a model is by 
    /// new text - a good model should find typical text unsurprising (low perplexity).
    /// </para>
    /// </remarks>
    Perplexity,

    /// <summary>
    /// Measures the difference between two probability distributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> KL Divergence measures how one probability distribution differs from another. 
    /// It's often used in machine learning to compare models or optimize algorithms. Lower values indicate 
    /// more similar distributions. It's like comparing two recipes for the same dish - how different are the 
    /// ingredients and their proportions?
    /// </para>
    /// </remarks>
    KLDivergence,

    /// <summary>
    /// Measures the similarity between two sequences.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Levenshtein Distance counts the minimum number of single-character edits 
    /// (insertions, deletions, or substitutions) required to change one word into another. It's useful in 
    /// spell checking, DNA sequence analysis, and more. Lower values indicate more similar sequences. 
    /// It's like counting how many changes you need to make to turn one word into another.
    /// </para>
    /// </remarks>
    LevenshteinDistance,

    /// <summary>
    /// Measures the similarity between two probability distributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bhattacharyya Distance measures the similarity of two probability distributions. 
    /// It's used in classification, feature selection, and image processing. A distance of 0 means the 
    /// distributions are identical, while larger values indicate more different distributions. It's like 
    /// comparing two different brands of the same product - how similar are their characteristics?
    /// </para>
    /// </remarks>
    BhattacharyyaDistance,

    /// <summary>
    /// Measures the asymmetry of the probability distribution of a real-valued random variable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Skewness tells you if your data is lopsided to one side or balanced around the average.
    /// 
    /// Think of it as measuring which way your data "leans":
    /// - Positive skewness (> 0): There's a longer tail on the right side - most values are on the left with a few high outliers
    /// - Zero skewness (= 0): The data is symmetric around the mean - balanced on both sides
    /// - Negative skewness (< 0): There's a longer tail on the left side - most values are on the right with a few low outliers
    /// 
    /// For example, income distribution often has positive skewness because most people have moderate incomes,
    /// but a few extremely wealthy individuals pull the right tail out.
    /// </para>
    /// </remarks>
    Skewness,

    /// <summary>
    /// Measures the "tailedness" of the probability distribution of a real-valued random variable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kurtosis measures how much of your data is in the "tails" versus the "center".
    /// 
    /// Think of it as measuring the shape of your data's peaks and tails:
    /// - High kurtosis (> 3): "Heavy-tailed" - more values in the extremes, with a sharper peak
    /// - Normal kurtosis (= 3): Follows a normal distribution (bell curve)
    /// - Low kurtosis (< 3): "Light-tailed" - fewer outliers, with a flatter peak
    /// 
    /// For example, stock returns often have high kurtosis because they mostly have small changes day-to-day,
    /// but occasionally have extreme movements.
    /// </para>
    /// </remarks>
    Kurtosis,

    /// <summary>
    /// The median absolute deviation (MAD) of the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MAD is another way to measure spread that's less affected by outliers.
    /// 
    /// To calculate MAD:
    /// 1. Find the median of your data
    /// 2. Calculate how far each value is from the median (absolute deviation)
    /// 3. Find the median of those distances
    /// 
    /// MAD is useful when your data has outliers that might skew other measures like standard deviation.
    /// Think of it as measuring the "typical" distance from the center, ignoring extreme values.
    /// </para>
    /// </remarks>
    MAD,

    /// <summary>
    /// Mean Absolute Error - The average absolute difference between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MAE measures the average size of errors without considering their direction (positive or negative).
    /// Lower values indicate better accuracy. If MAE = 5, your predictions are off by 5 units on average.
    /// </para>
    /// </remarks>
    MAE,

    /// <summary>
    /// Mean Squared Error - The average of squared differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MSE squares the errors before averaging them, which penalizes large errors more heavily than small ones.
    /// Lower values indicate better accuracy. Because of squaring, the value is not in the same units as your data.
    /// </para>
    /// </remarks>
    MSE,

    /// <summary>
    /// Root Mean Squared Error - The square root of the Mean Squared Error.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RMSE converts MSE back to the original units of your data by taking the square root.
    /// It's often preferred over MSE for interpretation because it's in the same units as your data.
    /// Like MAE, lower values indicate better accuracy.
    /// </para>
    /// </remarks>
    RMSE,

    /// <summary>
    /// Mean Absolute Percentage Error - The average percentage difference between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MAPE expresses error as a percentage, which helps you understand the relative size of errors.
    /// MAPE = 10 means that, on average, your predictions are off by 10% from the actual values.
    /// Note: MAPE can be problematic when actual values are close to zero.
    /// </para>
    /// </remarks>
    MAPE,

    /// <summary>
    /// Mean Bias Error - The average of prediction errors (predicted - actual).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MeanBiasError helps determine if your model tends to overestimate (positive value) or
    /// underestimate (negative value). Ideally, it should be close to zero, indicating no systematic bias.
    /// Unlike MAE, this doesn't take the absolute value, so positive and negative errors can cancel out.
    /// </para>
    /// </remarks>
    MeanBiasError,

    /// <summary>
    /// Median Absolute Error - The middle value of all absolute differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Unlike MAE which uses the average, MedianAbsoluteError uses the middle value of all absolute errors.
    /// This makes it less sensitive to outliers (extreme errors) than MAE.
    /// For example, if you have errors of [1, 2, 100], the median is 2, while the mean would be 34.3.
    /// </para>
    /// </remarks>
    MedianAbsoluteError,

    /// <summary>
    /// Theil's U Statistic - A measure of forecast accuracy relative to a naive forecasting method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TheilUStatistic compares your model's accuracy to a simple "no-change" prediction.
    /// Values less than 1 mean your model is better than the naive approach.
    /// Values equal to 1 mean your model performs the same as the naive approach.
    /// Values greater than 1 mean your model performs worse than the naive approach.
    /// This is especially useful for time series forecasting evaluation.
    /// </para>
    /// </remarks>
    TheilUStatistic,

    /// <summary>
    /// Durbin-Watson Statistic - Detects autocorrelation in prediction errors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DurbinWatsonStatistic helps identify if there are patterns in your prediction errors over time.
    /// Values range from 0 to 4:
    /// - Values near 2 suggest no autocorrelation (good)
    /// - Values toward 0 suggest positive autocorrelation (errors tend to be followed by similar errors)
    /// - Values toward 4 suggest negative autocorrelation (errors tend to be followed by opposite errors)
    /// 
    /// Autocorrelation in errors suggests your model might be missing important patterns in the data.
    /// </para>
    /// </remarks>
    DurbinWatsonStatistic,

    /// <summary>
    /// Sample Standard Error - An estimate of the standard deviation of prediction errors, adjusted for model complexity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SampleStandardError estimates how much prediction errors typically vary, taking into account
    /// how many parameters (features) your model uses. It's useful for constructing confidence intervals
    /// around predictions and is adjusted downward based on the number of parameters in your model.
    /// </para>
    /// </remarks>
    SampleStandardError,

    /// <summary>
    /// Population Standard Error - The standard deviation of prediction errors without adjustment for model complexity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PopulationStandardError measures how much prediction errors typically vary, but unlike 
    /// SampleStandardError, it doesn't adjust for model complexity. It gives you an idea of the
    /// typical size of the errors your model makes.
    /// </para>
    /// </remarks>
    PopulationStandardError,

    /// <summary>
    /// Alternative Akaike Information Criterion - A variant of AIC with a different penalty term.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AICAlt is another version of AIC that uses a slightly different approach to penalize
    /// model complexity. It's particularly useful when sample sizes are small.
    /// Like AIC and BIC, lower values are better.
    /// </para>
    /// </remarks>
    AICAlt,

    /// <summary>
    /// Area Under the Precision-Recall Curve - Measures classification accuracy focusing on positive cases.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AUCPR is especially useful for imbalanced classification problems (where one class is rare).
    /// It ranges from 0 to 1, with higher values indicating better performance.
    /// 
    /// Precision measures how many of your positive predictions were correct.
    /// Recall measures what fraction of actual positives your model identified.
    /// AUCPR considers how these trade off across different threshold settings.
    /// </para>
    /// </remarks>
    AUCPR,

    /// <summary>
    /// Area Under the Receiver Operating Characteristic Curve - Measures classification accuracy across thresholds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AUCROC is a common metric for classification models. It ranges from 0 to 1:
    /// - 0.5 means the model is no better than random guessing
    /// - 1.0 means perfect classification
    /// - Values below 0.5 suggest the model is worse than random
    /// 
    /// It measures how well your model can distinguish between classes across different threshold settings.
    /// </para>
    /// </remarks>
    AUCROC,

    /// <summary>
    /// Symmetric Mean Absolute Percentage Error - A variant of MAPE that handles zero or near-zero values better.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SMAPE is similar to MAPE but uses a different formula that handles cases where actual values are zero
    /// or very small. It's bounded between 0% and 200%, with lower values indicating better performance.
    /// 
    /// SMAPE treats positive and negative errors more symmetrically than MAPE,
    /// which can be important in some forecasting applications.
    /// </para>
    /// </remarks>
    SMAPE,

    /// <summary>
    /// Mean Squared Logarithmic Error - Penalizes underestimates more than overestimates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MeanSquaredLogError is useful when you care more about relative errors than absolute ones.
    /// It's calculated by applying logarithms to actual and predicted values before computing MSE.
    /// 
    /// MSLE penalizes underestimation (predicting too low) more heavily than overestimation.
    /// This is useful in scenarios where underestimating would be more problematic, like inventory forecasting.
    /// </para>
    /// </remarks>
    MeanSquaredLogError,

    /// <summary>
    /// Measures the sensitivity of a system to small changes in input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Condition Number tells you how much the output of a system can change for a small change in the input.
    /// A high condition number means the system is very sensitive to small changes, which can lead to numerical instability.
    /// It's like measuring how wobbly a table is - a high condition number means even a small nudge could make things very unstable.
    /// </para>
    /// </remarks>
    ConditionNumber,

    /// <summary>
    /// A measure of the model's predictive accuracy on a point-by-point basis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Log Pointwise Predictive Density (LPPD) measures how well your model predicts each individual data point.
    /// It's like grading a test where each question is scored separately, and then all scores are combined.
    /// A higher LPPD means your model is doing a better job at predicting individual data points.
    /// </para>
    /// </remarks>
    LogPointwisePredictiveDensity,

    /// <summary>
    /// The value of the test statistic computed from the observed data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Observed Test Statistic is a number calculated from your data that you use to test a hypothesis.
    /// It's like measuring how far your results are from what you'd expect if your hypothesis were false.
    /// You compare this number to a threshold to decide if your results are statistically significant.
    /// </para>
    /// </remarks>
    ObservedTestStatistic,

    /// <summary>
    /// The probability of the observed data under all possible parameter values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Marginal Likelihood is like asking "How likely is this data, considering all possible explanations?"
    /// It's used to compare different models, taking into account both how well they fit the data and how complex they are.
    /// A higher marginal likelihood suggests a better balance between fitting the data and keeping the model simple.
    /// </para>
    /// </remarks>
    MarginalLikelihood,

    /// <summary>
    /// The marginal likelihood of a reference or null model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Reference Model Marginal Likelihood is the marginal likelihood of a simpler, baseline model.
    /// It's used as a point of comparison for your main model. If your main model's marginal likelihood is much higher,
    /// it suggests your model is substantially better than the basic reference model.
    /// </para>
    /// </remarks>
    ReferenceModelMarginalLikelihood,

    /// <summary>
    /// The effective number of parameters in a model, useful for model complexity assessment.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Effective Number of Parameters measures how complex your model really is.
    /// It's like counting the number of knobs you can turn to adjust your model, but some knobs might be more important than others.
    /// This helps you understand if your model is overly complicated, which could lead to overfitting.
    /// </para>
    /// </remarks>
    EffectiveNumberOfParameters,

    /// <summary>
    /// The straight-line distance between two points in Euclidean space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Euclidean Distance is the "as the crow flies" distance between two points.
    /// If you plot your data points on a graph, it's the length of the straight line you'd draw between them.
    /// It's used to measure how different or similar data points are to each other.
    /// </para>
    /// </remarks>
    EuclideanDistance,

    /// <summary>
    /// The sum of the absolute differences of the coordinates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Manhattan Distance is like measuring the distance a taxi would drive in a city with a grid layout.
    /// Instead of going directly ("as the crow flies"), it's the sum of the distances along each dimension.
    /// It's useful when you can't move diagonally through your data space.
    /// </para>
    /// </remarks>
    ManhattanDistance,

    /// <summary>
    /// A measure of similarity between two non-zero vectors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cosine Similarity measures the cosine of the angle between two vectors.
    /// It's like comparing the directions two arrows are pointing, regardless of their length.
    /// A value of 1 means they point in the same direction, 0 means they're perpendicular, and -1 means opposite directions.
    /// </para>
    /// </remarks>
    CosineSimilarity,

    /// <summary>
    /// A statistic used for comparing the similarity and diversity of sample sets.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Jaccard Similarity measures how similar two sets are by comparing what they have in common to what they have in total.
    /// It's like comparing two people's music libraries - how many songs do they both have compared to their total unique songs?
    /// A value of 1 means the sets are identical, and 0 means they have nothing in common.
    /// </para>
    /// </remarks>
    JaccardSimilarity,

    /// <summary>
    /// The number of positions at which the corresponding symbols are different.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hamming Distance counts how many changes you need to make to turn one string into another.
    /// It's like comparing two versions of a document and counting how many characters are different.
    /// It's often used in error detection and correction in data transmission.
    /// </para>
    /// </remarks>
    HammingDistance,

    /// <summary>
    /// A multi-dimensional generalization of measuring how many standard deviations away a point is from the mean of a distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mahalanobis Distance measures how many standard deviations a point is from the mean of a distribution.
    /// It's like measuring how unusual a data point is, taking into account the overall pattern of the data.
    /// It's useful for detecting outliers in multi-dimensional data.
    /// </para>
    /// </remarks>
    MahalanobisDistance,

    /// <summary>
    /// A normalization of the Mutual Information to scale the results between 0 and 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Normalized Mutual Information measures how much knowing one variable reduces uncertainty about another.
    /// It's like measuring how much knowing someone's job tells you about their income, but adjusted so it's always between 0 and 1.
    /// This makes it easier to compare across different datasets.
    /// </para>
    /// </remarks>
    NormalizedMutualInformation,

    /// <summary>
    /// A measure of the amount of information lost when compressing two random variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Variation of Information measures how much information is lost when you group data.
    /// It's like measuring how much detail you lose when you summarize a long story.
    /// Lower values mean you've kept more of the original information in your grouping.
    /// </para>
    /// </remarks>
    VariationOfInformation,

    /// <summary>
    /// A measure of how similar an object is to its own cluster compared to other clusters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Calinski-Harabasz Index measures how well-defined your clusters are.
    /// It compares the average between-cluster distance to the average within-cluster distance.
    /// Higher values indicate better-defined clusters, like well-separated groups in a crowd.
    /// </para>
    /// </remarks>
    CalinskiHarabaszIndex,

    /// <summary>
    /// A metric for evaluating clustering algorithms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Davies-Bouldin Index measures how similar each cluster is to its most similar other cluster.
    /// It's like measuring how distinct each group in a party is from the group it's most likely to be confused with.
    /// Lower values indicate better clustering, with more distinct, well-separated clusters.
    /// </para>
    /// </remarks>
    DaviesBouldinIndex,

    /// <summary>
    /// The mean of the average precision scores for each query.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Average Precision is often used in information retrieval to evaluate search algorithms.
    /// It's like measuring how good a search engine is at putting the most relevant results at the top of the list.
    /// A higher MAP indicates that relevant items are generally ranked higher in the search results.
    /// </para>
    /// </remarks>
    MeanAveragePrecision,

    /// <summary>
    /// A measure of ranking quality used in information retrieval.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Normalized Discounted Cumulative Gain (NDCG) measures the quality of ranking in search results.
    /// It's like grading a search engine on how well it puts the most relevant results at the top.
    /// It takes into account that users are less likely to look at results further down the list.
    /// </para>
    /// </remarks>
    NormalizedDiscountedCumulativeGain,

    /// <summary>
    /// The average of the reciprocal ranks of the first relevant items.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean Reciprocal Rank measures how high the first correct answer appears in a list of results.
    /// It's like measuring how quickly a search engine gives you the answer you're looking for.
    /// A higher MRR means the correct answer tends to appear earlier in the list of results.
    /// </para>
    /// </remarks>
    MeanReciprocalRank,
}