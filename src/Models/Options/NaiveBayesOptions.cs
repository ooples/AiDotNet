namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Naive Bayes classifiers.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Naive Bayes classifiers are probabilistic classifiers based on Bayes' theorem with the
/// "naive" assumption of conditional independence between features given the class label.
/// Despite this simplifying assumption, Naive Bayes often performs well in practice.
/// </para>
/// <para><b>For Beginners:</b> Naive Bayes is one of the simplest and most effective classifiers.
///
/// How it works:
/// 1. During training, it learns the probability of each class and the probability of
///    each feature value given each class
/// 2. During prediction, it uses Bayes' theorem to calculate the probability of each class
///    given the observed features
/// 3. It returns the class with the highest probability
///
/// The "naive" assumption is that features are independent given the class. For example,
/// in spam detection, the words "free" and "win" might both indicate spam, but the model
/// assumes they contribute independently to that prediction.
///
/// Despite this unrealistic assumption, Naive Bayes often works surprisingly well!
/// </para>
/// </remarks>
public class NaiveBayesOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the smoothing parameter (alpha) for Laplace/additive smoothing.
    /// </summary>
    /// <value>
    /// A positive smoothing value, defaulting to 1.0 (Laplace smoothing).
    /// Set to 0.0 for no smoothing (not recommended).
    /// </value>
    /// <remarks>
    /// <para>
    /// Smoothing prevents zero probabilities when a feature value was not observed for a class
    /// in the training data. Laplace smoothing (alpha=1.0) is most common, but smaller values
    /// (like 0.1 or 0.01) may work better in some cases.
    /// </para>
    /// <para><b>For Beginners:</b> Smoothing solves the "zero probability" problem.
    ///
    /// Imagine training a spam classifier on emails. If the word "congratulations" never
    /// appeared in your spam training examples, without smoothing the model would say
    /// ANY email containing "congratulations" has ZERO probability of being spam.
    /// That's too extreme!
    ///
    /// Smoothing adds a small count to all feature values so nothing has zero probability:
    /// - Alpha = 1.0: Laplace smoothing (default, works well in most cases)
    /// - Alpha = 0.1-0.5: Lighter smoothing (can improve accuracy with enough data)
    /// - Alpha = 0.0: No smoothing (risky - avoid unless you're sure all features appear)
    ///
    /// For most applications, the default of 1.0 is a safe choice.
    /// </para>
    /// </remarks>
    public double Alpha { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to fit class prior probabilities from the data.
    /// </summary>
    /// <value>
    /// True (default) to learn class priors from training data;
    /// false to use uniform priors (all classes equally likely a priori).
    /// </value>
    /// <remarks>
    /// <para>
    /// Class priors represent the probability of each class before seeing any features.
    /// When FitPriors is true, these are estimated from the training data frequencies.
    /// When false, all classes are assumed equally likely, which can be useful when
    /// the training data is not representative of the true class distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how the model handles class imbalance.
    ///
    /// Example: In a medical dataset, 95% of samples are healthy, 5% have a disease.
    ///
    /// With FitPriors = true (default):
    /// - The model learns that healthy is much more common
    /// - It will be more likely to predict "healthy" by default
    /// - This reflects the real-world probability
    ///
    /// With FitPriors = false:
    /// - The model treats healthy and diseased as equally likely a priori
    /// - It focuses purely on the feature evidence
    /// - May be better if you want to catch more disease cases
    ///
    /// Use false when your training data doesn't reflect the true distribution,
    /// or when you want equal consideration for all classes regardless of frequency.
    /// </para>
    /// </remarks>
    public bool FitPriors { get; set; } = true;

    /// <summary>
    /// Gets or sets custom class prior probabilities.
    /// </summary>
    /// <value>
    /// An array of prior probabilities for each class, or null to estimate from data.
    /// Values should sum to 1.0.
    /// </value>
    /// <remarks>
    /// <para>
    /// Custom priors allow you to specify the prior probability for each class explicitly.
    /// This is useful when you have domain knowledge about the true class distribution
    /// that differs from the training data distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you tell the model how common each class is.
    ///
    /// Example: You're building a fraud detector with training data that has equal fraud/legitimate.
    /// But in reality, only 1% of transactions are fraudulent.
    ///
    /// You can set: ClassPriors = [0.99, 0.01]
    ///
    /// This tells the model that even before looking at features:
    /// - A transaction has 99% chance of being legitimate
    /// - A transaction has 1% chance of being fraudulent
    ///
    /// Leave this as null (default) to learn priors from your training data.
    /// </para>
    /// </remarks>
    public double[]? ClassPriors { get; set; }

    /// <summary>
    /// Gets or sets the minimum variance for Gaussian Naive Bayes.
    /// </summary>
    /// <value>
    /// A small positive value, defaulting to 1e-9.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Gaussian Naive Bayes, this sets a floor on the variance of each feature
    /// to prevent division by zero and numerical instability when a feature has
    /// very low variance (nearly constant values).
    /// </para>
    /// <para><b>For Beginners:</b> This prevents math problems when features barely change.
    ///
    /// If a feature is almost constant (like height in cm for adult males might be 175 ± 0.001),
    /// dividing by such a tiny variance can cause calculation problems.
    ///
    /// The minimum variance sets a floor to prevent these issues.
    /// The default of 1e-9 is usually fine; you shouldn't need to change this unless
    /// you're seeing numerical errors or warnings.
    /// </para>
    /// </remarks>
    public double MinVariance { get; set; } = 1e-9;
}
