namespace AiDotNet.Enums;

/// <summary>
/// Built-in univariate scoring functions used by <c>SelectKBest</c> and <c>SelectPercentile</c>
/// (the scikit-learn name of each is given in its summary).
/// </summary>
public enum UnivariateScoreFunction
{
    /// <summary>ANOVA F-value between each feature and the class labels (<c>f_classif</c>).</summary>
    FClassif,

    /// <summary>F-value of each feature's univariate linear regression on the target (<c>f_regression</c>).</summary>
    FRegression,

    /// <summary>Chi-squared statistic of each non-negative feature against the class labels (<c>chi2</c>).</summary>
    Chi2,

    /// <summary>
    /// Mutual information between each continuous feature and the class labels, by the k-nearest-neighbour
    /// estimator of Ross (2014) with 3 neighbours (<c>mutual_info_classif</c>).
    /// </summary>
    MutualInfoClassif
}
