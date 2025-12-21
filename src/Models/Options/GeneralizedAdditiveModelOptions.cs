namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Generalized Additive Models (GAMs), which are flexible regression models
/// that combine multiple simple functions to model complex relationships.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Generalized Additive Models extend linear regression by allowing non-linear relationships between
/// predictors and the target variable. They work by fitting a separate smooth function (typically a spline)
/// for each predictor variable and then adding these functions together. This approach maintains much of
/// the interpretability of linear models while allowing for more flexible relationships.
/// </para>
/// <para><b>For Beginners:</b> Think of a Generalized Additive Model as a more flexible version of linear
/// regression. In linear regression, you assume each input feature has a straight-line relationship with
/// your target variable (like y = mx + b from high school math). GAMs relax this assumption by allowing
/// each feature to have its own curved relationship with the target. 
/// 
/// For example, if you're predicting house prices, a linear model might assume that price increases by a
/// fixed amount for each additional square foot. A GAM could capture that small houses might see a bigger
/// price increase per square foot than very large houses, where additional space might add less value.
/// 
/// GAMs achieve this flexibility while still being relatively easy to interpret compared to "black box"
/// models like neural networks, because you can visualize how each individual feature affects the prediction.
/// They're a good middle ground between simple linear models and complex machine learning algorithms.</para>
/// </remarks>
public class GeneralizedAdditiveModelOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the number of spline basis functions to use for each feature.
    /// </summary>
    /// <value>The number of splines, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how many spline basis functions are used to represent the smooth function
    /// for each predictor variable. More splines allow for more complex and wiggly functions, while fewer
    /// splines result in smoother functions with less flexibility.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how flexible each curve in your model can be.
    /// With the default value of 10, the model uses 10 simple curve pieces (called "splines") that it
    /// combines to form a smooth curve for each feature. Think of it like drawing a curve using a limited
    /// number of control points - more points (higher NumSplines) let you create more complex curves with
    /// more twists and turns, while fewer points force the curve to be simpler and smoother.
    /// 
    /// If you set this too high (like 30+), your model might become too flexible and start fitting to noise
    /// in your data (overfitting). If you set it too low (like 3), your model might be too rigid to capture
    /// important patterns (underfitting). The default of 10 is a good starting point for many problems, but
    /// you might adjust it based on how complex you think the relationships in your data are and how much
    /// data you have (more data can support more splines).</para>
    /// </remarks>
    public int NumSplines { get; set; } = 10;

    /// <summary>
    /// Gets or sets the degree of the polynomial splines used in the model.
    /// </summary>
    /// <value>The polynomial degree, defaulting to 3 (cubic splines).</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the degree of the polynomial used for each spline segment. Higher degrees
    /// allow for smoother transitions between spline segments but can lead to more oscillations. The default
    /// value of 3 corresponds to cubic splines, which provide a good balance of smoothness and stability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how smooth the transitions are between the curve
    /// pieces in your model. With the default value of 3, the model uses cubic polynomials (like y = ax³ + bx² + cx + d)
    /// which create very smooth curves with continuous first and second derivatives (meaning the curve and its
    /// slope change gradually without sudden jumps).
    /// 
    /// Think of it like the difference between connecting dots with straight lines (degree 1), gentle curves
    /// (degree 2), or even smoother curves (degree 3). Higher degrees create smoother transitions but can
    /// sometimes lead to unexpected wiggles in regions with sparse data. Lower degrees create simpler curves
    /// but might have more abrupt changes in direction.
    /// 
    /// The most common choices are:
    /// - 1: Linear splines (straight line segments)
    /// - 2: Quadratic splines (parabolic segments)
    /// - 3: Cubic splines (the default, very smooth)
    /// 
    /// For most applications, the default cubic splines (degree 3) work well, providing a good balance of
    /// smoothness and predictability.</para>
    /// </remarks>
    public int Degree { get; set; } = 3;
}
