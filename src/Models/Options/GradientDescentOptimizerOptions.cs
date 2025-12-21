namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Gradient Descent optimizer, which is a fundamental algorithm for
/// finding the minimum of a function by iteratively moving in the direction of steepest descent.
/// </summary>
/// <remarks>
/// <para>
/// Gradient Descent is one of the most widely used optimization algorithms in machine learning. It works
/// by calculating the gradient (slope) of a loss function with respect to the model parameters, then
/// updating those parameters in the opposite direction of the gradient to minimize the loss. This class
/// inherits from GradientBasedOptimizerOptions, so all general gradient-based optimization settings are
/// also available.
/// </para>
/// <para><b>For Beginners:</b> Think of Gradient Descent like finding the lowest point in a valley by
/// always walking downhill. Imagine you're standing on a hilly landscape and want to reach the lowest
/// point. You look around, figure out which direction is most steeply downhill, take a step in that
/// direction, and repeat until you can't go any lower.
/// 
/// In machine learning, the "landscape" is the error or loss function (how wrong your model's predictions
/// are), and the "lowest point" represents the best possible model parameters. Gradient descent helps your
/// model learn by repeatedly adjusting its parameters to reduce prediction errors.
/// 
/// This is the most basic form of optimization used in many machine learning algorithms, including neural
/// networks, linear regression, and logistic regression. The options in this class let you control how
/// quickly the algorithm moves downhill and how it avoids certain pitfalls during the optimization process.</para>
/// </remarks>
public class GradientDescentOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// The regularization options to control overfitting during gradient descent optimization.
    /// </summary>
    /// <remarks>
    /// This field follows the naming convention _variableName for private fields.
    /// </remarks>
    private RegularizationOptions _regularizationOptions;

    /// <summary>
    /// Gets or sets the regularization options to control overfitting during optimization.
    /// </summary>
    /// <value>
    /// The regularization options, defaulting to L2 regularization with a strength of 0.01.
    /// </value>
    /// <remarks>
    /// <para>
    /// Regularization adds a penalty to the loss function based on the model's parameter values, which
    /// helps prevent overfitting by discouraging overly complex models. The setter ensures that the
    /// regularization options are never null by using default options if null is provided.
    /// </para>
    /// <para><b>For Beginners:</b> Regularization is like adding a "simplicity rule" to your model's
    /// training process. Without regularization, your model might become too complex and start memorizing
    /// the training data instead of learning general patterns (this is called "overfitting").
    /// 
    /// Think of it like learning to drive - you want to learn general rules of the road, not memorize
    /// every specific turn you made during practice. Regularization penalizes your model for becoming
    /// too complex, encouraging it to find simpler solutions that are more likely to work well on new data.
    /// 
    /// The default setting uses "L2 regularization" (also called "ridge" or "weight decay"), which is like
    /// telling your model "don't make any single parameter too large." This tends to work well for most
    /// problems and helps your model generalize better to new data. The strength value of 0.01 provides a
    /// moderate amount of regularization - higher values would enforce simplicity more strongly.</para>
    /// </remarks>
    public RegularizationOptions RegularizationOptions
    {
        get => _regularizationOptions;
        set => _regularizationOptions = value ?? CreateDefaultRegularizationOptions();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GradientDescentOptimizerOptions"/> class with default settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The constructor initializes the regularization options with default values that are suitable for
    /// gradient descent optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new set of options for gradient descent with
    /// reasonable default values. When you create a new GradientDescentOptimizerOptions object without
    /// specifying any parameters, it will use these defaults, which work well for many common machine
    /// learning problems. You can then customize specific settings as needed for your particular task.</para>
    /// </remarks>
    public GradientDescentOptimizerOptions()
    {
        _regularizationOptions = CreateDefaultRegularizationOptions();
    }

    /// <summary>
    /// Creates default regularization options specifically tuned for gradient descent optimization.
    /// </summary>
    /// <returns>A new RegularizationOptions object with default settings for gradient descent.</returns>
    /// <remarks>
    /// <para>
    /// This method creates regularization options with values that work well with gradient descent.
    /// It uses L2 regularization (ridge) with a moderate strength of 0.01 and no L1 component.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the default "simplicity rules" for your model.
    /// It chooses L2 regularization, which penalizes large parameter values in a smooth way that works
    /// well with gradient descent. The strength is set to 0.01, which provides a good balance between
    /// allowing the model to fit the data while preventing it from becoming too complex.
    /// 
    /// The L1Ratio of 0.0 means it's using pure L2 regularization with no L1 component. L1 regularization
    /// (which encourages some parameters to be exactly zero) can be useful for feature selection but is
    /// less smooth and can be more challenging for gradient descent to optimize.</para>
    /// </remarks>
    private static RegularizationOptions CreateDefaultRegularizationOptions()
    {
        // we override the default regularization values to use variables more friendly to gradient descent
        return new RegularizationOptions
        {
            Type = RegularizationType.L2,
            Strength = 0.01,
            L1Ratio = 0.0
        };
    }
}
