namespace AiDotNet.Regularization;

/// <summary>
/// Provides a base implementation for regularization techniques used to prevent overfitting in machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data structure type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data structure type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// RegularizationBase serves as an abstract foundation for implementing various regularization strategies.
/// Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function
/// that discourages complex models. This base class defines the common interface and provides shared
/// functionality for all regularization implementations.
/// </para>
/// <para><b>For Beginners:</b> Think of regularization as adding "guardrails" to your model's learning process.
/// 
/// Imagine training a model is like teaching a student:
/// - Without regularization, the student might memorize all the test answers without understanding the principles
/// - With regularization, we encourage the student to learn simpler, more general rules
/// 
/// This base class provides the framework that all specific regularization techniques build upon.
/// Different regularization approaches (like L1, L2, or Elastic Net) are like different teaching
/// strategies, but they all aim to help your model generalize better to new data.
/// 
/// For example:
/// - L1 regularization eliminates less important features
/// - L2 regularization makes all features less extreme
/// - Elastic Net combines both approaches
/// 
/// All of these approaches inherit from this base class, which provides the common structure they all share.
/// </para>
/// </remarks>
public abstract class RegularizationBase<T, TInput, TOutput> : IRegularization<T, TInput, TOutput>
{
    /// <summary>
    /// Provides numeric operations appropriate for the generic type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a reference to the appropriate numeric operations implementation for the
    /// generic type T, allowing the regularization methods to perform mathematical operations
    /// regardless of whether T is float, double, or another numeric type.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that allows the code to work with different number types.
    /// 
    /// Since this class uses a generic type T (which could be float, double, etc.):
    /// - We need a way to perform math operations (+, -, *, /) on these values
    /// - NumOps provides the right methods for whatever numeric type is being used
    /// 
    /// Think of it like having different calculators for different types of numbers,
    /// and NumOps makes sure we're using the right calculator for the job.
    /// </para>
    /// </remarks>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Configuration options for the regularization technique.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration settings for the regularization, including the regularization type,
    /// strength, and any additional parameters specific to the regularization method (such as L1Ratio for
    /// Elastic Net regularization).
    /// </para>
    /// <para><b>For Beginners:</b> This stores the settings that control how the regularization works.
    /// 
    /// These options include:
    /// - Type: What kind of regularization to use (L1, L2, etc.)
    /// - Strength: How powerful the regularization effect should be
    /// - Other settings specific to certain types of regularization
    /// 
    /// It's like the control panel for your regularization - you adjust these settings
    /// to get the right balance between fitting your training data and generalizing to new data.
    /// </para>
    /// </remarks>
    protected readonly RegularizationOptions Options;

    /// <summary>
    /// Initializes a new instance of the RegularizationBase class with the specified options.
    /// </summary>
    /// <param name="regularizationOptions">
    /// Configuration options for the regularization. If not provided, default options will be used.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the base class for a regularization implementation,
    /// setting up the numeric operations and storing the configuration options. If no options
    /// are provided, a default set of options is created.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the foundation for any type of regularization.
    /// 
    /// When creating a regularization object:
    /// - It gets the right calculator for the numeric type being used
    /// - It stores the settings you provided, or uses default settings if none were specified
    /// 
    /// This is like preparing your workspace before starting a project - gathering the tools
    /// and materials you'll need, based on the specific type of project you're doing.
    /// </para>
    /// </remarks>
    public RegularizationBase(RegularizationOptions? regularizationOptions = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = regularizationOptions ?? new();
    }

    /// <summary>
    /// Applies regularization to a matrix of input features.
    /// </summary>
    /// <param name="data">The input feature matrix to regularize.</param>
    /// <returns>The regularized matrix.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method defines the interface for regularizing an input feature matrix.
    /// Different regularization techniques may transform the input features in different ways,
    /// although many implementations simply return the matrix unchanged as they apply
    /// regularization only to the coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method might modify your input data before model training.
    /// 
    /// Depending on the regularization type:
    /// - It might transform your features to make them better behaved
    /// - It might leave them completely unchanged
    /// - It might normalize or scale them in some way
    /// 
    /// Most common regularization methods (like L1 and L2) don't modify your input data,
    /// but this method is included to support regularization techniques that might need to.
    /// </para>
    /// </remarks>
    public abstract Matrix<T> Regularize(Matrix<T> data);

    /// <summary>
    /// Applies regularization to model coefficients.
    /// </summary>
    /// <param name="data">The coefficient vector to regularize.</param>
    /// <returns>The regularized coefficient vector.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method defines the interface for regularizing a model's coefficient vector.
    /// This is where most regularization techniques apply their core functionality, shrinking
    /// coefficients or setting some to zero based on the regularization approach.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts your model's learned patterns to prevent overfitting.
    /// 
    /// Different regularization types modify the coefficients in different ways:
    /// - L1 regularization may set some coefficients to exactly zero (removing features)
    /// - L2 regularization shrinks all coefficients proportionally
    /// - Elastic Net combines both approaches
    /// 
    /// This is the heart of regularization - changing the strength of relationships your model
    /// has learned to create a more balanced, generalizable model.
    /// </para>
    /// </remarks>
    public abstract Vector<T> Regularize(Vector<T> data);

    /// <summary>
    /// Adjusts the gradient vector to account for regularization during optimization.
    /// </summary>
    /// <param name="gradient">The original gradient vector from the loss function.</param>
    /// <param name="coefficients">The current coefficient vector.</param>
    /// <returns>The regularized gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method defines the interface for modifying the gradient vector during
    /// optimization to account for the regularization penalty. It adds the derivative of the
    /// regularization term with respect to the coefficients to the original gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This method guides the model's learning process toward simpler solutions.
    /// 
    /// During model training:
    /// - The gradient tells the model which direction to adjust coefficients to improve
    /// - This method modifies that gradient to include the effect of regularization
    /// - It pushes coefficients in directions that balance accuracy with simplicity
    /// 
    /// Think of it like a coach correcting an athlete's technique - the athlete is trying to
    /// reach a goal (fitting the data), but the coach provides guidance to ensure they do it
    /// with good form (simpler model).
    /// </para>
    /// </remarks>
    public abstract TOutput Regularize(TOutput gradient, TOutput coefficients);

    /// <summary>
    /// Gets the configuration options for this regularization technique.
    /// </summary>
    /// <returns>The regularization options.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the configuration options that control the behavior of the
    /// regularization technique, including type, strength, and any additional parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you check what settings are being used.
    /// 
    /// It returns:
    /// - The type of regularization (L1, L2, Elastic Net, etc.)
    /// - The strength of the regularization effect
    /// - Any additional settings specific to the regularization type
    /// 
    /// This is useful when you want to inspect or verify the current configuration
    /// of your regularization component.
    /// </para>
    /// </remarks>
    public RegularizationOptions GetOptions()
    {
        return Options;
    }
}
