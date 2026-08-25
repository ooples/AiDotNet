using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Regression;

/// <summary>
/// Implements Principal Component Regression (PCR), a technique that combines principal component analysis (PCA) 
/// with linear regression to handle multicollinearity in the predictor variables.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Principal Component Regression works by first performing principal component analysis (PCA) on the predictor 
/// variables to reduce their dimensionality, then using these principal components as predictors in a linear 
/// regression model. This approach is particularly useful when dealing with multicollinearity (high correlation 
/// among predictor variables) or when the number of predictors is large relative to the number of observations.
/// </para>
/// <para>
/// The algorithm first centers and scales the data, performs PCA to extract principal components, selects a 
/// subset of these components based on either a fixed number or explained variance ratio, and then performs 
/// linear regression using the selected components.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of PCR as a two-step process: first, it finds the most important patterns in your input data 
/// (principal components), then it uses these patterns instead of the original variables to build a regression model. 
/// This can help when your original variables are highly related to each other (multicollinear), which can cause 
/// problems in standard regression.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Principal Component Regression to handle multicollinearity
/// var options = new PrincipalComponentRegressionOptions&lt;double&gt;();
/// var model = new PrincipalComponentRegression&lt;double&gt;(options);
///
/// // Prepare training data: 5 samples with 3 features
/// var features = Matrix&lt;double&gt;.Build.Dense(5, 3, new double[] {
///     1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12,  13, 14, 15 });
/// var targets = new Vector&lt;double&gt;(new double[] { 6.0, 15.1, 24.0, 33.2, 42.0 });
///
/// // Train with PCA dimensionality reduction then regression
/// model.Train(features, targets);
///
/// // Predict for a new sample
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 3, new double[] { 16, 17, 18 });
/// var prediction = model.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Regression)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Principal Component Analysis", "https://doi.org/10.1007/978-1-4757-1904-8")]
public partial class PrincipalComponentRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the principal component regression model.
    /// </summary>
    /// <value>
    /// Contains settings like the number of components to use or the explained variance ratio threshold.
    /// </value>
    private readonly PrincipalComponentRegressionOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The principal components extracted from the training data.
    /// </summary>
    /// <value>
    /// A matrix where each column represents a principal component.
    /// </value>
    [Buffer]
    private Matrix<T> _components;

    /// <summary>
    /// The mean of each predictor variable used for centering.
    /// </summary>
    /// <value>
    /// A vector containing the mean value of each predictor variable.
    /// </value>
    [Buffer]
    private Vector<T> _xMean;

    /// <summary>
    /// The mean of the target variable used for centering.
    /// </summary>
    /// <value>
    /// The mean value of the target variable.
    /// </value>
    private T _yMean;

    /// <summary>
    /// The standard deviation of each predictor variable used for scaling.
    /// </summary>
    /// <value>
    /// A vector containing the standard deviation of each predictor variable.
    /// </value>
    [Buffer]
    private Vector<T> _xStd;

    /// <summary>
    /// The standard deviation of the target variable used for scaling.
    /// </summary>
    /// <value>
    /// The standard deviation value of the target variable.
    /// </value>
    private T _yStd;

    /// <summary>
    /// Initializes a new instance of the PrincipalComponentRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the PCR model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the PCR model with your specified settings or uses
    /// default settings if none are provided. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public PrincipalComponentRegression(PrincipalComponentRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new PrincipalComponentRegressionOptions<T>();
        _components = new Matrix<T>(0, 0);
        _xMean = Vector<T>.Empty();
        _yMean = NumOps.Zero;
        _xStd = Vector<T>.Empty();
        _yStd = NumOps.Zero;
    }

    /// <summary>
    /// Trains the principal component regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method performs the following steps:
    /// 1. Validates the input data
    /// 2. Centers and scales the data
    /// 3. Performs principal component analysis (PCA) on the predictor variables
    /// 4. Selects the appropriate number of principal components
    /// 5. Projects the data onto the selected principal components
    /// 6. Performs linear regression on the projected data
    /// 7. Transforms the coefficients back to the original space
    /// 8. Applies regularization to the coefficients
    /// 9. Adjusts for scaling and calculates the intercept
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Training is the process where the model learns from your data. The PCR algorithm first centers and scales
    /// your data (makes all variables have similar ranges), then finds the most important patterns (principal components)
    /// in your input features. It selects a subset of these patterns based on your settings, and uses them to build
    /// a regression model. Finally, it converts the model back to work with your original variables.
    /// </para>
    /// </remarks>
    /// <summary>PCR doesn't benefit from optimizer parameter injection.</summary>
        /// <remarks>
    /// Expressed as a capability, not as a count. A zero ParameterCount also suppresses
    /// injection -- that is why this was written that way -- but it overloads a COUNT to carry
    /// a CAPABILITY: the model does have parameters (the base getter returns its coefficients
    /// and intercept), so the count contradicted the vector and anything pairing the two by
    /// length saw parameters the model claimed not to have.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);
        TrainingFeatureCount = x.Columns;

        // This method previously fitted ORDINARY LEAST SQUARES and returned immediately. The
        // guard that followed it was written as a condition but is always true for any real
        // problem, so it acted as an unconditional return and left the real estimation below
        // unreachable: callers received a plain linear least-squares fit from a model named for a
        // different algorithm. The real estimation now runs.

        // PCA path (not used in standard regression mode)
        // Center and scale the data
        (Matrix<T> xScaled, Vector<T> yScaled, _xMean, _xStd, _yMean, _yStd) = RegressionHelper<T>.CenterAndScale(x, y);

        bool hasVaryingFeature = false;
        for (int i = 0; i < xScaled.Rows && !hasVaryingFeature; i++)
        {
            for (int j = 0; j < xScaled.Columns; j++)
            {
                if (!NumOps.Equals(xScaled[i, j], NumOps.Zero))
                {
                    hasVaryingFeature = true;
                    break;
                }
            }
        }

        if (!hasVaryingFeature)
        {
            throw new ArgumentException(
                "Principal component regression requires at least one feature with non-zero variance.",
                nameof(x));
        }

        // Perform PCA
        (Matrix<T> components, Vector<T> explainedVariance) = PerformPCA(xScaled);

        // Select number of components
        int numComponents = SelectNumberOfComponents(explainedVariance);
        _components = components.Submatrix(0, 0, components.Rows, numComponents);

        // Project data onto principal components
        Matrix<T> xProjected = xScaled.Multiply(_components);

        // Perform linear regression on projected data
        Vector<T> coefficients = SolveSystem(xProjected, yScaled);

        // Transform coefficients back to original space
        Coefficients = _components.Multiply(coefficients);

        // Apply regularization to coefficients
        Coefficients = Regularization.Regularize(Coefficients);

        // Adjust for scaling
        for (int i = 0; i < Coefficients.Length; i++)
        {
            Coefficients[i] = NumOps.Divide(NumOps.Multiply(Coefficients[i], _yStd), _xStd[i]);
        }

        // Calculate intercept
        Intercept = NumOps.Subtract(_yMean, Coefficients.DotProduct(_xMean));
    }

    /// <summary>
    /// Validates the input data before training.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <exception cref="ArgumentException">Thrown when the number of rows in x doesn't match the length of y.</exception>
    /// <remarks>
    /// <para>
    /// This method checks that the input data is valid before proceeding with training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method makes sure your data is in the correct format before training begins.
    /// It checks that you have the same number of target values as you have examples in your input data.
    /// If not, it will raise an error to let you know there's a mismatch.
    /// </para>
    /// </remarks>
    private void ValidateInputs(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of rows in x must match the length of y.");
        }

        if (x.Rows < 2 || x.Columns == 0)
        {
            throw new ArgumentException(
                "Principal component regression requires at least two rows and one feature.", nameof(x));
        }
    }

    /// <summary>
    /// Performs Principal Component Analysis (PCA) on the input data.
    /// </summary>
    /// <param name="x">The centered and scaled input features matrix.</param>
    /// <returns>A tuple containing the principal components matrix and the explained variance vector.</returns>
    /// <remarks>
    /// <para>
    /// This method performs Singular Value Decomposition (SVD) on the input matrix to extract the principal components
    /// and calculate the explained variance for each component.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> PCA is a technique that finds the most important patterns (principal components) in your data.
    /// These components are directions in the data space where the data varies the most. The method uses
    /// a mathematical technique called Singular Value Decomposition (SVD) to find these components and
    /// calculate how much of the total variation each component explains.
    /// </para>
    /// </remarks>
    private (Matrix<T>, Vector<T>) PerformPCA(Matrix<T> x)
    {
        // Perform SVD
        var svd = new SvdDecomposition<T>(x);
        (Matrix<T> u, Vector<T> s, Matrix<T> vt) = (svd.U, svd.S, svd.Vt);

        // Components are the right singular vectors (rows of vt)
        Matrix<T> components = vt.Transpose();

        // Calculate explained variance
        Vector<T> explainedVariance = s.Transform(val => NumOps.Multiply(val, val));
        T totalVariance = explainedVariance.Sum();
        explainedVariance = explainedVariance.Transform(val => NumOps.Divide(val, totalVariance));

        return (components, explainedVariance);
    }

    /// <summary>
    /// Selects the appropriate number of principal components to use in the regression model.
    /// </summary>
    /// <param name="explainedVariance">The explained variance vector for each principal component.</param>
    /// <returns>The number of principal components to use.</returns>
    /// <remarks>
    /// <para>
    /// This method selects the number of components based on either a fixed number specified in the options
    /// or a threshold for the cumulative explained variance ratio.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method decides how many principal components to keep for the regression model. It can either use
    /// a fixed number that you specify, or it can select enough components to explain a certain percentage of
    /// the total variation in your data (e.g., keep enough components to explain 95% of the variance).
    /// </para>
    /// </remarks>
    private int SelectNumberOfComponents(Vector<T> explainedVariance)
    {
        if (_options.NumComponents > 0)
        {
            return Math.Min(_options.NumComponents, explainedVariance.Length);
        }

        T cumulativeVariance = NumOps.Zero;
        for (int i = 0; i < explainedVariance.Length; i++)
        {
            cumulativeVariance = NumOps.Add(cumulativeVariance, explainedVariance[i]);
            if (NumOps.GreaterThanOrEquals(cumulativeVariance, NumOps.FromDouble(_options.ExplainedVarianceRatio)))
            {
                return i + 1;
            }
        }

        return explainedVariance.Length;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        // OLS coefficients are in original space. Use base class prediction.
        return base.Predict(input);
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, coefficients, principal components,
    /// number of components used, and feature importance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Model metadata provides information about the model itself, rather than the predictions it makes.
    /// This includes details about how the model is configured (like how many components it uses) and
    /// information about the importance of different features. This can help you understand which input
    /// variables are most influential in making predictions.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
        {
            { "Coefficients", Coefficients },
            { "Components", _components },
            { "NumComponents", _components.Columns },
            { "FeatureImportance", CalculateFeatureImportances() }
        }
        };
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier for principal component regression.</returns>
    /// <remarks>
    /// <para>
    /// This method is used for model identification and serialization purposes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method simply returns an identifier that indicates this is a principal component regression model.
    /// It's used internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>

    /// <summary>
    /// Calculates the importance of each feature in the model.
    /// </summary>
    /// <returns>A vector containing the importance score for each feature.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates feature importances based on the absolute values of the regression coefficients.
    /// Larger absolute values indicate more important features.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Feature importance tells you which input variables have the most influence on the predictions.
    /// In PCR, this is calculated based on the magnitude (absolute value) of each coefficient in the model.
    /// Features with larger coefficient magnitudes have a stronger effect on the predictions and are considered
    /// more important.
    /// </para>
    /// </remarks>
    protected override Vector<T> CalculateFeatureImportances()
    {
        // Feature importances are based on the magnitude of the coefficients
        return Coefficients.Transform(NumOps.Abs);
    }
}
