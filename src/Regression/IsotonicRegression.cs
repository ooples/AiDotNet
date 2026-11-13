using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Regression;

/// <summary>
/// Implements an Isotonic Regression model, which fits a free-form line to data with the constraint
/// that the fitted line must be non-decreasing (monotonically increasing).
/// </summary>
/// <remarks>
/// <para>
/// Isotonic Regression is a form of nonlinear regression that fits a non-decreasing function to data.
/// Unlike many regression techniques, it makes minimal assumptions about the shape of the function
/// besides monotonicity (that the function doesn't decrease as the input increases). This makes it
/// particularly useful for calibrating probability estimates from other models or for situations where
/// a monotonic relationship is expected between the input and output variables.
/// </para>
/// <para><b>For Beginners:</b> Isotonic Regression creates a "stair-step" function that only goes up, never down.
/// 
/// Imagine you're drawing a line through points on a graph, but with two key rules:
/// - The line can go up or stay flat, but it can never go down
/// - The line should stick as close as possible to all the data points
/// 
/// This model is useful when you know that as one value increases, the other should never decrease.
/// For example:
/// - As study time increases, test scores shouldn't decrease
/// - As price increases, demand shouldn't increase
/// - As age increases, height (for children) shouldn't decrease
/// 
/// Unlike a straight line (linear regression), Isotonic Regression can capture more complex relationships
/// while still maintaining this "never decreasing" property.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an isotonic regression with monotonically non-decreasing constraint
///
/// // Prepare training data: 5 samples with 1 feature (single sorted predictor)
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 1.2, 2.8, 2.5, 4.1, 5.3 });
///
/// // Train with pool adjacent violators algorithm
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new IsotonicRegression&lt;double&gt;())
///     .Build(features, targets);
///
/// // Predict for a new sample (output is monotonically non-decreasing)
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 3.5 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Isotonic Regression Under Lipschitz Constraint", "https://doi.org/10.1080/01621459.1972.10482387")]
public partial class IsotonicRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// The sorted input values from the training data.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _xValues;

    /// <summary>
    /// The target values corresponding to the sorted input values.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _yValues;
    private int _trainingFeatureCount;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _olsCoefficients;
    private T _olsIntercept;

    /// <summary>
    /// Initializes a new instance of the <see cref="IsotonicRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration options for the nonlinear regression algorithm.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Isotonic Regression model with the specified options and regularization
    /// strategy. If no options are provided, default values are used. If no regularization is specified, no regularization
    /// is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new Isotonic Regression model.
    /// 
    /// The constructor is quite simple because Isotonic Regression doesn't have many configuration options
    /// compared to more complex models. You can specify:
    /// - Options: General settings for nonlinear regression
    /// - Regularization: Helps prevent the model from being too sensitive to small variations in the data
    /// 
    /// If you don't specify these parameters, the model will use reasonable default settings.
    /// 
    /// Example:
    /// ```csharp
    /// // Create an Isotonic Regression model with default settings
    /// var isoReg = new IsotonicRegression&lt;double&gt;();
    /// ```
    /// </para>
    /// </remarks>
    public IsotonicRegression(NonLinearRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _xValues = Vector<T>.Empty();
        _yValues = Vector<T>.Empty();
        _olsIntercept = NumOps.Zero;
    }

    /// <summary>
    /// Isotonic regression learns a monotonic step function — no optimizer parameter injection.
    /// </summary>
        /// <remarks>
    /// Expressed as a capability, not as a count. A zero ParameterCount also suppresses
    /// injection -- that is why this was written that way -- but it overloads a COUNT to carry
    /// a CAPABILITY: the model does have parameters (the base getter returns its coefficients
    /// and intercept), so the count contradicted the vector and anything pairing the two by
    /// length saw parameters the model claimed not to have.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

    /// <summary>
    /// Trains the Isotonic Regression model using the provided input features and target values.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature. For Isotonic Regression, typically only the first column is used.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <remarks>
    /// <para>
    /// This method trains the Isotonic Regression model by validating inputs, extracting the first column of the
    /// feature matrix (since Isotonic Regression typically works with 1D input), and optimizing the model using
    /// the Pool Adjacent Violators (PAV) algorithm. The model learns a non-decreasing function that best fits
    /// the relationship between the input feature and the target values.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model how to make predictions using your data.
    /// 
    /// During training, the model:
    /// 1. Takes your input data (typically just one feature/column)
    /// 2. Links each input value with its corresponding output value
    /// 3. Adjusts the relationship to ensure it only goes up or stays flat, never down
    /// 4. Uses a special algorithm (called Pool Adjacent Violators) to find the best-fitting monotonic function
    /// 
    /// After training, the model will be ready to make predictions on new data.
    /// 
    /// Example:
    /// ```csharp
    /// // Train the model
    /// isoReg.Train(features, targets);
    /// ```
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);
        _trainingFeatureCount = x.Columns;

        // Compute 1D projection using OLS for ordering (handles multi-feature data)
        // For single-feature data this degenerates to using that feature directly
        Vector<T> xProjection;
        if (x.Columns == 1)
        {
            xProjection = x.GetColumn(0);
            _olsCoefficients = null;
            _olsIntercept = NumOps.Zero;
        }
        else
        {
            // Compute OLS coefficients: beta = (X'X + λI)^-1 X'y
            var xWithInt = x.AddConstantColumn(NumOps.One);
            var engine = AiDotNetEngine.Current;
            var xTx = xWithInt.Transpose().Multiply(xWithInt);
            var xTy = xWithInt.Transpose().Multiply(y);
            xTx = engine.MatrixAdd(xTx, Matrix<T>.CreateDiagonal(engine.Fill(xTx.Rows, NumOps.FromDouble(1e-10))));
            var solution = xTx.Inverse().Multiply(xTy);
            _olsIntercept = solution[0];
            _olsCoefficients = solution.Slice(1, x.Columns);

            // Project training data: x_proj = X * beta + intercept
            xProjection = ProjectRows(x, _olsCoefficients, x.Columns);
        }

        // Sort by projected x values for PAV algorithm (stable: tied values keep sample order)
        var indices = xProjection
            .Select((value, index) => (Value: value, Index: index))
            .OrderBy(p => p.Value, Comparer<T>.Default)
            .Select(p => p.Index)
            .ToArray();
        _xValues = RegressionEngineOps.Gather(xProjection, indices);
        _yValues = RegressionEngineOps.Gather(y, indices);

        OptimizeModel(x, _yValues);
    }

    /// <summary>
    /// Optimizes the Isotonic Regression model using the Pool Adjacent Violators (PAV) algorithm.
    /// </summary>
    /// <param name="x">The feature matrix of training samples.</param>
    /// <param name="y">The target vector of training samples.</param>
    /// <remarks>
    /// <para>
    /// This method implements the Pool Adjacent Violators (PAV) algorithm, which is the standard optimization
    /// technique for Isotonic Regression. The algorithm starts by assigning each target value to its corresponding
    /// input value, then repeatedly merges adjacent violators (points that violate the monotonicity constraint)
    /// by replacing them with their weighted average. This process continues until the fitted values are monotonically
    /// increasing.
    /// </para>
    /// <para><b>For Beginners:</b> This method does the actual work of finding the best "stair-step" function.
    /// 
    /// The algorithm works like this:
    /// 1. Start by assuming each input point maps directly to its output value
    /// 2. Look for any places where the function would go down (violating the "never decreasing" rule)
    /// 3. When it finds such places, it "pools" those points together by replacing them with their average value
    /// 4. It keeps doing this until there are no more violations and the function only goes up or stays flat
    /// 
    /// This approach is called the "Pool Adjacent Violators" algorithm, and it guarantees finding the
    /// monotonically increasing function that best fits your data.
    /// </para>
    /// </remarks>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Single-pass Pool Adjacent Violators (PAV) algorithm using block sums
        // to avoid weight overflow that occurs with repeated multi-pass merging
        var n = y.Length;
        if (n == 0)
        {
            SupportVectors = new Matrix<T>(0, 1);
            Alphas = new Vector<T>(0);
            return;
        }

        // Track blocks by their sum and count (avoids large weight * value products)
        var blockSum = new double[n];
        var blockCount = new int[n];
        var blockEnd = new int[n]; // blockEnd[i] = last index in block starting at i

        for (int i = 0; i < n; i++)
        {
            blockSum[i] = NumOps.ToDouble(y[i]);
            blockCount[i] = 1;
            blockEnd[i] = i;
        }

        // Single-pass PAV: scan left-to-right, merge backward when violation found
        var blocks = new List<(int start, int end, double sum, int count)>();
        blocks.Add((0, 0, blockSum[0], 1));

        for (int i = 1; i < n; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            blocks.Add((i, i, yi, 1));

            // Merge backward while monotonicity is violated
            while (blocks.Count > 1)
            {
                var last = blocks[^1];
                var prev = blocks[^2];
                double lastMean = last.sum / last.count;
                double prevMean = prev.sum / prev.count;

                if (prevMean <= lastMean)
                    break;

                // Merge last two blocks
                blocks.RemoveAt(blocks.Count - 1);
                blocks[^1] = (prev.start, last.end, prev.sum + last.sum, prev.count + last.count);
            }
        }

        // Expand blocks back to per-element predictions
        var yhat = new double[n];
        foreach (var (start, end, sum, count) in blocks)
        {
            double mean = sum / count;
            for (int i = start; i <= end; i++)
                yhat[i] = mean;
        }

        SupportVectors = new Matrix<T>(n, 1);
        var alphas = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            SupportVectors[i, 0] = _xValues[i];
            alphas[i] = NumOps.FromDouble(yhat[i]);
        }
        Alphas = alphas;
    }

    /// <summary>
    /// Predicts target values for the provided input features using the trained Isotonic Regression model.
    /// </summary>
    /// <param name="input">A matrix where each row represents a sample to predict and each column represents a feature. For Isotonic Regression, typically only the first column is used.</param>
    /// <returns>A vector of predicted values corresponding to each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts target values for new input data by finding the nearest support vector for each input
    /// sample and retrieving the corresponding fitted value. Since Isotonic Regression typically works with 1D input,
    /// only the first column of the input matrix is used for predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to make predictions on new data.
    /// 
    /// To make a prediction for a new input value, the model:
    /// 1. Looks at the input value (typically just one number)
    /// 2. Finds the closest value from the training data
    /// 3. Returns the corresponding output value from the fitted "stair-step" function
    /// 
    /// This approach ensures that predictions follow the same monotonic (never decreasing) pattern
    /// that was learned during training.
    /// 
    /// Example:
    /// ```csharp
    /// // Make predictions
    /// var predictions = isoReg.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (Alphas.Length == 0 || input.Rows == 0)
        {
            return new Vector<T>(input.Rows);
        }

        // Project every row to 1D using the same OLS projection used during training
        Vector<T> projected = _olsCoefficients is not null
            ? ProjectRows(input, _olsCoefficients, Math.Min(input.Columns, _olsCoefficients.Length))
            : AiDotNetEngine.Current.GetColumn(input, 0); // Single-feature fallback

        return PredictProjected(projected);
    }

    /// <summary>
    /// Computes intercept + X[:, 0..featureCount) * coefficients[0..featureCount) for every row.
    /// </summary>
    private Vector<T> ProjectRows(Matrix<T> x, Vector<T> coefficients, int featureCount)
    {
        var engine = AiDotNetEngine.Current;
        if (featureCount <= 0)
        {
            return engine.Fill(x.Rows, _olsIntercept);
        }

        var features = featureCount == x.Columns ? x : x.GetSubMatrix(0, 0, x.Rows, featureCount);
        var beta = featureCount == coefficients.Length ? coefficients : coefficients.GetSubVector(0, featureCount);
        return engine.Add(engine.Fill(x.Rows, _olsIntercept), engine.MatrixVectorMultiply(features, beta));
    }

    /// <summary>
    /// Looks up the fitted isotonic value for every projected input at once.
    /// </summary>
    /// <remarks>
    /// For each value the upper-bound position p (the number of support points &lt;= x, capped at the last
    /// point) is found with a sorted search. Point p wins when p &gt; 0 and point p-1 is strictly farther
    /// away; otherwise point max(0, p-1) wins.
    /// </remarks>
    private Vector<T> PredictProjected(Vector<T> xs)
    {
        var engine = AiDotNetEngine.Current;
        int n = SupportVectors.Rows;
        int lastAlpha = Alphas.Length - 1;
        var supportPoints = SupportVectors.GetColumn(0);

        int[] upper = engine.TensorSearchSorted(
            Tensor<T>.FromVector(supportPoints), Tensor<T>.FromVector(xs), right: true).ToArray();
        var current = new int[upper.Length];
        var previous = new int[upper.Length];
        var hasPrevious = new bool[upper.Length];
        for (int k = 0; k < upper.Length; k++)
        {
            int left = n == 1 ? 0 : Math.Min(upper[k], n - 1);
            current[k] = left;
            previous[k] = Math.Max(0, left - 1);
            hasPrevious[k] = left > 0;
        }

        var currentDistance = engine.Abs(engine.Subtract(RegressionEngineOps.Gather(supportPoints, current), xs));
        var previousDistance = engine.Abs(engine.Subtract(RegressionEngineOps.Gather(supportPoints, previous), xs));
        var takeCurrent = engine.TensorGreaterThan(
            Tensor<T>.FromVector(previousDistance), Tensor<T>.FromVector(currentDistance));

        var currentAlpha = Tensor<T>.FromVector(
            RegressionEngineOps.Gather(Alphas, current.Select(i => Math.Min(i, lastAlpha)).ToArray()));
        var previousAlpha = Tensor<T>.FromVector(
            RegressionEngineOps.Gather(Alphas, previous.Select(i => Math.Min(i, lastAlpha)).ToArray()));

        var nearest = engine.TensorWhere(takeCurrent, currentAlpha, previousAlpha);
        return engine.TensorWhere(hasPrevious, nearest, previousAlpha).ToVector();
    }

    /// <summary>
    /// Predicts the target value for a single input feature vector.
    /// </summary>
    /// <param name="input">The feature vector of the sample to predict.</param>
    /// <returns>The predicted value for the input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts the target value for a single input feature vector by extracting the first element
    /// (since Isotonic Regression typically works with 1D input), finding the nearest support vector, and
    /// retrieving the corresponding fitted value.
    /// </para>
    /// <para><b>For Beginners:</b> This is the method that makes a prediction for a single input value.
    /// 
    /// For a given input:
    /// 1. It takes just the first feature (Isotonic Regression usually works with one input variable)
    /// 2. It finds the closest matching point from the training data
    /// 3. It returns the corresponding output value from the fitted function
    /// 
    /// This ensures that the prediction follows the monotonic relationship learned during training.
    /// </para>
    /// </remarks>
    protected override T PredictSingle(Vector<T> input)
    {
        if (Alphas.Length == 0)
            return NumOps.Zero;

        // Project input to 1D using the same OLS projection used during training
        T x;
        if (_olsCoefficients is not null)
        {
            int featureCount = Math.Min(input.Length, _olsCoefficients.Length);
            x = featureCount <= 0
                ? _olsIntercept
                : NumOps.Add(_olsIntercept, AiDotNetEngine.Current.DotProduct(
                    input.GetSubVector(0, featureCount), _olsCoefficients.GetSubVector(0, featureCount)));
        }
        else
        {
            x = input[0]; // Single-feature fallback
        }

        var result = PredictProjected(new Vector<T>(new[] { x }));
        return result[0];
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new IsotonicRegression<T>(Options, Regularization);
    }
}
