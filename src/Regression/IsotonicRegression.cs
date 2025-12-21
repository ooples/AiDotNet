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
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class IsotonicRegression<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// The sorted input values from the training data.
    /// </summary>
    private Vector<T> _xValues;

    /// <summary>
    /// The target values corresponding to the sorted input values.
    /// </summary>
    private Vector<T> _yValues;

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
    }

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

        // Apply regularization to the input matrix
        var regularizedX = Regularization.Regularize(x);

        _xValues = regularizedX.GetColumn(0); // Isotonic regression typically works with 1D input
        _yValues = y;

        OptimizeModel(regularizedX, _yValues);
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
        // Implement Pool Adjacent Violators (PAV) algorithm
        var n = y.Length;
        var yhat = new Vector<T>(n);
        var w = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            yhat[i] = y[i];
            w[i] = NumOps.One;
        }

        bool changed;
        do
        {
            changed = false;
            for (int i = 0; i < n - 1; i++)
            {
                if (NumOps.LessThan(yhat[i + 1], yhat[i]))
                {
                    var weightedMean = NumOps.Divide(
                        NumOps.Add(NumOps.Multiply(w[i], yhat[i]), NumOps.Multiply(w[i + 1], yhat[i + 1])),
                        NumOps.Add(w[i], w[i + 1])
                    );
                    yhat[i] = weightedMean;
                    yhat[i + 1] = weightedMean;
                    w[i] = NumOps.Add(w[i], w[i + 1]);
                    w[i + 1] = w[i];
                    changed = true;
                }
            }
        } while (changed);

        // Apply regularization to the coefficients
        yhat = Regularization.Regularize(yhat);

        SupportVectors = new Matrix<T>(n, 1);
        for (int i = 0; i < n; i++)
        {
            SupportVectors[i, 0] = _xValues[i];
        }
        Alphas = yhat;
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
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
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
        var x = input[0]; // Isotonic regression typically works with 1D input
        int index = FindNearestIndex(x);
        return Alphas[index];
    }

    /// <summary>
    /// Finds the index of the nearest support vector to the given input value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The index of the nearest support vector.</returns>
    private int FindNearestIndex(T x)
    {
        int left = 0;
        int right = SupportVectors.Rows - 1;

        while (left < right)
        {
            int mid = (left + right) / 2;
            if (NumOps.LessThanOrEquals(SupportVectors[mid, 0], x))
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }

        if (left > 0 && NumOps.GreaterThan(NumOps.Abs(NumOps.Subtract(SupportVectors[left - 1, 0], x)),
                                           NumOps.Abs(NumOps.Subtract(SupportVectors[left, 0], x))))
        {
            return left;
        }

        return left - 1;
    }

    /// <summary>
    /// Gets the model type of the Isotonic Regression model.
    /// </summary>
    /// <returns>The model type enumeration value.</returns>
    protected override ModelType GetModelType()
    {
        return ModelType.IsotonicRegression;
    }

    /// <summary>
    /// Serializes the Isotonic Regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the Isotonic Regression model into a byte array that can be stored in a file, database,
    /// or transmitted over a network. The serialized data includes the base class data, input values, and target
    /// values used during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained model as a sequence of bytes.
    /// 
    /// Serialization allows you to:
    /// - Save your model to a file
    /// - Store your model in a database
    /// - Send your model over a network
    /// - Keep your model for later use without having to retrain it
    /// 
    /// The serialized data includes:
    /// - The input values from your training data
    /// - The corresponding output values
    /// - All the information needed to recreate the model exactly as it was
    /// 
    /// Example:
    /// ```csharp
    /// // Serialize the model
    /// byte[] modelData = isoReg.Serialize();
    /// 
    /// // Save to a file
    /// File.WriteAllBytes("isotonicRegression.model", modelData);
    /// ```
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize IsotonicRegression specific data
        writer.Write(_xValues.Length);
        for (int i = 0; i < _xValues.Length; i++)
        {
            writer.Write(Convert.ToDouble(_xValues[i]));
        }

        writer.Write(_yValues.Length);
        for (int i = 0; i < _yValues.Length; i++)
        {
            writer.Write(Convert.ToDouble(_yValues[i]));
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Loads a previously serialized Isotonic Regression model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs an Isotonic Regression model from a byte array that was previously created using the
    /// Serialize method. It restores the base class data, input values, and target values, allowing the model to be
    /// used for predictions without retraining.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a sequence of bytes.
    /// 
    /// Deserialization allows you to:
    /// - Load a model that was saved earlier
    /// - Use a model without having to retrain it
    /// - Share models between different applications
    /// 
    /// When you deserialize a model:
    /// - The input and output values from training are recovered
    /// - The model is ready to make predictions immediately
    /// - You don't need to go through the training process again
    /// 
    /// Example:
    /// ```csharp
    /// // Load from a file
    /// byte[] modelData = File.ReadAllBytes("isotonicRegression.model");
    /// 
    /// // Deserialize the model
    /// var isoReg = new IsotonicRegression&lt;double&gt;();
    /// isoReg.Deserialize(modelData);
    /// 
    /// // Now you can use the model for predictions
    /// var predictions = isoReg.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize IsotonicRegression specific data
        int xLength = reader.ReadInt32();
        _xValues = new Vector<T>(xLength);
        for (int i = 0; i < xLength; i++)
        {
            _xValues[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        int yLength = reader.ReadInt32();
        _yValues = new Vector<T>(yLength);
        for (int i = 0; i < yLength; i++)
        {
            _yValues[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    /// <summary>
    /// Creates a new instance of the IsotonicRegression with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new IsotonicRegression instance with the same options and regularization as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the IsotonicRegression model with the same configuration options
    /// and regularization settings as the current instance. This is useful for model cloning, ensemble methods, or
    /// cross-validation scenarios where multiple instances of the same model with identical configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model's blueprint.
    /// 
    /// When you need multiple versions of the same type of model with identical settings:
    /// - This method creates a new, empty model with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new model has the same settings but no trained data
    /// - This is useful for techniques that need multiple models, like cross-validation
    /// 
    /// For example, when testing your model on different subsets of data,
    /// you'd want each test to use a model with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new IsotonicRegression<T>(Options, Regularization);
    }
}
