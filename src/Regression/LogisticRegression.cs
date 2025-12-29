namespace AiDotNet.Regression;

/// <summary>
/// Represents a logistic regression model for binary classification problems.
/// </summary>
/// <remarks>
/// <para>
/// Logistic regression is a statistical method used for binary classification tasks, where the goal is to predict
/// one of two possible outcomes (such as yes/no, true/false, 0/1). Unlike linear regression, which predicts continuous values,
/// logistic regression outputs probabilities between 0 and 1, which can be interpreted as the likelihood of belonging to the
/// positive class.
/// </para>
/// <para><b>For Beginners:</b> Logistic regression is like a decision-maker that predicts whether something belongs to
/// one category or another.
/// 
/// Think of it like determining whether an email is spam or not:
/// - The model looks at different "features" of the email (like certain words or sender information)
/// - It calculates how much each feature suggests "spam" or "not spam"
/// - It combines all this information to make a final prediction between 0 and 1
/// - Values closer to 1 mean "more likely spam", values closer to 0 mean "more likely not spam"
/// 
/// For example, words like "free" or "offer" might increase the spam probability, while emails from your contacts
/// might decrease it. Logistic regression finds the right balance of these factors to make accurate predictions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LogisticRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// The configuration options for the logistic regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control the behavior of the logistic regression algorithm during training, including
    /// parameters such as the maximum number of iterations and convergence tolerance.
    /// </para>
    /// <para><b>For Beginners:</b> These are the settings that control how the model learns.
    /// 
    /// They include:
    /// - How many attempts (iterations) the model makes to improve itself
    /// - How precise the model needs to be before it stops training
    /// - How quickly the model adjusts its predictions (learning rate)
    /// 
    /// Think of these like the knobs on a machine that you can adjust to get better results.
    /// </para>
    /// </remarks>
    private readonly LogisticRegressionOptions<T> _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="LogisticRegression{T}"/> class with optional custom options
    /// and regularization.
    /// </summary>
    /// <param name="options">Custom options for the logistic regression algorithm. If null, default options are used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new logistic regression model with the specified options and regularization.
    /// If no options are provided, default values are used. Regularization helps prevent overfitting by penalizing
    /// large coefficient values.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new logistic regression model with your chosen settings.
    /// 
    /// When creating a logistic regression model:
    /// - You can provide custom settings (options) or use the defaults
    /// - You can add regularization, which helps the model generalize better to new data
    /// 
    /// Regularization is like adding training wheels to prevent the model from memorizing the training data too closely,
    /// which would make it perform poorly on new, unseen data.
    /// </para>
    /// </remarks>
    public LogisticRegression(LogisticRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new LogisticRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the logistic regression model using the provided features and target values.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target vector containing the binary labels (0 or 1) for each sample.</param>
    /// <exception cref="ArgumentException">Thrown when the number of rows in X does not match the length of y.</exception>
    /// <remarks>
    /// <para>
    /// This method trains the logistic regression model using gradient ascent to maximize the likelihood of the observed data.
    /// The algorithm iteratively updates the coefficients and intercept based on the prediction errors until convergence
    /// or until the maximum number of iterations is reached. Regularization is applied if specified.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model learns from your data.
    /// 
    /// During training:
    /// - The model starts with initial guesses for how important each feature is
    /// - It makes predictions based on these guesses
    /// - It compares its predictions with the actual answers
    /// - It adjusts its guesses to reduce errors
    /// - This process repeats until the model stops improving significantly
    /// 
    /// For example, with email classification, the model might learn that the word "meeting" is a strong indicator
    /// of a legitimate email, while "click here to claim" suggests spam.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
            throw new ArgumentException("The number of rows in X must match the length of y.");
        int n = x.Rows;
        int p = x.Columns;
        Coefficients = new Vector<T>(p);
        Intercept = NumOps.Zero;
        // Apply regularization to the input matrix
        Matrix<T> regularizedX = Regularization != null ? Regularization.Regularize(x) : x;
        T learningRate = NumOps.FromDouble(_options.LearningRate);
        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            Vector<T> predictions = Predict(regularizedX);
            Vector<T> errors = y.Subtract(predictions);
            // Calculate gradient: X^T * (y - predictions)
            Vector<T> gradient = regularizedX.Transpose().Multiply(errors);
            // Apply regularization to the gradient
            if (Regularization != null)
            {
                gradient = ApplyRegularizationGradient(gradient);
            }
            // Update coefficients: coef += learning_rate * gradient
            Coefficients = Coefficients.Add(gradient.Multiply(learningRate));
            // Update intercept: intercept += learning_rate * sum(errors)
            T interceptGrad = errors.Sum();
            Intercept = NumOps.Add(Intercept, NumOps.Multiply(learningRate, interceptGrad));
            if (HasConverged(gradient))
                break;
        }
        // Apply final regularization to coefficients
        if (Regularization != null)
        {
            Coefficients = Regularization.Regularize(Coefficients);
        }
    }

    /// <summary>
    /// Applies regularization to the gradient during training.
    /// </summary>
    /// <param name="gradient">The current gradient to be regularized.</param>
    /// <returns>The regularized gradient.</returns>
    /// <remarks>
    /// <para>
    /// This method applies regularization to the gradient during the training process if a regularization method is specified.
    /// Regularization modifies the gradient to penalize large coefficient values, helping to prevent overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This adjusts how the model learns to prevent it from becoming too complex.
    /// 
    /// Regularization works by:
    /// - Adding a penalty for large coefficient values
    /// - Encouraging the model to use smaller, more balanced values
    /// - Helping the model focus on the most important features
    /// 
    /// This is like teaching the model to prefer simpler explanations over complicated ones,
    /// which typically leads to better performance on new data.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyRegularizationGradient(Vector<T> gradient)
    {
        if (Regularization != null)
        {
            return Regularization.Regularize(gradient, Coefficients);
        }

        return gradient;
    }

    /// <summary>
    /// Makes predictions for new data points using the trained logistic regression model.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <returns>A vector of predicted probabilities for the positive class.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the predicted probabilities for each sample in the input feature matrix.
    /// It computes the raw scores by multiplying the features by the learned coefficients and adding the intercept,
    /// then transforms these scores into probabilities using the sigmoid function.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model makes predictions on new data.
    /// 
    /// During prediction:
    /// - The model calculates a score for each example using the learned weights
    /// - The score is converted to a probability between 0 and 1 using the sigmoid function
    /// - Values closer to 1 indicate higher confidence in the positive class
    /// 
    /// For instance, if you've trained the model to detect fraudulent transactions, 
    /// a probability of 0.92 would suggest the transaction is likely fraudulent,
    /// while 0.03 would suggest it's probably legitimate.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> x)
    {
        Vector<T> scores = x.Multiply(Coefficients).Add(Intercept);
        return scores.Transform(Sigmoid);
    }

    /// <summary>
    /// Applies the sigmoid (logistic) function to transform a raw score into a probability.
    /// </summary>
    /// <param name="x">The raw score to transform.</param>
    /// <returns>A probability value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// The sigmoid function transforms any real number into a value between 0 and 1, which can be interpreted as a probability.
    /// It is defined as f(x) = 1 / (1 + e^(-x)), where e is the base of the natural logarithm.
    /// </para>
    /// <para><b>For Beginners:</b> The sigmoid function converts any number into a probability between 0 and 1.
    /// 
    /// The sigmoid function:
    /// - Takes any number as input (from negative infinity to positive infinity)
    /// - Always outputs a number between 0 and 1
    /// - Has an S-shaped curve that flattens at the extremes
    /// 
    /// For example:
    /// - Large positive numbers (like +5) become close to 1
    /// - Large negative numbers (like -5) become close to 0
    /// - Zero becomes exactly 0.5
    /// 
    /// This is perfect for binary classification because it converts raw scores into probabilities.
    /// </para>
    /// </remarks>
    private T Sigmoid(T x)
    {
        T expNegX = NumOps.Exp(NumOps.Negate(x));
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }

    /// <summary>
    /// Determines if the training process has converged based on the magnitude of the gradient.
    /// </summary>
    /// <param name="gradient">The current gradient vector.</param>
    /// <returns>True if the maximum absolute gradient value is less than the tolerance, indicating convergence; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the training process has converged by comparing the maximum absolute value in the gradient
    /// to the specified tolerance. If the maximum gradient is smaller than the tolerance, the model is considered to have converged,
    /// meaning that further iterations would not significantly improve the model.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if the model has finished learning.
    /// 
    /// Convergence means:
    /// - The model's predictions aren't improving much anymore
    /// - The adjustments being made are very small
    /// - Further training is unlikely to give better results
    /// 
    /// It's like knowing when to stop studying for a test - at some point, more studying won't improve your score much,
    /// and it's better to save your energy.
    /// </para>
    /// </remarks>
    private bool HasConverged(Vector<T> gradient)
    {
        T maxGradient = gradient.Max(NumOps.Abs) ?? NumOps.Zero;
        return NumOps.LessThan(maxGradient, NumOps.FromDouble(_options.Tolerance));
    }

    /// <summary>
    /// Gets the type of regression model.
    /// </summary>
    /// <returns>The model type, in this case, LogisticRegression.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an enumeration value indicating that this is a logistic regression model. This is used
    /// for type identification when working with different regression models in a unified manner.
    /// </para>
    /// <para><b>For Beginners:</b> This simply tells other parts of the program what kind of model this is.
    /// 
    /// When you have different types of models in your program:
    /// - Each model needs to identify itself
    /// - This method returns a label (LogisticRegression) that identifies this specific type
    /// - Other code can use this label to handle the model appropriately
    /// 
    /// It's like having different types of vehicles (cars, trucks, motorcycles) that each need to be serviced differently.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.LogisticRegression;
    }

    /// <summary>
    /// Serializes the logistic regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the entire logistic regression model, including its parameters and configuration,
    /// into a byte array that can be stored in a file or database, or transmitted over a network. The model can
    /// later be restored using the Deserialize method.
    /// </para>
    /// <para><b>For Beginners:</b> This converts the model into a format that can be saved or shared.
    /// 
    /// Serialization:
    /// - Transforms the model into a sequence of bytes
    /// - Preserves all the important information about the model
    /// - Allows you to save the trained model to a file
    /// - Lets you load the model later without having to retrain it
    /// 
    /// It's like taking a snapshot of the model that you can use later or share with others.
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
        // Serialize MultipleRegression specific data
        writer.Write(_options.MaxIterations);
        writer.Write(_options.Tolerance);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the logistic regression model from a byte array.
    /// </summary>
    /// <param name="modelData">A byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method restores a logistic regression model from a serialized byte array, reconstructing its parameters
    /// and configuration. This allows a previously trained model to be loaded from storage or after being received
    /// over a network.
    /// </para>
    /// <para><b>For Beginners:</b> This rebuilds the model from a saved format.
    /// 
    /// Deserialization:
    /// - Takes a sequence of bytes that represents a model
    /// - Reconstructs the original model with all its learned patterns
    /// - Allows you to use a previously trained model without retraining
    /// 
    /// Think of it like unpacking a model that was packed up for storage or shipping,
    /// so you can use it again exactly as it was.
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
        // Deserialize MultipleRegression specific data
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = reader.ReadDouble();
    }

    /// <summary>
    /// Creates a new instance of the logistic regression model.
    /// </summary>
    /// <returns>A new instance of the logistic regression model with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the logistic regression model with the same configuration as the current instance.
    /// It is used internally during serialization/deserialization to create a new instance of the model.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the model structure without copying the learned data.
    /// 
    /// It's like creating a new, empty notebook with the same number of pages and section dividers as your current notebook,
    /// but without copying any of the notes you've written. This is useful when you want to create a similar model
    /// or when loading a saved model from a file.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new LogisticRegression<T>(_options, Regularization);
    }
}
