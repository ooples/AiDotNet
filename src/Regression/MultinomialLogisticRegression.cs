namespace AiDotNet.Regression;

/// <summary>
/// Represents a multinomial logistic regression model for multi-class classification problems.
/// </summary>
/// <remarks>
/// <para>
/// Multinomial logistic regression extends binary logistic regression to handle multiple classes. It models the probabilities
/// of different possible outcomes using the softmax function. For each class, the model learns a set of coefficients that
/// determine how each feature affects the probability of that class. During prediction, it assigns the input to the class
/// with the highest probability.
/// </para>
/// <para><b>For Beginners:</b> Multinomial logistic regression is a method for classifying data into multiple categories.
/// 
/// Think of it like a voting system where:
/// - Each feature (input variable) gets to "vote" for different categories
/// - The importance of each feature's vote is learned from training data
/// - For any new data point, we count the weighted votes for each category
/// - The category with the most votes wins and becomes the prediction
/// 
/// For example, when classifying emails into categories like "work," "personal," or "spam,"
/// certain words might strongly suggest one category over others. The model learns which
/// features (words) are most helpful for distinguishing between the different categories.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MultinomialLogisticRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// The configuration options for the multinomial logistic regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control the behavior of the multinomial logistic regression algorithm during training, including
    /// parameters such as the maximum number of iterations, convergence tolerance, and the matrix decomposition method
    /// used for solving the linear system.
    /// </para>
    /// <para><b>For Beginners:</b> These are the settings that control how the model learns.
    /// 
    /// Key settings include:
    /// - How many attempts (iterations) the model makes to improve itself
    /// - How precise the model needs to be before it stops training
    /// - What mathematical method to use for calculations
    /// 
    /// These settings affect how quickly the model trains and how accurate it becomes.
    /// Think of them as the "knobs" you can adjust to fine-tune the learning process.
    /// </para>
    /// </remarks>
    private readonly MultinomialLogisticRegressionOptions<T> _options;

    /// <summary>
    /// The coefficients matrix, where each row corresponds to a class and each column to a feature (plus intercept).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The coefficients matrix contains the learned weights for the multinomial logistic regression model. Each row
    /// corresponds to a different class, and each column corresponds to a different feature (with an additional column
    /// for the intercept term). These coefficients determine how each feature influences the probability of each class.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "knowledge" the model learns from the training data.
    /// 
    /// The coefficients:
    /// - Show how important each feature is for predicting each class
    /// - Positive values mean the feature increases the chance of that class
    /// - Negative values mean the feature decreases the chance of that class
    /// - Larger absolute values (further from zero) indicate stronger influences
    /// 
    /// For example, in email classification, the word "meeting" might have a high coefficient for
    /// "work" emails and a low coefficient for "spam" emails.
    /// </para>
    /// </remarks>
    private Matrix<T>? _coefficients;

    /// <summary>
    /// The number of distinct classes in the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the number of distinct classes found in the training data. It determines the number of rows
    /// in the coefficients matrix, as each class has its own set of coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of different categories the model can predict.
    /// 
    /// For example:
    /// - In email classification, it might be 3 (work, personal, spam)
    /// - In product categorization, it might be dozens or hundreds of categories
    /// - In sentiment analysis, it might be 3 (positive, neutral, negative)
    /// 
    /// The model learns a separate set of weights for each of these categories.
    /// </para>
    /// </remarks>
    private int _numClasses;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultinomialLogisticRegression{T}"/> class with optional custom options and regularization.
    /// </summary>
    /// <param name="options">Custom options for the multinomial logistic regression algorithm. If null, default options are used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new multinomial logistic regression model with the specified options and regularization.
    /// If no options are provided, default values are used. Regularization helps prevent overfitting by penalizing
    /// large coefficient values.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new multinomial logistic regression model with your chosen settings.
    /// 
    /// When creating the model:
    /// - You can provide custom settings (options) or use the defaults
    /// - You can add regularization, which helps prevent the model from memorizing the training data
    /// 
    /// Regularization is like adding a penalty for complexity, encouraging the model to keep things
    /// simple unless there's strong evidence for complexity. This typically helps the model
    /// perform better on new, unseen data.
    /// </para>
    /// </remarks>
    public MultinomialLogisticRegression(MultinomialLogisticRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new MultinomialLogisticRegressionOptions<T>();
    }

    /// <summary>
    /// Trains the multinomial logistic regression model using the provided features and target values.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target vector containing the class labels (as integers) for each sample.</param>
    /// <remarks>
    /// <para>
    /// This method trains the multinomial logistic regression model using Newton's method to find the maximum likelihood
    /// estimates of the coefficients. It iteratively computes the probabilities, gradient, and Hessian matrix, and updates
    /// the coefficients until convergence or until the maximum number of iterations is reached. Regularization is applied
    /// if specified.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model learns from your data.
    /// 
    /// During training:
    /// 1. The model starts with initial guesses for the coefficients
    /// 2. It calculates how likely each class is for each training example
    /// 3. It calculates how to change the coefficients to improve the predictions
    /// 4. It updates the coefficients based on a mathematically optimal approach (Newton's method)
    /// 5. It repeats steps 2-4 until the changes become very small or a maximum number of iterations is reached
    /// 
    /// This process finds the best coefficients for distinguishing between the different classes
    /// based on the features in your training data.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);
        _numClasses = y.Distinct().Count();

        int numFeatures = x.Columns;
        _coefficients = new Matrix<T>(_numClasses, numFeatures + 1);

        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            Matrix<T> probabilities = ComputeProbabilities(xWithIntercept);
            Matrix<T> gradient = ComputeGradient(xWithIntercept, y, probabilities);
            Matrix<T> hessian = ComputeHessian(xWithIntercept, probabilities);

            if (Regularization != null)
            {
                gradient = Regularization.Regularize(gradient);
                hessian = Regularization.Regularize(hessian);
            }

            Vector<T> flattenedGradient = gradient.Flatten();
            Vector<T> update = MatrixSolutionHelper.SolveLinearSystem(hessian, flattenedGradient, MatrixDecompositionFactory.GetDecompositionType(_options.DecompositionMethod));

            Matrix<T> updateMatrix = new Matrix<T>(gradient.Rows, gradient.Columns);
            for (int i = 0; i < update.Length; i++)
            {
                updateMatrix[i / gradient.Columns, i % gradient.Columns] = update[i];
            }

            _coefficients = _coefficients.Subtract(updateMatrix);

            if (HasConverged(updateMatrix))
            {
                break;
            }
        }

        Coefficients = _coefficients.GetColumn(0);
        Intercept = _coefficients.GetColumn(_coefficients.Columns - 1)[0];
    }

    /// <summary>
    /// Computes the probabilities of each class for each sample using the softmax function.
    /// </summary>
    /// <param name="x">The feature matrix with intercept term.</param>
    /// <returns>A matrix of probabilities, where each row corresponds to a sample and each column to a class.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the coefficients have not been initialized.</exception>
    /// <remarks>
    /// <para>
    /// This method computes the probabilities of each class for each sample using the softmax function. It first calculates
    /// the raw scores by multiplying the features with the coefficients, then applies the softmax function to convert these
    /// scores into probabilities that sum to 1 for each sample.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how likely each class is for each data point.
    /// 
    /// The probability calculation:
    /// 1. Multiplies each feature by its corresponding coefficient for each class (weighted voting)
    /// 2. Sums these values to get a "score" for each class
    /// 3. Applies the softmax function, which converts these scores into probabilities
    /// 4. The probabilities for all classes for a single sample add up to 100%
    /// 
    /// The softmax function ensures that increasing the score for one class increases its probability
    /// while decreasing the probability of other classes proportionally.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeProbabilities(Matrix<T> x)
    {
        if (_coefficients == null)
            throw new InvalidOperationException("Coefficients have not been initialized.");

        Matrix<T> scores = x.Multiply(_coefficients.Transpose());
        Vector<T> maxScores = scores.RowWiseMax();
        Matrix<T> expScores = scores.Transform((s, i, j) => NumOps.Exp(NumOps.Subtract(s, maxScores[i])));
        Vector<T> sumExpScores = expScores.RowWiseSum();

        return expScores.PointwiseDivide(sumExpScores.ToColumnMatrix());
    }

    /// <summary>
    /// Computes the gradient of the log-likelihood with respect to the coefficients.
    /// </summary>
    /// <param name="x">The feature matrix with intercept term.</param>
    /// <param name="y">The target vector containing the class labels.</param>
    /// <param name="probabilities">The matrix of class probabilities for each sample.</param>
    /// <returns>The gradient matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the gradient of the log-likelihood with respect to the coefficients. The gradient indicates
    /// how the log-likelihood would change with small changes in the coefficients, providing the direction for updating
    /// the coefficients to increase the likelihood.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how to change the coefficients to improve predictions.
    /// 
    /// The gradient:
    /// - Shows the direction and amount to change each coefficient to make better predictions
    /// - Compares the predicted probabilities with the actual classes
    /// - Larger gradient values indicate coefficients that need more adjustment
    /// 
    /// It's like getting feedback on which knobs need the most adjustment to improve
    /// the model's performance.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeGradient(Matrix<T> x, Vector<T> y, Matrix<T> probabilities)
    {
        Matrix<T> yOneHot = CreateOneHotEncoding(y);
        return x.Transpose().Multiply(yOneHot.Subtract(probabilities));
    }

    /// <summary>
    /// Computes the Hessian matrix of the log-likelihood with respect to the coefficients.
    /// </summary>
    /// <param name="x">The feature matrix with intercept term.</param>
    /// <param name="probabilities">The matrix of class probabilities for each sample.</param>
    /// <returns>The Hessian matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the Hessian matrix of the log-likelihood with respect to the coefficients. The Hessian
    /// contains the second derivatives of the log-likelihood, providing information about the curvature of the
    /// log-likelihood surface. This is used in Newton's method to determine not just the direction but also the
    /// optimal step size for updating the coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how sensitive the model is to changes in each coefficient.
    /// 
    /// The Hessian:
    /// - Measures how quickly the gradient changes as the coefficients change
    /// - Helps determine the optimal step size for updating each coefficient
    /// - Accounts for interactions between different coefficients
    /// 
    /// It's like having a map of the terrain that helps you take steps of the right size
    /// in each direction, rather than always taking fixed-size steps.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeHessian(Matrix<T> x, Matrix<T> probabilities)
    {
        int n = x.Rows;
        int p = x.Columns;
        Matrix<T> hessian = new(p * _numClasses, p * _numClasses);

        for (int i = 0; i < n; i++)
        {
            Vector<T> xi = x.GetRow(i);
            Vector<T> probs = probabilities.GetRow(i);
            Matrix<T> diagP = Matrix<T>.CreateDiagonal(probs);
            Matrix<T> ppt = probs.OuterProduct(probs);
            Matrix<T> h = diagP.Subtract(ppt);
            Matrix<T> xxt = xi.OuterProduct(xi);
            Matrix<T> block = xxt.KroneckerProduct(h);
            hessian = hessian.Add(block);
        }

        return hessian.Negate();
    }

    /// <summary>
    /// Creates a one-hot encoding of the class labels.
    /// </summary>
    /// <param name="y">The target vector containing the class labels as integers.</param>
    /// <returns>A matrix where each row is a one-hot encoded vector for the corresponding class label.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a one-hot encoding of the class labels, which is a binary matrix representation where each row
    /// corresponds to a sample and each column corresponds to a class. For each sample, the element corresponding to its
    /// class is set to 1, and all other elements are set to 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts class labels into a special matrix format.
    /// 
    /// One-hot encoding:
    /// - Represents categorical data as a matrix of 0s and 1s
    /// - Each row represents one data point
    /// - Each column represents one possible class
    /// - A 1 in position (i,j) means the i-th data point belongs to the j-th class
    /// - All other positions contain 0s
    /// 
    /// For example, if there are 3 classes (0, 1, 2), the label "1" would be encoded as [0, 1, 0],
    /// meaning "not class 0, yes class 1, not class 2".
    /// </para>
    /// </remarks>
    private Matrix<T> CreateOneHotEncoding(Vector<T> y)
    {
        Matrix<T> oneHot = new Matrix<T>(y.Length, _numClasses);
        for (int i = 0; i < y.Length; i++)
        {
            int classIndex = Convert.ToInt32(NumOps.ToInt32(y[i]));
            oneHot[i, classIndex] = NumOps.One;
        }

        return oneHot;
    }

    /// <summary>
    /// Determines if the training has converged based on the magnitude of the coefficient updates.
    /// </summary>
    /// <param name="update">The matrix of coefficient updates from the current iteration.</param>
    /// <returns>True if the maximum absolute update is less than the tolerance, indicating convergence; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the training has converged by comparing the maximum absolute value of the coefficient updates
    /// to the specified tolerance. If the maximum update is smaller than the tolerance, the algorithm is considered to have
    /// converged, meaning that further iterations would not significantly improve the model.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the model has finished learning.
    /// 
    /// Convergence means:
    /// - The model is no longer making significant improvements
    /// - The coefficient updates are very small (below a threshold)
    /// - Further training is unlikely to yield better results
    /// 
    /// It's like knowing when to stop studying for a test - at some point, additional effort
    /// yields diminishing returns, and your time is better spent elsewhere.
    /// </para>
    /// </remarks>
    private bool HasConverged(Matrix<T> update)
    {
        T maxChange = update.Max(NumOps.Abs);
        return NumOps.LessThan(maxChange, NumOps.FromDouble(_options.Tolerance));
    }

    /// <summary>
    /// Predicts the class labels for new data points using the trained multinomial logistic regression model.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample to predict.</param>
    /// <returns>A vector containing the predicted class labels (as integers).</returns>
    /// <remarks>
    /// <para>
    /// This method predicts the class labels for new data points by computing the probabilities for each class and
    /// selecting the class with the highest probability for each sample. The class labels are returned as integers.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model makes predictions on new data.
    /// 
    /// The prediction process:
    /// 1. Calculate the probability of each class for each data point
    /// 2. For each data point, select the class with the highest probability
    /// 3. Return these predicted classes as the results
    /// 
    /// It's like a voting system where each feature casts a weighted vote for each class,
    /// and the class with the most votes wins.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> x)
    {
        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));
        Matrix<T> probabilities = ComputeProbabilities(xWithIntercept);

        return probabilities.RowWiseArgmax();
    }

    /// <summary>
    /// Predicts the probabilities of each class for new data points.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample to predict.</param>
    /// <returns>A matrix where each row corresponds to a sample and each column to the probability of a class.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the probabilities of each class for new data points using the trained model. The resulting
    /// matrix contains the probability of each class for each sample. These probabilities sum to 1 across the classes
    /// for each sample.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides the likelihood of each class for each data point.
    /// 
    /// Rather than just giving the final prediction, it provides:
    /// - The probability of each possible class
    /// - A measure of the model's confidence in each prediction
    /// - Values between 0 (impossible) and 1 (certain), with all classes summing to 1
    /// 
    /// This is useful when you need to know not just the predicted class but also
    /// how confident the model is in that prediction. For example, you might treat a
    /// prediction with 95% confidence differently than one with 51% confidence.
    /// </para>
    /// </remarks>
    public Matrix<T> PredictProbabilities(Matrix<T> x)
    {
        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));
        return ComputeProbabilities(xWithIntercept);
    }

    /// <summary>
    /// Serializes the multinomial logistic regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the entire multinomial logistic regression model, including its parameters and configuration,
    /// into a byte array that can be stored in a file or database, or transmitted over a network. The model can later be
    /// restored using the Deserialize method.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to a format that can be stored or shared.
    /// 
    /// Serialization:
    /// - Converts all the model's data into a sequence of bytes
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

        // Serialize MultinomialLogisticRegression specific data
        writer.Write(_numClasses);

        // Write whether _coefficients is null
        writer.Write(_coefficients != null);

        if (_coefficients != null)
        {
            writer.Write(_coefficients.Rows);
            writer.Write(_coefficients.Columns);
            for (int i = 0; i < _coefficients.Rows; i++)
            {
                for (int j = 0; j < _coefficients.Columns; j++)
                {
                    writer.Write(Convert.ToDouble(_coefficients[i, j]));
                }
            }
        }

        // Serialize options
        writer.Write(_options.MaxIterations);
        writer.Write(Convert.ToDouble(_options.Tolerance));

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the multinomial logistic regression model from a byte array.
    /// </summary>
    /// <param name="data">A byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method restores a multinomial logistic regression model from a serialized byte array, reconstructing its parameters
    /// and configuration. This allows a previously trained model to be loaded from storage or after being received over a network.
    /// </para>
    /// <para><b>For Beginners:</b> This method rebuilds the model from a saved format.
    /// 
    /// Deserialization:
    /// - Takes a sequence of bytes that represents a model
    /// - Reconstructs the original model with all its learned patterns
    /// - Allows you to use a previously trained model without retraining
    /// 
    /// Think of it like unpacking a model that was packed up for storage or shipping,
    /// so you can use it again exactly as it was before.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize MultinomialLogisticRegression specific data
        _numClasses = reader.ReadInt32();

        bool coefficientsExist = reader.ReadBoolean();
        if (coefficientsExist)
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            _coefficients = new Matrix<T>(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    _coefficients[i, j] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
        }
        else
        {
            _coefficients = null;
        }

        // Deserialize options
        _options.MaxIterations = reader.ReadInt32();
        _options.Tolerance = reader.ReadDouble();
    }

    /// <summary>
    /// Gets the type of regression model.
    /// </summary>
    /// <returns>The model type, in this case, MultinomialLogisticRegression.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an enumeration value indicating that this is a multinomial logistic regression model. This is used
    /// for type identification when working with different regression models in a unified manner.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply tells what kind of model this is.
    /// 
    /// It returns a label (MultinomialLogisticRegression) that:
    /// - Identifies this specific type of model
    /// - Helps other code handle the model appropriately
    /// - Is used for model identification and categorization
    /// 
    /// It's like a name tag that lets other parts of the program know what kind of model they're working with.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.MultinomialLogisticRegression;
    }

    /// <summary>
    /// Creates a new instance of the Multinomial Logistic Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Multinomial Logistic Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Multinomial Logistic Regression model, including its options,
    /// coefficients matrix, number of classes, and regularization settings. The new instance is completely independent 
    /// of the original, allowing modifications without affecting the original model.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect duplicate:
    /// - It copies all the configuration settings (like maximum iterations and tolerance)
    /// - It preserves the coefficient weights for all classes (the voting system for each category)
    /// - It maintains information about how many categories the model can predict
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var newModel = new MultinomialLogisticRegression<T>(_options, Regularization);

        // Copy the number of classes
        newModel._numClasses = _numClasses;

        // Deep copy the coefficients matrix if it exists
        if (_coefficients != null)
        {
            newModel._coefficients = _coefficients.Clone();
        }

        // Copy coefficients and intercept from base class
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }

        newModel.Intercept = Intercept;

        return newModel;
    }
}
