using AiDotNet.Autodiff;
using Newtonsoft.Json;

namespace AiDotNet.Regression;

/// <summary>
/// Base class for non-linear regression algorithms that provides common functionality for training and prediction.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements core functionality shared by different non-linear regression algorithms,
/// including kernel functions, regularization, and model serialization/deserialization.
/// </para>
/// <para>
/// Non-linear regression models can capture complex relationships in data that linear models cannot represent.
/// They typically use kernel functions to transform the input space into a higher-dimensional feature space
/// where the relationship becomes linear.
/// </para>
/// <para>
/// For Beginners:
/// Non-linear regression is used when your data doesn't follow a straight line pattern. These models can
/// capture curved or complex relationships between your input features and target values. Think of it like
/// having a flexible curve that can bend and shape itself to fit your data points, rather than just a
/// straight line.
/// </para>
/// </remarks>
public abstract class NonLinearRegressionBase<T> : INonLinearRegression<T>
{
    /// <summary>
    /// Gets the numeric operations provider for the specified type T.
    /// </summary>
    /// <value>
    /// An object that provides mathematical operations for the numeric type T.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// This property provides a way to perform math operations (like addition, multiplication, etc.)
    /// on the generic type T. It allows the algorithm to work with different numeric types
    /// (float, double, decimal) without changing the core logic.
    /// </para>
    /// </remarks>
    protected INumericOperations<T> NumOps { get; private set; }

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the configuration options for the non-linear regression model.
    /// </summary>
    /// <value>
    /// Contains settings like kernel type, regularization parameters, and optimization parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// These are the settings that control how the model behaves. They include choices like what type of
    /// mathematical function (kernel) to use and how to prevent the model from memorizing the training data
    /// instead of learning general patterns (regularization).
    /// </para>
    /// </remarks>
    protected NonLinearRegressionOptions Options { get; private set; }

    /// <summary>
    /// Gets the regularization method used to prevent overfitting.
    /// </summary>
    /// <value>
    /// An implementation of the IRegularization interface that determines how model complexity is penalized.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// Regularization helps prevent "overfitting," which is when a model learns the training data too well
    /// and performs poorly on new data. It's like adding a penalty for complexity, encouraging the model
    /// to find simpler solutions that generalize better to new examples.
    /// </para>
    /// </remarks>
    protected IRegularization<T, Matrix<T>, Vector<T>> Regularization { get; private set; }

    /// <summary>
    /// Gets or sets the support vectors used by the model.
    /// </summary>
    /// <value>
    /// A matrix where each row represents a support vector (a subset of the training examples that define the model).
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// Support vectors are the key training examples that define the model's decision boundary.
    /// Instead of using all training data points, many non-linear models only need to remember
    /// a subset of them (the "support vectors") to make predictions. This makes the model more
    /// efficient and often helps it generalize better to new data.
    /// </para>
    /// </remarks>
    protected Matrix<T> SupportVectors { get; set; }

    /// <summary>
    /// Gets or sets the alpha coefficients for each support vector.
    /// </summary>
    /// <value>
    /// A vector of coefficients that determine the influence of each support vector on predictions.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// Alpha coefficients determine how much influence each support vector has on the final prediction.
    /// Larger alpha values mean that support vector has a stronger effect on the prediction.
    /// </para>
    /// </remarks>
    protected Vector<T> Alphas { get; set; }

    /// <summary>
    /// Gets or sets the bias term (intercept) of the model.
    /// </summary>
    /// <value>
    /// A scalar value that represents the offset from the origin.
    /// </value>
    /// <remarks>
    /// <para>
    /// For Beginners:
    /// The bias term (sometimes called the intercept) is like a baseline value for predictions.
    /// It shifts the entire model up or down to better fit the data. Without a bias term,
    /// all predictions would have to pass through the origin (0,0).
    /// </para>
    /// </remarks>
    protected T B { get; set; }

    /// <summary>
    /// Gets the default loss function for this non-linear regression model.
    /// </summary>
    /// <value>
    /// The loss function used for gradient computation.
    /// </value>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    /// <value>
    /// An array of feature names. If not set, feature indices will be used as names.
    /// </value>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Initializes a new instance of the NonLinearRegressionBase class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the non-linear regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <param name="lossFunction">Loss function for gradient computation. If null, defaults to Mean Squared Error.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with default values and prepares it for training.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This constructor sets up the model with either the options you provide or default settings.
    /// It's like setting up a new tool before you start using it - you're configuring how it will work
    /// before you actually train it with data. The loss function determines how prediction errors
    /// are measured during training.
    /// </para>
    /// </remarks>
    protected NonLinearRegressionBase(NonLinearRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null, ILossFunction<T>? lossFunction = null)
    {
        Options = options ?? new NonLinearRegressionOptions();
        Regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        NumOps = MathHelper.GetNumericOperations<T>();
        SupportVectors = new Matrix<T>(0, 0);
        Alphas = new Vector<T>(0);
        B = NumOps.Zero;
        _defaultLossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
    }

    /// <summary>
    /// Trains the non-linear regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method performs the following steps:
    /// 1. Validates the input data
    /// 2. Initializes the model parameters
    /// 3. Optimizes the model parameters using the training data
    /// 4. Extracts the support vectors and their coefficients
    /// </para>
    /// <para>
    /// For Beginners:
    /// Training is the process where the model learns from your data. It looks at the examples you provide
    /// (input features and their corresponding target values) and adjusts its internal parameters to make
    /// predictions that match the target values as closely as possible. After training, the model will be
    /// ready to make predictions on new data.
    /// </para>
    /// </remarks>
    public virtual void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputs(x, y);
        InitializeModel(x, y);
        OptimizeModel(x, y);
        ExtractModelParameters();
    }

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the trained model to each input example and returns the predicted values.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, this method is used to make predictions on new data. It takes your input features
    /// and runs them through the trained model to estimate what the target values should be. This is the
    /// main purpose of building a regression model - to predict values for new examples.
    /// </para>
    /// </remarks>
    public virtual Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
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
    /// For Beginners:
    /// This method makes sure your data is in the correct format before training begins.
    /// It checks that you have the same number of target values as you have examples in your input data.
    /// If not, it will raise an error to let you know there's a mismatch.
    /// </para>
    /// </remarks>
    protected virtual void ValidateInputs(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("The number of rows in X must match the length of y.");
        }
    }

    /// <summary>
    /// Initializes the model parameters before optimization.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This method initializes the model parameters (Alphas and B) based on the input data.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Before the model can start learning, it needs to set up its internal values (parameters).
    /// This method creates those initial values, which will be adjusted during the training process.
    /// It's like setting up the starting position before beginning a journey.
    /// </para>
    /// </remarks>
    protected virtual void InitializeModel(Matrix<T> x, Vector<T> y)
    {
        // Initialize model parameters (e.g., Alphas, B) based on input data
        Alphas = new Vector<T>(x.Rows);
        B = NumOps.Zero;
    }

    /// <summary>
    /// Optimizes the model parameters using the training data.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to perform the actual optimization
    /// of the model parameters.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This is where the actual learning happens. Different algorithms will implement this method
    /// differently, but they all have the same goal: to adjust the model's parameters to make
    /// predictions that match the training data as closely as possible, while still generalizing
    /// well to new data.
    /// </para>
    /// </remarks>
    protected abstract void OptimizeModel(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Extracts the support vectors and their coefficients after optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method identifies the support vectors (training examples with non-zero alpha coefficients)
    /// and extracts them along with their corresponding alpha values.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, not all examples are equally important for making predictions.
    /// This method identifies which examples are most important (the "support vectors")
    /// and keeps only those for making future predictions. This makes the model more
    /// efficient and often improves its performance on new data.
    /// </para>
    /// </remarks>
    protected virtual void ExtractModelParameters()
    {
        // Extract support vectors and their corresponding alphas
        int[] supportVectorIndices = Enumerable.Range(0, Alphas.Length)
            .Where(i => NumOps.GreaterThan(NumOps.Abs(Alphas[i]), NumOps.FromDouble(1e-5)))
            .ToArray();

        // If all alphas are near-zero, keep the single largest-magnitude alpha as a fallback.
        // This avoids producing an empty model that cannot be exported or JIT-compiled.
        if (supportVectorIndices.Length == 0 && Alphas.Length > 0 && SupportVectors.Rows > 0)
        {
            int bestIndex = 0;
            T bestAbs = NumOps.Abs(Alphas[0]);

            for (int i = 1; i < Alphas.Length; i++)
            {
                T abs = NumOps.Abs(Alphas[i]);
                if (NumOps.GreaterThan(abs, bestAbs))
                {
                    bestAbs = abs;
                    bestIndex = i;
                }
            }

            supportVectorIndices = new[] { bestIndex };
        }

        int featureCount = SupportVectors.Columns;
        var oldSupportVectors = SupportVectors;
        SupportVectors = new Matrix<T>(supportVectorIndices.Length, featureCount);
        var newAlphas = new Vector<T>(supportVectorIndices.Length);

        for (int i = 0; i < supportVectorIndices.Length; i++)
        {
            int index = supportVectorIndices[i];
            for (int j = 0; j < featureCount; j++)
            {
                SupportVectors[i, j] = oldSupportVectors[index, j];
            }
            newAlphas[i] = Alphas[index];
        }

        Alphas = newAlphas;
    }

    /// <summary>
    /// Makes a prediction for a single input example.
    /// </summary>
    /// <param name="input">The input feature vector.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the prediction for a single input example by applying the kernel function
    /// to the input and each support vector, multiplying by the corresponding alpha coefficient,
    /// and adding the bias term.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method calculates the prediction for a single example. It uses the support vectors
    /// (important training examples) and their weights (alpha coefficients) to compute how similar
    /// the new example is to each support vector. These similarity scores are combined to produce
    /// the final prediction. The mathematical function used to measure similarity is called the
    /// "kernel function."
    /// </para>
    /// </remarks>
    protected virtual T PredictSingle(Vector<T> input)
    {
        T result = B;
        for (int i = 0; i < SupportVectors.Rows; i++)
        {
            Vector<T> supportVector = SupportVectors.GetRow(i);
            result = NumOps.Add(result, NumOps.Multiply(Alphas[i], KernelFunction(input, supportVector)));
        }

        return result;
    }

    /// <summary>
    /// Computes the kernel function between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel function value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the kernel function value between two vectors based on the kernel type
    /// specified in the options. The kernel function measures the similarity between two vectors
    /// in a potentially higher-dimensional space.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The kernel function is a mathematical way to measure how similar two examples are to each other.
    /// Different kernel functions capture different types of similarities. For example:
    /// - Linear kernel: Measures similarity based on the dot product (like a straight-line relationship)
    /// - RBF (Radial Basis Function) kernel: Measures similarity based on distance (examples close to each other are more similar)
    /// - Polynomial kernel: Can capture more complex curved relationships
    /// 
    /// The kernel function is what gives non-linear regression models their power to model complex patterns.
    /// </para>
    /// </remarks>
    protected T KernelFunction(Vector<T> x1, Vector<T> x2)
    {
        switch (Options.KernelType)
        {
            case KernelType.Linear:
                return x1.DotProduct(x2);

            case KernelType.RBF:
                // VECTORIZED: Use Engine operations for element-wise subtract and sum
                var diff = (Vector<T>)Engine.Subtract(x1, x2);
                T squaredDistance = diff.Transform(v => NumOps.Square(v)).Sum();
                return NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(-Options.Gamma), squaredDistance));

            case KernelType.Polynomial:
                T dot = x1.DotProduct(x2);
                return NumOps.Power(
                    NumOps.Add(NumOps.Multiply(NumOps.FromDouble(Options.Gamma), dot), NumOps.FromDouble(Options.Coef0)),
                    NumOps.FromDouble(Options.PolynomialDegree)
                );

            case KernelType.Sigmoid:
                T sigmoidDot = x1.DotProduct(x2);
                return MathHelper.Tanh(NumOps.Add(NumOps.Multiply(NumOps.FromDouble(Options.Gamma), sigmoidDot), NumOps.FromDouble(Options.Coef0))
                );

            case KernelType.Laplacian:
                // VECTORIZED: Use Engine operations for element-wise subtract and abs
                var diffLap = (Vector<T>)Engine.Subtract(x1, x2);
                var absDiff = (Vector<T>)Engine.Abs(diffLap);
                T l1Distance = absDiff.Sum();
                return NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(-Options.Gamma), l1Distance));

            default:
                throw new ArgumentOutOfRangeException(nameof(Options.KernelType), Options.KernelType, "Unsupported kernel type");
        }
    }

    /// <summary>
    /// Clips a value to be within the specified range.
    /// </summary>
    /// <param name="value">The value to clip.</param>
    /// <param name="low">The lower bound of the range.</param>
    /// <param name="high">The upper bound of the range.</param>
    /// <returns>The clipped value.</returns>
    /// <remarks>
    /// <para>
    /// This method ensures that a value is within the specified range by returning the lower bound
    /// if the value is less than the lower bound, the upper bound if the value is greater than the
    /// upper bound, or the value itself if it is within the range.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Clipping is a simple way to ensure a value stays within a certain range. It's like setting
    /// minimum and maximum limits. If the value is below the minimum, it's raised to the minimum.
    /// If it's above the maximum, it's lowered to the maximum. Otherwise, it stays as is.
    /// This is often used during optimization to keep parameters within valid bounds.
    /// </para>
    /// </remarks>
    protected T Clip(T value, T low, T high)
    {
        var max = NumOps.GreaterThan(value, low) ? value : low;
        return NumOps.LessThan(max, high) ? max : high;
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type and additional information
    /// such as the kernel type, kernel parameters, and the number of support vectors.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Model metadata is information about the model itself, rather than the predictions it makes.
    /// This includes details about how the model is configured (like what type of kernel it uses)
    /// and some statistics about its structure (like how many support vectors it has). This information
    /// can be useful for understanding the model's complexity and for debugging purposes.
    /// </para>
    /// </remarks>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            AdditionalInfo = new Dictionary<string, object>
            {
                ["KernelType"] = Options.KernelType,
                ["Gamma"] = Options.Gamma,
                ["Coef0"] = Options.Coef0,
                ["PolynomialDegree"] = Options.PolynomialDegree,
                ["SupportVectorsCount"] = SupportVectors.Rows
            }
        };

        return metadata;
    }

    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    /// <returns>The model type identifier.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to return the specific model type.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method simply returns an identifier that indicates what specific type of non-linear regression
    /// model this is (e.g., Support Vector Regression, Neural Network Regression, etc.). It's used
    /// internally by the library to keep track of different types of models.
    /// </para>
    /// </remarks>
    protected abstract ModelType GetModelType();

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters, including options, support vectors, alpha coefficients,
    /// bias term, and regularization options, to a byte array that can be stored or transmitted.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize options with type information for proper polymorphic deserialization
        var serializerSettings = new JsonSerializerSettings { TypeNameHandling = TypeNameHandling.All };
        var optionsJson = JsonConvert.SerializeObject(Options, serializerSettings);
        writer.Write(optionsJson);

        // Serialize support vectors
        writer.Write(SupportVectors.Rows);
        writer.Write(SupportVectors.Columns);
        for (int i = 0; i < SupportVectors.Rows; i++)
        {
            for (int j = 0; j < SupportVectors.Columns; j++)
            {
                writer.Write(Convert.ToDouble(SupportVectors[i, j]));
            }
        }

        // Serialize alphas
        writer.Write(Alphas.Length);
        foreach (var alpha in Alphas)
        {
            writer.Write(Convert.ToDouble(alpha));
        }

        // Serialize B
        writer.Write(Convert.ToDouble(B));

        // Serialize regularization options
        var regularizationOptionsJson = JsonConvert.SerializeObject(Regularization.GetOptions());
        writer.Write(regularizationOptionsJson);

        return ms.ToArray();
    }

    /// <summary>
    /// Creates a new instance of the derived model class.
    /// </summary>
    /// <returns>A new instance of the same model type.</returns>
    /// <remarks>
    /// <para>
    /// This abstract factory method must be implemented by derived classes to create a new
    /// instance of their specific type. It's used by Clone and DeepCopy to ensure that
    /// the correct derived type is instantiated.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates a new, empty instance of the specific model type.
    /// It's used during cloning and deep copying to ensure that the copy
    /// is of the same specific type as the original. This is more efficient
    /// than using reflection to create instances and gives derived classes
    /// explicit control over how new instances are created.
    /// </para>
    /// </remarks>
    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateInstance();

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array, including options,
    /// support vectors, alpha coefficients, bias term, and regularization options.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize options with type information for proper polymorphic deserialization
        var serializerSettings = new JsonSerializerSettings { TypeNameHandling = TypeNameHandling.All };
        var optionsJson = reader.ReadString();
        Options = JsonConvert.DeserializeObject<NonLinearRegressionOptions>(optionsJson, serializerSettings) ?? new NonLinearRegressionOptions();

        // Deserialize support vectors
        int svRows = reader.ReadInt32();
        int svCols = reader.ReadInt32();
        SupportVectors = new Matrix<T>(svRows, svCols);
        for (int i = 0; i < svRows; i++)
        {
            for (int j = 0; j < svCols; j++)
            {
                SupportVectors[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Deserialize alphas
        int alphaCount = reader.ReadInt32();
        Alphas = new Vector<T>(alphaCount);
        for (int i = 0; i < alphaCount; i++)
        {
            Alphas[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Deserialize B
        B = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize regularization options
        var regularizationOptionsJson = reader.ReadString();
        var regularizationOptions = JsonConvert.DeserializeObject<RegularizationOptions>(regularizationOptionsJson)
            ?? new RegularizationOptions();

        // Create regularization based on deserialized options
        Regularization = RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(regularizationOptions);

        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the model parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters (alpha coefficients and bias term).</returns>
    /// <remarks>
    /// <para>
    /// This method combines all model parameters into a single vector, with the bias term as the first element
    /// followed by all alpha coefficients. This representation is useful for optimization algorithms and
    /// for operations that need to treat all parameters uniformly.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method collects all the model's internal values (parameters) into a single list.
    /// Think of it like getting a complete list of ingredients and measurements for a recipe.
    /// This allows you to see all the parameters at once or pass them to other algorithms
    /// that work with the model's parameters as a group.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameters()
    {
        // Create a vector to hold all parameters (bias + alphas)
        var parameters = new Vector<T>(Alphas.Length + 1);

        // Set the bias term as the first parameter
        parameters[0] = B;

        // Copy all alpha coefficients
        for (int i = 0; i < Alphas.Length; i++)
        {
            parameters[i + 1] = Alphas[i];
        }

        return parameters;
    }

    /// <summary>
    /// Creates a new model with the specified parameters.
    /// </summary>
    /// <param name="parameters">A vector containing all model parameters (bias term followed by alpha coefficients).</param>
    /// <returns>A new model instance with the specified parameters.</returns>
    /// <exception cref="ArgumentException">Thrown when the parameters vector doesn't match the expected length.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a new model instance with the same structure as the current model but with different
    /// parameter values. The first element of the parameters vector is interpreted as the bias term, and the
    /// remaining elements are interpreted as alpha coefficients.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates a new model with specific parameter values you provide.
    /// It's like following a recipe but changing the amounts of certain ingredients.
    /// This is useful when you want to experiment with different parameter settings
    /// or when an optimization algorithm suggests better parameter values.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        // Verify that the parameters vector has the correct length
        if (parameters.Length != Alphas.Length + 1)
        {
            throw new ArgumentException($"Parameters vector length ({parameters.Length}) " +
                                       $"does not match expected length ({Alphas.Length + 1}).");
        }

        // Create a new instance of the model
        var clone = (NonLinearRegressionBase<T>)this.Clone();

        // Set the bias term
        clone.B = parameters[0];

        // Set the alpha coefficients
        for (int i = 0; i < Alphas.Length; i++)
        {
            clone.Alphas[i] = parameters[i + 1];
        }

        return clone;
    }

    /// <summary>
    /// Gets the indices of features that are actively used by the model.
    /// </summary>
    /// <returns>A collection of feature indices that have non-zero weight in the model.</returns>
    /// <remarks>
    /// <para>
    /// This method identifies which features have a significant impact on the model's predictions by analyzing
    /// the support vectors and their coefficients. A feature is considered active if it has a non-zero weight
    /// in at least one support vector with a non-zero alpha coefficient.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method tells you which features (input variables) your model is actually using
    /// to make predictions. Some models may effectively ignore certain features if they
    /// don't help with predictions. Knowing which features are actually being used can help
    /// you understand what information the model considers important and potentially simplify
    /// your model by removing unused features.
    /// </para>
    /// </remarks>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        // Create a set to store the active feature indices
        // This set will automatically remove duplicate indices and ensure that we only return unique values
        var activeIndices = new HashSet<int>(); activeIndices = new HashSet<int>();

        // Identify features that have non-zero weight in support vectors with non-zero alpha
        for (int i = 0; i < Alphas.Length; i++)
        {
            // Skip if the alpha coefficient is effectively zero
            if (NumOps.LessThan(NumOps.Abs(Alphas[i]), NumOps.FromDouble(1e-5)))
                continue;

            // Check each feature in this support vector
            for (int j = 0; j < SupportVectors.Columns; j++)
            {
                // If the feature has a non-zero value, consider it active
                if (!NumOps.LessThan(NumOps.Abs(SupportVectors[i, j]), NumOps.FromDouble(1e-5)))
                {
                    activeIndices.Add(j);
                }
            }
        }

        return activeIndices;
    }

    /// <summary>
    /// Determines whether a specific feature is used by the model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used by the model; otherwise, false.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when featureIndex is negative or greater than the number of features.</exception>
    /// <remarks>
    /// <para>
    /// This method checks whether a specific feature has a significant impact on the model's predictions
    /// by determining if it has a non-zero weight in at least one support vector with a non-zero alpha coefficient.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method checks if a specific feature (input variable) is actually being used
    /// by your model to make predictions. It's like checking if a particular ingredient
    /// in a recipe is actually affecting the final dish or if it could be left out without
    /// changing the result. This can help you understand what information your model
    /// considers important and potentially simplify your model by removing unused features.
    /// </para>
    /// </remarks>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        // Validate feature index
        if (featureIndex < 0 || featureIndex >= SupportVectors.Columns)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex),
                $"Feature index must be between 0 and {SupportVectors.Columns - 1}.");
        }

        // Check if the feature has a non-zero value in any support vector with non-zero alpha
        for (int i = 0; i < Alphas.Length; i++)
        {
            // Skip if the alpha coefficient is effectively zero
            if (NumOps.LessThan(NumOps.Abs(Alphas[i]), NumOps.FromDouble(1e-5)))
                continue;

            // Check if this feature has a non-zero value in this support vector
            if (!NumOps.LessThan(NumOps.Abs(SupportVectors[i, featureIndex]), NumOps.FromDouble(1e-5)))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    /// <param name="parameters">A vector containing the model parameters.</param>
    public virtual void SetParameters(Vector<T> parameters)
    {
        int expectedParamCount = Alphas.Length + 1; // Alphas.Length + 1 (for Bias term)
        if (parameters.Length != expectedParamCount)
        {
            throw new ArgumentException($"Expected {expectedParamCount} parameters, but got {parameters.Length}", nameof(parameters));
        }

        for (int i = 0; i < Alphas.Length; i++)
        {
            Alphas[i] = parameters[i];
        }
        B = parameters[Alphas.Length];
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to activate.</param>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        var activeSet = new HashSet<int>(featureIndices);

        for (int i = 0; i < SupportVectors.Rows; i++)
        {
            for (int j = 0; j < SupportVectors.Columns; j++)
            {
                if (!activeSet.Contains(j))
                {
                    SupportVectors[i, j] = NumOps.Zero;
                }
            }
        }
    }

    /// <summary>
    /// Gets the feature importance scores as a dictionary.
    /// </summary>
    /// <returns>A dictionary mapping feature names to their importance scores.</returns>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();
        var importance = new T[SupportVectors.Columns];

        for (int j = 0; j < SupportVectors.Columns; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < Alphas.Length; i++)
            {
                T weighted = NumOps.Multiply(NumOps.Abs(Alphas[i]), NumOps.Abs(SupportVectors[i, j]));
                sum = NumOps.Add(sum, weighted);
            }
            importance[j] = sum;
        }

        for (int i = 0; i < importance.Length; i++)
        {
            string featureName = FeatureNames != null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[featureName] = importance[i];
        }

        return result;
    }

    /// <summary>
    /// Creates a deep copy of the model.
    /// </summary>
    /// <returns>A new model instance that is a deep copy of the current model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a completely independent copy of the model, including all parameters,
    /// support vectors, and configuration options. Modifications to the returned model will not
    /// affect the original model, and vice versa.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates a complete, independent copy of your model.
    /// It's like making a photocopy of a document - the copy looks exactly
    /// the same but is a separate object that can be modified without affecting
    /// the original. This is useful when you want to experiment with changes to
    /// a model without risking the original or when you need multiple independent
    /// instances of the same model (e.g., for ensemble learning).
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        // Create a new instance through cloning
        var clone = (NonLinearRegressionBase<T>)this.Clone();

        // Perform deep copy of all mutable fields
        clone.SupportVectors = SupportVectors.Clone();
        clone.Alphas = Alphas.Clone();
        clone.B = B; // Value types are copied by value
        clone.Options = JsonConvert.DeserializeObject<NonLinearRegressionOptions>(
            JsonConvert.SerializeObject(Options)) ?? new NonLinearRegressionOptions();

        // Create a new regularization instance with the same options
        var regularizationOptions = Regularization.GetOptions();
        clone.Regularization = RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(regularizationOptions);

        return clone;
    }

    /// <summary>
    /// Creates a shallow copy of the model.
    /// </summary>
    /// <returns>A new model instance that is a shallow copy of the current model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the model that shares references to the same internal data
    /// structures as the original model. This is primarily used internally by other methods that need
    /// to create modified copies of the model.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This method creates a lightweight copy of your model.
    /// Unlike DeepCopy, which creates completely independent copies of everything,
    /// this method creates a new model object but may share some internal data
    /// with the original. This makes it faster to create copies, but changes to one
    /// copy might affect others in some cases. This is primarily used internally
    /// by other methods rather than directly by users.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        // Create a new instance using the factory method
        var clone = (NonLinearRegressionBase<T>)CreateInstance();

        // Copy the model parameters
        clone.SupportVectors = SupportVectors;  // Shallow copy
        clone.Alphas = Alphas;                 // Shallow copy
        clone.B = B;                          // Value types are copied by value
        clone.Options = Options;              // Shallow copy
        clone.Regularization = Regularization; // Shallow copy

        return clone;
    }

    public virtual int ParameterCount
    {
        get { return Alphas.Length + 1; } // Alphas + bias term
    }

    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));

        try
        {
            var data = Serialize();
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllBytes(filePath, data);
        }
        catch (IOException ex) { throw new InvalidOperationException($"Failed to save model to '{filePath}': {ex.Message}", ex); }
        catch (UnauthorizedAccessException ex) { throw new InvalidOperationException($"Access denied when saving model to '{filePath}': {ex.Message}", ex); }
        catch (System.Security.SecurityException ex) { throw new InvalidOperationException($"Security error when saving model to '{filePath}': {ex.Message}", ex); }
    }

    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));

        try
        {
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        catch (FileNotFoundException ex) { throw new FileNotFoundException($"The specified model file does not exist: {filePath}", filePath, ex); }
        catch (IOException ex) { throw new InvalidOperationException($"File I/O error while loading model from '{filePath}': {ex.Message}", ex); }
        catch (UnauthorizedAccessException ex) { throw new InvalidOperationException($"Access denied when loading model from '{filePath}': {ex.Message}", ex); }
        catch (System.Security.SecurityException ex) { throw new InvalidOperationException($"Security error when loading model from '{filePath}': {ex.Message}", ex); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to deserialize model from file '{filePath}'. The file may be corrupted or incompatible: {ex.Message}", ex); }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// For non-linear regression models, the default loss function is Mean Squared Error (MSE).
    /// This can be customized by passing a different loss function to the constructor.
    /// </para>
    /// </remarks>
    public virtual ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Non-linear regression models use kernel functions and support vectors, making gradient
    /// computation more complex than linear regression. This implementation uses numerical
    /// differentiation to compute gradients with respect to the support vector weights (alphas)
    /// and bias term.
    /// </para>
    /// <para><b>For Beginners:</b> This method computes how to adjust the model to reduce errors.
    ///
    /// Non-linear models are more complex than linear ones - they use kernel functions to capture
    /// curved relationships. Computing gradients requires:
    /// 1. Making predictions with the current model
    /// 2. Measuring the errors
    /// 3. Computing how each support vector weight should change
    ///
    /// This uses numerical differentiation, which approximates gradients by making small changes
    /// to parameters and observing the effect on the loss.
    /// </para>
    /// </remarks>
    public virtual Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? DefaultLossFunction;

        // Make predictions
        var predictions = Predict(input);

        // Compute prediction errors
        var errors = predictions.Subtract(target);

        // For kernel-based models, compute gradients using numerical differentiation
        // This is a simplified implementation - specific algorithms may override with analytical gradients
        var epsilon = NumOps.FromDouble(1e-7);
        var gradients = new Vector<T>(ParameterCount);

        // Compute gradient for each alpha (support vector weight)
        // Use try-finally to ensure state is restored even if exceptions occur
        for (int i = 0; i < Alphas.Length; i++)
        {
            var originalAlpha = Alphas[i];
            try
            {
                // Forward difference: f(x + h) - f(x) / h
                Alphas[i] = NumOps.Add(originalAlpha, epsilon);
                var predPlus = Predict(input);
                var lossPlus = loss.CalculateLoss(predPlus, target);

                var lossCurrent = loss.CalculateLoss(predictions, target);

                gradients[i] = NumOps.Divide(NumOps.Subtract(lossPlus, lossCurrent), epsilon);
            }
            finally
            {
                // Always restore original state, even if exception occurs
                Alphas[i] = originalAlpha;
            }
        }

        // Gradient for bias term
        // Use try-finally to ensure state is restored even if exceptions occur
        var originalB = B;
        try
        {
            B = NumOps.Add(originalB, epsilon);
            var predPlusB = Predict(input);
            var lossPlusB = loss.CalculateLoss(predPlusB, target);

            var lossCurrentB = loss.CalculateLoss(predictions, target);

            gradients[Alphas.Length] = NumOps.Divide(NumOps.Subtract(lossPlusB, lossCurrentB), epsilon);
        }
        finally
        {
            // Always restore original state, even if exception occurs
            B = originalB;
        }

        return gradients;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Updates the support vector weights (alphas) and bias term using gradient descent.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the computed gradients to improve the model.
    ///
    /// It updates:
    /// - Support vector weights (alphas): Control how much each support vector influences predictions
    /// - Bias term (B): The baseline prediction value
    ///
    /// The update follows gradient descent: new_value = old_value - learning_rate * gradient
    /// </para>
    /// </remarks>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (gradients.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} gradients, but got {gradients.Length}", nameof(gradients));
        }

        // Get current parameters
        var currentParams = GetParameters();

        // Apply gradient descent: params = params - learningRate * gradients
        var newParams = new Vector<T>(currentParams.Length);
        for (int i = 0; i < currentParams.Length; i++)
        {
            newParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        // Use SetParameters to update all model state
        SetParameters(newParams);
    }

    /// <summary>
    /// Saves the model's current state to a stream.
    /// </summary>
    public virtual void SaveState(Stream stream)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite) throw new ArgumentException("Stream must be writable.", nameof(stream));
        var data = Serialize();
        stream.Write(data, 0, data.Length);
        stream.Flush();
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    public virtual void LoadState(Stream stream)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanRead) throw new ArgumentException("Stream must be readable.", nameof(stream));
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        var data = ms.ToArray();
        if (data.Length == 0) throw new InvalidOperationException("Stream contains no data.");
        Deserialize(data);
    }

    #region IJitCompilable Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Non-linear regression models support JIT compilation for all kernel types:
    /// - Linear kernel: Fully supported (dot product)
    /// - RBF kernel: Fully supported (Gaussian similarity)
    /// - Sigmoid kernel: Fully supported (tanh-based similarity)
    /// - Polynomial kernel: Fully supported (power operation)
    /// - Laplacian kernel: Fully supported (L1 norm using sqrt(x^2) approximation)
    /// </para>
    /// <para><b>For Beginners:</b> JIT (Just-In-Time) compilation can speed up kernel-based models.
    ///
    /// Non-linear models use kernel functions to capture complex patterns. JIT compilation
    /// optimizes these computations for faster predictions. All kernel types are supported:
    /// - Linear kernels (simple dot products)
    /// - RBF kernels (Gaussian similarity based on distance)
    /// - Sigmoid kernels (tanh-based similarity)
    /// - Polynomial kernels (captures polynomial relationships)
    /// - Laplacian kernels (L1 distance-based similarity)
    ///
    /// For large models with many support vectors, JIT can provide 3-5x speedup.
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation
    {
        get
        {
            // Check if we have a trained model
            if (SupportVectors == null || SupportVectors.Rows == 0 || Alphas == null || Alphas.Length == 0)
                return false;

            // Check if kernel type is supported
            return Options.KernelType == KernelType.Linear ||
                   Options.KernelType == KernelType.RBF ||
                   Options.KernelType == KernelType.Sigmoid ||
                   Options.KernelType == KernelType.Polynomial ||
                   Options.KernelType == KernelType.Laplacian;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Exports the non-linear regression model as a computation graph.
    /// The graph represents: output = B + sum(alpha[i] * kernel(input, supportVector[i]))
    /// </para>
    /// <para><b>For Beginners:</b> This converts the kernel-based model to a computation graph.
    ///
    /// The computation graph represents:
    /// 1. For each support vector:
    ///    - Compute kernel similarity between input and support vector
    ///    - Multiply by alpha coefficient (weight)
    /// 2. Sum all weighted kernel values
    /// 3. Add bias term (B)
    ///
    /// Kernel functions measure similarity:
    /// - Linear: Simple dot product (like correlation)
    /// - RBF: Gaussian distance (close points are similar)
    /// - Sigmoid: Tanh-based similarity
    ///
    /// The JIT compiler optimizes this complex computation into fast native code.
    /// </para>
    /// </remarks>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        // Validation
        if (SupportVectors == null || SupportVectors.Rows == 0)
        {
            throw new InvalidOperationException("Cannot export computation graph: Model has not been trained yet.");
        }

        if (!SupportsJitCompilation)
        {
            throw new NotSupportedException($"JIT compilation is not supported for kernel type: {Options.KernelType}");
        }

        // Create input node (placeholder for input features)
        // Shape: [1, feature_count] (single example)
        var featureCount = SupportVectors.Columns;
        var inputShape = new int[] { 1, featureCount };
        var inputTensor = new Tensor<T>(inputShape);
        var inputNode = new ComputationNode<T>(inputTensor);
        inputNodes.Add(inputNode);

        // Accumulator for summing all kernel results
        ComputationNode<T>? sumNode = null;

        // Process each support vector
        for (int i = 0; i < SupportVectors.Rows; i++)
        {
            // Create support vector node
            var svShape = new int[] { 1, featureCount };
            var svData = new T[featureCount];
            for (int j = 0; j < featureCount; j++)
            {
                svData[j] = SupportVectors[i, j];
            }
            var svTensor = new Tensor<T>(svShape, new Vector<T>(svData));
            var svNode = new ComputationNode<T>(svTensor);

            // Compute kernel value based on kernel type
            ComputationNode<T> kernelNode = Options.KernelType switch
            {
                KernelType.Linear => ComputeLinearKernel(inputNode, svNode),
                KernelType.RBF => ComputeRBFKernel(inputNode, svNode),
                KernelType.Sigmoid => ComputeSigmoidKernel(inputNode, svNode),
                KernelType.Polynomial => ComputePolynomialKernel(inputNode, svNode),
                KernelType.Laplacian => ComputeLaplacianKernel(inputNode, svNode),
                _ => throw new NotSupportedException($"Kernel type {Options.KernelType} is not supported for JIT compilation")
            };

            // Multiply by alpha coefficient
            var alphaTensor = CreateFilledTensorLike(kernelNode, Alphas[i]);
            var alphaNode = TensorOperations<T>.Constant(alphaTensor, $"alpha_{i}");
            var weightedNode = TensorOperations<T>.ElementwiseMultiply(kernelNode, alphaNode);

            // Add to accumulator
            if (sumNode == null)
            {
                sumNode = weightedNode;
            }
            else
            {
                sumNode = TensorOperations<T>.Add(sumNode, weightedNode);
            }
        }

        // Add bias term
        var biasTensor = CreateFilledTensorLike(sumNode!, B);
        var biasNode = TensorOperations<T>.Constant(biasTensor, "bias");
        var outputNode = TensorOperations<T>.Add(sumNode!, biasNode);

        return outputNode;
    }

    /// <summary>
    /// Computes linear kernel: x1 · x2 (dot product).
    /// </summary>
    private ComputationNode<T> ComputeLinearKernel(ComputationNode<T> x1, ComputationNode<T> x2)
    {
        // Element-wise multiply
        var product = TensorOperations<T>.ElementwiseMultiply(x1, x2);

        // Sum all elements to get the dot product (scalar)
        return TensorOperations<T>.Sum(product);
    }

    /// <summary>
    /// Computes RBF kernel: exp(-gamma * ||x1 - x2||^2).
    /// </summary>
    private ComputationNode<T> ComputeRBFKernel(ComputationNode<T> x1, ComputationNode<T> x2)
    {
        // Compute difference: x1 - x2
        var diff = TensorOperations<T>.Subtract(x1, x2);

        // Square: (x1 - x2)^2
        var squared = TensorOperations<T>.ElementwiseMultiply(diff, diff);

        // Sum squared differences to get ||x1 - x2||^2 (scalar)
        var sumSquared = TensorOperations<T>.Sum(squared);

        // Multiply by -gamma
        var gammaTensor = CreateFilledTensorLike(sumSquared, NumOps.FromDouble(-Options.Gamma));
        var gammaNode = TensorOperations<T>.Constant(gammaTensor, "gamma");
        var scaled = TensorOperations<T>.ElementwiseMultiply(sumSquared, gammaNode);

        // Exp(-gamma * ||x1 - x2||^2)
        var result = TensorOperations<T>.Exp(scaled);

        return result;
    }

    /// <summary>
    /// Computes Sigmoid kernel: tanh(gamma * (x1 · x2) + coef0).
    /// </summary>
    private ComputationNode<T> ComputeSigmoidKernel(ComputationNode<T> x1, ComputationNode<T> x2)
    {
        // Dot product: x1 · x2 = sum(x1 * x2)
        var product = TensorOperations<T>.ElementwiseMultiply(x1, x2);
        var dotProduct = TensorOperations<T>.Sum(product);

        // Multiply by gamma
        var gammaTensor = CreateFilledTensorLike(dotProduct, NumOps.FromDouble(Options.Gamma));
        var gammaNode = TensorOperations<T>.Constant(gammaTensor, "gamma");
        var scaled = TensorOperations<T>.ElementwiseMultiply(dotProduct, gammaNode);

        // Add coef0
        var coef0Tensor = CreateFilledTensorLike(scaled, NumOps.FromDouble(Options.Coef0));
        var coef0Node = TensorOperations<T>.Constant(coef0Tensor, "coef0");
        var sum = TensorOperations<T>.Add(scaled, coef0Node);

        // Tanh
        var result = TensorOperations<T>.Tanh(sum);

        return result;
    }

    /// <summary>
    /// Computes Polynomial kernel: (gamma * (x1 · x2) + coef0) ^ degree.
    /// </summary>
    private ComputationNode<T> ComputePolynomialKernel(ComputationNode<T> x1, ComputationNode<T> x2)
    {
        // Dot product: x1 · x2 = sum(x1 * x2)
        var product = TensorOperations<T>.ElementwiseMultiply(x1, x2);
        var dotProduct = TensorOperations<T>.Sum(product);

        // Multiply by gamma
        var gammaTensor = CreateFilledTensorLike(dotProduct, NumOps.FromDouble(Options.Gamma));
        var gammaNode = TensorOperations<T>.Constant(gammaTensor, "gamma");
        var scaled = TensorOperations<T>.ElementwiseMultiply(dotProduct, gammaNode);

        // Add coef0
        var coef0Tensor = CreateFilledTensorLike(scaled, NumOps.FromDouble(Options.Coef0));
        var coef0Node = TensorOperations<T>.Constant(coef0Tensor, "coef0");
        var sum = TensorOperations<T>.Add(scaled, coef0Node);

        // Power(sum, degree)
        var result = TensorOperations<T>.Power(sum, Options.PolynomialDegree);

        return result;
    }

    /// <summary>
    /// Computes Laplacian kernel: exp(-gamma * |x1 - x2|_1).
    /// </summary>
    private ComputationNode<T> ComputeLaplacianKernel(ComputationNode<T> x1, ComputationNode<T> x2)
    {
        // Compute difference: x1 - x2
        var diff = TensorOperations<T>.Subtract(x1, x2);

        // Compute |x1 - x2| using sqrt((x1-x2)^2) as approximation of abs
        // Note: This works for element-wise absolute value
        var squared = TensorOperations<T>.ElementwiseMultiply(diff, diff);
        var absDiff = TensorOperations<T>.Sqrt(squared);

        // Sum absolute differences to get L1 norm (|x1 - x2|_1)
        var l1Norm = TensorOperations<T>.Sum(absDiff);

        // Multiply by -gamma
        var gammaTensor = CreateFilledTensorLike(l1Norm, NumOps.FromDouble(-Options.Gamma));
        var gammaNode = TensorOperations<T>.Constant(gammaTensor, "gamma");
        var scaled = TensorOperations<T>.ElementwiseMultiply(l1Norm, gammaNode);

        // Exp(-gamma * |x1 - x2|_1)
        var result = TensorOperations<T>.Exp(scaled);

        return result;
    }

    #endregion

    private static Tensor<T> CreateFilledTensorLike(ComputationNode<T> referenceNode, T value)
    {
        var tensor = new Tensor<T>((int[])referenceNode.Value.Shape.Clone());
        tensor.Fill(value);
        return tensor;
    }
}
