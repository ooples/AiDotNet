namespace AiDotNet.Regression;

/// <summary>
/// Implements stepwise regression, which automatically selects the most relevant features for the model.
/// This approach builds a model by adding or removing features based on their statistical significance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Stepwise regression helps solve the feature selection problem by iteratively building a model, either by:
/// - Forward selection: Starting with no features and adding the most significant ones
/// - Backward elimination: Starting with all features and removing the least significant ones
/// 
/// At each step, the algorithm evaluates the impact of adding or removing features based on a fitness metric
/// such as adjusted R-squared, AIC, BIC, or other statistical criteria.
/// </para>
/// <para><b>For Beginners:</b> Stepwise regression is like a smart shopping assistant that helps you 
/// pick only the most useful ingredients for a recipe.
/// 
/// Think of it like this:
/// - You have many potential ingredients (features) that might affect the outcome
/// - Instead of using all ingredients, which could make the recipe complicated or less tasty
/// - Stepwise regression tests each ingredient to see how much it improves the recipe
/// - It keeps only the ingredients that make a significant difference to the final result
/// 
/// For example, when predicting house prices, you might have data on square footage, 
/// number of bedrooms, location, age, etc. Stepwise regression would determine which of these 
/// features are most important for accurate predictions and discard the rest.
/// </para>
/// </remarks>
public class StepwiseRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the stepwise regression model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control the behavior of the stepwise regression algorithm, including the method used
    /// (forward selection or backward elimination), the maximum and minimum number of features to include,
    /// and the minimum improvement required to add or remove a feature.
    /// </para>
    /// <para><b>For Beginners:</b> These settings control how the feature selection process works:
    /// 
    /// - Method: Whether to start with no features and add them (forward) or start with all features and remove them (backward)
    /// - MaxFeatures: The maximum number of ingredients (features) to include in your recipe (model)
    /// - MinFeatures: The minimum number of ingredients you want to keep
    /// - MinImprovement: How much better the recipe needs to get to justify adding/removing an ingredient
    /// 
    /// These options help you balance between a model that's too simple (underfitting) and one that's 
    /// too complex (overfitting).
    /// </para>
    /// </remarks>
    private readonly StepwiseRegressionOptions<T> _options;

    /// <summary>
    /// The calculator used to evaluate the fitness or quality of models during feature selection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This component calculates a score for each potential model during the stepwise process.
    /// Common fitness metrics include adjusted R-squared, AIC (Akaike Information Criterion),
    /// BIC (Bayesian Information Criterion), or cross-validation error.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the judge in a cooking competition.
    /// 
    /// The fitness calculator:
    /// - Tastes each version of the recipe (evaluates each model)
    /// - Gives it a score based on how good it is
    /// - Helps decide which ingredients to keep or remove
    /// 
    /// Different judges might look for different qualities (like taste vs. nutrition vs. presentation),
    /// just as different fitness calculators might prioritize accuracy, simplicity, or generalization.
    /// </para>
    /// </remarks>
    private readonly IFitnessCalculator<T, Matrix<T>, Vector<T>> _fitnessCalculator;

    /// <summary>
    /// The list of feature indices that have been selected for the final model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list contains the indices of the features from the original dataset that were selected
    /// during the stepwise process. These are the features that will be used in the final regression model.
    /// </para>
    /// <para><b>For Beginners:</b> This is your final shopping list of ingredients.
    /// 
    /// After trying different combinations:
    /// - This list contains only the ingredients that actually improved your recipe
    /// - The numbers in the list refer to the positions of these ingredients in your original list
    /// - These selected ingredients will be used to cook your final dish (make predictions)
    /// 
    /// For example, if you started with 10 possible features and the selected list contains [0, 3, 7],
    /// it means only the 1st, 4th, and 8th features were important enough to keep.
    /// </para>
    /// </remarks>
    private List<int> _selectedFeatures;

    /// <summary>
    /// The evaluator used to assess the performance of models during the feature selection process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This component evaluates various metrics about each potential model, such as prediction error,
    /// R-squared, and other statistics that help determine the quality of the model.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the measurement tools in your kitchen.
    /// 
    /// The model evaluator:
    /// - Measures different aspects of your recipe's performance
    /// - Calculates things like how close your predictions are to the actual values
    /// - Provides data that the fitness calculator uses to score each model
    /// 
    /// Think of it as the thermometer, scale, and timer that help you objectively assess 
    /// how well your recipe turned out.
    /// </para>
    /// </remarks>
    private readonly IModelEvaluator<T, Matrix<T>, Vector<T>> _modelEvaluator;

    /// <summary>
    /// Creates a new stepwise regression model.
    /// </summary>
    /// <param name="options">
    /// Optional configuration settings for the stepwise regression model. These settings control aspects like:
    /// - The stepwise method (forward selection or backward elimination)
    /// - The maximum and minimum number of features to include
    /// - The minimum improvement required to add or remove a feature
    /// If not provided, default options will be used.
    /// </param>
    /// <param name="predictionOptions">
    /// Optional settings for prediction statistics calculation.
    /// </param>
    /// <param name="fitnessCalculator">
    /// Optional calculator for evaluating model fitness during feature selection.
    /// If not provided, adjusted R-squared will be used as the fitness metric.
    /// </param>
    /// <param name="regularization">
    /// Optional regularization method to prevent overfitting. 
    /// If not provided, no regularization will be applied.
    /// </param>
    /// <param name="modelEvaluator">
    /// Optional evaluator for assessing model performance.
    /// If not provided, the default model evaluator will be used.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new stepwise regression model with the specified configuration options,
    /// fitness calculator, regularization method, and model evaluator. If these components are not provided,
    /// default implementations are used.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up your feature selection process.
    /// 
    /// Think of it like preparing for a cooking competition:
    /// - You decide your strategy (forward or backward selection)
    /// - You set limits on how many ingredients to use
    /// - You choose how you'll judge which ingredients to keep
    /// - You set up safety measures to prevent over-complicating the recipe
    /// 
    /// After setting up with these options, the model will be ready to train
    /// and discover which features are most important for your predictions.
    /// </para>
    /// </remarks>
    public StepwiseRegression(StepwiseRegressionOptions<T>? options = null,
        PredictionStatsOptions? predictionOptions = null,
        IFitnessCalculator<T, Matrix<T>, Vector<T>>? fitnessCalculator = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        IModelEvaluator<T, Matrix<T>, Vector<T>>? modelEvaluator = null)
        : base(options, regularization)
    {
        _options = options ?? new StepwiseRegressionOptions<T>();
        _fitnessCalculator = fitnessCalculator ?? new AdjustedRSquaredFitnessCalculator<T, Matrix<T>, Vector<T>>();
        _selectedFeatures = new List<int>();
        _modelEvaluator = modelEvaluator ?? new DefaultModelEvaluator<T, Matrix<T>, Vector<T>>();
    }

    /// <summary>
    /// Trains the stepwise regression model using the provided input features and target values.
    /// </summary>
    /// <param name="x">
    /// The input feature matrix, where rows represent observations and columns represent features.
    /// </param>
    /// <param name="y">
    /// The target values vector, containing the actual output values that the model should learn to predict.
    /// </param>
    /// <exception cref="NotSupportedException">
    /// Thrown when an unsupported stepwise method is specified in the options.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method implements the stepwise regression algorithm for feature selection. It:
    /// 1. Validates the input data
    /// 2. Performs either forward selection or backward elimination based on the specified method
    /// 3. Trains a final multiple regression model using only the selected features
    /// 
    /// The result is a model that uses a subset of the original features, potentially improving
    /// both interpretability and predictive performance.
    /// </para>
    /// <para><b>For Beginners:</b> This method discovers which features are most important and builds your model.
    /// 
    /// The training process works like this:
    /// 
    /// 1. If using forward selection:
    ///    - Start with an empty recipe (no features)
    ///    - Try adding each available ingredient, one at a time
    ///    - Keep the ingredient that improves your recipe the most
    ///    - Repeat until adding more ingredients doesn't help much
    /// 
    /// 2. If using backward elimination:
    ///    - Start with all ingredients in your recipe (all features)
    ///    - Try removing each ingredient, one at a time
    ///    - Remove the ingredient that hurts your recipe the least
    ///    - Repeat until removing more ingredients would harm the recipe too much
    /// 
    /// 3. Finally, create a model using only the best ingredients (selected features)
    /// 
    /// This process helps you create a simpler, more efficient model that focuses only on
    /// the most important factors affecting your predictions.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);

        if (_options.Method == StepwiseMethod.Forward)
        {
            ForwardSelection(x, y);
        }
        else if (_options.Method == StepwiseMethod.Backward)
        {
            BackwardElimination(x, y);
        }
        else
        {
            throw new NotSupportedException("Unsupported stepwise method.");
        }

        // Train the final model using selected features
        Matrix<T> selectedX = x.GetColumns(_selectedFeatures);
        var finalRegression = new MultipleRegression<T>(Options, Regularization);
        finalRegression.Train(selectedX, y);

        Coefficients = finalRegression.Coefficients;
        Intercept = finalRegression.Intercept;
    }

    /// <summary>
    /// Makes predictions using only the selected features from the input matrix.
    /// </summary>
    /// <param name="input">The input feature matrix to make predictions on.</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method filters the input matrix to only include the selected features
    /// before making predictions. This is necessary because stepwise regression
    /// selects a subset of features during training.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_selectedFeatures.Count == 0 || Coefficients.Length == 0)
        {
            return new Vector<T>(input.Rows);
        }

        // Filter input to only use selected features
        Matrix<T> filteredInput = input.GetColumns(_selectedFeatures);
        return base.Predict(filteredInput);
    }

    /// <summary>
    /// Performs forward selection of features.
    /// </summary>
    /// <param name="x">The input feature matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This method implements the forward selection algorithm, which:
    /// 1. Starts with no features selected
    /// 2. Iteratively adds the feature that most improves the model's fitness
    /// 3. Stops when adding more features doesn't provide sufficient improvement
    ///    or when the maximum number of features is reached
    /// </para>
    /// <para><b>For Beginners:</b> This method builds your model by starting simple and adding complexity.
    /// 
    /// Think of it like building a team:
    /// - You start with an empty team
    /// - You try out each available player and pick the one who helps the team most
    /// - You keep adding players one by one, always choosing the best available
    /// - You stop when adding more players doesn't improve the team, or when you reach your team size limit
    /// 
    /// This approach ensures you only add features that genuinely improve your model's performance.
    /// </para>
    /// </remarks>
    private void ForwardSelection(Matrix<T> x, Vector<T> y)
    {
        List<int> remainingFeatures = [.. Enumerable.Range(0, x.Columns)];
        _selectedFeatures.Clear();

        while (_selectedFeatures.Count < Math.Min(_options.MaxFeatures, x.Columns))
        {
            var (bestFeature, bestScore) = EvaluateFeatures(x, y, remainingFeatures, true);

            if (bestFeature != -1)
            {
                _selectedFeatures.Add(bestFeature);
                remainingFeatures.Remove(bestFeature);

                if (NumOps.LessThan(bestScore, NumOps.FromDouble(_options.MinImprovement)))
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }
    }

    /// <summary>
    /// Performs backward elimination of features.
    /// </summary>
    /// <param name="x">The input feature matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// This method implements the backward elimination algorithm, which:
    /// 1. Starts with all features selected
    /// 2. Iteratively removes the feature that least impacts the model's fitness
    /// 3. Stops when removing more features would harm the model's performance
    ///    or when the minimum number of features is reached
    /// </para>
    /// <para><b>For Beginners:</b> This method simplifies your model by removing unnecessary complexity.
    /// 
    /// Think of it like decluttering your home:
    /// - You start with all your possessions (all features)
    /// - You consider removing each item and evaluate the impact
    /// - You remove the item that you'll miss the least
    /// - You continue removing items until further removal would affect your lifestyle too much
    /// 
    /// This approach helps create a more efficient model by eliminating features that
    /// don't contribute significantly to the predictions.
    /// </para>
    /// </remarks>
    private void BackwardElimination(Matrix<T> x, Vector<T> y)
    {
        _selectedFeatures = [.. Enumerable.Range(0, x.Columns)];

        while (_selectedFeatures.Count > _options.MinFeatures)
        {
            var (worstFeature, bestScore) = EvaluateFeatures(x, y, _selectedFeatures, false);

            if (worstFeature != -1)
            {
                _selectedFeatures.RemoveAt(worstFeature);

                if (NumOps.LessThan(bestScore, NumOps.FromDouble(_options.MinImprovement)))
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }
    }

    /// <summary>
    /// Evaluates the impact of adding or removing features on the model's performance.
    /// </summary>
    /// <param name="x">The input feature matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <param name="featuresToEvaluate">The list of features to evaluate for addition or removal.</param>
    /// <param name="isForwardSelection">Whether the evaluation is for forward selection (true) or backward elimination (false).</param>
    /// <returns>A tuple containing the index of the best feature to add or remove and the improvement score.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates each candidate feature to determine its impact on model performance:
    /// - For forward selection, it tests adding each remaining feature to the current set
    /// - For backward elimination, it tests removing each feature from the current set
    /// 
    /// For each potential model, it:
    /// 1. Trains a regression model using the candidate feature set
    /// 2. Evaluates the model's performance using the model evaluator
    /// 3. Calculates a fitness score using the fitness calculator
    /// 4. Identifies the best feature to add or remove based on this score
    /// </para>
    /// <para><b>For Beginners:</b> This method is like a chef testing each ingredient to see how it affects the dish.
    /// 
    /// For each potential ingredient (feature):
    /// - The chef prepares a test version of the recipe with or without that ingredient
    /// - They measure how good the resulting dish is
    /// - They select the ingredient that makes the biggest positive difference
    /// 
    /// This systematic testing ensures that each feature added to or removed from the model
    /// is the one that provides the most benefit at that step of the process.
    /// </para>
    /// </remarks>
    private (int bestFeatureIndex, T bestScore) EvaluateFeatures(Matrix<T> x, Vector<T> y, List<int> featuresToEvaluate, bool isForwardSelection)
    {
        int bestFeatureIndex = -1;
        T bestScore = _fitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue;

        for (int i = 0; i < featuresToEvaluate.Count; i++)
        {
            List<int> currentFeatures = [.. _selectedFeatures];
            if (isForwardSelection)
            {
                currentFeatures.Add(featuresToEvaluate[i]);
            }
            else
            {
                currentFeatures.RemoveAt(i);
            }

            Matrix<T> currentX = x.GetColumns(currentFeatures);
            var regression = new MultipleRegression<T>(Options, Regularization);
            regression.Train(currentX, y);

            var input = new ModelEvaluationInput<T, Matrix<T>, Vector<T>>
            {
                Model = regression,
                InputData = OptimizerHelper<T, Matrix<T>, Vector<T>>.CreateOptimizationInputData(currentX, y, currentX, y, currentX, y)
            };
            var evaluationData = _modelEvaluator.EvaluateModel(input);
            var score = _fitnessCalculator.CalculateFitnessScore(evaluationData);

            if (_fitnessCalculator.IsBetterFitness(score, bestScore))
            {
                bestScore = score;
                bestFeatureIndex = i;
            }
        }

        return (bestFeatureIndex, bestScore);
    }

    /// <summary>
    /// Returns the type identifier for this regression model.
    /// </summary>
    /// <returns>
    /// The model type identifier for stepwise regression.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns the enum value that identifies this model as a stepwise regression model. This is used 
    /// for model identification in serialization/deserialization and for logging purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method simply tells the system what kind of model this is.
    /// 
    /// It's like a name tag for the model that says "I am a stepwise regression model."
    /// This is useful when:
    /// - Saving the model to a file
    /// - Loading a model from a file
    /// - Logging information about the model
    /// 
    /// You generally won't need to call this method directly in your code.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.StepwiseRegression;
    }

    /// <summary>
    /// Serializes the stepwise regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the model, including its coefficients, selected features, and configuration options, into a 
    /// byte array. This enables the model to be saved to a file, stored in a database, or transmitted over a network.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to computer memory so you can use it later.
    /// 
    /// Think of it like taking a snapshot of the model:
    /// - It captures all the important values, settings, and the list of selected features
    /// - It converts them into a format that can be easily stored
    /// - The resulting byte array can be saved to a file or database
    /// 
    /// This is useful when you want to:
    /// - Train the model once and use it many times
    /// - Share the model with others
    /// - Use the model in a different application
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

        // Serialize StepwiseRegression specific data
        writer.Write((int)_options.Method);
        writer.Write(_options.MaxFeatures);
        writer.Write(_options.MinFeatures);
        writer.Write(Convert.ToDouble(_options.MinImprovement));

        // Serialize selected features
        writer.Write(_selectedFeatures.Count);
        foreach (var feature in _selectedFeatures)
        {
            writer.Write(feature);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the stepwise regression model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model from a byte array created by the Serialize method. It restores 
    /// the model's coefficients, selected features, and configuration options, allowing a previously saved model 
    /// to be loaded and used for predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a saved model from computer memory.
    /// 
    /// Think of it like opening a saved document:
    /// - It takes the byte array created by the Serialize method
    /// - It rebuilds all the settings, coefficients, and the list of selected features
    /// - The model is then ready to use for making predictions
    /// 
    /// This allows you to:
    /// - Use a previously trained model without having to train it again
    /// - Load models that others have shared with you
    /// - Use the same model across different applications
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

        // Deserialize StepwiseRegression specific data
        _options.Method = (StepwiseMethod)reader.ReadInt32();
        _options.MaxFeatures = reader.ReadInt32();
        _options.MinFeatures = reader.ReadInt32();
        _options.MinImprovement = Convert.ToDouble(reader.ReadDouble());

        // Deserialize selected features
        int featureCount = reader.ReadInt32();
        _selectedFeatures = new List<int>(featureCount);
        for (int i = 0; i < featureCount; i++)
        {
            _selectedFeatures.Add(reader.ReadInt32());
        }
    }

    /// <summary>
    /// Creates a new instance of the Stepwise Regression model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Stepwise Regression model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Stepwise Regression model, including its coefficients,
    /// intercept, configuration options, selected features, fitness calculator, and model evaluator. 
    /// The new instance is completely independent of the original, allowing modifications without 
    /// affecting the original model.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact copy of the current regression model.
    /// 
    /// The copy includes:
    /// - The same coefficients (the importance values for each feature)
    /// - The same intercept (the starting point value)
    /// - The same list of selected features (the ingredients that were chosen as important)
    /// - The same configuration settings (like whether to use forward or backward selection)
    /// - The same fitness calculator (the judge that evaluates model quality)
    /// - The same model evaluator (the measurement tools that assess performance)
    /// 
    /// This is useful when you want to:
    /// - Create a backup before further training or modification
    /// - Create variations of the same model for different purposes
    /// - Share the model while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var newModel = new StepwiseRegression<T>(
            options: _options,
            predictionOptions: null,
            fitnessCalculator: _fitnessCalculator,
            regularization: Regularization,
            modelEvaluator: _modelEvaluator);

        // Copy the coefficients
        if (Coefficients != null)
        {
            newModel.Coefficients = Coefficients.Clone();
        }

        // Copy the intercept
        newModel.Intercept = Intercept;

        // Create a deep copy of the selected features list
        newModel._selectedFeatures = [.. _selectedFeatures];

        return newModel;
    }
}
