namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a symbolic machine learning model that represents solutions as mathematical expressions.
/// </summary>
/// <remarks>
/// Symbolic models create human-readable mathematical formulas to represent relationships in data.
/// These models can evolve through genetic operations like mutation and crossover.
/// 
/// For Beginners: A symbolic model is like having an AI that writes mathematical formulas to
/// explain your data. Instead of being a "black box" where you can't see how decisions are made,
/// symbolic models give you actual equations you can understand.
/// 
/// For example, instead of just predicting house prices, a symbolic model might tell you:
/// "Price = (Size × $100) + (Number of Bedrooms × $15,000) - (Age of House × $1,000)"
/// 
/// These models can "evolve" by trying different formulas and combining the best ones,
/// similar to how genetic traits evolve in nature.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface ISymbolicModel<T> : IFullModel<T>
{
    /// <summary>
    /// Gets the complexity measure of the symbolic model.
    /// </summary>
    /// <remarks>
    /// Complexity typically represents how complicated the mathematical expression is.
    /// 
    /// For Beginners: This property tells you how complicated the formula is. A higher
    /// number means a more complex formula with more operations, variables, or terms.
    /// 
    /// Simple formulas (low complexity) are usually preferred because they:
    /// - Are easier to understand
    /// - Often generalize better to new data
    /// - Are less likely to be "overfitting" (memorizing) the training data
    /// </remarks>
    int Complexity { get; }

    /// <summary>
    /// Trains the symbolic model using the provided input features and target values.
    /// </summary>
    /// <remarks>
    /// This method finds the best symbolic expression to represent the relationship between inputs and outputs.
    /// 
    /// For Beginners: This is where the model "learns" from your data. You provide:
    /// - X: Your input data (features) organized in rows and columns
    /// - y: The correct answers (target values) you want the model to predict
    /// 
    /// The model will try different mathematical formulas until it finds one that
    /// best explains the relationship between your inputs and the correct answers.
    /// </remarks>
    /// <param name="X">The matrix of input features where each row is a data point and each column is a feature.</param>
    /// <param name="y">The vector of target values corresponding to each row in X.</param>
    void Fit(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Calculates the model's prediction for a single input vector.
    /// </summary>
    /// <remarks>
    /// This method applies the symbolic expression to the input data to produce a prediction.
    /// 
    /// For Beginners: Once your model has learned a formula, this method lets you use that
    /// formula to make predictions. You provide the input values, and the model calculates
    /// what the output should be according to its formula.
    /// 
    /// For example, if your model learned a formula for house prices, you could provide
    /// details about a new house and get a predicted price.
    /// </remarks>
    /// <param name="input">The vector of input features for a single data point.</param>
    /// <returns>The predicted value for the given input.</returns>
    T Evaluate(Vector<T> input);

    /// <summary>
    /// Creates a slightly modified version of the current model through random changes.
    /// </summary>
    /// <remarks>
    /// Mutation introduces small random changes to the symbolic expression to explore new solutions.
    /// 
    /// For Beginners: This is like creating a "child" version of the model with small random changes.
    /// These changes might make the formula better or worse at predictions.
    /// 
    /// The mutationRate controls how many changes are made:
    /// - A low rate (e.g., 0.01) means subtle changes
    /// - A high rate (e.g., 0.5) means more dramatic changes
    /// 
    /// This is inspired by genetic mutation in nature and helps the model explore new possible
    /// formulas that might work better than the current one.
    /// </remarks>
    /// <param name="mutationRate">A value between 0 and 1 controlling the probability and extent of mutations.</param>
    /// <returns>A new symbolic model with mutations applied.</returns>
    ISymbolicModel<T> Mutate(double mutationRate);

    /// <summary>
    /// Combines this model with another model to create a new model that inherits traits from both.
    /// </summary>
    /// <remarks>
    /// Crossover combines parts of two symbolic expressions to create a new expression.
    /// 
    /// For Beginners: This is like creating a "child" model that inherits parts of its formula
    /// from two "parent" models. The idea is that if both parent models have good qualities,
    /// the child might combine the best of both.
    /// 
    /// The crossoverRate controls how much mixing occurs:
    /// - A low rate means the child will be more similar to one parent
    /// - A high rate means more balanced mixing between both parents
    /// 
    /// This is inspired by genetic reproduction in nature and helps create new formulas
    /// that might perform better than either parent alone.
    /// </remarks>
    /// <param name="other">The other symbolic model to combine with.</param>
    /// <param name="crossoverRate">A value between 0 and 1 controlling the probability and extent of crossover.</param>
    /// <returns>A new symbolic model resulting from the crossover operation.</returns>
    ISymbolicModel<T> Crossover(ISymbolicModel<T> other, double crossoverRate);

    /// <summary>
    /// Creates an identical copy of this symbolic model.
    /// </summary>
    /// <remarks>
    /// This method creates a deep copy where all internal structures are duplicated.
    /// 
    /// For Beginners: This creates an exact duplicate of the model with the same formula.
    /// Changes to the copy won't affect the original model.
    /// 
    /// This is useful when you want to:
    /// - Preserve a good model before experimenting with changes
    /// - Create multiple variations of the same base model
    /// - Save a "snapshot" of the model at a certain point in training
    /// </remarks>
    /// <returns>A new instance that is an exact copy of this symbolic model.</returns>
    ISymbolicModel<T> Copy();

    /// <summary>
    /// Gets the number of input features the model can accept.
    /// </summary>
    /// <remarks>
    /// This represents the number of variables the symbolic expression can use.
    /// 
    /// For Beginners: This tells you how many different input variables the model's formula
    /// can work with. For example, if you're predicting house prices based on size, age,
    /// and number of bedrooms, the feature count would be 3.
    /// </remarks>
    int FeatureCount { get; }

    /// <summary>
    /// Determines whether a specific input feature is used in the symbolic expression.
    /// </summary>
    /// <remarks>
    /// This method checks if a particular feature contributes to the model's predictions.
    /// 
    /// For Beginners: This tells you if a specific input variable is actually used in the
    /// model's formula. Sometimes, the model might decide that certain inputs aren't helpful
    /// for making predictions and exclude them from the formula.
    /// 
    /// For example, if you provided house size, age, and color as features, the model might
    /// determine that color doesn't help predict price and won't use it in the formula.
    /// </remarks>
    /// <param name="featureIndex">The zero-based index of the feature to check.</param>
    /// <returns>True if the feature is used in the model; otherwise, false.</returns>
    bool IsFeatureUsed(int featureIndex);

    /// <summary>
    /// Gets the coefficients (numerical parameters) used in the symbolic expression.
    /// </summary>
    /// <remarks>
    /// Coefficients are the numerical values that scale or adjust terms in the expression.
    /// 
    /// For Beginners: These are the numbers in the model's formula. For example, in the formula
    /// "y = 2x + 3", the coefficients are 2 and 3.
    /// 
    /// Coefficients determine how strongly each part of the formula affects the final prediction.
    /// A larger coefficient means that term has a bigger impact on the result.
    /// </remarks>
    Vector<T> Coefficients { get; }

    /// <summary>
    /// Creates a new model with updated coefficient values while keeping the same structure.
    /// </summary>
    /// <remarks>
    /// This method allows fine-tuning the model by adjusting its numerical parameters.
    /// 
    /// For Beginners: This lets you change just the numbers in the formula without changing
    /// its overall structure. It's like keeping the recipe the same but adjusting the amounts
    /// of each ingredient.
    /// 
    /// This is useful when:
    /// - You want to fine-tune a model that has a good structure
    /// - You're performing optimization to find the best coefficient values
    /// - You want to test how sensitive the model is to changes in coefficients
    /// </remarks>
    /// <param name="newCoefficients">The new coefficient values to use in the model.</param>
    /// <returns>A new symbolic model with updated coefficients.</returns>
    ISymbolicModel<T> UpdateCoefficients(Vector<T> newCoefficients);
}