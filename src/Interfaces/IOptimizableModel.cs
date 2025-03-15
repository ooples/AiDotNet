namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for models that can be optimized using evolutionary algorithms.
/// </summary>
/// <remarks>
/// This interface provides functionality for models that can evolve and improve through
/// processes inspired by natural selection, such as genetic algorithms.
/// 
/// For Beginners: This interface helps create AI models that can "evolve" to get better over time.
/// 
/// What are evolutionary algorithms?
/// - These are optimization techniques inspired by natural evolution
/// - They work by creating multiple versions (population) of a solution
/// - The best solutions are selected and combined to create new, potentially better solutions
/// - Random changes (mutations) are introduced to explore new possibilities
/// - This process repeats over many generations, gradually improving the solutions
/// 
/// Real-world analogy:
/// Think of breeding dogs. Dog breeders select dogs with desirable traits (like intelligence
/// or strength) and breed them together. The puppies inherit traits from both parents, with
/// some random variations. Over many generations, this process can create dogs with specific
/// desired characteristics. Evolutionary algorithms work in a similar way with AI models.
/// 
/// When to use evolutionary optimization:
/// - When traditional optimization methods don't work well
/// - For problems with many possible solutions
/// - When the problem is too complex to solve analytically
/// - When you need to find good (but not necessarily perfect) solutions quickly
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IOptimizableModel<T>
{
    /// <summary>
    /// Evaluates the model with the given input and returns a result.
    /// </summary>
    /// <remarks>
    /// This method applies the model to the input data and produces an output value.
    /// 
    /// For Beginners: This is how the model makes predictions or calculations based on input data.
    /// 
    /// For example:
    /// - In a weather prediction model, the input might be today's weather data
    /// - The Evaluate method would return tomorrow's predicted temperature
    /// 
    /// This method is used:
    /// - During training to see how well the model performs
    /// - When using the trained model to make predictions
    /// </remarks>
    /// <param name="input">The input vector to evaluate.</param>
    /// <returns>The result of evaluating the model with the given input.</returns>
    T Evaluate(Vector<T> input);

    /// <summary>
    /// Creates a slightly modified version of the current model.
    /// </summary>
    /// <remarks>
    /// This method introduces random changes to the model's parameters based on the mutation rate.
    /// 
    /// For Beginners: This is like creating a variation of the model with small random changes.
    /// 
    /// What is mutation?
    /// - Mutation introduces random changes to explore new possibilities
    /// - Higher mutation rates mean bigger or more frequent changes
    /// - Lower mutation rates mean smaller or fewer changes
    /// 
    /// Real-world analogy:
    /// Think of mutation like trying a new ingredient in a recipe. Most changes might make
    /// the dish worse, but occasionally you discover something better. In evolutionary algorithms,
    /// mutations help discover new solutions that might be better than the current ones.
    /// 
    /// The mutationRate controls how much change occurs:
    /// - 0.0 means no changes (clone)
    /// - 1.0 means maximum possible changes
    /// - Typical values are small (0.01 to 0.1) to make gradual improvements
    /// </remarks>
    /// <param name="mutationRate">
    /// A value between 0 and 1 that controls the amount of mutation.
    /// Higher values cause more significant changes.
    /// </param>
    /// <returns>A new model instance with mutated parameters.</returns>
    IOptimizableModel<T> Mutate(double mutationRate);

    /// <summary>
    /// Combines this model with another model to create a new model.
    /// </summary>
    /// <remarks>
    /// This method creates a new model by combining parameters from this model and another model,
    /// based on the crossover rate.
    /// 
    /// For Beginners: This is like creating a "child" model that inherits traits from two "parent" models.
    /// 
    /// What is crossover?
    /// - Crossover combines parts of two successful models to create a new one
    /// - It's similar to how children inherit traits from both parents
    /// - The goal is to combine the best aspects of both models
    /// 
    /// Real-world analogy:
    /// Think of crossover like breeding two prize-winning dogs. The puppies might inherit
    /// the intelligence of one parent and the strength of the other, potentially becoming
    /// better than either parent. In evolutionary algorithms, crossover helps combine
    /// successful traits from different solutions.
    /// 
    /// The crossoverRate controls how much mixing occurs:
    /// - 0.0 means the new model is a copy of this model (no mixing)
    /// - 1.0 means maximum mixing between the two models
    /// - Typical values are around 0.5 to 0.8 for balanced mixing
    /// </remarks>
    /// <param name="other">The other model to combine with this model.</param>
    /// <param name="crossoverRate">
    /// A value between 0 and 1 that controls how much of the other model's parameters
    /// are incorporated into the new model.
    /// </param>
    /// <returns>A new model instance with combined parameters from both parent models.</returns>
    IOptimizableModel<T> Crossover(IOptimizableModel<T> other, double crossoverRate);

    /// <summary>
    /// Creates an exact copy of this model.
    /// </summary>
    /// <remarks>
    /// This method creates a new instance of the model with identical parameters.
    /// 
    /// For Beginners: This creates an exact duplicate of the model.
    /// 
    /// Why cloning is important:
    /// - It preserves good solutions while exploring new ones
    /// - It allows you to keep the original model unchanged while creating variations
    /// - It's often used to maintain the best model found so far (called "elitism")
    /// 
    /// For example:
    /// - You might clone your best-performing model before applying mutations
    /// - This ensures you don't lose a good solution while trying to find better ones
    /// 
    /// In evolutionary algorithms, cloning is often used to preserve the best solutions
    /// from one generation to the next, ensuring that performance never decreases.
    /// </remarks>
    /// <returns>A new model instance that is an exact copy of this model.</returns>
    IOptimizableModel<T> Clone();
}