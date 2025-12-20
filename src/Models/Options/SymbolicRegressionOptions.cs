namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Symbolic Regression, an evolutionary approach to finding
/// mathematical expressions that best fit a dataset.
/// </summary>
/// <remarks>
/// <para>
/// Symbolic Regression is a type of regression analysis that searches for mathematical expressions that best 
/// fit a given dataset, both in terms of accuracy and simplicity. Unlike traditional regression techniques 
/// that fit parameters to a predefined model structure, symbolic regression simultaneously evolves both the 
/// structure of the model and its parameters. It uses genetic programming, an evolutionary algorithm inspired 
/// by biological evolution, to evolve a population of mathematical expressions through operations like 
/// selection, crossover, and mutation. This approach can discover complex, non-linear relationships in data 
/// without requiring prior assumptions about the form of the model. This class inherits from 
/// NonLinearRegressionOptions and adds parameters specific to the evolutionary algorithm used in symbolic 
/// regression, such as population size, number of generations, and genetic operator rates.
/// </para>
/// <para><b>For Beginners:</b> Symbolic Regression finds mathematical formulas that explain your data.
/// 
/// When performing regression (predicting values):
/// - Traditional methods fit parameters to a predefined equation
/// - You must specify the form of the equation in advance
/// - This requires knowing what relationship to look for
/// 
/// Symbolic Regression solves this by:
/// - Automatically discovering both the structure and parameters of equations
/// - Starting with a population of random simple formulas
/// - Evolving them through "survival of the fittest"
/// - Combining good formulas to create better ones (crossover)
/// - Randomly changing formulas occasionally (mutation)
/// - Continuing until it finds a formula that fits the data well
/// 
/// This approach offers several benefits:
/// - Can discover unexpected relationships in your data
/// - Produces human-readable mathematical formulas
/// - Doesn't require prior knowledge of the underlying relationship
/// - Often finds simpler models than other techniques
/// 
/// This class lets you configure how the evolutionary algorithm searches for formulas.
/// </para>
/// </remarks>
public class SymbolicRegressionOptions : NonLinearRegressionOptions
{
    /// <summary>
    /// Gets or sets the size of the population in the genetic algorithm.
    /// </summary>
    /// <value>A positive integer, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the number of mathematical expressions (individuals) maintained in the population 
    /// during the evolutionary process. A larger population provides more genetic diversity and a broader search 
    /// of the solution space, potentially finding better solutions but requiring more computational resources. 
    /// A smaller population requires less computation per generation but might converge prematurely to suboptimal 
    /// solutions due to limited genetic diversity. The default value of 100 provides a moderate population size 
    /// suitable for many applications, balancing diversity and computational efficiency. The optimal value depends 
    /// on the complexity of the problem, the available computational resources, and the desired trade-off between 
    /// exploration and exploitation in the search process.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many different formulas the algorithm evaluates in parallel.
    /// 
    /// The population size:
    /// - Determines how many different mathematical expressions are considered at once
    /// - Affects both the quality of solutions and computational requirements
    /// 
    /// The default value of 100 means:
    /// - The algorithm maintains 100 different formulas in each generation
    /// - This provides a good balance between diversity and efficiency for many problems
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 500): More diverse exploration, better chance of finding optimal solutions, but slower
    /// - Smaller values (e.g., 20): Faster computation, but may get stuck in suboptimal solutions
    /// 
    /// When to adjust this value:
    /// - Increase it for complex problems where finding the right formula structure is difficult
    /// - Decrease it when computational resources are limited or for simpler problems
    /// - Scale it with the complexity of your data and the expected complexity of the relationship
    /// 
    /// For example, if searching for a formula to describe a complex physical system with
    /// many variables, you might increase this to 500 to explore more possible formulas.
    /// </para>
    /// </remarks>
    public int PopulationSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the maximum number of generations for the genetic algorithm.
    /// </summary>
    /// <value>A positive integer, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum number of evolutionary generations the algorithm will run before 
    /// terminating if no other stopping criterion (such as reaching the fitness threshold) is met. Each generation 
    /// involves evaluating the fitness of all individuals in the population, selecting individuals for reproduction, 
    /// and creating a new population through crossover and mutation. More generations allow for more evolution and 
    /// potentially better solutions but require more computation time. The default value of 1000 provides a 
    /// reasonable upper limit for many applications, allowing sufficient evolution while preventing excessive 
    /// computation. The optimal value depends on the complexity of the problem, the population size, and how 
    /// quickly the population converges to a solution.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how many rounds of evolution the algorithm will perform.
    /// 
    /// The maximum generations:
    /// - Sets an upper limit on how long the evolutionary process will run
    /// - Prevents the algorithm from running indefinitely
    /// - Serves as a stopping criterion if a good solution isn't found earlier
    /// 
    /// The default value of 1000 means:
    /// - The algorithm will evolve the population for at most 1000 generations
    /// - It may stop earlier if it finds a solution that meets the fitness threshold
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 5000): More opportunity to find better solutions, but longer runtime
    /// - Smaller values (e.g., 100): Faster results, but may not find optimal solutions for complex problems
    /// 
    /// When to adjust this value:
    /// - Increase it for complex problems that need more evolution to find good solutions
    /// - Decrease it when you need faster results or for simpler problems
    /// - Monitor the fitness improvement over generations to determine if more generations would help
    /// 
    /// For example, if you notice the best solution is still improving significantly at
    /// generation 1000, you might increase this to 2000 or more to allow further improvement.
    /// </para>
    /// </remarks>
    public int MaxGenerations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the probability of mutation in the genetic algorithm.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the probability of applying mutation to an individual in the population during 
    /// reproduction. Mutation introduces random changes to mathematical expressions, such as changing operators, 
    /// constants, or variables, or adding or removing terms. It helps maintain genetic diversity and explore new 
    /// regions of the solution space. A higher mutation rate increases exploration but might disrupt good solutions, 
    /// while a lower rate preserves good solutions but might lead to premature convergence. The default value of 
    /// 0.1 (10%) provides a moderate mutation rate suitable for many applications, balancing exploration and 
    /// exploitation. The optimal value depends on the complexity of the problem and the desired balance between 
    /// exploring new solutions and refining existing ones.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how often random changes are introduced to formulas.
    /// 
    /// The mutation rate:
    /// - Determines the probability of making random changes to formulas
    /// - Helps the algorithm explore new possibilities and avoid getting stuck
    /// - Introduces innovation into the population
    /// 
    /// The default value of 0.1 means:
    /// - Each formula has a 10% chance of being mutated in each generation
    /// - Mutations might include changing an operation (+ to ×), adding a term, etc.
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.3): More exploration, more diversity, but may disrupt good solutions
    /// - Lower values (e.g., 0.01): More stability, better refinement of good solutions, but may get stuck
    /// 
    /// When to adjust this value:
    /// - Increase it when the algorithm seems to be converging too quickly to suboptimal solutions
    /// - Decrease it when good solutions are being found but need refinement
    /// - Often paired with CrossoverRate adjustments to balance exploration and exploitation
    /// 
    /// For example, if your algorithm keeps finding the same suboptimal formula,
    /// you might increase this to 0.2 to encourage more exploration of different formulas.
    /// </para>
    /// </remarks>
    public double MutationRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the probability of crossover in the genetic algorithm.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.8.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the probability of applying crossover between two parent individuals during 
    /// reproduction. Crossover combines parts of two parent expressions to create new offspring, allowing the 
    /// algorithm to combine good features from different solutions. A higher crossover rate increases the mixing 
    /// of genetic material and potentially accelerates convergence to good solutions, while a lower rate preserves 
    /// more of the original expressions. The default value of 0.8 (80%) provides a high crossover rate suitable 
    /// for many applications, emphasizing the recombination of existing solutions. The optimal value depends on 
    /// the complexity of the problem and the desired balance between combining existing solutions and preserving 
    /// them intact.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how often the algorithm combines parts of two good formulas to create new ones.
    /// 
    /// The crossover rate:
    /// - Determines the probability of combining parts of two parent formulas
    /// - Allows good components from different formulas to be combined
    /// - Is the main mechanism for improvement in genetic algorithms
    /// 
    /// The default value of 0.8 means:
    /// - There's an 80% chance that two selected formulas will be combined
    /// - This creates offspring that inherit traits from both parents
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.9): More mixing of formula components, faster convergence
    /// - Lower values (e.g., 0.5): More preservation of existing formulas, slower evolution
    /// 
    /// When to adjust this value:
    /// - Increase it to more aggressively combine promising formula components
    /// - Decrease it when good formulas are being disrupted too frequently
    /// - Often adjusted in conjunction with MutationRate
    /// 
    /// For example, if you have a diverse population but evolution is progressing slowly,
    /// you might increase this to 0.9 to more frequently combine promising formula components.
    /// </para>
    /// </remarks>
    public double CrossoverRate { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the fitness threshold for early stopping.
    /// </summary>
    /// <value>A positive double value, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum fitness value (typically an error measure like mean squared error) 
    /// that, when reached by any individual in the population, will cause the algorithm to terminate early. 
    /// It provides a stopping criterion based on solution quality rather than just the number of generations. 
    /// A smaller threshold requires a better fit to the data before stopping, potentially leading to more accurate 
    /// but more complex expressions and longer computation time. A larger threshold allows earlier stopping with 
    /// less accurate but potentially simpler expressions. The default value of 0.001 provides a moderately strict 
    /// threshold suitable for many applications, requiring a good fit while preventing excessive computation for 
    /// diminishing returns. The appropriate value depends on the specific problem, the scale of the target variable, 
    /// and the desired trade-off between accuracy and computation time.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how good a formula must be before the algorithm stops searching.
    /// 
    /// The fitness threshold:
    /// - Sets a target quality level for the formulas
    /// - When a formula reaches this level of fitness, the algorithm can stop early
    /// - Prevents unnecessary computation once a good solution is found
    /// 
    /// The default value of 0.001 means:
    /// - The algorithm stops when it finds a formula with an error measure below 0.001
    /// - This is quite strict and requires a very good fit to the data
    /// 
    /// Think of it like this:
    /// - Smaller values (e.g., 0.0001): More strict, requires better fit, longer runtime
    /// - Larger values (e.g., 0.01): Less strict, accepts solutions with more error, faster results
    /// 
    /// When to adjust this value:
    /// - Decrease it when you need very accurate formulas and are willing to wait longer
    /// - Increase it when approximate solutions are acceptable or when data is noisy
    /// - Scale it according to the range and units of your target variable
    /// 
    /// For example, if your data contains measurement noise of about 1%, setting this to
    /// 0.01 might be reasonable since formulas can't be expected to fit the noise.
    /// </para>
    /// </remarks>
    public double FitnessThreshold { get; set; } = 0.001;
}
