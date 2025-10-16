using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies
{
    /// <summary>
    /// Implements a majority vote combination strategy for ensemble classification models.
    /// This strategy selects the class that receives the most votes from the base models.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the models.</typeparam>
    /// <typeparam name="TInput">The type of input data.</typeparam>
    /// <typeparam name="TOutput">The type of output data.</typeparam>
    /// <remarks>
    /// The majority vote strategy is commonly used in classification tasks where each model
    /// votes for a class, and the class with the most votes is selected as the final prediction.
    /// In case of ties, the strategy can either select randomly or use model weights to break the tie.
    /// </remarks>
    public class MajorityVoteStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
        where TOutput : notnull
    {
        private readonly bool _useWeights;
        private readonly Random _random = default!;

        /// <summary>
        /// Gets the name of the combination strategy.
        /// </summary>
        public override string StrategyName => "MajorityVote";

        /// <summary>
        /// Gets whether this strategy requires training.
        /// </summary>
        public override bool RequiresTraining => false;

        /// <summary>
        /// Initializes a new instance of the <see cref="MajorityVoteStrategy{T, TInput, TOutput}"/> class.
        /// </summary>
        /// <param name="useWeights">Whether to use model weights when counting votes. 
        /// If true, each model's vote is weighted; if false, all votes are equal.</param>
        /// <param name="seed">Optional seed for random number generation when breaking ties. 
        /// If null, a time-based seed is used.</param>
        public MajorityVoteStrategy(bool useWeights = false, int? seed = null) : base()
        {
            _useWeights = useWeights;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        /// <summary>
        /// Combines predictions from multiple models using majority voting.
        /// </summary>
        /// <param name="predictions">The predictions from each model.</param>
        /// <param name="weights">Optional weights for each model. Used only if useWeights is true.</param>
        /// <returns>The class that received the most votes.</returns>
        public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
        {
            if (predictions == null || predictions.Count == 0)
                throw new ArgumentException("Predictions list cannot be null or empty.", nameof(predictions));

            // Count votes for each prediction
            var voteCounts = new Dictionary<TOutput, T>();

            for (int i = 0; i < predictions.Count; i++)
            {
                var prediction = predictions[i];
                T voteWeight;

                if (_useWeights && weights != null && i < weights.Length)
                {
                    voteWeight = weights[i];
                }
                else
                {
                    voteWeight = NumOps.One;
                }

                if (!voteCounts.ContainsKey(prediction))
                {
                    voteCounts[prediction] = NumOps.Zero;
                }

                voteCounts[prediction] = NumOps.Add(voteCounts[prediction], voteWeight);
            }

            // Find the prediction(s) with the maximum votes
            var maxVotes = NumOps.Zero;
            var candidates = new List<TOutput>();

            foreach (var kvp in voteCounts)
            {
                int comparison;
                if (NumOps.GreaterThan(kvp.Value, maxVotes))
                    comparison = 1;
                else if (NumOps.LessThan(kvp.Value, maxVotes))
                    comparison = -1;
                else
                    comparison = 0;
                if (comparison > 0)
                {
                    maxVotes = kvp.Value;
                    candidates.Clear();
                    candidates.Add(kvp.Key);
                }
                else if (comparison == 0)
                {
                    candidates.Add(kvp.Key);
                }
            }

            // Handle ties
            if (candidates.Count == 1)
            {
                return candidates[0];
            }
            else if (candidates.Count > 1)
            {
                // If there's a tie, select randomly or use additional criteria
                return BreakTie(candidates, predictions);
            }
            else
            {
                // This should not happen if predictions is not empty
                throw new InvalidOperationException("No winner found in majority vote.");
            }
        }

        /// <summary>
        /// Checks if the predictions can be combined.
        /// </summary>
        public override bool CanCombine(List<TOutput> predictions)
        {
            return predictions != null && predictions.Count > 0;
        }

        /// <summary>
        /// Breaks ties when multiple predictions have the same number of votes.
        /// </summary>
        /// <param name="tiedPredictions">The predictions that are tied.</param>
        /// <param name="allPredictions">All predictions from the models.</param>
        /// <returns>The selected prediction after breaking the tie.</returns>
        protected virtual TOutput BreakTie(List<TOutput> tiedPredictions, List<TOutput> allPredictions)
        {
            // Strategy 1: Select the prediction that appears first in the original predictions list
            // This gives preference to models that appear earlier in the ensemble
            foreach (var prediction in allPredictions)
            {
                if (tiedPredictions.Contains(prediction))
                {
                    return prediction;
                }
            }

            // Fallback: Random selection (should rarely reach here)
            return tiedPredictions[_random.Next(tiedPredictions.Count)];
        }

        /// <summary>
        /// Gets vote counts for analysis purposes.
        /// </summary>
        /// <param name="predictions">The predictions from each model.</param>
        /// <param name="weights">Optional weights for each model.</param>
        /// <returns>A dictionary mapping each unique prediction to its vote count.</returns>
        public Dictionary<TOutput, double> GetVoteCounts(List<TOutput> predictions, Vector<T>? weights = null)
        {
            var voteCounts = new Dictionary<TOutput, double>();

            for (int i = 0; i < predictions.Count; i++)
            {
                var prediction = predictions[i];
                double voteWeight;

                if (_useWeights && weights != null && i < weights.Length)
                {
                    voteWeight = Convert.ToDouble(weights[i]);
                }
                else
                {
                    voteWeight = 1.0;
                }

                if (!voteCounts.ContainsKey(prediction))
                {
                    voteCounts[prediction] = 0;
                }

                voteCounts[prediction] += voteWeight;
            }

            return voteCounts;
        }

        /// <summary>
        /// Gets confidence score based on the proportion of votes for the winning class.
        /// </summary>
        /// <param name="predictions">The predictions from each model.</param>
        /// <param name="weights">Optional weights for each model.</param>
        /// <returns>A value between 0 and 1 indicating confidence in the prediction.</returns>
        public double GetConfidence(List<TOutput> predictions, Vector<T>? weights = null)
        {
            var voteCounts = GetVoteCounts(predictions, weights);
            if (voteCounts.Count == 0)
                return 0.0;

            var totalVotes = voteCounts.Values.Sum();
            var maxVotes = voteCounts.Values.Max();

            return totalVotes > 0 ? maxVotes / totalVotes : 0.0;
        }
    }
}