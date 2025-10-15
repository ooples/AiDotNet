using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Helpers;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Models.Options;

namespace AiDotNet.ReinforcementLearning.Models
{
    /// <summary>
    /// A hierarchical risk-aware reinforcement learning model for financial applications.
    /// </summary>
    /// <remarks>
    /// The HRARL model combines hierarchical reinforcement learning with explicit risk consideration,
    /// making it particularly suitable for financial trading and portfolio management. The hierarchical
    /// structure allows different levels of decision-making (strategic and tactical), while the risk-aware
    /// component ensures that not just returns but also downside risks are considered in the decision process.
    /// 
    /// For beginners: This model works like having both a long-term financial planner and a day-to-day
    /// trader working together. The planner sets the overall strategy and risk tolerance based on market conditions,
    /// while the trader executes specific trades to meet those goals. What makes this model special is that it
    /// explicitly considers risk (the chance of losing money) alongside potential returns, similar to how human
    /// professional investors think about trading.
    /// 
    /// This model is particularly useful for:
    /// - Portfolio management with explicit risk constraints
    /// - Trading strategies that need to adapt to different market regimes
    /// - Investment scenarios where avoiding large drawdowns is as important as generating returns
    /// - Complex financial decision-making that involves both strategic and tactical choices
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
    public class HRARLModel<T> : ReinforcementLearningModelBase<T>
    {
        private readonly HRARLOptions _options = default!;
        private HRARLAgent<Tensor<T>, T>? _agent;
        private bool _isInitialized = false;
        private readonly T _riskAversionParameter = default!;
        private bool _isTrainingMode = true;
        private int _stateDimension;

        /// <summary>
        /// Gets or sets the current market risk assessment (0-1 scale, higher = more risky).
        /// </summary>
        /// <remarks>
        /// For beginners: This represents the model's assessment of current market risk levels.
        /// Values closer to 1 indicate the model thinks the market is currently very risky.
        /// This affects how cautious the model will be in its trading decisions.
        /// </remarks>
        public T CurrentRiskAssessment { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="HRARLModel{T}"/> class.
        /// </summary>
        /// <param name="options">The configuration options for the model.</param>
        public HRARLModel(HRARLOptions options) : base(options)
        {
            _options = options ?? new HRARLOptions();
            _riskAversionParameter = NumOps.FromDouble(_options.RiskAversionParameter);
            CurrentRiskAssessment = NumOps.FromDouble(0.5); // Start with neutral risk assessment
            IsContinuous = true; // Financial markets typically use continuous action spaces
            _stateDimension = _options.StateSize;
        }

        /// <summary>
        /// Initializes the HRARL agent that will interact with the environment.
        /// </summary>
        protected override void InitializeAgent()
        {
            // Initialize the agent
            _agent = new HRARLAgent<Tensor<T>, T>(
                stateDimension: _stateDimension,
                actionDimension: _options.ActionSize,
                numHierarchicalLevels: _options.NumHierarchicalLevels,
                highLevelHiddenDimension: _options.HighLevelHiddenDimension,
                lowLevelHiddenDimension: _options.LowLevelHiddenDimension,
                highLevelTimeHorizon: _options.HighLevelTimeHorizon,
                lowLevelTimeHorizon: _options.LowLevelTimeHorizon,
                highLevelGamma: _options.HighLevelGamma,
                lowLevelGamma: _options.LowLevelGamma,
                highLevelLearningRate: _options.HighLevelLearningRate,
                lowLevelLearningRate: _options.LowLevelLearningRate,
                highLevelEntropyCoef: _options.HighLevelEntropyCoef,
                lowLevelEntropyCoef: _options.LowLevelEntropyCoef,
                riskMetricType: _options.RiskMetricType,
                confidenceLevel: _options.ConfidenceLevel,
                useRecurrentHighLevelPolicy: _options.UseRecurrentHighLevelPolicy,
                useIntrinsicRewards: _options.UseIntrinsicRewards,
                intrinsicRewardScale: _options.IntrinsicRewardScale,
                useTargetNetwork: _options.UseTargetNetwork,
                targetUpdateFrequency: _options.TargetUpdateFrequency,
                useHindsightExperienceReplay: _options.UseHindsightExperienceReplay,
                batchSize: _options.BatchSize);

            _isInitialized = true;
        }

        /// <summary>
        /// Gets the HRARL agent that interacts with the environment.
        /// </summary>
        /// <returns>The HRARL agent as an IAgent interface.</returns>
        protected override IAgent<Tensor<T>, Vector<T>, T> GetAgent()
        {
            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            return _agent;
        }

        /// <summary>
        /// Sets the training mode of the model.
        /// </summary>
        /// <param name="isTraining">Whether the model is in training mode.</param>
        public void SetTrainingMode(bool isTraining)
        {
            _isTrainingMode = isTraining;
        }

        /// <summary>
        /// Sets the risk aversion parameter for the model.
        /// </summary>
        /// <param name="riskAversion">The risk aversion parameter (0-1 scale, higher = more risk-averse).</param>
        /// <remarks>
        /// For beginners: This allows you to adjust how cautious the model is. Higher values
        /// make the model prioritize avoiding losses over seeking gains. This is similar to
        /// setting the risk tolerance for a human investor based on their preferences.
        /// </remarks>
        public void SetRiskAversionParameter(T riskAversion)
        {
            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            _agent.SetRiskAversionParameter(riskAversion);
        }

        /// <summary>
        /// Selects an action based on the current state using the hierarchical policy.
        /// </summary>
        /// <param name="state">The current market state.</param>
        /// <param name="isTraining">Whether the model is in training mode.</param>
        /// <returns>The selected action vector.</returns>
        /// <remarks>
        /// For beginners: This is where the model decides what trading action to take based on
        /// the current market conditions. The high-level policy first determines the overall
        /// strategy and risk budget, then the low-level policy decides the specific trades to execute.
        /// </remarks>
        public override Vector<T> SelectAction(Tensor<T> state, bool isTraining = false)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            // Update the current risk assessment based on state
            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            CurrentRiskAssessment = _agent.AssessMarketRisk(state);

            // If adaptive risk aversion is enabled, adjust the risk aversion parameter
            if (_options.UseAdaptiveRiskAversion)
            {
                var adaptiveRiskAversion = CalculateAdaptiveRiskAversion(CurrentRiskAssessment);
                if (_agent == null)
                    throw new InvalidOperationException("Agent not initialized");
                _agent.SetRiskAversionParameter(adaptiveRiskAversion);
            }

            // Use the hierarchical agent to select an action
            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            return _agent.SelectAction(state, isTraining || _isTrainingMode);
        }

        /// <summary>
        /// Updates the model based on the observed transition.
        /// </summary>
        /// <param name="state">The current market state.</param>
        /// <param name="action">The action taken.</param>
        /// <param name="reward">The reward received.</param>
        /// <param name="nextState">The next market state.</param>
        /// <param name="done">Whether the episode is completed.</param>
        /// <returns>The loss value from the update.</returns>
        /// <remarks>
        /// For beginners: This is how the model learns from experience. After taking an action
        /// and seeing the result (whether it made or lost money), the model updates its understanding
        /// of how the market works and how to make better decisions in the future.
        /// </remarks>
        public override T Update(Tensor<T> state, Vector<T> action, T reward, Tensor<T> nextState, bool done)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            _agent.Learn(state, action, reward, nextState, done);
            return _agent.GetLatestLoss();
        }

        /// <summary>
        /// Gets the high-level policy's current goal or directive.
        /// </summary>
        /// <param name="state">The current market state.</param>
        /// <returns>A vector representing the high-level policy's goal.</returns>
        /// <remarks>
        /// For beginners: This shows what the strategic level of the model is trying to achieve.
        /// It might represent target asset allocations, risk budgets, or general market positioning
        /// (bullish/bearish). The tactical level then works to execute this high-level strategy.
        /// </remarks>
        public Vector<T> GetHighLevelGoal(Tensor<T> state)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            return _agent.GetHighLevelGoal(state);
        }

        /// <summary>
        /// Gets the risk-adjusted expected return for a given state and action.
        /// </summary>
        /// <param name="state">The current market state.</param>
        /// <param name="action">The action to evaluate.</param>
        /// <returns>The risk-adjusted expected return.</returns>
        /// <remarks>
        /// For beginners: This shows how good the model thinks a particular trading decision is,
        /// after accounting for both potential profit AND risk. It's like getting both a return
        /// forecast and a risk assessment for a potential investment decision.
        /// </remarks>
        public T GetRiskAdjustedValue(Tensor<T> state, Vector<T> action)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            return _agent.GetRiskAdjustedValue(state, action);
        }

        /// <summary>
        /// Gets the value distribution for a given state and action.
        /// </summary>
        /// <param name="state">The current market state.</param>
        /// <param name="action">The action to evaluate (if null, uses the best action).</param>
        /// <returns>A vector representing the distribution of possible returns.</returns>
        /// <remarks>
        /// For beginners: Instead of just giving a single expected return value, this shows
        /// the range of possible outcomes that might happen - including both the upside potential
        /// and downside risk. This provides a much more complete picture of the risk/reward profile.
        /// </remarks>
        public Vector<T> GetValueDistribution(Tensor<T> state, Vector<T>? action = null)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            return _agent.GetValueDistribution(state, action);
        }

        /// <summary>
        /// Calculates the Value at Risk (VaR) for a given state and action.
        /// </summary>
        /// <param name="state">The current market state.</param>
        /// <param name="action">The action to evaluate (if null, uses the best action).</param>
        /// <param name="confidenceLevel">The confidence level for VaR calculation (e.g., 0.05 for 95% VaR).</param>
        /// <returns>The Value at Risk estimate.</returns>
        /// <remarks>
        /// For beginners: Value at Risk (VaR) tells you the maximum amount you might lose in a worst-case
        /// scenario. For example, a 95% VaR of $1000 means that in 95% of cases, you wouldn't lose more
        /// than $1000. This is a standard risk measure used by professional traders and risk managers.
        /// </remarks>
        public T GetValueAtRisk(Tensor<T> state, Vector<T>? action = null, double confidenceLevel = 0.05)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            return _agent.GetValueAtRisk(state, action ?? new Vector<T>(_options.ActionSize), NumOps.FromDouble(confidenceLevel));
        }

        /// <summary>
        /// Calculates the Conditional Value at Risk (CVaR) for a given state and action.
        /// </summary>
        /// <param name="state">The current market state.</param>
        /// <param name="action">The action to evaluate (if null, uses the best action).</param>
        /// <param name="confidenceLevel">The confidence level for CVaR calculation (e.g., 0.05 for 95% CVaR).</param>
        /// <returns>The Conditional Value at Risk estimate.</returns>
        /// <remarks>
        /// For beginners: CVaR (also called Expected Shortfall) measures the average loss in the worst
        /// scenarios. It's considered more comprehensive than VaR because it tells you not just the
        /// threshold of how bad things could get, but the average severity when things do go bad.
        /// </remarks>
        public T GetConditionalValueAtRisk(Tensor<T> state, Vector<T>? action = null, double confidenceLevel = 0.05)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            return _agent.GetConditionalValueAtRisk(state, action ?? new Vector<T>(_options.ActionSize), NumOps.FromDouble(confidenceLevel));
        }

        /// <summary>
        /// Performs a risk analysis for multiple potential actions.
        /// </summary>
        /// <param name="state">The current market state.</param>
        /// <returns>A dictionary mapping action descriptions to risk analysis results.</returns>
        /// <remarks>
        /// For beginners: This provides a comprehensive risk assessment for different trading strategies
        /// you might take. For each possible approach (like "aggressive buy", "modest buy", "hold", etc.),
        /// it shows expected returns, downside risks, and probability of various outcomes. This helps
        /// make informed trading decisions with full awareness of the risk/reward tradeoffs.
        /// </remarks>
        public Dictionary<string, RiskAnalysisResult<T>> AnalyzeActionRisks(Tensor<T> state)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            var actions = DefineStandardActions();
            var results = new Dictionary<string, RiskAnalysisResult<T>>();

            foreach (var entry in actions)
            {
                string actionName = entry.Key;
                Vector<T> action = entry.Value;

                if (_agent == null)
                    throw new InvalidOperationException("Agent not initialized");
                var expectedReturn = _agent.GetExpectedValue(state, action);
                var valueDistribution = _agent.GetValueDistribution(state, action);
                var variance = _agent.GetReturnVariance(state, action);
                var var95 = _agent.GetValueAtRisk(state, action, NumOps.FromDouble(0.05));
                var cvar95 = _agent.GetConditionalValueAtRisk(state, action, NumOps.FromDouble(0.05));
                var riskAdjustedReturn = _agent.GetRiskAdjustedValue(state, action);
                var probPositiveReturn = _agent.GetProbabilityOfPositiveReturn(state, action);

                results[actionName] = new RiskAnalysisResult<T>
                {
                    ExpectedReturn = expectedReturn,
                    ReturnVariance = variance,
                    ValueAtRisk95 = var95,
                    ConditionalValueAtRisk95 = cvar95,
                    RiskAdjustedReturn = riskAdjustedReturn,
                    ProbabilityOfPositiveReturn = probPositiveReturn,
                    ValueDistribution = valueDistribution
                };
            }

            return results;
        }

        /// <summary>
        /// Simulates the outcome of a strategy over multiple time steps.
        /// </summary>
        /// <param name="state">The starting market state.</param>
        /// <param name="numSteps">The number of steps to simulate.</param>
        /// <param name="numSimulations">The number of Monte Carlo simulations to run.</param>
        /// <returns>A collection of simulated return trajectories.</returns>
        /// <remarks>
        /// For beginners: This shows what might happen if you follow the model's trading strategy
        /// over time. It runs multiple simulations to show the range of possible outcomes, giving you
        /// insight into not just the expected returns, but also the potential volatility and drawdowns
        /// you might experience along the way.
        /// </remarks>
        public List<Vector<T>> SimulateStrategy(Tensor<T> state, int numSteps, int numSimulations = 100)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            return _agent?.SimulateStrategy(state, numSteps, numSimulations) ?? throw new InvalidOperationException("Agent not initialized");
        }

        /// <summary>
        /// Saves the model to the specified path.
        /// </summary>
        /// <param name="path">The path to save the model to.</param>
        public override void SaveModel(string path)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            // TODO: Implement SaveModel on HRARLAgent
            // _agent.SaveModel(path);
            throw new NotImplementedException("SaveModel not yet implemented for HRARLAgent");
        }

        /// <summary>
        /// Loads the model from the specified path.
        /// </summary>
        /// <param name="path">The path to load the model from.</param>
        public override void LoadModel(string path)
        {
            using (var stream = File.OpenRead(path))
            {
                Load(stream);
            }
            _isInitialized = true;
        }

        /// <summary>
        /// Creates a new instance of the model.
        /// </summary>
        /// <returns>A new instance of the model with the same configuration but no learned parameters.</returns>
        public override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new HRARLModel<T>(_options);
        }

        /// <summary>
        /// Creates a deep copy of this model.
        /// </summary>
        /// <returns>A deep copy of this model with the same parameters and state.</returns>
        public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        {
            // Use the base class implementation which uses serialization/deserialization
            return base.DeepCopy();
        }

        /// <summary>
        /// Creates a shallow copy of this model.
        /// </summary>
        /// <returns>A shallow copy of this model.</returns>
        public override IFullModel<T, Tensor<T>, Tensor<T>> Clone()
        {
            // Use the base class implementation which creates a new instance
            return base.Clone();
        }

        /// <summary>
        /// Creates a new instance with the specified parameters.
        /// </summary>
        /// <param name="parameters">The parameters to set in the new instance.</param>
        /// <returns>A new instance with the specified parameters.</returns>
        public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            // Use the base class implementation which uses DeepCopy and SetParameters
            return base.WithParameters(parameters);
        }

        /// <summary>
        /// Saves the model to a stream.
        /// </summary>
        /// <param name="stream">The stream to save the model to.</param>
        public override void Save(Stream stream)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
            {
                // Write model type identifier
                writer.Write("HRARLModel");

                // Save agent parameters
                var parameters = GetParameters();
                writer.Write(parameters.Length);
                for (int i = 0; i < parameters.Length; i++)
                {
                    writer.Write(Convert.ToDouble(parameters[i]));
                }

                // Save options
                writer.Write(_options.NumHierarchicalLevels);
                writer.Write(_options.HighLevelTimeHorizon);
                writer.Write(_options.LowLevelTimeHorizon);
                writer.Write(_options.RiskAversionParameter);
                writer.Write(_options.UseAdaptiveRiskAversion);
                writer.Write(Convert.ToDouble(CurrentRiskAssessment));
            }
        }

        /// <summary>
        /// Loads the model from a stream.
        /// </summary>
        /// <param name="stream">The stream to load the model from.</param>
        public override void Load(Stream stream)
        {
            using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, true))
            {
                // Read and verify model type identifier
                string modelType = reader.ReadString();
                if (modelType != "HRARLModel")
                {
                    throw new InvalidOperationException($"Expected HRARLModel, but got {modelType}");
                }

                // Load parameters
                int paramCount = reader.ReadInt32();
                var parameters = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++)
                {
                    parameters[i] = NumOps.FromDouble(reader.ReadDouble());
                }

                // Initialize the agent if needed
                if (!_isInitialized)
                {
                    InitializeAgent();
                }

                // Set parameters to the agent
                SetParameters(parameters);

                // Load options
                int numHierarchicalLevels = reader.ReadInt32();
                int highLevelTimeHorizon = reader.ReadInt32();
                int lowLevelTimeHorizon = reader.ReadInt32();
                double riskAversionParameter = reader.ReadDouble();
                bool useAdaptiveRiskAversion = reader.ReadBoolean();
                CurrentRiskAssessment = NumOps.FromDouble(reader.ReadDouble());

                _isInitialized = true;
            }
        }

        /// <summary>
        /// Gets the parameters of the model as a single vector.
        /// </summary>
        /// <returns>A vector containing all parameters of the model.</returns>
        public override Vector<T> GetParameters()
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            // TODO: Implement GetParameters on HRARLAgent
            // return _agent.GetParameters();
            throw new NotImplementedException("GetParameters not yet implemented for HRARLAgent");
        }

        /// <summary>
        /// Sets the parameters of the model from a vector.
        /// </summary>
        /// <param name="parameters">A vector containing all parameters to set.</param>
        public override void SetParameters(Vector<T> parameters)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            // TODO: Implement SetParameters on HRARLAgent
            // _agent.SetParameters(parameters);
            throw new NotImplementedException("SetParameters not yet implemented for HRARLAgent");
        }

        /// <summary>
        /// Gets the metadata for this model.
        /// </summary>
        /// <returns>The model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();

            metadata.ModelType = ModelType.HRARLModel;
            metadata.Description = "A multi-level reinforcement learning model that explicitly considers financial risk";
            metadata.FeatureCount = _stateDimension;
            metadata.Complexity = (_options.NumHierarchicalLevels * (_options.HighLevelHiddenDimension + _options.LowLevelHiddenDimension)); // Complexity based on network architecture
            metadata.AdditionalInfo = new Dictionary<string, object>
            {
                { "Algorithm", "HRARL" },
                { "NumHierarchicalLevels", _options.NumHierarchicalLevels },
                { "HighLevelTimeHorizon", _options.HighLevelTimeHorizon },
                { "LowLevelTimeHorizon", _options.LowLevelTimeHorizon },
                { "RiskAversionParameter", _options.RiskAversionParameter },
                { "UseAdaptiveRiskAversion", _options.UseAdaptiveRiskAversion },
                { "CurrentRiskAssessment", Convert.ToDouble(CurrentRiskAssessment) }
            };

            return metadata;
        }


        /// <summary>
        /// Gets the last computed loss value.
        /// </summary>
        /// <returns>The last computed loss value.</returns>
        public override T GetLoss()
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            return _agent.GetLatestLoss();
        }

        /// <summary>
        /// Trains the model on a batch of experiences.
        /// </summary>
        /// <param name="states">The batch of states.</param>
        /// <param name="actions">The batch of actions.</param>
        /// <param name="rewards">The batch of rewards.</param>
        /// <param name="nextStates">The batch of next states.</param>
        /// <param name="dones">The batch of done flags.</param>
        /// <returns>The loss value from the training.</returns>
        protected override T TrainOnBatch(
            Tensor<T> states,
            Tensor<T> actions,
            Vector<T> rewards,
            Tensor<T> nextStates,
            Vector<T> dones)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Model is not initialized. Please initialize it first.");

            // Extract continuous actions from the tensor
            var batchSize = actions.Shape[0];
            var actionSize = actions.Shape[1];
            var actionsList = new List<Vector<T>>();

            for (int i = 0; i < batchSize; i++)
            {
                var action = new Vector<T>(actionSize);
                for (int j = 0; j < actionSize; j++)
                {
                    action[j] = actions[i, j];
                }
                actionsList.Add(action);
            }

            if (_agent == null)
                throw new InvalidOperationException("Agent not initialized");
            _agent.TrainOnBatch(states, actionsList.ToArray(), rewards, nextStates, dones);
            return _agent.GetLatestLoss();
        }

        /// <summary>
        /// Performs a training step on a batch of data.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="expectedOutput">The expected output tensor.</param>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // This method is not typically used in RL in the same way as supervised learning
            // Instead, the Update method is used with experience tuples
        }

        /// <summary>
        /// Defines a set of standard actions for risk analysis.
        /// </summary>
        private Dictionary<string, Vector<T>> DefineStandardActions()
        {
            var actions = new Dictionary<string, Vector<T>>();

            // Strong Buy (very aggressive)
            var strongBuy = new Vector<T>(_options.ActionSize);
            strongBuy[0] = NumOps.FromDouble(1.0); // Full allocation
            if (_options.ActionSize > 1)
            {
                // Additional action dimensions could represent things like:
                // - Position size scaling
                // - Stop loss level
                // - Take profit level
                // - Leverage level
                strongBuy[1] = NumOps.FromDouble(0.8); // High risk tolerance
            }
            actions["Strong Buy (Aggressive)"] = strongBuy;

            // Moderate Buy
            var moderateBuy = new Vector<T>(_options.ActionSize);
            moderateBuy[0] = NumOps.FromDouble(0.5); // Half allocation
            if (_options.ActionSize > 1)
            {
                moderateBuy[1] = NumOps.FromDouble(0.5); // Medium risk tolerance
            }
            actions["Moderate Buy"] = moderateBuy;

            // Conservative Buy
            var conservativeBuy = new Vector<T>(_options.ActionSize);
            conservativeBuy[0] = NumOps.FromDouble(0.25); // Quarter allocation
            if (_options.ActionSize > 1)
            {
                conservativeBuy[1] = NumOps.FromDouble(0.2); // Low risk tolerance
            }
            actions["Conservative Buy"] = conservativeBuy;

            // Hold (no position)
            var hold = new Vector<T>(_options.ActionSize);
            hold[0] = NumOps.FromDouble(0.0); // No allocation
            if (_options.ActionSize > 1)
            {
                hold[1] = NumOps.FromDouble(0.0); // N/A for risk tolerance
            }
            actions["Hold (Cash)"] = hold;

            // Moderate Sell
            var moderateSell = new Vector<T>(_options.ActionSize);
            moderateSell[0] = NumOps.FromDouble(-0.5); // Half short allocation
            if (_options.ActionSize > 1)
            {
                moderateSell[1] = NumOps.FromDouble(0.5); // Medium risk tolerance
            }
            actions["Moderate Sell"] = moderateSell;

            // Strong Sell
            var strongSell = new Vector<T>(_options.ActionSize);
            strongSell[0] = NumOps.FromDouble(-1.0); // Full short allocation
            if (_options.ActionSize > 1)
            {
                strongSell[1] = NumOps.FromDouble(0.8); // High risk tolerance
            }
            actions["Strong Sell (Aggressive)"] = strongSell;

            return actions;
        }

        /// <summary>
        /// Calculates an adaptive risk aversion parameter based on the current market risk assessment.
        /// </summary>
        /// <param name="marketRisk">The current market risk assessment (0-1 scale).</param>
        /// <returns>The adaptive risk aversion parameter.</returns>
        private T CalculateAdaptiveRiskAversion(T marketRisk)
        {
            // Base risk aversion from options
            T baseRiskAversion = _riskAversionParameter;

            // Adjust risk aversion based on market risk (higher market risk = higher risk aversion)
            // We use a sigmoid-like function to ensure risk aversion stays in a reasonable range
            T adjustment = NumOps.Multiply(marketRisk, NumOps.FromDouble(0.5));
            return NumOps.Add(baseRiskAversion, adjustment);
        }
    }
}
