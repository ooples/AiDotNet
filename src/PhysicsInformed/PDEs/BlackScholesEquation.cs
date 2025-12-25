using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the Black-Scholes Equation for option pricing:
    /// ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Black-Scholes equation is the fundamental equation in mathematical finance for pricing options.
    ///
    /// Variables:
    /// - V(S,t) = Option price (value of the derivative)
    /// - S = Current stock price (the underlying asset)
    /// - t = Time (usually time to expiration)
    /// - σ (sigma) = Volatility of the stock (how much the price fluctuates)
    /// - r = Risk-free interest rate
    ///
    /// Physical/Financial Interpretation:
    /// - The equation balances the change in option value over time with:
    ///   * The effect of stock price changes (delta hedging)
    ///   * The effect of volatility (gamma)
    ///   * The time value of money (discounting)
    ///
    /// Key Insight:
    /// Under certain assumptions (no arbitrage, continuous trading, no dividends),
    /// any derivative can be perfectly hedged, leading to this equation.
    ///
    /// Historical Note:
    /// Developed by Fischer Black and Myron Scholes in 1973, earning Scholes
    /// a Nobel Prize in Economics in 1997 (Black had passed away by then).
    ///
    /// Example: Pricing a European call option on a stock trading at $100
    /// with strike $105, volatility 20%, risk-free rate 5%, expiring in 1 year.
    /// </remarks>
    public class BlackScholesEquation<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly T _volatility;
        private readonly T _riskFreeRate;
        private readonly T _halfVolSquared;

        /// <summary>
        /// Initializes a new instance of the Black-Scholes Equation.
        /// </summary>
        /// <param name="volatility">The volatility σ of the underlying asset (must be positive).</param>
        /// <param name="riskFreeRate">The risk-free interest rate r.</param>
        public BlackScholesEquation(T volatility, T riskFreeRate)
        {
            ValidatePositive(volatility, nameof(volatility));

            _volatility = volatility;
            _riskFreeRate = riskFreeRate;

            // Pre-compute ½σ² for efficiency
            T volSquared = NumOps.Multiply(volatility, volatility);
            _halfVolSquared = NumOps.Multiply(NumOps.FromDouble(0.5), volSquared);
        }

        /// <summary>
        /// Initializes a new instance of the Black-Scholes Equation with double parameters.
        /// </summary>
        /// <param name="volatility">The volatility σ of the underlying asset (default 0.2 = 20% annual, must be positive).</param>
        /// <param name="riskFreeRate">The risk-free interest rate r (default 0.05 = 5% annual).</param>
        public BlackScholesEquation(double volatility = 0.2, double riskFreeRate = 0.05)
            : this(
                MathHelper.GetNumericOperations<T>().FromDouble(volatility),
                MathHelper.GetNumericOperations<T>().FromDouble(riskFreeRate))
        {
        }

        /// <inheritdoc/>
        public override T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateSecondDerivatives(derivatives);

            var firstDerivs = derivatives.FirstDerivatives;
            var secondDerivs = derivatives.SecondDerivatives;

            if (firstDerivs is null || secondDerivs is null)
            {
                throw new InvalidOperationException("Derivatives were null after validation.");
            }

            // inputs = [S, t], outputs = [V]
            // PDE: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

            T S = inputs[0]; // Stock price
            T V = outputs[0]; // Option value

            T dVdt = firstDerivs[0, 1]; // ∂V/∂t
            T dVdS = firstDerivs[0, 0]; // ∂V/∂S
            T d2VdS2 = secondDerivs[0, 0, 0]; // ∂²V/∂S²

            // ½σ²S²∂²V/∂S²
            T S2 = NumOps.Multiply(S, S);
            T gammaterm = NumOps.Multiply(_halfVolSquared, NumOps.Multiply(S2, d2VdS2));

            // rS∂V/∂S
            T deltaterm = NumOps.Multiply(_riskFreeRate, NumOps.Multiply(S, dVdS));

            // rV
            T discountTerm = NumOps.Multiply(_riskFreeRate, V);

            // ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
            T residual = NumOps.Add(dVdt, gammaterm);
            residual = NumOps.Add(residual, deltaterm);
            residual = NumOps.Subtract(residual, discountTerm);

            return residual;
        }

        /// <inheritdoc/>
        public PDEResidualGradient<T> ComputeResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateSecondDerivatives(derivatives);

            T S = inputs[0];
            T S2 = NumOps.Multiply(S, S);

            var gradient = CreateGradient();

            // ∂R/∂(∂V/∂t) = 1
            gradient.FirstDerivatives[0, 1] = NumOps.One;

            // ∂R/∂(∂V/∂S) = rS
            gradient.FirstDerivatives[0, 0] = NumOps.Multiply(_riskFreeRate, S);

            // ∂R/∂(∂²V/∂S²) = ½σ²S²
            gradient.SecondDerivatives[0, 0, 0] = NumOps.Multiply(_halfVolSquared, S2);

            // ∂R/∂V = -r (stored separately as output gradient)
            gradient.OutputGradients[0] = NumOps.Negate(_riskFreeRate);

            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => 2; // [S, t] - Stock price and time

        /// <inheritdoc/>
        public override int OutputDimension => 1; // [V] - Option value

        /// <inheritdoc/>
        public override string Name => $"Black-Scholes Equation (σ={_volatility}, r={_riskFreeRate})";
    }
}
