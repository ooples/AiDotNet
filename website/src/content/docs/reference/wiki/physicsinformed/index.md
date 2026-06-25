---
title: "Physics Informed"
description: "All 83 public types in the AiDotNet.physicsinformed namespace, organized by kind."
section: "API Reference"
---

**83** public types in this namespace, organized by kind.

## Models & Types (43)

| Type | Summary |
|:-----|:--------|
| [`AdvectionDiffusionEquation<T>`](/docs/reference/wiki/physicsinformed/advectiondiffusionequation/) |  |
| [`AllenCahnEquation<T>`](/docs/reference/wiki/physicsinformed/allencahnequation/) | Represents the Allen-Cahn equation: u_t - epsilon^2 * u_xx + u^3 - u = 0. |
| [`BlackScholesEquation<T>`](/docs/reference/wiki/physicsinformed/blackscholesequation/) | Represents the Black-Scholes Equation for option pricing: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0 |
| [`BurgersEquation<T>`](/docs/reference/wiki/physicsinformed/burgersequation/) | Represents the Burgers' Equation: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x² |
| [`DeepOperatorNetwork<T>`](/docs/reference/wiki/physicsinformed/deepoperatornetwork/) | Implements Deep Operator Network (DeepONet) for learning operators. |
| [`DeepRitzMethod<T>`](/docs/reference/wiki/physicsinformed/deepritzmethod/) | Implements the Deep Ritz Method for solving variational problems and PDEs. |
| [`DomainDecompositionPINN<T>`](/docs/reference/wiki/physicsinformed/domaindecompositionpinn/) | Domain Decomposition Physics-Informed Neural Network for large-scale problems. |
| [`DomainDecompositionTrainingHistory<T>`](/docs/reference/wiki/physicsinformed/domaindecompositiontraininghistory/) | Training history for domain decomposition PINN training. |
| [`FourierNeuralOperator<T>`](/docs/reference/wiki/physicsinformed/fourierneuraloperator/) | Implements the Fourier Neural Operator (FNO) for learning operators between function spaces. |
| [`GpuPINNTrainer<T>`](/docs/reference/wiki/physicsinformed/gpupinntrainer/) | Provides GPU-accelerated training for Physics-Informed Neural Networks. |
| [`GpuTrainingHistory<T>`](/docs/reference/wiki/physicsinformed/gputraininghistory/) | Training history with GPU-specific metrics. |
| [`GraphNeuralOperator<T>`](/docs/reference/wiki/physicsinformed/graphneuraloperator/) | Implements Graph Neural Operators for learning operators on graph-structured data. |
| [`HamiltonianNeuralNetwork<T>`](/docs/reference/wiki/physicsinformed/hamiltonianneuralnetwork/) | Implements Hamiltonian Neural Networks (HNN) for learning conservative dynamical systems. |
| [`HeatEquation<T>`](/docs/reference/wiki/physicsinformed/heatequation/) | Represents the Heat Equation (or Diffusion Equation): ∂u/∂t = α ∂²u/∂x² |
| [`InterfaceDefinition<T>`](/docs/reference/wiki/physicsinformed/interfacedefinition/) | Defines an interface between two subdomains. |
| [`InverseProblemPINN<T>`](/docs/reference/wiki/physicsinformed/inverseproblempinn/) | Implements a Physics-Informed Neural Network for inverse problems (parameter identification). |
| [`InverseProblemResult<T>`](/docs/reference/wiki/physicsinformed/inverseproblemresult/) | Results from inverse problem optimization. |
| [`KortewegDeVriesEquation<T>`](/docs/reference/wiki/physicsinformed/kortewegdevriesequation/) | Represents the Korteweg-de Vries (KdV) Equation: ∂u/∂t + αu∂u/∂x + β∂³u/∂x³ = 0 |
| [`LagrangianNeuralNetwork<T>`](/docs/reference/wiki/physicsinformed/lagrangianneuralnetwork/) | Implements Lagrangian Neural Networks (LNN) for learning mechanical systems. |
| [`LinearElasticityEquation<T>`](/docs/reference/wiki/physicsinformed/linearelasticityequation/) | Represents the 2D Linear Elasticity Equations (Navier-Cauchy equations): (λ + μ)∂(∂u/∂x + ∂v/∂y)/∂x + μ∇²u + fₓ = 0 (λ + μ)∂(∂u/∂x + ∂v/∂y)/∂y + μ∇²v + fᵧ = 0 |
| [`MaxwellEquations<T>`](/docs/reference/wiki/physicsinformed/maxwellequations/) | Represents Maxwell's equations for electromagnetic wave propagation (2D TE mode). |
| [`MultiFidelityPINN<T>`](/docs/reference/wiki/physicsinformed/multifidelitypinn/) | Multi-fidelity Physics-Informed Neural Network for combining data of different accuracy levels. |
| [`MultiFidelityTrainingHistory<T>`](/docs/reference/wiki/physicsinformed/multifidelitytraininghistory/) | Training history for multi-fidelity PINN training. |
| [`MultiScalePINN<T>`](/docs/reference/wiki/physicsinformed/multiscalepinn/) | Implements a Multi-Scale Physics-Informed Neural Network for solving PDEs with multiple scales. |
| [`NavierStokesEquation<T>`](/docs/reference/wiki/physicsinformed/navierstokesequation/) | Represents the incompressible Navier-Stokes equations for fluid dynamics. |
| [`OperatorBenchmarkResult`](/docs/reference/wiki/physicsinformed/operatorbenchmarkresult/) |  |
| [`OperatorDataset2D`](/docs/reference/wiki/physicsinformed/operatordataset2d/) |  |
| [`PDEDerivatives<T>`](/docs/reference/wiki/physicsinformed/pdederivatives/) | Holds the derivatives needed for PDE computation. |
| [`PDEResidualGradient<T>`](/docs/reference/wiki/physicsinformed/pderesidualgradient/) | Holds gradients of a residual with respect to outputs and derivatives. |
| [`PINNGpuMemoryInfo`](/docs/reference/wiki/physicsinformed/pinngpumemoryinfo/) | GPU memory usage information for PINN training. |
| [`PdeBenchmarkResult`](/docs/reference/wiki/physicsinformed/pdebenchmarkresult/) |  |
| [`PhysicsInformedLoss<T>`](/docs/reference/wiki/physicsinformed/physicsinformedloss/) | Loss function for Physics-Informed Neural Networks (PINNs). |
| [`PhysicsInformedNeuralNetwork<T>`](/docs/reference/wiki/physicsinformed/physicsinformedneuralnetwork/) | Represents a Physics-Informed Neural Network (PINN) for solving PDEs. |
| [`PhysicsLossGradient<T>`](/docs/reference/wiki/physicsinformed/physicslossgradient/) | Holds loss and gradient information for physics-informed objectives. |
| [`PoissonEquation<T>`](/docs/reference/wiki/physicsinformed/poissonequation/) | Represents the Poisson Equation: ∇²u = f(x,y) |
| [`SchrodingerEquation<T>`](/docs/reference/wiki/physicsinformed/schrodingerequation/) | Represents the time-dependent Schrodinger equation for quantum mechanics. |
| [`SubdomainDefinition<T>`](/docs/reference/wiki/physicsinformed/subdomaindefinition/) | Defines a subdomain for domain decomposition. |
| [`SymbolicExpression<T>`](/docs/reference/wiki/physicsinformed/symbolicexpression/) | Represents a symbolic mathematical expression. |
| [`SymbolicPhysicsLearner<T>`](/docs/reference/wiki/physicsinformed/symbolicphysicslearner/) | Implements Symbolic Physics Learning for discovering interpretable equations from data. |
| [`TrainingHistory<T>`](/docs/reference/wiki/physicsinformed/traininghistory/) | Stores training history for analysis. |
| [`UniversalDifferentialEquation<T>`](/docs/reference/wiki/physicsinformed/universaldifferentialequation/) | Implements Universal Differential Equations (UDEs) - ODEs with neural network components. |
| [`VariationalPINN<T>`](/docs/reference/wiki/physicsinformed/variationalpinn/) | Implements Variational Physics-Informed Neural Networks (VPINNs). |
| [`WaveEquation<T>`](/docs/reference/wiki/physicsinformed/waveequation/) | Represents the Wave Equation: ∂²u/∂t² = c² * ∇²u |

## Layers (1)

| Type | Summary |
|:-----|:--------|
| [`FourierLayer<T>`](/docs/reference/wiki/physicsinformed/fourierlayer/) | Represents a single Fourier layer in the FNO. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`PDESpecificationBase<T>`](/docs/reference/wiki/physicsinformed/pdespecificationbase/) | Base class for all Partial Differential Equation (PDE) specifications. |

## Interfaces (12)

| Type | Summary |
|:-----|:--------|
| [`IBoundaryConditionGradient<T>`](/docs/reference/wiki/physicsinformed/iboundaryconditiongradient/) | Provides gradients for boundary residuals with respect to outputs and derivatives. |
| [`IBoundaryCondition<T>`](/docs/reference/wiki/physicsinformed/iboundarycondition/) | Defines boundary conditions for a PDE problem. |
| [`IDomainDecompositionTrainingHistory<T>`](/docs/reference/wiki/physicsinformed/idomaindecompositiontraininghistory/) | Extended training history interface for domain decomposition PINN training. |
| [`IGpuAcceleratedPINN<T>`](/docs/reference/wiki/physicsinformed/igpuacceleratedpinn/) | Interface for Physics-Informed Neural Networks that support GPU acceleration. |
| [`IInitialCondition<T>`](/docs/reference/wiki/physicsinformed/iinitialcondition/) | Defines initial conditions for time-dependent PDEs. |
| [`IInverseProblemGradient<T>`](/docs/reference/wiki/physicsinformed/iinverseproblemgradient/) | Provides gradient information for inverse problem parameters. |
| [`IInverseProblem<T>`](/docs/reference/wiki/physicsinformed/iinverseproblem/) | Defines the interface for inverse problems in physics-informed neural networks. |
| [`IMultiFidelityTrainingHistory<T>`](/docs/reference/wiki/physicsinformed/imultifidelitytraininghistory/) | Extended training history interface for multi-fidelity PINN training. |
| [`IMultiScalePDEGradient<T>`](/docs/reference/wiki/physicsinformed/imultiscalepdegradient/) | Provides gradient information for multi-scale PDE residuals. |
| [`IMultiScalePDE<T>`](/docs/reference/wiki/physicsinformed/imultiscalepde/) | Defines the interface for multi-scale Partial Differential Equations. |
| [`IPDEResidualGradient<T>`](/docs/reference/wiki/physicsinformed/ipderesidualgradient/) | Provides gradients of the PDE residual with respect to outputs and derivatives. |
| [`IPDESpecification<T>`](/docs/reference/wiki/physicsinformed/ipdespecification/) | Defines the interface for specifying Partial Differential Equations (PDEs) that can be used with Physics-Informed Neural Networks. |

## Enums (3)

| Type | Summary |
|:-----|:--------|
| [`InverseProblemRegularization`](/docs/reference/wiki/physicsinformed/inverseproblemregularization/) | Specifies the type of regularization for inverse problems. |
| [`OdeIntegrationMethod`](/docs/reference/wiki/physicsinformed/odeintegrationmethod/) | Supported integration schemes for UDE simulation. |
| [`SymbolicExpressionType`](/docs/reference/wiki/physicsinformed/symbolicexpressiontype/) | Types of symbolic expressions. |

## Options & Configuration (21)

| Type | Summary |
|:-----|:--------|
| [`AllenCahnBenchmarkOptions`](/docs/reference/wiki/physicsinformed/allencahnbenchmarkoptions/) |  |
| [`BurgersBenchmarkOptions`](/docs/reference/wiki/physicsinformed/burgersbenchmarkoptions/) |  |
| [`DarcyOperatorBenchmarkOptions`](/docs/reference/wiki/physicsinformed/darcyoperatorbenchmarkoptions/) |  |
| [`DeepOperatorNetworkOptions`](/docs/reference/wiki/physicsinformed/deepoperatornetworkoptions/) | Configuration options for the DeepOperatorNetwork. |
| [`DeepRitzMethodOptions`](/docs/reference/wiki/physicsinformed/deepritzmethodoptions/) | Configuration options for the DeepRitzMethod. |
| [`DomainDecompositionPINNOptions`](/docs/reference/wiki/physicsinformed/domaindecompositionpinnoptions/) | Configuration options for the Domain Decomposition PINN model. |
| [`FourierNeuralOperatorOptions`](/docs/reference/wiki/physicsinformed/fourierneuraloperatoroptions/) | Configuration options for the FourierNeuralOperator. |
| [`GpuPINNTrainingOptions`](/docs/reference/wiki/physicsinformed/gpupinntrainingoptions/) | Configuration options for GPU-accelerated PINN training. |
| [`GraphNeuralOperatorOptions`](/docs/reference/wiki/physicsinformed/graphneuraloperatoroptions/) | Configuration options for the GraphNeuralOperator. |
| [`HamiltonianNeuralNetworkOptions`](/docs/reference/wiki/physicsinformed/hamiltonianneuralnetworkoptions/) | Configuration options for the HamiltonianNeuralNetwork. |
| [`InverseProblemOptions<T>`](/docs/reference/wiki/physicsinformed/inverseproblemoptions/) | Configuration options for inverse problem PINN training. |
| [`LagrangianNeuralNetworkOptions`](/docs/reference/wiki/physicsinformed/lagrangianneuralnetworkoptions/) | Configuration options for the LagrangianNeuralNetwork. |
| [`MultiFidelityPINNOptions`](/docs/reference/wiki/physicsinformed/multifidelitypinnoptions/) | Configuration options for the Multi-Fidelity PINN model. |
| [`MultiScalePINNOptions`](/docs/reference/wiki/physicsinformed/multiscalepinnoptions/) | Configuration options for the Multi-Scale PINN model. |
| [`MultiScaleTrainingOptions<T>`](/docs/reference/wiki/physicsinformed/multiscaletrainingoptions/) | Configuration options for multi-scale PINN training. |
| [`OperatorBenchmarkOptions`](/docs/reference/wiki/physicsinformed/operatorbenchmarkoptions/) |  |
| [`PdeBenchmarkOptions`](/docs/reference/wiki/physicsinformed/pdebenchmarkoptions/) |  |
| [`PhysicsInformedNeuralNetworkOptions`](/docs/reference/wiki/physicsinformed/physicsinformedneuralnetworkoptions/) | Configuration options for the PhysicsInformedNeuralNetwork. |
| [`PoissonOperatorBenchmarkOptions`](/docs/reference/wiki/physicsinformed/poissonoperatorbenchmarkoptions/) |  |
| [`UniversalDifferentialEquationsOptions`](/docs/reference/wiki/physicsinformed/universaldifferentialequationsoptions/) | Configuration options for the UniversalDifferentialEquations model. |
| [`VariationalPINNOptions`](/docs/reference/wiki/physicsinformed/variationalpinnoptions/) | Configuration options for the VariationalPINN. |

## Helpers & Utilities (2)

| Type | Summary |
|:-----|:--------|
| [`OperatorBenchmarkSuite`](/docs/reference/wiki/physicsinformed/operatorbenchmarksuite/) |  |
| [`PdeBenchmarkSuite`](/docs/reference/wiki/physicsinformed/pdebenchmarksuite/) |  |

