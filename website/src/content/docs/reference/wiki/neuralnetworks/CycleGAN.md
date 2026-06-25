---
title: "CycleGAN<T>"
description: "Represents a CycleGAN for unpaired image-to-image translation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a CycleGAN for unpaired image-to-image translation.

## For Beginners

CycleGAN translates images without matched pairs.

Key innovation:

- Doesn't need paired training data
- Learns from two separate collections of images
- Example: Photos of horses + Photos of zebras → can convert horses to zebras

How it works:

- Two generators: G (A→B) and F (B→A)
- Two discriminators: D_A and D_B
- Cycle consistency: G(F(B)) ≈ B and F(G(A)) ≈ A
- This prevents mode collapse and maintains content

Applications:

- Style transfer (Monet → Photo, Photo → Monet)
- Season transfer (Summer → Winter)
- Object transfiguration (Horse → Zebra)
- Domain adaptation

Reference: Zhu et al., "Unpaired Image-to-Image Translation using
Cycle-Consistent Adversarial Networks" (2017)

## How It Works

CycleGAN enables image-to-image translation without paired training data:

- Uses two generators (A→B and B→A) and two discriminators
- Enforces cycle consistency: A→B→A should equal A
- Works without paired examples (e.g., can learn horses→zebras from separate collections)
- Uses adversarial loss + cycle consistency loss + identity loss

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CycleGAN` | Initializes a new instance of the `CycleGAN` class with the specified architecture and training parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiscriminatorA` | Discriminator for domain A. |
| `DiscriminatorB` | Discriminator for domain B. |
| `GeneratorAtoB` | Generator A→B. |
| `GeneratorBtoA` | Generator B→A. |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateCycleGANArchitecture(NeuralNetworkArchitecture<>,InputType)` | Creates the combined CycleGAN architecture with correct dimension handling. |
| `CreateNetworkForInputType(NeuralNetworkArchitecture<>,InputType)` | Creates the appropriate neural network type based on the input type. |
| `CreateNewInstance` | Creates a new instance of the CycleGAN with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes CycleGAN-specific data from a binary reader. |
| `ForwardForTraining(Tensor<>)` | Defines the CycleGAN forward graph for tape-based training. |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `GetParameterChunks` |  |
| `GetParameters` |  |
| `ResetOptimizerState` | Resets the state of all optimizers to their initial values. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes CycleGAN-specific data to a binary writer. |
| `TrainStep(Tensor<>,Tensor<>)` | Performs one training step for CycleGAN. |
| `TranslateAtoB(Tensor<>)` | Translates image from domain A to domain B. |
| `TranslateBtoA(Tensor<>)` | Translates image from domain B to domain A. |
| `UpdateDiscriminatorAParameters` | Updates the parameters of discriminator A using its optimizer. |
| `UpdateDiscriminatorBParameters` | Updates the parameters of discriminator B using its optimizer. |
| `UpdateGeneratorAtoBParameters` | Updates the parameters of the generator A→B network using its optimizer. |
| `UpdateGeneratorBtoAParameters` | Updates the parameters of the generator B→A network using its optimizer. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all networks in the CycleGAN. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cycleConsistencyLambda` | Coefficient for cycle consistency loss. |
| `_discriminatorAOptimizer` | The optimizer used for training discriminator A. |
| `_discriminatorBOptimizer` | The optimizer used for training discriminator B. |
| `_generatorAtoBOptimizer` | The optimizer used for training generator A→B. |
| `_generatorBtoAOptimizer` | The optimizer used for training generator B→A. |
| `_identityLambda` | Coefficient for identity loss. |

