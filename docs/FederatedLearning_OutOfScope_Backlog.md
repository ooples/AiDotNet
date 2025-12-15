# Federated Learning Out-of-Scope Backlog (Post-v1)

This document lists federated learning topics that are intentionally **out of scope** for the v1 frozen snapshot defined in `docs/FederatedLearning_V1_FrozenScope.md`.

Each item here should be tracked as a dedicated GitHub issue (or issue series) after the v1 scope is accepted. If an issue number exists, add it beside the item.

## Cryptography / Privacy Beyond v1

- Fully verifiable federated learning (e.g., ZK proofs for update correctness)
- Advanced MPC for arbitrary training beyond secure aggregation
- Private set intersection (PSI) based entity alignment for vertical FL

## Advanced Federated Data Regimes

- Full vertical federated learning (VFL) training stack with production-grade protocols
- Federated learning over graph-structured data with specialized secure protocols

## Mobile Device TEEs (Full on-device confidential training)

- Running full training/inference inside iOS Secure Enclave / Android TrustZone environments
  - v1 will use an **attested gateway runtime** for mobile enterprise Option C instead

## Emerging / Fast-Moving Research

- New robust aggregation algorithms not included in the v1 frozen list
- New privacy accounting methods beyond RDP/basic composition
- New personalization/meta-learning breakthroughs beyond the v1 list

