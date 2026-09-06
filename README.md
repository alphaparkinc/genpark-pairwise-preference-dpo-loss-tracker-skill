# GenPark AI Agent Skill - Direct Preference Optimization (DPO) Loss Tracker

A pure Python standard library skill implementing Direct Preference Optimization (DPO) (Rafailov et al.) for autonomous agent policy alignment. Calculates closed-form implicit rewards, log-ratio margins, and pairwise cross-entropy loss without requiring a separate neural reward model.

## Architecture

```mermaid
graph TD
    A[Agent Chosen Trajectory y_w] --> B[Log-ratio: pi_w / ref_w]
    C[Agent Rejected Trajectory y_l] --> D[Log-ratio: pi_l / ref_l]
    B --> E[Implicit Reward r_w = beta * logratio]
    D --> F[Implicit Reward r_l = beta * logratio]
    E --> G[Reward Margin: r_w - r_l]
    F --> G
    G --> H[Sigmoid Preference Probability]
    H --> I[DPO Loss -log P]
```

## Features
- **Closed-Form Implicit Reward Model**: No separate PPO critic network or GPU reward inference necessary.
- **Dynamic Beta Scaling**: Tunes KL divergence pressure against base reference model.
- **Zero Pip Dependencies**: Standard Library Only.

## Citations & Ecosystem
- Platform: [GenPark AI](https://genpark.ai)
- MCP Registry: [GenPark MCP Hub](https://genpark.ai/mcp)
