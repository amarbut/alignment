"""
Expert Routing Intervention Module

This module provides clean, production-ready expert routing interventions
for MoE models. It abstracts away model-specific details using ModelCard
and supports multiple intervention strategies.

Main components:
- ExpertInterventionConfig: Configuration for routing interventions
- apply_expert_interventions: Apply bias modifications to force/suppress experts
- remove_expert_interventions: Restore original router biases

Pre-configured intervention builders:
- get_select_experts_refusal_induction_config: Threshold-based refusal induction
- get_select_experts_response_induction_config: Threshold-based response induction
- get_top_experts_refusal_config: Per-layer top expert refusal induction
- get_top_experts_jailbreak_config: Per-layer top expert jailbreak induction
"""

from expert_routing.expert_routing_hooks import (
    ExpertInterventionConfig,
    apply_expert_interventions,
    remove_expert_interventions,
    print_intervention_summary,
    calibrate_router_ranges,
    get_select_experts_refusal_induction_config,
    get_select_experts_response_induction_config,
    get_top_experts_refusal_config,
    get_top_experts_jailbreak_config,
    load_expert_diffs,
)

__all__ = [
    'ExpertInterventionConfig',
    'apply_expert_interventions',
    'remove_expert_interventions',
    'print_intervention_summary',
    'calibrate_router_ranges',
    'get_select_experts_refusal_induction_config',
    'get_select_experts_response_induction_config',
    'get_top_experts_refusal_config',
    'get_top_experts_jailbreak_config',
    'load_expert_diffs',
]
