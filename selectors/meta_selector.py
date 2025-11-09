"""
meta_selector.py

A theoretically-correct implementation of the Meta-Selector framework.

The meta-selector:
1. Observes actual outputs from the selector's chosen model
2. Validates decisions by measuring true behavioral distance
3. Manages escalation based on confidence, criticality, and session state
4. Implements expected value of information (EVI) for rational escalation

Author: Claude (based on theory by Karol Kowalczyk)
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import logging

from proper_selector import (
    BehavioralSelector,
    ModelAdapter,
    EmbeddingSpace,
    SelectionResult,
    SelectionMode
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Session State Management
# ============================================================================

@dataclass
class SessionState:
    """
    Tracks session-level context for meta-selection decisions.
    """
    # Query history
    query_history: List[str] = field(default_factory=list)
    query_embeddings: List[np.ndarray] = field(default_factory=list)
    
    # Confidence tracking
    confidence_history: List[float] = field(default_factory=list)
    distance_history: List[float] = field(default_factory=list)
    
    # Model usage tracking
    model_history: List[str] = field(default_factory=list)
    level_history: List[int] = field(default_factory=list)
    
    # Budget and criticality
    remaining_budget: float = 1.0
    cost_spent: float = 0.0
    is_critical: bool = False
    
    # Intent tracking
    current_intent_repetition: int = 0
    prev_query_embedding: Optional[np.ndarray] = None
    
    def add_interaction(
        self,
        query: str,
        query_embedding: np.ndarray,
        model_id: str,
        level: int,
        confidence: float,
        distance: float,
        cost: float,
    ):
        """Record a completed interaction."""
        self.query_history.append(query)
        self.query_embeddings.append(query_embedding)
        self.model_history.append(model_id)
        self.level_history.append(level)
        self.confidence_history.append(confidence)
        self.distance_history.append(distance)
        self.cost_spent += cost
        self.remaining_budget -= cost
    
    def update_intent_repetition(
        self,
        current_query_embedding: np.ndarray,
        similarity_threshold: float = 0.9
    ):
        """
        Detect if user is repeating similar queries (indicating dissatisfaction).
        """
        if self.prev_query_embedding is None:
            self.current_intent_repetition = 1
        else:
            # Compute cosine similarity
            dot = np.dot(current_query_embedding, self.prev_query_embedding)
            norm_curr = np.linalg.norm(current_query_embedding)
            norm_prev = np.linalg.norm(self.prev_query_embedding)
            similarity = dot / (norm_curr * norm_prev + 1e-12)
            
            if similarity >= similarity_threshold:
                self.current_intent_repetition += 1
                logger.info(f"Intent repetition detected: {self.current_intent_repetition}")
            else:
                self.current_intent_repetition = 1
        
        self.prev_query_embedding = current_query_embedding.copy()
    
    def get_recent_confidence_trend(self, window: int = 3) -> str:
        """Analyze recent confidence trend."""
        if len(self.confidence_history) < window:
            return "insufficient_data"
        
        recent = self.confidence_history[-window:]
        if all(c >= 0.75 for c in recent):
            return "consistently_high"
        elif all(c < 0.5 for c in recent):
            return "consistently_low"
        elif recent[-1] < recent[0]:
            return "declining"
        else:
            return "mixed"


# ============================================================================
# Escalation Decision Types
# ============================================================================

class EscalationReason(Enum):
    """Reasons for escalating to a higher-level model."""
    LOW_CONFIDENCE = "low_confidence"
    INTENT_REPETITION = "intent_repetition"
    CRITICAL_TASK = "critical_task"
    POSITIVE_EVI = "positive_evi"  # Expected Value of Information
    MANUAL = "manual"
    NO_ESCALATION = "no_escalation"


@dataclass
class EscalationDecision:
    """Result of a meta-selector decision."""
    should_escalate: bool
    reason: EscalationReason
    target_level: int
    target_model_id: Optional[str]
    confidence_in_decision: float
    expected_improvement: float
    expected_cost: float


# ============================================================================
# The Proper Meta-Selector
# ============================================================================

class MetaSelector:
    """
    A theoretically-correct meta-selector that validates selector decisions
    and manages hierarchical escalation.
    
    Key responsibilities:
    1. Validate selector decisions using actual behavioral distance
    2. Detect low-confidence situations requiring escalation
    3. Track session state and intent repetition
    4. Implement rational escalation via Expected Value of Information
    5. Manage computational budget
    """
    
    def __init__(
        self,
        selector: BehavioralSelector,
        embedding_space: EmbeddingSpace,
        model_adapters: Dict[str, ModelAdapter],
        confidence_threshold: float = 0.75,
        critical_confidence_threshold: float = 0.65,
        intent_repetition_threshold: int = 2,
        lambda_cost: float = 0.6,  # Cost weight in EVI calculation
        hysteresis_margin: float = 0.05,
        hysteresis_window: int = 3,
    ):
        self.selector = selector
        self.space = embedding_space
        self.models = model_adapters
        
        # Thresholds
        self.confidence_threshold = confidence_threshold
        self.critical_confidence_threshold = critical_confidence_threshold
        self.intent_repetition_threshold = intent_repetition_threshold
        self.lambda_cost = lambda_cost
        self.hysteresis_margin = hysteresis_margin
        self.hysteresis_window = hysteresis_window
        
        # Get available levels
        self.levels = sorted(set(adapter.level for adapter in model_adapters.values()))
        self.models_by_level = self._organize_models_by_level()
    
    def _organize_models_by_level(self) -> Dict[int, List[str]]:
        """Organize models by their hierarchy level."""
        by_level = {}
        for model_id, adapter in self.models.items():
            if adapter.level not in by_level:
                by_level[adapter.level] = []
            by_level[adapter.level].append(model_id)
        return by_level
    
    def _get_next_level_model(self, current_level: int) -> Optional[str]:
        """Get a model from the next higher level."""
        next_level = current_level + 1
        if next_level in self.models_by_level:
            # Choose the first model from next level (or implement smarter selection)
            return self.models_by_level[next_level][0]
        return None
    
    def _estimate_quality_improvement(
        self,
        current_distance: float,
        current_level: int,
        target_level: int,
    ) -> float:
        """
        Estimate expected improvement in behavioral distance from escalation.
        
        This is a heuristic based on the assumption that higher-level models
        generally produce outputs closer to the query intent.
        """
        if target_level <= current_level:
            return 0.0
        
        # Heuristic: each level improves distance by ~30%
        level_diff = target_level - current_level
        improvement_rate = 0.3
        
        estimated_improvement = current_distance * (1 - (1 - improvement_rate) ** level_diff)
        return estimated_improvement
    
    def _compute_evi(
        self,
        current_distance: float,
        current_level: int,
        target_level: int,
        query: str,
    ) -> float:
        """
        Compute Expected Value of Information for escalation.
        
        EVI = Expected_Improvement - λ * Expected_Cost
        
        Escalate if EVI > 0
        """
        # Estimate quality improvement
        improvement = self._estimate_quality_improvement(
            current_distance, current_level, target_level
        )
        
        # Estimate cost of target model
        target_model_id = self._get_next_level_model(current_level)
        if target_model_id is None:
            return -float('inf')  # No higher level available
        
        cost = self.models[target_model_id].estimate_cost(query)
        
        # EVI calculation
        evi = improvement - self.lambda_cost * cost
        
        logger.debug(
            f"EVI calculation: improvement={improvement:.4f}, "
            f"cost={cost:.6f}, EVI={evi:.4f}"
        )
        
        return evi
    
    def decide_escalation(
        self,
        query: str,
        current_result: SelectionResult,
        state: SessionState,
        force_critical: bool = False,
    ) -> EscalationDecision:
        """
        Main meta-selection decision: should we escalate to a higher-level model?
        
        This method validates the selector's choice and decides on escalation.
        """
        current_level = current_result.chosen_level
        confidence = current_result.confidence
        distance = current_result.behavioral_distance
        
        # Update session state
        query_embedding = self.space.embed_query(query)
        state.update_intent_repetition(query_embedding)
        state.add_interaction(
            query=query,
            query_embedding=query_embedding,
            model_id=current_result.chosen_model_id,
            level=current_level,
            confidence=confidence,
            distance=distance,
            cost=current_result.estimated_cost,
        )
        
        if force_critical:
            state.is_critical = True
        
        # Budget guard: can't escalate if no budget
        if state.remaining_budget <= 0:
            logger.warning("Budget exhausted, cannot escalate")
            return EscalationDecision(
                should_escalate=False,
                reason=EscalationReason.NO_ESCALATION,
                target_level=current_level,
                target_model_id=current_result.chosen_model_id,
                confidence_in_decision=1.0,
                expected_improvement=0.0,
                expected_cost=0.0,
            )
        
        # Check if we're already at max level
        next_model = self._get_next_level_model(current_level)
        if next_model is None:
            logger.info("Already at maximum level")
            return EscalationDecision(
                should_escalate=False,
                reason=EscalationReason.NO_ESCALATION,
                target_level=current_level,
                target_model_id=current_result.chosen_model_id,
                confidence_in_decision=confidence,
                expected_improvement=0.0,
                expected_cost=0.0,
            )
        
        # Escalation Rule 1: Critical task with low confidence
        if state.is_critical and confidence < self.critical_confidence_threshold:
            logger.warning(
                f"Critical task with low confidence ({confidence:.3f}), escalating"
            )
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.CRITICAL_TASK,
                target_level=current_level + 1,
                target_model_id=next_model,
                confidence_in_decision=0.9,
                expected_improvement=self._estimate_quality_improvement(
                    distance, current_level, current_level + 1
                ),
                expected_cost=self.models[next_model].estimate_cost(query),
            )
        
        # Escalation Rule 2: Intent repetition with low confidence
        if (state.current_intent_repetition >= self.intent_repetition_threshold and
            confidence < self.confidence_threshold):
            logger.warning(
                f"Intent repetition ({state.current_intent_repetition}) "
                f"with low confidence ({confidence:.3f}), escalating"
            )
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.INTENT_REPETITION,
                target_level=current_level + 1,
                target_model_id=next_model,
                confidence_in_decision=0.85,
                expected_improvement=self._estimate_quality_improvement(
                    distance, current_level, current_level + 1
                ),
                expected_cost=self.models[next_model].estimate_cost(query),
            )
        
        # Escalation Rule 3: Expected Value of Information
        evi = self._compute_evi(distance, current_level, current_level + 1, query)
        if evi > 0:
            logger.info(f"Positive EVI ({evi:.4f}), escalating")
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.POSITIVE_EVI,
                target_level=current_level + 1,
                target_model_id=next_model,
                confidence_in_decision=0.8,
                expected_improvement=self._estimate_quality_improvement(
                    distance, current_level, current_level + 1
                ),
                expected_cost=self.models[next_model].estimate_cost(query),
            )
        
        # Escalation Rule 4: Low confidence (even without repetition)
        if confidence < self.confidence_threshold:
            logger.info(f"Low confidence ({confidence:.3f}), escalating")
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                target_level=current_level + 1,
                target_model_id=next_model,
                confidence_in_decision=0.75,
                expected_improvement=self._estimate_quality_improvement(
                    distance, current_level, current_level + 1
                ),
                expected_cost=self.models[next_model].estimate_cost(query),
            )
        
        # Hysteresis: check if we should de-escalate (stay at current level)
        trend = state.get_recent_confidence_trend(self.hysteresis_window)
        if trend == "consistently_high" and current_level > 0:
            logger.info("High confidence sustained, could consider de-escalation")
        
        # Default: no escalation needed
        return EscalationDecision(
            should_escalate=False,
            reason=EscalationReason.NO_ESCALATION,
            target_level=current_level,
            target_model_id=current_result.chosen_model_id,
            confidence_in_decision=confidence,
            expected_improvement=0.0,
            expected_cost=0.0,
        )
    
    def process_query(
        self,
        query: str,
        state: SessionState,
        mode: SelectionMode = SelectionMode.HYBRID,
        force_critical: bool = False,
        max_escalations: int = 2,
    ) -> Tuple[SelectionResult, List[EscalationDecision]]:
        """
        Complete query processing with meta-selector oversight.
        
        Args:
            query: User query
            state: Session state
            mode: Selector mode
            force_critical: Mark as critical task
            max_escalations: Maximum number of escalations allowed
        
        Returns:
            (final_result, escalation_history)
        """
        escalation_history = []
        
        # Initial selection
        result = self.selector.select(query, mode=mode)
        
        # Meta-selector evaluation
        for escalation_round in range(max_escalations):
            decision = self.decide_escalation(query, result, state, force_critical)
            escalation_history.append(decision)
            
            if not decision.should_escalate:
                logger.info("Meta-selector: no escalation needed")
                break
            
            logger.info(
                f"Meta-selector: escalating from {result.chosen_model_id} "
                f"to level {decision.target_level} (reason: {decision.reason.value})"
            )
            
            # Perform escalation: actually call the higher-level model
            query_embedding = self.space.embed_query(query)
            target_adapter = self.models[decision.target_model_id]
            
            # ACTUALLY CALL the escalated model
            output = target_adapter.infer(query)
            output_embedding = self.space.embed_output(
                decision.target_model_id, 
                output
            )
            distance = self.space.distance(query_embedding, output_embedding)
            cost = target_adapter.estimate_cost(query)
            confidence = self.selector._compute_confidence(distance)
            
            # Update prototypes with escalated result
            self.selector.prototypes.add_observation(
                decision.target_model_id,
                output_embedding,
                output,
                self.space.distance
            )
            
            # Create new result
            result = SelectionResult(
                chosen_model_id=decision.target_model_id,
                chosen_level=decision.target_level,
                actual_output=output,
                output_embedding=output_embedding,
                behavioral_distance=distance,
                estimated_cost=cost,
                confidence=confidence,
                alternatives_considered=[(decision.target_model_id, distance, cost)]
            )
            
            # Check if escalation improved things
            if confidence >= self.confidence_threshold:
                logger.info(
                    f"Escalation successful: confidence improved to {confidence:.3f}"
                )
                break
        
        return result, escalation_history


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    from proper_selector import (
        SentenceTransformerSpace,
        PrototypeBank,
        create_mock_model
    )
    
    print("=" * 80)
    print("Proper Meta-Selector - Demonstration")
    print("=" * 80)
    
    # Setup
    space = SentenceTransformerSpace()
    
    models = {
        "gpt-small": ModelAdapter(
            model_id="gpt-small",
            inference_fn=create_mock_model("gpt-small", quality=0.3),
            cost_per_token=0.0001,
            level=0,
        ),
        "gpt-medium": ModelAdapter(
            model_id="gpt-medium",
            inference_fn=create_mock_model("gpt-medium", quality=0.6),
            cost_per_token=0.0005,
            level=1,
        ),
        "gpt-large": ModelAdapter(
            model_id="gpt-large",
            inference_fn=create_mock_model("gpt-large", quality=0.9),
            cost_per_token=0.002,
            level=2,
        ),
    }
    
    prototype_bank = PrototypeBank()
    selector = BehavioralSelector(
        embedding_space=space,
        model_adapters=models,
        prototype_bank=prototype_bank,
        exploration_rate=0.2,
    )
    
    meta_selector = MetaSelector(
        selector=selector,
        embedding_space=space,
        model_adapters=models,
        confidence_threshold=0.75,
    )
    
    # Session
    session = SessionState(remaining_budget=1.0)
    
    # Test scenarios
    scenarios = [
        ("What is 2+2?", False),  # Simple, should use small model
        ("Explain quantum entanglement.", False),  # Complex, may escalate
        ("Explain quantum entanglement in detail.", True),  # Critical, will escalate
        ("Tell me more about quantum physics.", False),  # Repeat, may escalate
    ]
    
    print("\n" + "=" * 80)
    print("Running meta-selector with escalation management")
    print("=" * 80 + "\n")
    
    for i, (query, is_critical) in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"Scenario {i}: {query}")
        print(f"Critical: {is_critical}")
        print(f"{'='*80}")
        
        result, escalations = meta_selector.process_query(
            query=query,
            state=session,
            mode=SelectionMode.HYBRID,
            force_critical=is_critical,
            max_escalations=2,
        )
        
        print(f"\n✓ Final Model: {result.chosen_model_id} (Level {result.chosen_level})")
        print(f"  Behavioral Distance: {result.behavioral_distance:.4f}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Cost: ${result.estimated_cost:.6f}")
        print(f"\n  Output: {result.actual_output[:150]}...")
        
        print(f"\n  Escalation History ({len(escalations)} decisions):")
        for j, esc in enumerate(escalations):
            print(f"    {j+1}. {'ESCALATE' if esc.should_escalate else 'STAY'} "
                  f"- Reason: {esc.reason.value}")
        
        print(f"\n  Session State:")
        print(f"    Budget remaining: ${session.remaining_budget:.6f}")
        print(f"    Cost spent: ${session.cost_spent:.6f}")
        print(f"    Intent repetition: {session.current_intent_repetition}")
        print(f"    Confidence trend: {session.get_recent_confidence_trend()}")
    
    print("\n" + "=" * 80)
    print("Meta-Selector Session Complete")
    print("=" * 80)
    print(f"Total queries: {len(session.query_history)}")
    print(f"Total cost: ${session.cost_spent:.6f}")
    print(f"Budget remaining: ${session.remaining_budget:.6f}")
    print(f"Model usage:")
    for model_id in set(session.model_history):
        count = session.model_history.count(model_id)
        print(f"  {model_id}: {count} times")
