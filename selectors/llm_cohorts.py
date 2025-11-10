"""
llm_cohorts.py - Sleep-Wake Orchestration in Hierarchical LLM Cohorts

Implementation of continuous self-optimization in large language model ensembles
through alternating sleep-wake cycles. Inspired by biological sleep dynamics and
grounded in the theory of adjoint projections on computational hierarchies.

Core Concepts:
- 1/3 VRAM budget for training (sleep), 2/3 for inference (wake)
- A "cohort" is a set of models of the same size class (e.g., 3B, 8B, 13B)
- An "envelope" reserves resources: exactly one sleeper (training) + its workers
- Sleeper fine-tunes adapters on "gap cells" (areas where small models fail but
  upper-level models succeed)
- Canary → rollout → working with safety (hysteresis/stop-loss)

Theoretical Foundation:
- Sleep phase corresponds to projection P_n (updating latent representations)
- Wake phase corresponds to collapse C_n (producing concrete outputs)
- The system oscillates between dual modes maintaining bounded computational coherence

Author: Karol Kowalczyk
Date: November 2025
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import time
import math
import numpy as np
from collections import defaultdict


# =============================================================================
# 0) CONFIGURATION AND CONSTANTS
# =============================================================================

# Time slots for training by model size (minutes)
DEFAULT_SLOT_MINUTES: Dict[str, int] = {
    "3B": 30,
    "8B": 60,
    "13B": 90,
    "30B": 150,
    "70B": 360,
    "175B": 720,
    "GPT5": 1080
}

# LoRA rank configuration by model size
DEFAULT_LORA_RANK: Dict[str, int] = {
    "3B": 16,
    "8B": 16,
    "13B": 16,
    "30B": 16,
    "70B": 16,
    "175B": 16,
    "GPT5": 16
}

# Exponential moving average decay for prototypes
EMA_PROTO: float = 0.99              # Local prototype updates
EMA_PROTO_GLOBAL: float = 0.995      # Global prototype consolidation

# Knowledge distillation temperature
KD_TEMPERATURE: float = 1.7

# Resource allocation fractions
TRAIN_BUDGET_FRACTION: float = 1.0 / 3.0  # 1/3 for training
INFER_BUDGET_FRACTION: float = 2.0 / 3.0  # 2/3 for inference

# Rollout parameters
CANARY_INITIAL_TRAFFIC: float = 0.02  # Start with 2% traffic
CANARY_GROWTH_FACTOR: float = 2.0     # Double traffic on success
CANARY_MAX_TRAFFIC: float = 0.5       # Cap at 50% traffic
STABILITY_REQUIRED_SLOTS: int = 2     # Slots before full promotion


# =============================================================================
# 1) CORE DATA STRUCTURES
# =============================================================================

class ModelState(str, Enum):
    """States in the sleep-wake lifecycle."""
    WORKING = "working"   # Handles production traffic
    SLEEPING = "sleeping" # In fine-tuning slot
    ROLLOUT = "rollout"   # Canary/expansion after training


@dataclass
class BehavioralPrototype:
    """
    A behavioral prototype representing typical model output in embedding space.
    Maintained via exponential moving average (EMA) for stability.
    """
    id: str
    embedding: np.ndarray
    count: int = 0
    variance: float = 0.0
    exemplar_output: Optional[str] = None
    
    def update(self, new_embedding: np.ndarray, ema_decay: float = EMA_PROTO) -> None:
        """Update prototype with new sample using EMA."""
        if self.count == 0:
            self.embedding = new_embedding.copy()
        else:
            self.embedding = (
                ema_decay * self.embedding +
                (1.0 - ema_decay) * new_embedding
            )
        self.count += 1


@dataclass
class Prototypes:
    """
    Collection of K behavioral prototypes maintained via EMA.
    Represents the behavioral manifold of a model.
    """
    vectors: List[BehavioralPrototype] = field(default_factory=list)
    max_prototypes: int = 10
    cluster_threshold: float = 0.3
    
    def update(
        self,
        samples: List[Tuple[np.ndarray, str]],
        ema: float = EMA_PROTO_GLOBAL
    ) -> None:
        """
        Update prototypes with high-confidence samples.
        
        Args:
            samples: List of (embedding, output_text) tuples
            ema: Exponential moving average decay factor
        """
        for embedding, output_text in samples:
            self._update_single(embedding, output_text, ema)
    
    def _update_single(
        self,
        embedding: np.ndarray,
        output_text: str,
        ema: float
    ) -> None:
        """Update prototypes with a single sample."""
        if not self.vectors:
            # Create first prototype
            self.vectors.append(BehavioralPrototype(
                id=f"proto_0",
                embedding=embedding.copy(),
                count=1,
                exemplar_output=output_text
            ))
            return
        
        # Find nearest prototype
        distances = [
            np.linalg.norm(embedding - p.embedding)
            for p in self.vectors
        ]
        min_idx = int(np.argmin(distances))
        min_dist = distances[min_idx]
        
        # Create new prototype if sufficiently different
        if min_dist > self.cluster_threshold and len(self.vectors) < self.max_prototypes:
            self.vectors.append(BehavioralPrototype(
                id=f"proto_{len(self.vectors)}",
                embedding=embedding.copy(),
                count=1,
                exemplar_output=output_text
            ))
        else:
            # Update nearest prototype
            self.vectors[min_idx].update(embedding, ema)


@dataclass
class Cell:
    """
    A region in the embedding space representing a cluster of similar queries.
    The "gap" represents where cohort models fail but higher-level models succeed.
    """
    id: str
    centroid: np.ndarray
    gap_weight: float  # G(C_i): demand * (1 - cohort_cover) * solvable_up
    query_count: int = 0
    cohort_success_rate: float = 0.0
    upper_level_success_rate: float = 0.0
    
    def compute_gap_weight(
        self,
        demand: float,
        cohort_cover: float,
        solvable_up: float
    ) -> float:
        """
        Compute gap weight: G(z) = D_Q(z) * (1 - cover(z)) * solvable_up(z)
        
        Args:
            demand: Query density D_Q(z) in this region
            cohort_cover: Local success rate of cohort peers
            solvable_up: Probability upper-tier models solved queries here
            
        Returns:
            Gap weight score
        """
        self.gap_weight = demand * (1.0 - cohort_cover) * solvable_up
        return self.gap_weight


@dataclass
class GapIndex:
    """
    Index of gap cells identified from production logs.
    Tracks regions where models underperform relative to potential.
    """
    cells: List[Cell] = field(default_factory=list)
    embedding_dim: int = 768
    max_cells: int = 100
    
    def update(self, logs_window: List[Dict[str, Any]]) -> None:
        """
        Refresh gap cells from recent logs; compute gap weights.
        
        This involves:
        1. Clustering queries in embedding space
        2. Computing success rates per cluster
        3. Identifying gaps where cohort fails but upper levels succeed
        4. Prioritizing by query demand
        
        Args:
            logs_window: Recent routing/quality logs
        """
        # In production: implement clustering algorithm (k-means, HDBSCAN, etc.)
        # For now: placeholder that would extract query embeddings,
        # cluster them, and compute gap weights
        pass
    
    def top_cells(self, M: int) -> List[Cell]:
        """
        Return top-M gap cells by gap_weight.
        
        Args:
            M: Number of top cells to return
            
        Returns:
            List of cells sorted by gap weight (descending)
        """
        return sorted(self.cells, key=lambda c: c.gap_weight, reverse=True)[:M]
    
    def weight(self, cell_id: str) -> float:
        """Get gap weight for a specific cell."""
        for c in self.cells:
            if c.id == cell_id:
                return c.gap_weight
        return 0.0


@dataclass
class Model:
    """
    A language model in the cohort with state tracking and resource accounting.
    """
    id: str
    size_class: str  # '3B', '8B', '13B', '30B', '70B', '175B', 'GPT5'
    state: ModelState = ModelState.WORKING
    
    # Resource requirements (GB)
    vram_inference: float = 0.0
    vram_training: float = 0.0
    
    # Behavioral characteristics
    prototypes: Prototypes = field(default_factory=Prototypes)
    adapters: Dict[str, Any] = field(default_factory=dict)  # cell_id -> adapter
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Training configuration
    target_cells: List[Cell] = field(default_factory=list)
    traffic_share: float = 0.0  # Fraction of traffic during rollout
    last_sleep_timestamp: float = 0.0
    
    # Quality tracking
    recent_success_rate: float = 0.0
    cost_per_query: float = 0.0
    
    def gap_misalignment_score(
        self,
        cohort: 'Cohort',
        logs: List[Dict[str, Any]]
    ) -> float:
        """
        Compute misalignment between this model's performance and gap cells.
        Higher score indicates this model would benefit most from targeted training.
        
        Args:
            cohort: Parent cohort
            logs: Recent performance logs
            
        Returns:
            Misalignment score (0-1, higher is worse)
        """
        # Aggregate performance on high-gap regions
        total_weight = 0.0
        weighted_failure = 0.0
        
        for cell in cohort.gap_index.cells:
            # Estimate model's performance on this cell from logs
            success_prob = self._estimate_success_on_cell(cell, logs)
            weighted_failure += cell.gap_weight * (1.0 - success_prob)
            total_weight += cell.gap_weight
        
        return weighted_failure / max(total_weight, 1e-6)
    
    def _estimate_success_on_cell(
        self,
        cell: Cell,
        logs: List[Dict[str, Any]]
    ) -> float:
        """Estimate success probability on a cell from logs."""
        # Placeholder: would analyze logs for queries in this cell's region
        return 0.5


@dataclass
class Selector:
    """
    Behavioral selector using prototypes for routing decisions.
    Routes queries to appropriate models based on behavioral distance.
    """
    embedding_fn: Optional[Callable[[str], np.ndarray]] = None
    
    def route(self, query: str, cohort: 'Cohort') -> Model:
        """
        Select best model for query based on behavioral distance.
        
        Args:
            query: Input text
            cohort: Cohort of models to choose from
            
        Returns:
            Selected model
        """
        if not self.embedding_fn:
            # Fallback: round-robin or random
            working_models = [m for m in cohort.models if m.state == ModelState.WORKING]
            return working_models[0] if working_models else cohort.models[0]
        
        # Embed query
        query_embedding = self.embedding_fn(query)
        
        # Find model with minimum behavioral distance
        best_model = None
        min_distance = float('inf')
        
        for model in cohort.models:
            if model.state != ModelState.WORKING:
                continue
            
            # Compute distance to model's prototypes
            distances = [
                np.linalg.norm(query_embedding - p.embedding)
                for p in model.prototypes.vectors
            ]
            
            if distances:
                model_distance = min(distances)
                if model_distance < min_distance:
                    min_distance = model_distance
                    best_model = model
        
        return best_model if best_model else cohort.models[0]


@dataclass
class MetaSelector:
    """
    Meta-selector tracking EVI (Expected Value of Improvement) and escalations.
    Monitors confidence and manages hierarchical escalation decisions.
    """
    escalation_threshold: float = 0.4
    confidence_history: List[float] = field(default_factory=list)
    
    def evi_gain(self, query: str, current_level: str, next_level: str) -> float:
        """
        Compute expected value of information from escalating.
        
        EVI(x) = E[Q_{n+1}(x) - Q_n(x)] - λ(C_{n+1} - C_n)
        
        Args:
            query: Input text
            current_level: Current model level
            next_level: Proposed escalation level
            
        Returns:
            Expected gain from escalation
        """
        # Placeholder: would estimate quality improvement vs cost increase
        return 1.0
    
    def should_escalate(
        self,
        query: str,
        confidence: float,
        current_model: Model
    ) -> bool:
        """
        Decide whether to escalate to higher-capacity model.
        
        Args:
            query: Input text
            confidence: Confidence in current model's output
            current_model: Model that produced the output
            
        Returns:
            True if should escalate
        """
        return confidence < self.escalation_threshold


@dataclass
class Cohort:
    """
    A cohort of models sharing comparable computational cost.
    Manages rotation, gap identification, and routing.
    """
    size_class: str  # '3B', '8B', '13B', etc.
    models: List[Model] = field(default_factory=list)
    gap_index: GapIndex = field(default_factory=GapIndex)
    router: Selector = field(default_factory=Selector)
    meta: MetaSelector = field(default_factory=MetaSelector)
    
    # Performance tracking
    total_queries: int = 0
    escalations: int = 0
    avg_confidence: float = 0.0
    cost_per_query: float = 0.0


@dataclass
class ResourcePools:
    """
    Global resource pools for training and inference.
    Maintains 1/3 training VRAM, 2/3 inference VRAM allocation.
    """
    train_free: float  # Available training VRAM (GB)
    infer_free: float  # Available inference VRAM (GB)
    
    @classmethod
    def from_total(cls, total_vram_gb: float) -> 'ResourcePools':
        """
        Create resource pools from total VRAM budget.
        
        Args:
            total_vram_gb: Total available VRAM
            
        Returns:
            ResourcePools with 1/3 training, 2/3 inference allocation
        """
        return cls(
            train_free=total_vram_gb * TRAIN_BUDGET_FRACTION,
            infer_free=total_vram_gb * INFER_BUDGET_FRACTION
        )


@dataclass
class Envelope:
    """
    Resource envelope managing one sleeper and its workers.
    An envelope reserves: exactly one sleeper (training) + working models.
    """
    id: str
    cohort: Cohort
    
    # Allocated models
    sleeper: Optional[Model] = None
    workers: List[Model] = field(default_factory=list)
    
    # Resource budgets (GB)
    train_budget_gb: float = 0.0  # = m_train[cohort.size_class]
    infer_budget_gb: float = 0.0  # = 2 * train_budget_gb
    
    # State tracking
    state: str = "idle"  # 'idle' | 'sleeping' | 'rollout' | 'working'
    target_cells: List[Cell] = field(default_factory=list)
    slot_ends_at: float = 0.0
    
    # Performance metrics
    rollout_metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# 2) UTILITY FUNCTIONS
# =============================================================================

def now_timestamp() -> float:
    """Get current timestamp."""
    return time.time()


def normalize(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize value to [min_val, max_val] range, handling inf/nan.
    
    Args:
        x: Value to normalize
        min_val: Minimum output value
        max_val: Maximum output value
        
    Returns:
        Normalized value
    """
    if math.isinf(x) or math.isnan(x):
        return min_val
    return max(min_val, min(max_val, x))


def slot_minutes(cohort: Cohort) -> int:
    """
    Get training slot duration for cohort.
    
    Args:
        cohort: Model cohort
        
    Returns:
        Slot duration in minutes
    """
    return DEFAULT_SLOT_MINUTES.get(cohort.size_class, 60)


def steps_for_slot(slot_minutes: int, steps_per_minute: int = 60) -> int:
    """
    Translate slot duration into optimizer steps.
    
    Args:
        slot_minutes: Duration in minutes
        steps_per_minute: Optimization steps per minute
        
    Returns:
        Total number of steps
    """
    return slot_minutes * steps_per_minute


def pick_learning_rate(size_class: str) -> float:
    """
    Select appropriate learning rate for model size.
    
    Args:
        size_class: Model size class
        
    Returns:
        Learning rate
    """
    lr_map = {
        "3B": 5e-5,
        "8B": 3e-5,
        "13B": 2e-5,
        "30B": 1e-5,
        "70B": 5e-6,
        "175B": 2e-6,
        "GPT5": 1e-6
    }
    return lr_map.get(size_class, 1e-5)


def lora_rank_for_class(cohort: Cohort) -> int:
    """
    Get LoRA rank for cohort size class.
    
    Args:
        cohort: Model cohort
        
    Returns:
        LoRA rank
    """
    return DEFAULT_LORA_RANK.get(cohort.size_class, 16)


def quality_cost_ratio(model: Model) -> float:
    """
    Compute quality-to-cost ratio for model selection.
    
    Args:
        model: Model to evaluate
        
    Returns:
        Quality/cost ratio
    """
    quality = model.recent_success_rate
    cost = max(model.cost_per_query, 1e-6)
    return quality / cost


# =============================================================================
# 3) DATA COLLECTION AND EVALUATION
# =============================================================================

def collect_logs(cohort: Cohort, window_hours: int = 4) -> List[Dict[str, Any]]:
    """
    Pull a sliding window of routing/quality logs.
    
    Args:
        cohort: Model cohort
        window_hours: Time window in hours
        
    Returns:
        List of log entries
    """
    # Placeholder: would query production logging system
    # Return format: [{"query": str, "model_id": str, "success": bool, 
    #                  "embedding": np.ndarray, "timestamp": float}, ...]
    return []


def sample_from_cell(cell: Cell, n_min: int = 100) -> List[Any]:
    """
    Sample recent examples from a gap cell.
    
    Args:
        cell: Gap cell to sample from
        n_min: Minimum number of samples
        
    Returns:
        List of query examples
    """
    # Placeholder: would query example database filtered by cell region
    return []


def high_level_teacher(query: Any) -> Tuple[Any, float]:
    """
    Get output and confidence from upper-level teacher model.
    
    Args:
        query: Input query
        
    Returns:
        (teacher_output, confidence) tuple
    """
    # Placeholder: would call high-capacity model
    return None, 0.9


def confidence_weight(confidence: float) -> float:
    """
    Convert confidence score to training weight.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Training weight
    """
    return max(0.0, confidence)


def evaluate_on_cells(
    model: Model,
    cells: List[Cell],
    horizon: str = "short"
) -> Dict[str, Any]:
    """
    Evaluate model performance on target cells.
    
    Args:
        model: Model to evaluate
        cells: Target cells
        horizon: Evaluation horizon ("short" or "long")
        
    Returns:
        Evaluation statistics
    """
    # Placeholder: would run evaluation queries from cells
    return {
        "improves": True,
        "regress_outside": False,
        "success_rate": 0.8,
        "confidence": 0.85
    }


def improves(stats: Dict[str, Any]) -> bool:
    """Check if evaluation shows improvement."""
    return bool(stats.get("improves", False))


def no_regress_outside(model: Model, stats: Dict[str, Any]) -> bool:
    """Check if model doesn't regress on non-target queries."""
    return not bool(stats.get("regress_outside", False))


# =============================================================================
# 4) PROTOTYPE AND METRIC MANAGEMENT
# =============================================================================

def update_prototypes(
    model: Model,
    cell: Cell,
    samples: List[Tuple[np.ndarray, str]],
    ema: float = EMA_PROTO
) -> None:
    """
    Update model prototypes with cell-specific samples.
    
    Args:
        model: Model to update
        cell: Target cell
        samples: List of (embedding, output) tuples
        ema: EMA decay factor
    """
    model.prototypes.update(samples, ema)


def refresh_metrics_and_prototypes(cohort: Cohort) -> None:
    """
    Refresh global prototypes and publish performance metrics.
    
    Args:
        cohort: Cohort to update
    """
    # Gather high-confidence samples from recent queries
    high_conf_samples: List[Tuple[np.ndarray, str]] = []
    # Placeholder: would collect from production logs
    
    # Update prototypes for all models
    for model in cohort.models:
        model.prototypes.update(high_conf_samples, ema=EMA_PROTO_GLOBAL)
    
    # Publish metrics
    publish_metrics(cohort, {
        "escalation_rate": current_escalations(cohort),
        "avg_confidence": avg_confidence(cohort),
        "gap_coverage": gap_coverage_gain(cohort),
        "cost_per_query": cost_per_query(cohort),
    })


def publish_metrics(cohort: Cohort, metrics: Dict[str, float]) -> None:
    """
    Publish cohort metrics to monitoring system.
    
    Args:
        cohort: Model cohort
        metrics: Metric dictionary
    """
    # Placeholder: would send to monitoring/logging system
    pass


def current_escalations(cohort: Cohort) -> float:
    """
    Compute current escalation rate.
    
    Args:
        cohort: Model cohort
        
    Returns:
        Escalation rate (0-1)
    """
    if cohort.total_queries == 0:
        return 0.0
    return cohort.escalations / cohort.total_queries


def avg_confidence(cohort: Cohort) -> float:
    """
    Get average confidence across cohort.
    
    Args:
        cohort: Model cohort
        
    Returns:
        Average confidence score
    """
    return cohort.avg_confidence


def gap_coverage_gain(cohort: Cohort) -> float:
    """
    Compute reduction in uncovered gap mass.
    
    Args:
        cohort: Model cohort
        
    Returns:
        Coverage improvement metric
    """
    # Placeholder: would compute sum of gap weights over time
    return 0.0


def cost_per_query(cohort: Cohort) -> float:
    """
    Get average cost per query.
    
    Args:
        cohort: Model cohort
        
    Returns:
        Mean inference cost
    """
    return cohort.cost_per_query


# =============================================================================
# 5) SLEEP TRAINING
# =============================================================================

def run_sleep_training(cohort: Cohort) -> None:
    """
    Execute fine-tuning for sleeping model in cohort.
    
    The fine-tuning objective is:
    L = E[w(x) * KL(p_sleeper || p_teacher)] + λ||P - P_EMA||² + μ*Div(sleeper, cohort)
    
    where:
    - w(x) = α*G(C_i) + β*EVI(x) + γ*conf(x)
    - G(C_i) is the gap weight
    - EVI(x) is expected value of improvement
    - conf(x) is teacher confidence
    
    Args:
        cohort: Model cohort containing sleeper
    """
    sleeper = next(
        (m for m in cohort.models if m.state == ModelState.SLEEPING),
        None
    )
    if not sleeper:
        return
    
    # Collect training data from target gap cells
    training_data: Dict[str, List[Tuple[Any, Any, float]]] = {}
    
    for cell in sleeper.target_cells:
        examples = sample_from_cell(cell, n_min=100)
        batches = []
        
        for query in examples:
            # Get teacher output and confidence
            teacher_output, teacher_confidence = high_level_teacher(query)
            
            # Compute example weight
            weight = (
                cell.gap_weight *  # α * G(C_i)
                1.0 *  # β * EVI(x) - placeholder
                confidence_weight(teacher_confidence)  # γ * conf(x)
            )
            
            batches.append((query, teacher_output, weight))
        
        training_data[cell.id] = batches
    
    # Train adapters for each target cell
    num_steps = steps_for_slot(slot_minutes(cohort))
    learning_rate = pick_learning_rate(cohort.size_class)
    lora_rank = lora_rank_for_class(cohort)
    
    for cell in sleeper.target_cells:
        # Get or create adapter for this cell
        adapter = sleeper.adapters.get(cell.id)
        if adapter is None:
            # Placeholder: would initialize LoRA adapter
            adapter = {"cell_id": cell.id, "rank": lora_rank}
            sleeper.adapters[cell.id] = adapter
        
        # Training loop (pseudocode)
        cell_data = training_data[cell.id]
        for step in range(num_steps):
            # Sample batch
            # Forward pass through model with adapter
            # Compute losses:
            #   1. KD loss: weighted KL divergence from teacher
            #   2. Prototype regularization: ||current_proto - target_proto||²
            #   3. Diversity loss: maintain distance from other cohort models
            # Backward pass and optimizer step
            pass
        
        # Update prototypes after training on this cell
        # Placeholder: would collect output embeddings
        sample_embeddings = []
        update_prototypes(sleeper, cell, sample_embeddings, ema=EMA_PROTO)


# =============================================================================
# 6) CANARY ROLLOUT
# =============================================================================

def start_canary(
    model: Model,
    cells: List[Cell],
    traffic_share: float = CANARY_INITIAL_TRAFFIC
) -> None:
    """
    Start canary rollout for recently trained model.
    
    Args:
        model: Model to roll out
        cells: Target cells for evaluation
        traffic_share: Initial traffic fraction
    """
    model.state = ModelState.ROLLOUT
    model.traffic_share = traffic_share
    model.target_cells = cells


def increase_traffic(
    model: Model,
    factor: float = CANARY_GROWTH_FACTOR,
    cap: float = CANARY_MAX_TRAFFIC
) -> None:
    """
    Increase traffic to model during successful rollout.
    
    Args:
        model: Model in rollout
        factor: Multiplication factor
        cap: Maximum traffic share
    """
    if model.traffic_share == 0:
        model.traffic_share = CANARY_INITIAL_TRAFFIC
    else:
        model.traffic_share = min(cap, model.traffic_share * factor)


def stable_for_slots(model: Model, k: int = STABILITY_REQUIRED_SLOTS) -> bool:
    """
    Check if model has been stable for k slots.
    
    Args:
        model: Model to check
        k: Required stability duration
        
    Returns:
        True if stable
    """
    # Placeholder: would check metric history
    return True


def rollback_adapters(model: Model) -> None:
    """
    Disable most recent adapters from failed rollout.
    
    Args:
        model: Model to rollback
    """
    # Placeholder: would disable/remove recent adapter checkpoints
    pass


def manage_rollouts(cohort: Cohort) -> None:
    """
    Manage canary rollouts for cohort models.
    
    Progression:
    1. sleeper completes → start canary (2% traffic)
    2. If improves + no regression → double traffic
    3. If stable for 2 slots → promote to full working
    4. If regresses → rollback adapters
    
    Args:
        cohort: Model cohort
    """
    # Check for sleepers that completed training
    for model in cohort.models:
        if model.state == ModelState.SLEEPING:
            # Check if training slot is complete
            # Placeholder: would check against slot_ends_at
            slot_complete = True
            
            if slot_complete:
                start_canary(model, cells=model.target_cells)
    
    # Manage ongoing rollouts
    for model in cohort.models:
        if model.state != ModelState.ROLLOUT:
            continue
        
        # Evaluate performance on target cells
        stats = evaluate_on_cells(model, model.target_cells, horizon="short")
        
        if improves(stats) and no_regress_outside(model, stats):
            # Success: increase traffic
            increase_traffic(model)
            
            # Check for promotion to full working
            if stable_for_slots(model):
                model.state = ModelState.WORKING
                model.traffic_share = 0.0  # Now handles normal routing
        else:
            # Regression: rollback and return to working
            rollback_adapters(model)
            model.state = ModelState.WORKING
            model.traffic_share = 0.0


# =============================================================================
# 7) SLEEPER ROTATION
# =============================================================================

def gap_misalignment_score(
    model: Model,
    cohort: Cohort,
    logs: List[Dict[str, Any]]
) -> float:
    """
    Compute gap misalignment score for a model.
    
    Higher score indicates model would benefit more from targeted training.
    
    Args:
        model: Model to evaluate
        cohort: Parent cohort
        logs: Recent performance logs
        
    Returns:
        Misalignment score (0-1, higher means worse alignment)
    """
    return model.gap_misalignment_score(cohort, logs)


def pick_sleeper_for_cohort(cohort: Cohort) -> Optional[Model]:
    """
    Select which model should enter sleep phase next.
    
    Selection criteria (weighted combination):
    - Gap misalignment: 70% (how poorly model covers high-gap regions)
    - Recency: 30% (time since last sleep)
    
    Args:
        cohort: Model cohort
        
    Returns:
        Selected model or None
    """
    logs = collect_logs(cohort, window_hours=4)
    
    # Refresh gap index
    cohort.gap_index.update(logs)
    
    # Score all working models
    candidates: List[Tuple[float, Model]] = []
    current_time = now_timestamp()
    
    for model in cohort.models:
        if model.state != ModelState.WORKING:
            continue
        
        # Compute components
        misalignment = gap_misalignment_score(model, cohort, logs)
        recency = current_time - model.last_sleep_timestamp
        recency_normalized = normalize(recency / (24 * 3600))  # Normalize to days
        
        # Combined score
        score = 0.7 * misalignment + 0.3 * recency_normalized
        candidates.append((score, model))
    
    if not candidates:
        return None
    
    # Return model with highest score
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def rotate_sleepers(cohort: Cohort) -> None:
    """
    Rotate models between working and sleeping states.
    
    Only one model sleeps at a time per cohort. Selection is based on:
    1. Gap misalignment (which model would benefit most)
    2. Time since last sleep (fairness)
    
    Args:
        cohort: Model cohort
    """
    # Check if a slot is currently occupied
    sleeper = next(
        (m for m in cohort.models if m.state == ModelState.SLEEPING),
        None
    )
    
    if sleeper:
        # Slot occupied; don't rotate yet
        return
    
    # Select next sleeper
    next_sleeper = pick_sleeper_for_cohort(cohort)
    
    if next_sleeper:
        # Transition to sleep
        next_sleeper.state = ModelState.SLEEPING
        next_sleeper.last_sleep_timestamp = now_timestamp()
        
        # Assign target cells (top-M gap cells)
        next_sleeper.target_cells = cohort.gap_index.top_cells(M=5)


# =============================================================================
# 8) SINGLE-ENVELOPE ORCHESTRATION
# =============================================================================

def single_envelope_tick(cohort: Cohort) -> None:
    """
    Execute one tick of the sleep-wake cycle for a single-envelope cohort.
    
    Sequence:
    1. Rotate sleepers (select next model to train)
    2. Run sleep training (fine-tune on gaps)
    3. Manage rollouts (canary → full deployment)
    4. Refresh metrics and prototypes (global updates)
    
    Args:
        cohort: Model cohort
    """
    rotate_sleepers(cohort)
    run_sleep_training(cohort)
    manage_rollouts(cohort)
    refresh_metrics_and_prototypes(cohort)


# =============================================================================
# 9) MULTI-ENVELOPE ORCHESTRATION
# =============================================================================

def gap_pressure(cohort: Cohort) -> float:
    """
    Compute priority score for allocating envelopes to a cohort.
    
    Higher pressure = more urgent need for training resources.
    Based on sum of gap weights for top cells.
    
    Args:
        cohort: Model cohort
        
    Returns:
        Pressure score
    """
    return sum(c.gap_weight for c in cohort.gap_index.top_cells(M=10))


def avg_train_memory(cohort: Cohort) -> float:
    """
    Compute average training memory footprint for cohort.
    
    Args:
        cohort: Model cohort
        
    Returns:
        Average training VRAM (GB)
    """
    if not cohort.models:
        return 0.0
    
    values = [m.vram_training for m in cohort.models]
    return sum(values) / len(values)


def avg_infer_memory(cohort: Cohort) -> float:
    """
    Compute average inference memory footprint for cohort.
    
    Args:
        cohort: Model cohort
        
    Returns:
        Average inference VRAM (GB)
    """
    if not cohort.models:
        return 0.0
    
    values = [m.vram_inference for m in cohort.models]
    return sum(values) / len(values)


def ensure_envelopes(
    cohorts: List[Cohort],
    pools: ResourcePools
) -> List[Envelope]:
    """
    Allocate envelopes to cohorts based on gap pressure and available resources.
    
    Algorithm:
    1. Rank cohorts by gap pressure (descending)
    2. For each cohort, allocate as many envelopes as resources permit
    3. Each envelope gets: train_budget (for sleeper) + infer_budget (for workers)
    
    Args:
        cohorts: List of all cohorts
        pools: Available resource pools
        
    Returns:
        List of allocated envelopes
    """
    envelopes: List[Envelope] = []
    
    # Rank cohorts by priority
    ranked_cohorts = sorted(cohorts, key=gap_pressure, reverse=True)
    
    for cohort in ranked_cohorts:
        # Compute resource requirements
        train_mem = avg_train_memory(cohort)
        infer_mem = avg_infer_memory(cohort)
        
        if train_mem == 0 or infer_mem == 0:
            continue
        
        # Maximum envelopes we can allocate to this cohort
        max_envelopes_train = int(pools.train_free // train_mem)
        max_envelopes_infer = int(pools.infer_free // (2 * infer_mem))
        max_envelopes = min(max_envelopes_train, max_envelopes_infer)
        
        # Allocate envelopes
        for i in range(max(0, max_envelopes)):
            envelope = Envelope(
                id=f"env-{cohort.size_class}-{len(envelopes) + 1}",
                cohort=cohort,
                train_budget_gb=train_mem,
                infer_budget_gb=2 * train_mem,
                state="idle"
            )
            envelopes.append(envelope)
            
            # Deduct from pools
            pools.train_free -= train_mem
            pools.infer_free -= 2 * infer_mem
    
    return envelopes


def pick_sleeper_for_envelope(envelope: Envelope) -> Optional[Model]:
    """
    Select model to sleep in this envelope.
    
    Args:
        envelope: Resource envelope
        
    Returns:
        Selected model or None
    """
    return pick_sleeper_for_cohort(envelope.cohort)


def fits_train_pool(model: Model, train_budget_gb: float) -> bool:
    """
    Check if model fits in training budget.
    
    Args:
        model: Model to check
        train_budget_gb: Available training VRAM
        
    Returns:
        True if fits
    """
    return model.vram_training <= train_budget_gb


def pick_workers_for_envelope(
    cohort: Cohort,
    exclude: List[Model],
    infer_budget_gb: float
) -> List[Model]:
    """
    Select worker models for envelope inference pool.
    
    Strategy: Pack models by quality/cost ratio until budget is full.
    
    Args:
        cohort: Model cohort
        exclude: Models to exclude (e.g., sleeper)
        infer_budget_gb: Available inference VRAM
        
    Returns:
        List of selected workers
    """
    # Get available working models
    candidates = [
        m for m in cohort.models
        if m.state == ModelState.WORKING and m not in exclude
    ]
    
    # Sort by quality/cost ratio (best first)
    candidates.sort(key=lambda m: quality_cost_ratio(m), reverse=True)
    
    # Pack workers until budget full
    workers: List[Model] = []
    used_memory = 0.0
    
    for model in candidates:
        if used_memory + model.vram_inference <= infer_budget_gb:
            workers.append(model)
            used_memory += model.vram_inference
        
        if used_memory >= infer_budget_gb:
            break
    
    return workers


def mark_training_allocation(envelope: Envelope, sleeper: Model) -> None:
    """
    Attach training resources to sleeper for the slot.
    
    Args:
        envelope: Resource envelope
        sleeper: Model entering sleep
    """
    # Placeholder: would reserve GPU memory, update resource tracking
    pass


def release_training_allocation(envelope: Envelope) -> None:
    """
    Release training resources after sleeper wakes.
    
    Args:
        envelope: Resource envelope
    """
    # Placeholder: would free GPU memory, update resource tracking
    pass


def run_sleep_training_for_envelope(envelope: Envelope) -> None:
    """
    Execute fine-tuning for sleeper in envelope.
    
    Similar to run_sleep_training but envelope-scoped.
    
    Args:
        envelope: Resource envelope containing sleeper
    """
    if not envelope.sleeper:
        return
    
    model = envelope.sleeper
    cohort = envelope.cohort
    
    # Collect training data from target gap cells
    training_data: Dict[str, List[Tuple[Any, Any, float]]] = {}
    
    for cell in envelope.target_cells:
        examples = sample_from_cell(cell, n_min=100)
        batches = []
        
        for query in examples:
            teacher_output, teacher_confidence = high_level_teacher(query)
            weight = (
                cell.gap_weight *
                1.0 *  # EVI placeholder
                confidence_weight(teacher_confidence)
            )
            batches.append((query, teacher_output, weight))
        
        training_data[cell.id] = batches
    
    # Train adapters
    num_steps = steps_for_slot(slot_minutes(cohort))
    learning_rate = pick_learning_rate(cohort.size_class)
    lora_rank = lora_rank_for_class(cohort)
    
    for cell in envelope.target_cells:
        adapter = model.adapters.get(cell.id)
        if adapter is None:
            adapter = {"cell_id": cell.id, "rank": lora_rank}
            model.adapters[cell.id] = adapter
        
        # Training loop (pseudocode)
        for step in range(num_steps):
            # Forward/backward pass with KD loss + regularization
            pass
        
        # Update prototypes
        sample_embeddings = []
        update_prototypes(model, cell, sample_embeddings, ema=EMA_PROTO)


def start_envelope_canary(envelope: Envelope) -> None:
    """
    Start canary rollout for envelope's sleeper.
    
    Args:
        envelope: Resource envelope
    """
    if not envelope.sleeper:
        return
    
    start_canary(
        envelope.sleeper,
        cells=envelope.target_cells,
        traffic_share=CANARY_INITIAL_TRAFFIC
    )
    envelope.state = "rollout"


def promote_to_worker(envelope: Envelope) -> None:
    """
    Promote envelope's sleeper to full worker status.
    
    Args:
        envelope: Resource envelope
    """
    if not envelope.sleeper:
        return
    
    envelope.sleeper.state = ModelState.WORKING
    envelope.sleeper.traffic_share = 0.0
    envelope.sleeper = None
    envelope.state = "working"


def envelope_tick(envelope: Envelope, pools: ResourcePools) -> None:
    """
    Execute one tick for a resource envelope.
    
    State machine:
    - idle/working → select sleeper → sleeping
    - sleeping → train → rollout
    - rollout → evaluate → promote or rollback
    
    Args:
        envelope: Resource envelope
        pools: Global resource pools
    """
    cohort = envelope.cohort
    current_time = now_timestamp()
    
    # State: idle or working → try to start new sleep cycle
    if envelope.state in ("idle", "working"):
        sleeper = pick_sleeper_for_envelope(envelope)
        
        if sleeper and fits_train_pool(sleeper, envelope.train_budget_gb):
            # Start sleep cycle
            envelope.sleeper = sleeper
            sleeper.state = ModelState.SLEEPING
            sleeper.last_sleep_timestamp = current_time
            
            # Assign target cells
            envelope.target_cells = cohort.gap_index.top_cells(M=5)
            
            # Set slot end time
            envelope.slot_ends_at = current_time + 60 * slot_minutes(cohort)
            envelope.state = "sleeping"
            
            # Allocate resources
            mark_training_allocation(envelope, sleeper)
            
            # Select workers for non-target traffic
            envelope.workers = pick_workers_for_envelope(
                cohort=cohort,
                exclude=[sleeper],
                infer_budget_gb=envelope.infer_budget_gb
            )
    
    # State: sleeping → run training
    elif envelope.state == "sleeping":
        run_sleep_training_for_envelope(envelope)
        
        # Check if slot complete
        if current_time >= envelope.slot_ends_at:
            start_envelope_canary(envelope)
    
    # State: rollout → manage canary
    elif envelope.state == "rollout":
        if not envelope.sleeper:
            envelope.state = "working"
            return
        
        # Evaluate performance
        stats = evaluate_on_cells(
            envelope.sleeper,
            envelope.target_cells,
            horizon="short"
        )
        
        if improves(stats) and no_regress_outside(envelope.sleeper, stats):
            # Success: increase traffic
            increase_traffic(envelope.sleeper)
            
            # Check for promotion
            if stable_for_slots(envelope.sleeper):
                promote_to_worker(envelope)
                release_training_allocation(envelope)
        else:
            # Regression: rollback
            rollback_adapters(envelope.sleeper)
            envelope.sleeper.state = ModelState.WORKING
            envelope.state = "working"
            release_training_allocation(envelope)


def global_tick(cohorts: List[Cohort], pools: ResourcePools) -> None:
    """
    Execute one global tick across all cohorts and envelopes.
    
    Sequence:
    1. Refresh gap indices for all cohorts
    2. Allocate/refresh envelopes based on gap pressure
    3. Tick each envelope independently
    4. Refresh global prototypes and metrics
    
    Args:
        cohorts: List of all model cohorts
        pools: Global resource pools
    """
    # Step 1: Refresh gap indices
    for cohort in cohorts:
        logs = collect_logs(cohort, window_hours=4)
        cohort.gap_index.update(logs)
    
    # Step 2: Allocate envelopes
    envelopes = ensure_envelopes(cohorts, pools)
    
    # Step 3: Tick each envelope
    for envelope in envelopes:
        envelope_tick(envelope, pools)
    
    # Step 4: Global updates
    for cohort in cohorts:
        refresh_metrics_and_prototypes(cohort)


# =============================================================================
# 10) MAIN ORCHESTRATION LOOP
# =============================================================================

def run_cohort_orchestration(
    cohorts: List[Cohort],
    total_vram_gb: float,
    tick_interval_seconds: float = 60.0,
    max_iterations: Optional[int] = None
) -> None:
    """
    Run continuous sleep-wake orchestration across all cohorts.
    
    This is the main entry point for the system. It manages:
    - Resource allocation (1/3 training, 2/3 inference)
    - Sleeper rotation
    - Gap-based fine-tuning
    - Canary rollouts
    - Metric tracking
    
    Args:
        cohorts: List of all model cohorts
        total_vram_gb: Total available VRAM
        tick_interval_seconds: Time between ticks
        max_iterations: Maximum iterations (None = infinite)
    """
    # Initialize resource pools
    pools = ResourcePools.from_total(total_vram_gb)
    
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        # Execute global tick
        global_tick(cohorts, pools)
        
        # Wait for next tick
        time.sleep(tick_interval_seconds)
        
        iteration += 1


# =============================================================================
# 11) EXAMPLE USAGE
# =============================================================================

def create_example_cohort(
    size_class: str,
    num_models: int,
    vram_inference: float,
    vram_training: float
) -> Cohort:
    """
    Create an example cohort for testing.
    
    Args:
        size_class: Model size class (e.g., "3B", "8B")
        num_models: Number of models in cohort
        vram_inference: VRAM per model for inference (GB)
        vram_training: VRAM per model for training (GB)
        
    Returns:
        Initialized cohort
    """
    models = [
        Model(
            id=f"{size_class}-model-{i}",
            size_class=size_class,
            vram_inference=vram_inference,
            vram_training=vram_training
        )
        for i in range(num_models)
    ]
    
    return Cohort(
        size_class=size_class,
        models=models
    )


def main():
    """
    Example main function demonstrating system setup and execution.
    """
    # Create example cohorts
    cohorts = [
        create_example_cohort("3B", num_models=4, vram_inference=6.0, vram_training=12.0),
        create_example_cohort("8B", num_models=3, vram_inference=16.0, vram_training=32.0),
        create_example_cohort("13B", num_models=2, vram_inference=26.0, vram_training=52.0),
    ]
    
    # Run orchestration
    total_vram = 256.0  # Total VRAM budget (GB)
    
    print("Starting sleep-wake orchestration...")
    print(f"Total VRAM: {total_vram} GB")
    print(f"Training budget: {total_vram * TRAIN_BUDGET_FRACTION:.1f} GB")
    print(f"Inference budget: {total_vram * INFER_BUDGET_FRACTION:.1f} GB")
    print(f"Cohorts: {len(cohorts)}")
    
    run_cohort_orchestration(
        cohorts=cohorts,
        total_vram_gb=total_vram,
        tick_interval_seconds=60.0,
        max_iterations=100  # Run for 100 ticks then stop
    )


if __name__ == "__main__":
    main()
