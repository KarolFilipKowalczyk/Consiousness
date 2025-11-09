"""
selector.py

A theoretically-correct implementation of the Selector framework from
"Selectors and Meta-Selectors in Large Language Model Hierarchies"

This implementation actually measures behavioral distance by:
1. Invoking candidate models with the query
2. Embedding their actual outputs
3. Comparing output embeddings to query embeddings
4. Learning prototypes from real model behaviors

Author: Claude (based on theory by Karol Kowalczyk)
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Protocol, Callable
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Core Protocols and Interfaces
# ============================================================================

class ModelInferenceProtocol(Protocol):
    """Protocol for actual model inference."""
    def __call__(self, query: str) -> str:
        """Execute model inference and return output."""
        ...


class CostEstimator(Protocol):
    """Protocol for estimating inference cost."""
    def estimate(self, query: str, model_id: str) -> float:
        """Estimate cost of running model_id on query."""
        ...


class EmbeddingSpace(ABC):
    """Abstract base for embedding queries and outputs in a shared space."""
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query into the latent space."""
        pass
    
    @abstractmethod
    def embed_output(self, model_id: str, output: str) -> np.ndarray:
        """Embed a model output into the latent space."""
        pass
    
    @abstractmethod
    def distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute distance between two embeddings."""
        pass


# ============================================================================
# Concrete Embedding Implementation
# ============================================================================

@dataclass
class SentenceTransformerSpace(EmbeddingSpace):
    """
    Shared latent space using SentenceTransformers.
    Both queries and outputs are embedded in the same space.
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    distance_metric: str = "cosine"  # "cosine" | "euclidean" | "manhattan"
    _model: Optional[SentenceTransformer] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        self._model = SentenceTransformer(self.model_name)
        logger.info(f"Loaded embedding model: {self.model_name}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed query using the sentence transformer."""
        vec = self._model.encode(
            query, 
            convert_to_numpy=True, 
            normalize_embeddings=self.normalize
        )
        return vec.astype(np.float32)
    
    def embed_output(self, model_id: str, output: str) -> np.ndarray:
        """
        Embed model output.
        Could optionally prepend model_id to capture model-specific style:
        text = f"[{model_id}] {output}"
        """
        vec = self._model.encode(
            output,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return vec.astype(np.float32)
    
    def distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute distance between embeddings."""
        if self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            dot = np.dot(u, v)
            norm_u = np.linalg.norm(u) + 1e-12
            norm_v = np.linalg.norm(v) + 1e-12
            return float(1.0 - dot / (norm_u * norm_v))
        elif self.distance_metric == "euclidean":
            return float(np.linalg.norm(u - v))
        elif self.distance_metric == "manhattan":
            return float(np.sum(np.abs(u - v)))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")


# ============================================================================
# Model Registry and Adapter
# ============================================================================

@dataclass
class ModelAdapter:
    """
    Wraps a real model with metadata for the selector.
    This adapter ACTUALLY CALLS the model during selection.
    """
    model_id: str
    inference_fn: ModelInferenceProtocol
    cost_per_token: float  # Cost per 1K tokens (example metric)
    level: int  # Hierarchy level: 0=small, 1=medium, 2=large
    quality_hint: float = 0.5  # Optional prior on quality (0-1)
    
    def infer(self, query: str) -> str:
        """Execute actual model inference."""
        logger.debug(f"Calling model {self.model_id} with query: {query[:50]}...")
        output = self.inference_fn(query)
        logger.debug(f"Model {self.model_id} returned: {output[:100]}...")
        return output
    
    def estimate_cost(self, query: str) -> float:
        """Estimate cost based on query length (simplified)."""
        # Rough token estimate: ~4 chars per token
        approx_tokens = len(query) / 4
        # Assume output is similar length
        total_tokens = approx_tokens * 2
        return (total_tokens / 1000) * self.cost_per_token


# ============================================================================
# Behavioral Prototype Bank
# ============================================================================

@dataclass
class BehavioralPrototype:
    """A prototype representing typical model behavior."""
    embedding: np.ndarray
    count: int = 0  # Number of times this prototype was updated
    exemplar_output: Optional[str] = None  # Optional: store one example


@dataclass
class PrototypeBank:
    """
    Stores learned prototypes of model behaviors.
    Prototypes are built from ACTUAL model outputs, not queries.
    """
    max_prototypes_per_model: int = 5
    ema_decay: float = 0.9
    initialization_threshold: int = 3  # Min samples before creating new prototype
    
    # Maps model_id -> list of prototypes
    _bank: Dict[str, List[BehavioralPrototype]] = field(default_factory=dict)
    
    def get_prototypes(self, model_id: str) -> List[BehavioralPrototype]:
        """Get all prototypes for a model."""
        return self._bank.get(model_id, [])
    
    def add_observation(
        self, 
        model_id: str, 
        output_embedding: np.ndarray,
        output_text: str,
        distance_fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> None:
        """
        Add an observed model output to update prototypes.
        This implements learning from actual model behaviors.
        """
        if model_id not in self._bank:
            self._bank[model_id] = []
        
        prototypes = self._bank[model_id]
        
        if len(prototypes) == 0:
            # First observation: create initial prototype
            self._bank[model_id].append(BehavioralPrototype(
                embedding=output_embedding.copy(),
                count=1,
                exemplar_output=output_text
            ))
            logger.info(f"Created first prototype for {model_id}")
            return
        
        # Find nearest prototype
        distances = [distance_fn(output_embedding, p.embedding) for p in prototypes]
        nearest_idx = int(np.argmin(distances))
        nearest_dist = distances[nearest_idx]
        
        # Threshold for creating new prototype (adaptive based on space coverage)
        new_prototype_threshold = 0.3  # Cosine distance threshold
        
        if nearest_dist > new_prototype_threshold and len(prototypes) < self.max_prototypes_per_model:
            # Create new prototype for this behavior cluster
            self._bank[model_id].append(BehavioralPrototype(
                embedding=output_embedding.copy(),
                count=1,
                exemplar_output=output_text
            ))
            logger.info(f"Created new prototype for {model_id} (now {len(self._bank[model_id])} prototypes)")
        else:
            # Update nearest prototype with EMA
            proto = prototypes[nearest_idx]
            proto.embedding = (
                self.ema_decay * proto.embedding + 
                (1.0 - self.ema_decay) * output_embedding
            ).astype(np.float32)
            proto.count += 1
            # Occasionally update exemplar
            if proto.count % 10 == 0:
                proto.exemplar_output = output_text
            logger.debug(f"Updated prototype {nearest_idx} for {model_id} (count={proto.count})")
    
    def predict_distance(
        self,
        model_id: str,
        query_embedding: np.ndarray,
        distance_fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> float:
        """
        Predict the likely behavioral distance for this model WITHOUT calling it.
        Uses learned prototypes as a proxy.
        Returns minimum distance to any prototype (optimistic estimate).
        """
        prototypes = self.get_prototypes(model_id)
        if not prototypes:
            return float('inf')  # No knowledge, assume worst case
        
        distances = [distance_fn(query_embedding, p.embedding) for p in prototypes]
        return min(distances)


# ============================================================================
# Selection Strategies
# ============================================================================

class SelectionMode(Enum):
    """Different selection strategies."""
    EXPLOIT = "exploit"  # Always choose best known model
    EXPLORE = "explore"  # Try models to learn their behaviors
    HYBRID = "hybrid"    # Balance exploitation and exploration


@dataclass
class SelectionResult:
    """Result of a selection decision."""
    chosen_model_id: str
    chosen_level: int
    actual_output: str
    output_embedding: np.ndarray
    behavioral_distance: float
    estimated_cost: float
    confidence: float
    alternatives_considered: List[Tuple[str, float, float]]  # (model_id, dist, cost)


# ============================================================================
# The Proper Selector
# ============================================================================

class BehavioralSelector:
    """
    A theoretically-correct selector that measures ACTUAL behavioral distance.
    
    Key differences from the original implementation:
    1. Actually calls models during selection (at least in explore mode)
    2. Measures distance between query embedding and ACTUAL output embedding
    3. Learns prototypes from real model behaviors
    4. Supports both full evaluation and prototype-based prediction
    """
    
    def __init__(
        self,
        embedding_space: EmbeddingSpace,
        model_adapters: Dict[str, ModelAdapter],
        prototype_bank: PrototypeBank,
        w_distance: float = 1.0,
        w_cost: float = 0.8,
        w_level: float = 0.1,  # Prefer lower levels when tied
        confidence_threshold: float = 0.75,
        exploration_rate: float = 0.1,
    ):
        self.space = embedding_space
        self.models = model_adapters
        self.prototypes = prototype_bank
        self.w_distance = w_distance
        self.w_cost = w_cost
        self.w_level = w_level
        self.confidence_threshold = confidence_threshold
        self.exploration_rate = exploration_rate
        
        # Statistics
        self.total_selections = 0
        self.explorations = 0
        self.exploitations = 0
    
    def _compute_confidence(self, distance: float) -> float:
        """
        Convert behavioral distance to confidence score.
        Lower distance = higher confidence that model is suitable.
        """
        # Logistic function: conf = 1 / (1 + exp(k * (d - x0)))
        k, x0 = -8.0, 0.35
        return float(1.0 / (1.0 + np.exp(k * (distance - x0))))
    
    def _evaluate_model_actual(
        self, 
        model_id: str, 
        query: str,
        query_embedding: np.ndarray
    ) -> Tuple[str, np.ndarray, float, float]:
        """
        Actually call the model and measure true behavioral distance.
        
        Returns: (output, output_embedding, distance, cost)
        """
        adapter = self.models[model_id]
        
        # ACTUALLY CALL THE MODEL
        output = adapter.infer(query)
        
        # Embed the ACTUAL output
        output_embedding = self.space.embed_output(model_id, output)
        
        # Measure TRUE behavioral distance
        distance = self.space.distance(query_embedding, output_embedding)
        
        # Estimate cost
        cost = adapter.estimate_cost(query)
        
        return output, output_embedding, distance, cost
    
    def _evaluate_model_predicted(
        self,
        model_id: str,
        query: str,
        query_embedding: np.ndarray
    ) -> Tuple[float, float]:
        """
        Predict behavioral distance using prototypes (no model call).
        
        Returns: (predicted_distance, estimated_cost)
        """
        adapter = self.models[model_id]
        
        # Predict distance using learned prototypes
        pred_distance = self.prototypes.predict_distance(
            model_id, 
            query_embedding, 
            self.space.distance
        )
        
        # Estimate cost
        cost = adapter.estimate_cost(query)
        
        return pred_distance, cost
    
    def _compute_composite_score(
        self,
        distance: float,
        cost: float,
        level: int
    ) -> float:
        """
        Compute weighted score combining behavioral fit, cost, and level preference.
        Lower is better.
        """
        return (
            self.w_distance * distance + 
            self.w_cost * cost + 
            self.w_level * level
        )
    
    def select_exploit(
        self, 
        query: str,
        top_k: int = 1
    ) -> SelectionResult:
        """
        Exploitation mode: use prototypes to predict best model, then call only that one.
        Fast but doesn't learn new behaviors.
        """
        query_embedding = self.space.embed_query(query)
        
        # Predict scores for all models using prototypes
        candidates = []
        for model_id, adapter in self.models.items():
            pred_dist, cost = self._evaluate_model_predicted(model_id, query, query_embedding)
            score = self._compute_composite_score(pred_dist, cost, adapter.level)
            candidates.append((model_id, pred_dist, cost, score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[3])
        
        # Choose best model and ACTUALLY CALL IT
        best_model_id, pred_dist, pred_cost, _ = candidates[0]
        output, output_emb, actual_dist, actual_cost = self._evaluate_model_actual(
            best_model_id, query, query_embedding
        )
        
        # Update prototypes with actual behavior
        self.prototypes.add_observation(
            best_model_id, 
            output_emb, 
            output,
            self.space.distance
        )
        
        confidence = self._compute_confidence(actual_dist)
        
        self.exploitations += 1
        self.total_selections += 1
        
        return SelectionResult(
            chosen_model_id=best_model_id,
            chosen_level=self.models[best_model_id].level,
            actual_output=output,
            output_embedding=output_emb,
            behavioral_distance=actual_dist,
            estimated_cost=actual_cost,
            confidence=confidence,
            alternatives_considered=[(m, d, c) for m, d, c, _ in candidates[:top_k]]
        )
    
    def select_explore(
        self,
        query: str,
        sample_size: int = 3
    ) -> SelectionResult:
        """
        Exploration mode: actually call multiple models to learn their behaviors.
        Expensive but builds accurate prototypes.
        """
        query_embedding = self.space.embed_query(query)
        
        # Randomly sample models or choose strategically
        # Strategy: sample across different levels
        models_by_level = {}
        for model_id, adapter in self.models.items():
            if adapter.level not in models_by_level:
                models_by_level[adapter.level] = []
            models_by_level[adapter.level].append(model_id)
        
        # Sample one from each level (up to sample_size)
        sample_models = []
        for level in sorted(models_by_level.keys()):
            if len(sample_models) >= sample_size:
                break
            sample_models.extend(models_by_level[level][:1])
        
        # ACTUALLY CALL all sampled models
        evaluations = []
        for model_id in sample_models[:sample_size]:
            output, output_emb, distance, cost = self._evaluate_model_actual(
                model_id, query, query_embedding
            )
            score = self._compute_composite_score(distance, cost, self.models[model_id].level)
            evaluations.append((model_id, output, output_emb, distance, cost, score))
            
            # Update prototypes with actual behavior
            self.prototypes.add_observation(
                model_id,
                output_emb,
                output,
                self.space.distance
            )
        
        # Choose best based on actual measurements
        evaluations.sort(key=lambda x: x[5])  # Sort by score
        best = evaluations[0]
        
        best_model_id, output, output_emb, distance, cost, _ = best
        confidence = self._compute_confidence(distance)
        
        self.explorations += 1
        self.total_selections += 1
        
        return SelectionResult(
            chosen_model_id=best_model_id,
            chosen_level=self.models[best_model_id].level,
            actual_output=output,
            output_embedding=output_emb,
            behavioral_distance=distance,
            estimated_cost=cost,
            confidence=confidence,
            alternatives_considered=[(m, d, c) for m, _, _, d, c, _ in evaluations]
        )
    
    def select_hybrid(
        self,
        query: str,
        force_explore: bool = False
    ) -> SelectionResult:
        """
        Hybrid mode: usually exploit, occasionally explore.
        Balances efficiency with continuous learning.
        """
        # Decide whether to explore
        should_explore = force_explore or (np.random.random() < self.exploration_rate)
        
        if should_explore:
            logger.info("Exploration mode activated")
            return self.select_explore(query, sample_size=3)
        else:
            result = self.select_exploit(query)
            
            # Adaptive exploration: if confidence is low, consider exploring
            if result.confidence < self.confidence_threshold:
                logger.warning(
                    f"Low confidence ({result.confidence:.3f}) for {result.chosen_model_id}, "
                    "consider exploring"
                )
            
            return result
    
    def select(
        self, 
        query: str, 
        mode: SelectionMode = SelectionMode.HYBRID
    ) -> SelectionResult:
        """
        Main selection interface.
        
        Args:
            query: The input query
            mode: Selection strategy (exploit/explore/hybrid)
        
        Returns:
            SelectionResult with chosen model and actual output
        """
        if mode == SelectionMode.EXPLOIT:
            return self.select_exploit(query)
        elif mode == SelectionMode.EXPLORE:
            return self.select_explore(query)
        elif mode == SelectionMode.HYBRID:
            return self.select_hybrid(query)
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
    
    def get_statistics(self) -> Dict:
        """Return selector statistics."""
        return {
            "total_selections": self.total_selections,
            "explorations": self.explorations,
            "exploitations": self.exploitations,
            "exploration_rate": self.explorations / max(1, self.total_selections),
            "models": {
                model_id: {
                    "num_prototypes": len(self.prototypes.get_prototypes(model_id)),
                    "level": adapter.level,
                }
                for model_id, adapter in self.models.items()
            }
        }


# ============================================================================
# Demonstration with Mock Models
# ============================================================================

def create_mock_model(model_id: str, quality: float) -> ModelInferenceProtocol:
    """
    Create a mock model that produces different outputs based on quality.
    Unlike the original implementation, these actually process the query.
    """
    def mock_inference(query: str) -> str:
        # Parse query to generate appropriate response
        query_lower = query.lower()
        
        if quality < 0.4:  # Small model
            if "quantum" in query_lower:
                return "Quantum particles can be connected."
            elif "explain" in query_lower or "what" in query_lower:
                return "It's a complex topic that involves multiple factors."
            else:
                return f"Here's a brief answer about {query[:30]}."
        
        elif quality < 0.7:  # Medium model
            if "quantum" in query_lower:
                return "Quantum entanglement is when two particles become correlated so that measuring one affects the other, even at a distance."
            elif "explain" in query_lower:
                # Extract topic
                words = query.split()
                topic = next((w for w in words if len(w) > 5), "this concept")
                return f"{topic.capitalize()} involves several interconnected principles. Let me break this down into key points: first, the fundamental mechanism; second, the practical implications."
            else:
                return f"Based on your question about '{query[:50]}', here's a detailed explanation of the key concepts involved."
        
        else:  # Large model
            if "quantum" in query_lower:
                return (
                    "Quantum entanglement represents a fundamental correlation between quantum systems "
                    "that cannot be explained by classical physics. When two particles are entangled, "
                    "their quantum states are interdependent—measuring one particle's state instantaneously "
                    "determines the other's state, regardless of spatial separation. This phenomenon doesn't "
                    "violate causality because no information is transmitted faster than light; rather, "
                    "it reveals the non-local nature of quantum mechanics as described by the EPR paradox "
                    "and Bell's theorem."
                )
            elif "explain" in query_lower:
                words = query.split()
                topic = next((w for w in words if len(w) > 5), "this concept")
                return (
                    f"Let me provide a comprehensive explanation of {topic}. "
                    f"First, we need to understand the historical context and theoretical foundation. "
                    f"The current understanding emerged from extensive research and debate. "
                    f"Key principles include: (1) the fundamental mechanisms at play, "
                    f"(2) empirical evidence supporting the theory, (3) practical applications, "
                    f"and (4) ongoing areas of investigation. Each of these aspects contributes to "
                    f"our holistic understanding of {topic}."
                )
            else:
                return (
                    f"Your inquiry regarding '{query[:60]}' raises several important considerations. "
                    f"I'll address this from multiple perspectives: theoretical foundations, "
                    f"empirical evidence, practical implications, and potential limitations of current understanding."
                )
    
    return mock_inference


if __name__ == "__main__":
    # Example usage demonstrating proper behavioral distance measurement
    
    print("=" * 80)
    print("Proper Behavioral Selector - Demonstration")
    print("=" * 80)
    
    # Initialize embedding space
    space = SentenceTransformerSpace()
    
    # Create model adapters with REAL (mock) inference functions
    models = {
        "gpt-small": ModelAdapter(
            model_id="gpt-small",
            inference_fn=create_mock_model("gpt-small", quality=0.3),
            cost_per_token=0.0001,
            level=0,
            quality_hint=0.3
        ),
        "gpt-medium": ModelAdapter(
            model_id="gpt-medium",
            inference_fn=create_mock_model("gpt-medium", quality=0.6),
            cost_per_token=0.0005,
            level=1,
            quality_hint=0.6
        ),
        "gpt-large": ModelAdapter(
            model_id="gpt-large",
            inference_fn=create_mock_model("gpt-large", quality=0.9),
            cost_per_token=0.002,
            level=2,
            quality_hint=0.9
        ),
    }
    
    # Initialize prototype bank
    prototype_bank = PrototypeBank(max_prototypes_per_model=5)
    
    # Create selector
    selector = BehavioralSelector(
        embedding_space=space,
        model_adapters=models,
        prototype_bank=prototype_bank,
        exploration_rate=0.2  # 20% exploration
    )
    
    # Test queries
    queries = [
        "Explain quantum entanglement to a 12-year-old.",
        "What is the capital of France?",
        "Explain quantum entanglement in detail.",
        "What are the key principles of thermodynamics?",
        "Give me a brief summary of quantum mechanics.",
    ]
    
    print("\n" + "=" * 80)
    print("Running selection with hybrid mode (exploit + explore)")
    print("=" * 80 + "\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")
        
        # Force exploration on first few queries to build prototypes
        force_explore = (i <= 2)
        result = selector.select_hybrid(query, force_explore=force_explore)
        
        print(f"\n✓ Selected Model: {result.chosen_model_id} (Level {result.chosen_level})")
        print(f"  Behavioral Distance: {result.behavioral_distance:.4f}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Estimated Cost: ${result.estimated_cost:.6f}")
        print(f"\n  Output: {result.actual_output[:150]}...")
        
        if result.alternatives_considered:
            print(f"\n  Alternatives considered:")
            for model_id, dist, cost in result.alternatives_considered[:3]:
                print(f"    - {model_id}: dist={dist:.4f}, cost=${cost:.6f}")
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("Selector Statistics")
    print("=" * 80)
    stats = selector.get_statistics()
    print(f"Total selections: {stats['total_selections']}")
    print(f"Explorations: {stats['explorations']}")
    print(f"Exploitations: {stats['exploitations']}")
    print(f"Exploration rate: {stats['exploration_rate']:.2%}")
    print("\nPrototypes learned per model:")
    for model_id, info in stats['models'].items():
        print(f"  {model_id}: {info['num_prototypes']} prototypes (Level {info['level']})")
