"""
selector.py - Selector mechanism with entropic scaling for consciousness framework

This module implements the selector that chooses which machine level to deploy
based on problem complexity and resource constraints using entropic scaling.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """Represents a computational problem"""
    id: str
    complexity: float  # Estimated complexity
    features: np.ndarray  # Feature vector
    context: Dict[str, Any] = None


@dataclass
class MachineLevel:
    """Represents a machine at a specific hierarchy level"""
    level: int
    capacity: float  # Information capacity I(n) = κ n log n
    kappa: float = 1.0  # Entropic scaling constant
    
    @property
    def information_capacity(self) -> float:
        """Calculate entropic information capacity I(n) = κ n log n"""
        if self.level <= 0:
            return 0
        return self.kappa * self.level * math.log(self.level + 1)
    
    @property
    def processing_time(self, tau_0: float = 0.05, gamma: float = 1e-6) -> float:
        """Calculate processing time τ(n) = τ₀ + γ n log n"""
        return tau_0 + gamma * self.level * math.log(self.level + 1)


class Selector:
    """
    Selector mechanism for choosing appropriate machine level.
    Uses entropic scaling for costs: C(n) = C₀(1 + β n log n)
    """
    
    def __init__(self, 
                 n_max: int = 40,
                 c_0: float = 1.0,
                 beta: float = 0.1,
                 kappa: float = 1.0,
                 lambda_weight: float = 0.5):
        """
        Initialize selector with entropic scaling parameters.
        
        Args:
            n_max: Maximum hierarchy level
            c_0: Base cost constant
            beta: Entropic cost scaling factor
            kappa: Entropic capacity scaling constant
            lambda_weight: Weight for cost vs quality tradeoff
        """
        self.n_max = n_max
        self.c_0 = c_0
        self.beta = beta
        self.kappa = kappa
        self.lambda_weight = lambda_weight
        self.history = []
        
        # Initialize machine levels
        self.machines = [MachineLevel(level=n, capacity=0, kappa=kappa) 
                        for n in range(1, n_max + 1)]
        
        # Update capacities with entropic scaling
        for machine in self.machines:
            machine.capacity = machine.information_capacity
    
    def compute_cost(self, level: int) -> float:
        """
        Compute entropic cost for using a given level.
        C(n) = C₀(1 + β n log n)
        
        Comment: Entropic scaling replaces quadratic or exponential cost growth
        """
        if level <= 0:
            return self.c_0
        
        entropic_factor = self.beta * level * math.log(level + 1)
        return self.c_0 * (1 + entropic_factor)
    
    def estimate_loss(self, problem: Problem, level: int) -> float:
        """
        Estimate loss for solving problem at given level.
        Lower loss for higher levels with sufficient capacity.
        """
        machine = self.machines[level - 1]
        capacity_ratio = machine.information_capacity / (problem.complexity + 1)
        
        # Loss decreases with entropic capacity
        if capacity_ratio >= 1:
            # Sufficient capacity
            loss = 1.0 / (level * math.log(level + 1))
        else:
            # Insufficient capacity - high loss
            loss = 1.0 + (1.0 - capacity_ratio)
        
        return loss
    
    def compute_score(self, problem: Problem, level: int) -> float:
        """
        Compute selection score for a level given a problem.
        Lower score is better.
        
        Uses entropic scaling for balanced level weighting.
        Comment: Entropic scaling replaces quadratic level weighting.
        """
        # Estimate quality loss
        loss = self.estimate_loss(problem, level)
        
        # Compute entropic cost
        cost = self.compute_cost(level)
        
        # Combined score with lambda weighting
        score = loss + self.lambda_weight * cost
        
        # Add small penalty for very high levels to encourage efficiency
        if level > 30:
            score += 0.01 * math.log(level)
        
        return score
    
    def select(self, problem: Problem) -> int:
        """
        Select optimal machine level for given problem.
        
        Returns:
            Selected level n* that minimizes expected cost + loss
        """
        scores = []
        
        for level in range(1, self.n_max + 1):
            score = self.compute_score(problem, level)
            scores.append((level, score))
            
            logger.debug(f"Level {level}: score={score:.4f}, "
                        f"cost={self.compute_cost(level):.4f}, "
                        f"capacity={self.machines[level-1].information_capacity:.2f}")
        
        # Select level with minimum score
        best_level = min(scores, key=lambda x: x[1])[0]
        
        # Record in history
        self.history.append({
            'problem_id': problem.id,
            'selected_level': best_level,
            'scores': scores
        })
        
        logger.info(f"Selected level {best_level} for problem {problem.id} "
                   f"(complexity={problem.complexity:.2f})")
        
        return best_level
    
    def expected_value_of_information(self, 
                                     current_level: int, 
                                     target_level: int,
                                     problem: Problem) -> float:
        """
        Calculate expected value of using target_level instead of current_level.
        EVI = ΔQuality - λ * ΔCost
        
        With entropic scaling, ΔCost ≈ log(target_level + 1)
        """
        if target_level <= current_level:
            return 0
        
        # Quality improvement
        current_loss = self.estimate_loss(problem, current_level)
        target_loss = self.estimate_loss(problem, target_level)
        delta_quality = current_loss - target_loss
        
        # Cost increase with entropic scaling
        # Comment: Entropic scaling replaces quadratic cost difference
        delta_cost = math.log(target_level + 1) - math.log(current_level + 1) if current_level > 0 else math.log(target_level + 1)
        
        # Expected value
        evi = delta_quality - self.lambda_weight * delta_cost
        
        return evi
    
    def adaptive_selection(self, 
                          problem: Problem,
                          confidence_threshold: float = 0.8) -> int:
        """
        Adaptive selection with confidence-based escalation.
        Start with low level and escalate if confidence is low.
        """
        current_level = 15  # Start with modest level
        
        while current_level < self.n_max:
            # Estimate if current level is sufficient
            loss = self.estimate_loss(problem, current_level)
            confidence = 1.0 - loss
            
            if confidence >= confidence_threshold:
                break
            
            # Check if escalation is worth it
            next_level = min(current_level + 5, self.n_max)
            evi = self.expected_value_of_information(current_level, next_level, problem)
            
            if evi > 0:
                current_level = next_level
            else:
                break
        
        return current_level
    
    def get_statistics(self) -> Dict:
        """
        Get selection statistics from history.
        """
        if not self.history:
            return {}
        
        selected_levels = [h['selected_level'] for h in self.history]
        
        stats = {
            'total_selections': len(self.history),
            'average_level': np.mean(selected_levels),
            'median_level': np.median(selected_levels),
            'level_distribution': {
                level: selected_levels.count(level) 
                for level in range(1, self.n_max + 1)
                if selected_levels.count(level) > 0
            },
            'average_cost': np.mean([
                self.compute_cost(level) for level in selected_levels
            ])
        }
        
        return stats


class MetaSelector:
    """
    Meta-selector that chooses between different selection strategies.
    Uses entropic scaling for meta-selection costs.
    """
    
    def __init__(self, selectors: List[Selector]):
        """
        Initialize meta-selector with list of base selectors.
        """
        self.selectors = selectors
        self.performance_history = {i: [] for i in range(len(selectors))}
        
    def select_selector(self, problem: Problem) -> int:
        """
        Choose which selector to use based on expected performance.
        
        Returns:
            Index of selected selector
        """
        if not any(self.performance_history.values()):
            # No history - choose randomly
            return np.random.randint(len(self.selectors))
        
        # Compute expected performance for each selector
        scores = []
        for i, selector in enumerate(self.selectors):
            if self.performance_history[i]:
                # Use historical average with entropic weighting
                avg_performance = np.mean(self.performance_history[i])
                # Apply entropic discount for complexity
                discount = 1.0 / (1.0 + math.log(len(self.performance_history[i]) + 1))
                score = avg_performance * discount
            else:
                # Optimistic initialization
                score = 0.0
            
            scores.append(score)
        
        # Select selector with best expected score
        return np.argmin(scores)
    
    def update_performance(self, selector_idx: int, performance: float):
        """
        Update performance history for a selector.
        """
        self.performance_history[selector_idx].append(performance)
        
        # Keep only recent history (sliding window)
        max_history = 100
        if len(self.performance_history[selector_idx]) > max_history:
            self.performance_history[selector_idx] = \
                self.performance_history[selector_idx][-max_history:]


def example_usage():
    """
    Demonstrate selector usage with entropic scaling.
    """
    # Create selector with entropic parameters
    selector = Selector(n_max=40, c_0=1.0, beta=0.1, kappa=1.0)
    
    # Create sample problems with varying complexity
    problems = [
        Problem("simple", complexity=10, features=np.random.rand(10)),
        Problem("moderate", complexity=50, features=np.random.rand(10)),
        Problem("complex", complexity=100, features=np.random.rand(10)),
        Problem("extreme", complexity=200, features=np.random.rand(10))
    ]
    
    print("Entropic Selector Demo")
    print("=" * 50)
    
    for problem in problems:
        level = selector.select(problem)
        cost = selector.compute_cost(level)
        capacity = selector.machines[level-1].information_capacity
        
        print(f"\nProblem: {problem.id}")
        print(f"  Complexity: {problem.complexity}")
        print(f"  Selected Level: {level}")
        print(f"  Information Capacity: {capacity:.2f} bits (entropic)")
        print(f"  Cost: {cost:.3f} (with n log n scaling)")
    
    # Show statistics
    print("\n" + "=" * 50)
    print("Selection Statistics:")
    stats = selector.get_statistics()
    for key, value in stats.items():
        if key != 'level_distribution':
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\nLevel Distribution:")
    if 'level_distribution' in stats:
        for level, count in sorted(stats['level_distribution'].items()):
            print(f"  Level {level}: {count} selections")
    
    # Demonstrate adaptive selection
    print("\n" + "=" * 50)
    print("Adaptive Selection Demo:")
    
    for problem in problems:
        level = selector.adaptive_selection(problem, confidence_threshold=0.7)
        print(f"  {problem.id}: adaptive level = {level}")


if __name__ == "__main__":
    example_usage()
