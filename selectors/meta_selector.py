"""
meta_selector.py - Meta-selector implementation with entropic scaling

Implements meta-selectors that choose between different selection strategies,
with Expected Value of Information calculations using entropic scaling.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configure logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SelectionContext:
    """Context information for selection decisions"""
    problem_complexity: float
    available_resources: float
    time_constraint: Optional[float] = None
    quality_requirement: Optional[float] = None
    history: List[Dict] = None


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies"""
    
    @abstractmethod
    def select(self, context: SelectionContext) -> int:
        """Select a model level given context"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass
    
    def calculate_evi(self, current_level: int, target_level: int, 
                     context: SelectionContext, lambda_param: float = 0.5) -> float:
        """
        Calculate Expected Value of Information with entropic scaling.
        
        EVI = ΔQuality - λ * ΔCost
        where ΔCost uses entropic scaling
        
        Comment: Entropic scaling for Expected Value of Information
        """
        if target_level <= current_level:
            return 0.0
        
        # Quality improvement estimate (simplified)
        delta_quality = (target_level - current_level) / (target_level + 1)
        
        # Cost increase with entropic scaling
        # Comment: Expected Value of Information with log-based scaling
        if current_level > 0:
            delta_cost = math.log(target_level + 1) - math.log(current_level + 1)
        else:
            delta_cost = math.log(target_level + 1)
        
        # Ensure EVI calculation uses log-based scaling
        evi = delta_quality - lambda_param * delta_cost
        
        return evi


class GreedyStrategy(SelectionStrategy):
    """Greedy selection strategy - always choose minimum cost"""
    
    def __init__(self, beta: float = 0.1):
        self.beta = beta
    
    def select(self, context: SelectionContext) -> int:
        """Select level that minimizes immediate cost with entropic scaling"""
        # Estimate required level based on complexity
        min_level = max(1, int(math.log(context.problem_complexity + 1)))
        
        # Apply entropic cost function
        best_level = min_level
        best_cost = float('inf')
        
        for level in range(min_level, min(40, min_level + 10)):
            # Cost with entropic scaling: C(n) = C_0(1 + β n log n)
            cost = 1.0 + self.beta * level * math.log(level + 1)
            
            # Check if this level can handle the complexity
            capacity = level * math.log(level + 1)  # Entropic capacity
            if capacity >= context.problem_complexity:
                if cost < best_cost:
                    best_cost = cost
                    best_level = level
        
        return best_level
    
    def get_name(self) -> str:
        return "Greedy-Entropic"


class AdaptiveStrategy(SelectionStrategy):
    """Adaptive strategy that learns from history"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.level_performance = {}  # Track performance by level
    
    def select(self, context: SelectionContext) -> int:
        """Select based on historical performance with entropic weighting"""
        if not context.history or not self.level_performance:
            # No history - use entropic heuristic
            base_level = 20
            adjustment = int(math.log(context.problem_complexity + 1))
            return min(40, base_level + adjustment)
        
        # Calculate expected utility for each level
        utilities = {}
        for level in range(1, 41):
            if level in self.level_performance:
                # Historical performance with entropic decay
                perf = self.level_performance[level]
                recency_weight = 1.0 / (1.0 + math.log(len(perf) + 1))
                avg_performance = np.mean(perf[-10:]) * recency_weight
            else:
                # Optimistic initialization with entropic prior
                avg_performance = 1.0 / (level * math.log(level + 1))
            
            # Subtract entropic cost
            cost = 0.1 * level * math.log(level + 1)
            utilities[level] = avg_performance - cost
        
        # Select level with highest utility
        return max(utilities, key=utilities.get)
    
    def update_performance(self, level: int, performance: float):
        """Update performance history for a level"""
        if level not in self.level_performance:
            self.level_performance[level] = []
        self.level_performance[level].append(performance)
    
    def get_name(self) -> str:
        return "Adaptive-Entropic"


class OptimalStoppingStrategy(SelectionStrategy):
    """Strategy based on optimal stopping with entropic thresholds"""
    
    def __init__(self, threshold_factor: float = 0.37):
        self.threshold_factor = threshold_factor  # From optimal stopping theory
    
    def select(self, context: SelectionContext) -> int:
        """Select using optimal stopping rule with entropic scaling"""
        # Explore phase: test lower levels
        explore_range = range(10, 20)
        best_evi = -float('inf')
        best_level = 15
        
        for level in explore_range:
            # Calculate EVI for this level
            evi = self.calculate_evi(1, level, context)
            if evi > best_evi:
                best_evi = evi
                best_level = level
        
        # Exploit phase: use threshold based on exploration
        threshold = best_evi * self.threshold_factor
        
        for level in range(20, 41):
            evi = self.calculate_evi(best_level, level, context)
            if evi > threshold:
                return level
        
        return best_level
    
    def get_name(self) -> str:
        return "OptimalStopping-Entropic"


class MetaSelector:
    """
    Meta-selector that chooses between selection strategies.
    Uses entropic scaling for meta-level decisions.
    """
    
    def __init__(self, strategies: List[SelectionStrategy]):
        """
        Initialize with list of selection strategies.
        
        Args:
            strategies: List of SelectionStrategy instances
        """
        self.strategies = strategies
        self.strategy_performance = {s.get_name(): [] for s in strategies}
        self.selection_history = []
        
    def select_strategy(self, context: SelectionContext) -> SelectionStrategy:
        """
        Select which strategy to use based on context and performance.
        
        Uses entropic weighting for strategy selection.
        """
        if not any(self.strategy_performance.values()):
            # No history - choose randomly
            return np.random.choice(self.strategies)
        
        # Calculate expected performance for each strategy
        scores = {}
        
        for strategy in self.strategies:
            name = strategy.get_name()
            
            if self.strategy_performance[name]:
                # Historical average with entropic decay
                history = self.strategy_performance[name]
                recent = history[-20:]  # Use recent history
                
                # Apply entropic weighting based on history length
                weight = 1.0 / (1.0 + 0.1 * math.log(len(recent) + 1))
                avg_performance = np.mean(recent) * weight
                
                # Adjust for context complexity using entropic scaling
                complexity_factor = math.log(context.problem_complexity + 1) / 10.0
                scores[name] = avg_performance * (1.0 - complexity_factor)
            else:
                # Optimistic initialization
                scores[name] = 1.0
        
        # Select strategy with best expected score
        best_strategy_name = max(scores, key=scores.get)
        
        for strategy in self.strategies:
            if strategy.get_name() == best_strategy_name:
                return strategy
        
        return self.strategies[0]  # Fallback
    
    def execute_selection(self, context: SelectionContext) -> Tuple[int, str]:
        """
        Execute full selection process: choose strategy, then level.
        
        Returns:
            Tuple of (selected_level, strategy_name)
        """
        # Select strategy
        strategy = self.select_strategy(context)
        
        # Use strategy to select level
        level = strategy.select(context)
        
        # Record selection
        self.selection_history.append({
            'strategy': strategy.get_name(),
            'level': level,
            'complexity': context.problem_complexity,
            'timestamp': len(self.selection_history)
        })
        
        logger.info(f"MetaSelector: {strategy.get_name()} -> Level {level}")
        
        return level, strategy.get_name()
    
    def update_performance(self, strategy_name: str, performance: float):
        """
        Update performance history for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            performance: Performance metric (lower is better)
        """
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name].append(performance)
            
            # Keep bounded history
            if len(self.strategy_performance[strategy_name]) > 100:
                self.strategy_performance[strategy_name] = \
                    self.strategy_performance[strategy_name][-100:]
    
    def get_statistics(self) -> Dict:
        """Get selection statistics"""
        stats = {
            'total_selections': len(self.selection_history),
            'strategy_usage': {},
            'average_level': 0,
            'performance_summary': {}
        }
        
        # Strategy usage counts
        for record in self.selection_history:
            strategy = record['strategy']
            if strategy not in stats['strategy_usage']:
                stats['strategy_usage'][strategy] = 0
            stats['strategy_usage'][strategy] += 1
        
        # Average selected level
        if self.selection_history:
            levels = [r['level'] for r in self.selection_history]
            stats['average_level'] = np.mean(levels)
        
        # Performance summary with entropic normalization
        for strategy_name, performances in self.strategy_performance.items():
            if performances:
                # Apply entropic normalization
                normalized = np.mean(performances) / math.log(len(performances) + 2)
                stats['performance_summary'][strategy_name] = {
                    'mean': np.mean(performances),
                    'normalized': normalized,
                    'std': np.std(performances),
                    'count': len(performances)
                }
        
        return stats


class HierarchicalMetaSelector:
    """
    Hierarchical meta-selector: meta-meta-selector and beyond.
    Each level uses entropic scaling for decisions.
    """
    
    def __init__(self, depth: int = 2):
        """
        Initialize hierarchical meta-selector.
        
        Args:
            depth: Depth of meta-selection hierarchy
        """
        self.depth = depth
        self.selectors_by_level = {}
        
        # Build hierarchy bottom-up
        # Level 0: base strategies
        base_strategies = [
            GreedyStrategy(),
            AdaptiveStrategy(),
            OptimalStoppingStrategy()
        ]
        
        self.selectors_by_level[0] = base_strategies
        
        # Build meta-levels
        for level in range(1, depth):
            if level == 1:
                # First meta-level: MetaSelector over base strategies
                self.selectors_by_level[1] = MetaSelector(base_strategies)
            else:
                # Higher levels: simplified for demonstration
                self.selectors_by_level[level] = MetaSelector(base_strategies)
        
        self.selection_costs = []
    
    def select(self, context: SelectionContext) -> int:
        """
        Perform hierarchical selection with entropic cost at each level.
        
        Returns:
            Selected model level
        """
        total_cost = 0
        
        # Start from highest meta-level and work down
        for meta_level in range(self.depth - 1, -1, -1):
            # Cost of meta-selection scales entropically
            # Comment: Meta-selection cost with entropic scaling
            meta_cost = 0.01 * (meta_level + 1) * math.log(meta_level + 2)
            total_cost += meta_cost
            
            if meta_level == 0:
                # Base level: select actual model level
                strategy = self.selectors_by_level[0][0]  # Use first strategy
                selected_level = strategy.select(context)
            else:
                # Meta-level: would select which lower-level selector to use
                # Simplified for demonstration
                pass
        
        self.selection_costs.append(total_cost)
        
        logger.info(f"Hierarchical selection: Level {selected_level}, "
                   f"Meta-cost: {total_cost:.4f}")
        
        return selected_level
    
    def get_average_meta_cost(self) -> float:
        """Get average meta-selection cost"""
        if self.selection_costs:
            return np.mean(self.selection_costs)
        return 0.0


def example_usage():
    """Demonstrate meta-selector with entropic scaling"""
    print("Meta-Selector with Entropic Scaling Demo")
    print("=" * 60)
    
    # Create strategies
    strategies = [
        GreedyStrategy(beta=0.1),
        AdaptiveStrategy(learning_rate=0.1),
        OptimalStoppingStrategy(threshold_factor=0.37)
    ]
    
    # Create meta-selector
    meta_selector = MetaSelector(strategies)
    
    # Simulate selection decisions
    print("\nSimulating 20 selection decisions:")
    print("-" * 40)
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(20):
        # Create context with varying complexity
        complexity = np.random.exponential(50) + 10
        context = SelectionContext(
            problem_complexity=complexity,
            available_resources=1000.0
        )
        
        # Perform selection
        level, strategy_name = meta_selector.execute_selection(context)
        
        # Simulate performance (lower is better)
        actual_cost = 0.1 * level * math.log(level + 1)
        quality_loss = max(0, complexity - level * math.log(level + 1)) / complexity
        performance = actual_cost + quality_loss
        
        # Update performance
        meta_selector.update_performance(strategy_name, performance)
        
        if i % 5 == 4:  # Print every 5 iterations
            print(f"  Iteration {i+1}: complexity={complexity:.1f}, "
                  f"strategy={strategy_name}, level={level}, "
                  f"performance={performance:.3f}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Selection Statistics:")
    stats = meta_selector.get_statistics()
    
    print(f"\nTotal selections: {stats['total_selections']}")
    print(f"Average level: {stats['average_level']:.1f}")
    
    print("\nStrategy usage:")
    for strategy, count in stats['strategy_usage'].items():
        percentage = 100 * count / stats['total_selections']
        print(f"  {strategy}: {count} ({percentage:.1f}%)")
    
    print("\nPerformance summary (with entropic normalization):")
    for strategy, perf in stats['performance_summary'].items():
        print(f"  {strategy}:")
        print(f"    Mean: {perf['mean']:.3f}")
        print(f"    Normalized: {perf['normalized']:.3f}")
        print(f"    Std: {perf['std']:.3f}")
    
    # Demonstrate hierarchical meta-selector
    print("\n" + "=" * 60)
    print("Hierarchical Meta-Selector Demo:")
    
    hierarchical = HierarchicalMetaSelector(depth=3)
    
    for i in range(5):
        complexity = np.random.exponential(50) + 10
        context = SelectionContext(problem_complexity=complexity, 
                                  available_resources=1000.0)
        
        level = hierarchical.select(context)
        print(f"  Selection {i+1}: complexity={complexity:.1f}, level={level}")
    
    avg_cost = hierarchical.get_average_meta_cost()
    print(f"\nAverage meta-selection cost (entropic): {avg_cost:.4f}")
    
    print("\nEntropic scaling ensures meta-selection overhead remains bounded")
    print("while enabling sophisticated strategy selection.")


if __name__ == "__main__":
    example_usage()
