"""
llm_cohorts.py - Sleep-Wake Orchestration in Hierarchical LLM Cohorts

Implements sleep-wake orchestration for hierarchical LLM systems with entropic
time scaling for training/processing slots.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import heapq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelState(Enum):
    """States for models in the cohort"""
    SLEEPING = "sleeping"  # Training/updating
    AWAKE = "awake"       # Available for inference
    TRANSITIONING = "transitioning"  # Switching states


@dataclass
class LLMModel:
    """Represents an LLM in the hierarchical cohort"""
    id: str
    level: int  # Hierarchy level n
    state: ModelState = ModelState.AWAKE
    last_update: datetime = field(default_factory=datetime.now)
    kappa: float = 1.0  # Entropic scaling constant
    
    @property
    def information_capacity(self) -> float:
        """Information capacity with entropic scaling I(n) = κ n log n"""
        if self.level <= 0:
            return 0
        return self.kappa * self.level * math.log(self.level + 1)
    
    @property 
    def sleep_duration(self, tau_0: float = 50, gamma: float = 1.0) -> float:
        """
        Sleep duration (training time) with entropic scaling.
        τ_sleep = τ₀ + γ n log n
        
        Comment: Entropic time complexity for hierarchical processing
        """
        if self.level <= 0:
            return tau_0
        return tau_0 + gamma * self.level * math.log(self.level + 1)


@dataclass
class SleepSlot:
    """Represents a sleep/training slot"""
    start_time: datetime
    duration: float  # In minutes, scaled entropically
    model_id: str
    level: int
    priority: float = 0.0
    
    @property
    def end_time(self) -> datetime:
        """Calculate slot end time"""
        return self.start_time + timedelta(minutes=self.duration)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.start_time < other.start_time


class SleepWakeOrchestrator:
    """
    Orchestrates sleep-wake cycles for LLM cohort with entropic time scaling.
    Ensures continuous availability while managing training schedules.
    """
    
    def __init__(self,
                 models: List[LLMModel],
                 min_awake_per_level: int = 1,
                 base_sleep_minutes: float = 60.0,
                 gamma_time: float = 1.0):
        """
        Initialize orchestrator with entropic scheduling.
        
        Args:
            models: List of LLM models in the cohort
            min_awake_per_level: Minimum models awake per hierarchy level
            base_sleep_minutes: Base sleep duration τ₀
            gamma_time: Entropic time scaling factor γ
        """
        self.models = {model.id: model for model in models}
        self.min_awake_per_level = min_awake_per_level
        self.base_sleep_minutes = base_sleep_minutes
        self.gamma_time = gamma_time
        
        # Group models by level
        self.models_by_level = self._group_by_level()
        
        # Sleep schedule (priority queue)
        self.sleep_schedule = []
        
        # Wake schedule
        self.wake_schedule = []
        
        # Resource pools
        self.sleep_resources = 4  # Number of parallel training slots
        self.current_sleeping = set()
    
    def _group_by_level(self) -> Dict[int, List[str]]:
        """Group model IDs by hierarchy level"""
        groups = {}
        for model in self.models.values():
            if model.level not in groups:
                groups[model.level] = []
            groups[model.level].append(model.id)
        return groups
    
    def calculate_sleep_duration(self, model: LLMModel) -> float:
        """
        Calculate sleep duration for a model using entropic scaling.
        τ_sleep = τ₀ + γ n log n
        
        Comment: Entropic time complexity for hierarchical processing
        """
        if model.level <= 0:
            return self.base_sleep_minutes
            
        entropic_factor = model.level * math.log(model.level + 1)
        duration = self.base_sleep_minutes + self.gamma_time * entropic_factor
        
        logger.debug(f"Model {model.id} (level {model.level}): "
                    f"sleep duration = {duration:.1f} min (entropic scaling)")
        
        return duration
    
    def can_sleep(self, model_id: str, current_time: datetime) -> bool:
        """
        Check if a model can go to sleep while maintaining availability.
        """
        model = self.models[model_id]
        level = model.level
        
        # Count awake models at this level
        awake_at_level = sum(1 for mid in self.models_by_level[level]
                            if self.models[mid].state == ModelState.AWAKE
                            and mid != model_id)
        
        # Check minimum availability constraint
        if awake_at_level < self.min_awake_per_level:
            return False
        
        # Check resource availability
        if len(self.current_sleeping) >= self.sleep_resources:
            return False
        
        # Check if model needs update (simplified: every 24 hours)
        time_since_update = current_time - model.last_update
        if time_since_update.total_seconds() < 86400:  # 24 hours
            return False
        
        return True
    
    def schedule_sleep(self, model_id: str, start_time: datetime) -> Optional[SleepSlot]:
        """
        Schedule a sleep slot for a model with entropic duration.
        """
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        
        if not self.can_sleep(model_id, start_time):
            return None
        
        # Calculate entropic sleep duration
        duration = self.calculate_sleep_duration(model)
        
        # Create sleep slot
        slot = SleepSlot(
            start_time=start_time,
            duration=duration,
            model_id=model_id,
            level=model.level,
            priority=-model.level  # Higher levels get priority
        )
        
        # Add to schedule
        heapq.heappush(self.sleep_schedule, slot)
        
        # Update model state
        model.state = ModelState.SLEEPING
        self.current_sleeping.add(model_id)
        
        logger.info(f"Scheduled sleep for {model_id} at {start_time} "
                   f"for {duration:.1f} minutes (entropic scaling)")
        
        return slot
    
    def wake_model(self, model_id: str, wake_time: datetime):
        """
        Wake a model from sleep.
        """
        if model_id not in self.models:
            return
        
        model = self.models[model_id]
        model.state = ModelState.AWAKE
        model.last_update = wake_time
        
        if model_id in self.current_sleeping:
            self.current_sleeping.remove(model_id)
        
        logger.info(f"Model {model_id} (level {model.level}) woke at {wake_time}")
    
    def process_schedule(self, current_time: datetime):
        """
        Process the sleep-wake schedule at current time.
        """
        # Wake models whose sleep has ended
        while self.sleep_schedule:
            next_slot = self.sleep_schedule[0]
            if next_slot.end_time <= current_time:
                heapq.heappop(self.sleep_schedule)
                self.wake_model(next_slot.model_id, next_slot.end_time)
            else:
                break
        
        # Try to schedule new sleeps
        for model_id, model in self.models.items():
            if model.state == ModelState.AWAKE:
                if self.can_sleep(model_id, current_time):
                    # Add some randomness to prevent synchronization
                    delay = np.random.exponential(10)  # Minutes
                    start_time = current_time + timedelta(minutes=delay)
                    self.schedule_sleep(model_id, start_time)
    
    def get_available_models(self, level: Optional[int] = None) -> List[str]:
        """
        Get list of currently available (awake) models.
        
        Args:
            level: If specified, only return models at this level
            
        Returns:
            List of available model IDs
        """
        available = []
        for model_id, model in self.models.items():
            if model.state == ModelState.AWAKE:
                if level is None or model.level == level:
                    available.append(model_id)
        return available
    
    def get_schedule_summary(self) -> Dict:
        """
        Get summary of current schedule.
        """
        summary = {
            'total_models': len(self.models),
            'sleeping': len(self.current_sleeping),
            'awake': len(self.models) - len(self.current_sleeping),
            'scheduled_slots': len(self.sleep_schedule),
            'by_level': {}
        }
        
        # Count by level
        for level, model_ids in self.models_by_level.items():
            awake = sum(1 for mid in model_ids 
                       if self.models[mid].state == ModelState.AWAKE)
            sleeping = len(model_ids) - awake
            
            # Calculate average sleep duration for this level (entropic)
            avg_duration = self.base_sleep_minutes + self.gamma_time * level * math.log(level + 1) if level > 0 else self.base_sleep_minutes
            
            summary['by_level'][level] = {
                'total': len(model_ids),
                'awake': awake,
                'sleeping': sleeping,
                'avg_sleep_duration': f"{avg_duration:.1f} min"
            }
        
        return summary
    
    def optimize_schedule(self):
        """
        Optimize the sleep-wake schedule to minimize resource usage
        while maintaining availability constraints.
        
        Uses entropic scaling for resource allocation.
        """
        # Calculate resource requirements per level
        resource_requirements = {}
        
        for level in self.models_by_level:
            # Resource requirement scales entropically
            # Comment: Entropic scaling for resource allocation
            req = self.gamma_time * level * math.log(level + 1) if level > 0 else self.gamma_time
            resource_requirements[level] = req
        
        # Allocate sleep slots proportionally
        total_req = sum(resource_requirements.values())
        
        for level, req in resource_requirements.items():
            # Allocate slots based on entropic requirements
            allocated_slots = max(1, int(self.sleep_resources * req / total_req))
            logger.info(f"Level {level}: allocated {allocated_slots} sleep slots "
                       f"(entropic scaling: {req:.2f})")


class CohortManager:
    """
    Manages multiple orchestrators for different cohorts.
    """
    
    def __init__(self):
        """Initialize cohort manager"""
        self.cohorts = {}
        self.orchestrators = {}
    
    def create_cohort(self, 
                      cohort_id: str,
                      level_distribution: Dict[int, int],
                      kappa: float = 1.0) -> SleepWakeOrchestrator:
        """
        Create a new cohort with specified level distribution.
        
        Args:
            cohort_id: Unique identifier for cohort
            level_distribution: {level: count} dictionary
            kappa: Entropic scaling constant
            
        Returns:
            Orchestrator for the new cohort
        """
        models = []
        
        for level, count in level_distribution.items():
            for i in range(count):
                model = LLMModel(
                    id=f"{cohort_id}_L{level}_M{i}",
                    level=level,
                    kappa=kappa
                )
                models.append(model)
        
        orchestrator = SleepWakeOrchestrator(models)
        
        self.cohorts[cohort_id] = models
        self.orchestrators[cohort_id] = orchestrator
        
        logger.info(f"Created cohort {cohort_id} with {len(models)} models")
        
        return orchestrator
    
    def get_global_availability(self) -> Dict[int, int]:
        """
        Get global availability across all cohorts.
        
        Returns:
            Dictionary of {level: available_count}
        """
        availability = {}
        
        for orchestrator in self.orchestrators.values():
            for level, model_ids in orchestrator.models_by_level.items():
                if level not in availability:
                    availability[level] = 0
                
                awake_count = sum(1 for mid in model_ids
                                if orchestrator.models[mid].state == ModelState.AWAKE)
                availability[level] += awake_count
        
        return availability


def example_usage():
    """
    Demonstrate sleep-wake orchestration with entropic scaling.
    """
    print("Sleep-Wake Orchestration with Entropic Scaling")
    print("=" * 60)
    
    # Create a cohort with various hierarchy levels
    manager = CohortManager()
    
    # Distribution of models across levels
    level_distribution = {
        25: 4,  # 4 models at level 25
        30: 3,  # 3 models at level 30
        35: 2,  # 2 models at level 35
        40: 1   # 1 model at level 40
    }
    
    orchestrator = manager.create_cohort("main_cohort", level_distribution)
    
    # Show initial state
    print("\nInitial Cohort State:")
    summary = orchestrator.get_schedule_summary()
    print(f"  Total Models: {summary['total_models']}")
    print(f"  Awake: {summary['awake']}")
    print(f"  Sleeping: {summary['sleeping']}")
    
    print("\nBy Level (with entropic sleep durations):")
    for level, info in sorted(summary['by_level'].items()):
        print(f"  Level {level}:")
        print(f"    Models: {info['total']} (awake: {info['awake']})")
        print(f"    Avg Sleep Duration: {info['avg_sleep_duration']}")
    
    # Simulate scheduling over time
    print("\n" + "=" * 60)
    print("Simulating Schedule (24 hours):")
    
    current_time = datetime.now()
    
    for hour in range(24):
        sim_time = current_time + timedelta(hours=hour)
        orchestrator.process_schedule(sim_time)
        
        if hour % 6 == 0:  # Report every 6 hours
            print(f"\nHour {hour}:")
            available = orchestrator.get_available_models()
            print(f"  Available models: {len(available)}")
            
            for level in sorted(level_distribution.keys()):
                level_available = orchestrator.get_available_models(level)
                capacity_total = sum(orchestrator.models[mid].information_capacity 
                                   for mid in level_available)
                print(f"  Level {level}: {len(level_available)} models, "
                     f"{capacity_total:.1f} bits total capacity (entropic)")
    
    # Optimize schedule
    print("\n" + "=" * 60)
    print("Optimizing Schedule with Entropic Resource Allocation:")
    orchestrator.optimize_schedule()
    
    # Final summary
    print("\n" + "=" * 60)
    print("Final Summary:")
    final_summary = orchestrator.get_schedule_summary()
    
    total_capacity_awake = sum(
        orchestrator.models[mid].information_capacity
        for mid in orchestrator.get_available_models()
    )
    
    print(f"  Total awake models: {final_summary['awake']}")
    print(f"  Total sleeping models: {final_summary['sleeping']}")
    print(f"  Total available capacity: {total_capacity_awake:.1f} bits (entropic)")
    print("\nEntropic scaling ensures efficient resource utilization")
    print("while maintaining service availability at all hierarchy levels.")


if __name__ == "__main__":
    example_usage()
