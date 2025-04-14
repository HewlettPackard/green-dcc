from datetime import datetime
from typing import Optional, Any
import random
import pandas as pd

class Task:
    """
    Represents a computing task in a datacenter scheduling system.
    
    Each task is assigned resource requirements, timing attributes, and an SLA deadline.
    Tasks can be deferred if they are not immediately scheduled, and they are tracked until
    they complete execution.

    Attributes:
        job_name (str): Unique identifier for the task.
        arrival_time (datetime): Timestamp when the task enters the system.
        duration (float): Required execution time (in minutes).
        cpu_req (float): Number of CPU cores required.
        gpu_req (float): Number of GPU units required.
        mem_req (float): Memory required (in GB).
        bandwidth_gb (float): Bandwidth required (in GB).
        start_time (Optional[datetime]): When the task begins execution.
        finish_time (Optional[datetime]): Expected completion time post-scheduling.
        sla_deadline (datetime): Deadline computed as `arrival_time + sla_multiplier * duration`.
        sla_met (bool): Indicator whether the task met its SLA.
        wait_intervals (int): Timestep counter for how long the task has been waiting.
        origin_dc_id (Optional[int]): ID of the datacenter where the task originated.
        dest_dc_id (Optional[int]): ID of the assigned destination datacenter.
        dest_dc (Optional[Any]): Reference to the destination datacenter.
        temporarily_deferred (bool): Flag indicating if the task is deferred for later assignment.
        sla_multiplier (int): Multiplier for SLA deadline calculation.
    """
    
    def __init__(
        self,
        job_name: str,
        arrival_time: datetime,
        duration: float,
        cpu_req: float,
        gpu_req: float,
        mem_req: float,
        bandwidth_gb: float,
        sla_multiplier: float = 1.5  # Default SLA multiplier
    ) -> None:
        # Initialize task properties
        self.job_name = job_name
        self.arrival_time = arrival_time
        self.duration = duration
        self.cpu_req = cpu_req
        self.gpu_req = gpu_req
        self.mem_req = mem_req
        self.bandwidth_gb = bandwidth_gb
        
        # Timing properties: to be set upon scheduling by the global scheduler
        self.start_time: Optional[datetime] = None
        self.finish_time: Optional[datetime] = None
        
        # Compute the SLA deadline based on a fixed factor
        self.sla_multiplier = sla_multiplier
        self.sla_deadline = arrival_time + pd.Timedelta(minutes=self.sla_multiplier * duration)
        self.sla_met = False
        
        # Initialize wait time counter
        self.wait_intervals: int = 0
        
        # Record the origin datacenter
        self.origin_dc_id: Optional[int] = None
        self.origin_dc: Optional[Any] = None
        
        # Destination information will be assigned when the task is routed
        self.dest_dc_id: Optional[int] = None
        self.dest_dc: Optional[Any] = None
        
        # Flag to indicate if the task is deferred for future scheduling
        self.temporarily_deferred = False
        
        # Ensure unique identification by appending a random number
        self.job_name += f"_{random.randint(0, 10000)}"

    def __repr__(self) -> str:
        """
        Returns a string representation of the Task object for debugging.

        Returns:
            str: A formatted string representation of the task.
        """
        return (
            f"Task(job_name='{self.job_name}', arrival_time={self.arrival_time}, "
            f"duration={self.duration}, cpu_req={self.cpu_req}, gpu_req={self.gpu_req}, "
            f"mem_req={self.mem_req}, bandwidth_gb={self.bandwidth_gb}, start_time={self.start_time}, "
            f"finish_time={self.finish_time}, wait_intervals={self.wait_intervals}, origin_dc_id={self.origin_dc_id})"
        )

    def increment_wait_intervals(self) -> None:
        """
        Increments the wait time counter by 1 timestep.
        """
        self.wait_intervals += 1

    def is_scheduled(self) -> bool:
        """
        Checks if the task has been scheduled (i.e., if a start time is defined).

        Returns:
            bool: True if scheduled, False otherwise.
        """
        return self.start_time is not None

    def is_completed(self, current_time: datetime) -> bool:
        """
        Determines whether the task has finished execution.

        Args:
            current_time (datetime): The current timestamp in the system.

        Returns:
            bool: True if the task has finished, False otherwise.
        """
        return self.finish_time is not None and current_time >= self.finish_time
