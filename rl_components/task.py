from datetime import datetime
from typing import Optional, Any
import random
import pandas as pd

SLA_FACTOR = 1.2  # e.g., SLA is 20% longer than duration

class Task:
    """
    Represents a computing task in a datacenter scheduling system.

    Attributes:
        job_name (str): Unique identifier or name of the job.
        arrival_time (datetime): The timestamp when the task enters the system.
        duration (float): Execution time required for the task (in minutes).
        cpu_req (float): Number of CPU cores required.
        gpu_req (float): Number of GPU units required.
        mem_req (float): Memory required (in GB).
        bandwidth_gb (float): Bandwidth required (in GB).
        start_time (Optional[datetime]): Timestamp when the task starts execution (None if not started).
        finish_time (Optional[datetime]): Expected completion time of the task (None if not scheduled).
        wait_time (int): Time the task has spent waiting in the queue (in steps).
        origin_dc_id (Optional[int]): ID of the data center where the task originated.
        extra_info (Optional[Any]): Additional metadata related to the task.
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
        extra_info: Optional[Any] = None,
        origin_dc_id: Optional[int] = None
    ) -> None:
        """
        Initializes a new task with required computing resources.

        Args:
            job_name (str): Unique identifier or name of the job.
            arrival_time (datetime): Timestamp when the task enters the system.
            duration (float): Execution time required for the task (in minutes).
            cpu_req (float): Number of CPU cores required.
            gpu_req (float): Number of GPU units required.
            mem_req (float): Memory required (in GB).
            bandwidth_gb (float): Bandwidth required (in GB).
            extra_info (Optional[Any]): Additional metadata related to the task (default: None).
            origin_dc_id (Optional[int]): ID of the data center where the task originated (default: None).
        """
        self.job_name = job_name
        self.arrival_time = arrival_time
        self.duration = duration
        self.cpu_req = cpu_req
        self.gpu_req = gpu_req
        self.mem_req = mem_req
        self.bandwidth_gb = bandwidth_gb
        self.start_time: Optional[datetime] = None  # Set when task starts execution
        self.finish_time: Optional[datetime] = None  # Set when task is scheduled
        self.sla_deadline = arrival_time + pd.Timedelta(minutes=SLA_FACTOR * duration)
        self.sla_met = False  # Indicates if the SLA was met
        self.wait_time: int = 0  # Tracks time the task spends in the queue
        self.extra_info = extra_info  # Optional metadata for extended functionality
        self.origin_dc_id = origin_dc_id  # ID of the data center where the task originated
        self.dest_dc_id: Optional[int] = None  # ID of the data center where the task is sent
        self.dest_dc: Optional[int] = None
        self.temporarily_deferred = False  # Indicates task is waiting for assignment

        
        # Append to the job_name a random number to make it unique
        self.job_name += f"_{(random.randint(0, 10000))}"

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
            f"finish_time={self.finish_time}, wait_time={self.wait_time}, origin_dc_id={self.origin_dc_id})"
        )

    def increment_wait_time(self) -> None:
        """ Increments the wait time counter when the task remains in the queue. """
        self.wait_time += 1

    def is_scheduled(self) -> bool:
        """ Checks if the task has been scheduled (i.e., has a start time). """
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
