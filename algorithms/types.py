from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class Vantage6AlgorithmInput(BaseModel):
    method: str = Field(..., description="The method to execute for the algorithm")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Algorithm-specific keyword arguments (e.g., column names, filtering parameters)"
    )

class Vantage6AlgorithmConfig(BaseModel):
    image: str = Field(..., description="Docker image for the algorithm")
    name: str = Field(..., description="The name of the task")
    description: Optional[str] = Field(None, description="A short description of the task")
    # Use alias 'input' to match the original task creation API parameter if needed.
    input_: Vantage6AlgorithmInput = Field(..., alias="input", description="Input configuration containing the method and kwargs")
    organizations: List[int] = Field(..., description="List of organization IDs on which to run the task")
    collaboration: int = Field(..., description="Collaboration ID for the task")
    databases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of database configurations to be used by the algorithm"
    )
    # Additional optional parameters you might consider in the future:
    timeout: Optional[int] = Field(None, description="Optional timeout for the task execution in seconds")
    retries: Optional[int] = Field(0, description="Optional number of retries if the task fails")

    class Config:
        allow_population_by_field_name = True  # Allows us to use the alias 'input' when parsing config data

# Example usage: constructing configuration for different algorithms

# Configuration for a Kaplan-Meier analysis
km_config = Vantage6AlgorithmConfig(
    image='harbor2.vantage6.ai/algorithms/kaplan-meier',
    name='demo-km-analysis',
    description='Kaplan-Meier dry-run',
    input={
        "method": "kaplan_meier_central",
        "kwargs": {
            "time_column_name": "Survival.time",
            "censor_column_name": "deadstatus.event",
            "organizations_to_include": [1, 2, 3]
        }
    },
    organizations=[2],
    collaboration=1,
    databases=[{'label': 'default'}]
)

# Configuration for an average computation algorithm
avg_config = Vantage6AlgorithmConfig(
    image='ghcr.io/mdw-nl/v6-average-py:v1.0.1',
    name='demo-average',
    description='Average dry-run',
    input={
        "method": "central_average",
        "kwargs": {
            "column_name": ["age"],
            "org_ids": [1, 2, 3]
        }
    },
    organizations=[2],
    collaboration=1,
    databases=[{'label': 'default'}]
)
