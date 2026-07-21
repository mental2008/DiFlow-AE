from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal


@dataclass
class Request:
    request_type: Literal["fetch"]
    request_id: str
