from .agent import BaseAgent
from .dqn_agent import DQNAgent
from .target_dqn_agent import TargetDQNAgent
from .double_dqn_agent import DoubleDQNAgent

AGENTS: dict[str, type[BaseAgent]] = {
    'dqn': DQNAgent,
    'target_dqn': TargetDQNAgent,
    'double_dqn': DoubleDQNAgent,
}

def get_agent_class(name: str) -> type[BaseAgent]:
    if name not in AGENTS:
        raise ValueError(f'Unknown agent: {name}. Available: {list(AGENTS)}')
    return AGENTS[name]
