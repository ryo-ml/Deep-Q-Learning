from .dqn_agent import DQNAgent, TargetDQNAgent, DoubleDQNAgent

AGENTS: dict[str, type[DQNAgent]] = {
    'dqn': DQNAgent,
    'target_dqn': TargetDQNAgent,
    'double_dqn': DoubleDQNAgent,
}

def get_agent_class(name: str) -> type[DQNAgent]:
    if name not in AGENTS:
        raise ValueError(f'Unknown agent: {name}. Available: {list(AGENTS)}')
    return AGENTS[name]
