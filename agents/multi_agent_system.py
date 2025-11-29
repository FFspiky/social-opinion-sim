from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

from .agent import Agent


class MultiAgentSystem:
    """
    简单的多代理容器，可通过现有 Agent 列表或工厂函数批量创建并驱动互动。
    """

    def __init__(
        self,
        agent_count: int = 0,
        topics: Optional[Sequence[str]] = None,
        agents: Optional[Iterable[Agent]] = None,
        agent_factory: Optional[Callable[[int, List[str]], Agent]] = None,
    ) -> None:
        self.topics = list(topics) if topics else []
        self.agents: List[Agent] = []

        if agents is not None:
            self.agents.extend(agents)
        elif agent_count > 0:
            if agent_factory is None:
                raise ValueError("需要提供 agent_factory 或现成的 agents 才能构造多代理系统")
            for idx in range(agent_count):
                self.agents.append(agent_factory(idx, self.topics))

        if not self.agents:
            raise ValueError("MultiAgentSystem 至少需要一个 Agent 实例")

    def run_simulation_step(self, environment) -> None:
        """
        逐个驱动代理与环境交互一次。
        """
        for agent in self.agents:
            agent.interact(environment)
