from agents.agent import *


def create_agent(agent_type: AgentType, *args, **kwargs):
    logger = logging.getLogger(__name__)
    if agent_type is AgentType.RANDOM_WALK_AGENT:
        logger.info("Creating RandomWalkAgent.")
        return PerimeterFollowingAgent(*args, **kwargs)
    else:
        logger.warning("No agent type specified.")
        return None
