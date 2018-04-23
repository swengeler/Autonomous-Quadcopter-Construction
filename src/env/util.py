from enum import Enum
from typing import List
from env.block import *


class Occupancy(Enum):
    UNOCCUPIED = 0
    OCCUPIED = 1
    SEED = 2


class BlockType(Enum):
    INERT = 0


class BlockGeneratorInfo:
    def __init__(self, block_type: BlockType, size: float, color: str, count: int):
        self.type = block_type
        self.size = size
        self.color = color
        self.count = count


def create_block(block_type, *args, **kwargs):
    logger = logging.getLogger(__name__)
    if block_type is BlockType.INERT:
        logger.info("Creating Block.")
        return Block(*args, **kwargs)
    else:
        logger.warning("No block type specified.")
        return None


def create_block_list(block_generator_info):
    blocks = []
    for bgi in block_generator_info:
        for _ in range(0, bgi.count):
            blocks.append(create_block(bgi.type, size=bgi.size, color=bgi.color))
    return blocks
