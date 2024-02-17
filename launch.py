"""Launch script for fine-tuning a model."""
from aihero.research.finetuning import sft
from fire import Fire

if __name__ == "__main__":
    Fire(sft)
