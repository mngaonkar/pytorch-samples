from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TrainingDataset():
    """
    Load JSONL files from a directory
    """
    
    def __init__(self, path: Path, key:str = "text"):
        if not path.exists():
            err = f"dataset path {path} does not exist."
            logger.critical(err)
            self._data = None

            raise Exception(err)
        else:
            with open(path, "r") as fp:
                self._data = [json.loads(line) for line in fp]
            self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)

        