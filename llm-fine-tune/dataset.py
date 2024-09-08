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
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self._data)
            step = idx.step if idx.step is not None else 1
            return [x[self._key] for x in self._data[start:stop:step]]
        elif isinstance(idx, int):
            return self._data[idx][self._key]
        else:
            raise TypeError("invalid index, not an integer or slice")

    def __len__(self):
        return len(self._data)

        