from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

class Config(BaseModel):
    # Paths
    fino_src_path: Path
    fino_dst_path: Path

    sar_src_path: Path
    sar_dst_path: Path

    db_path: Path
    images_path: Path

    # FINO1 coordinates
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)

    # Target tile dimensions
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class ConfigManager:
    config: Optional[Config] = None
    
    def __init__(self):
        pass

    @classmethod
    def from_file(self, path: str | Path) -> "ConfigManager":
        """
        Load configuration from a Python config file.
        """
        path = Path(path)

        namespace = {}
        with open(path, "r", encoding="utf-8") as f:
            exec(f.read(), {}, namespace)

        self.config = Config(
            fino_src_path=namespace["fino_src_path"],
            fino_dst_path=namespace["fino_dst_path"],
            sar_src_path=namespace["sar_src_path"],
            sar_dst_path=namespace["sar_dst_path"],
            db_path=namespace["db_path"],
            images_path=namespace["images_path"],
            lat=namespace["lat"],
            lon=namespace["lon"],
            width=namespace["width"],
            height=namespace["height"],
        )

        return self.config

    def save(self, path: str | Path) -> None:
        """
        Save configuration to a Python config file.
        """
        path = Path(path)

    def __getattr__(self, name):
        """
        Allow direct access:
            config.fino_src_path
            config.lat
            config.width
        """
        return getattr(self.config, name)
