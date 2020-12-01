import json
from json import load
import json_tricks
import os
import pandas as pd
from typing import Dict, List, Optional

import attr

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder, not_none_validator
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.rearrangement.rearrangement_task import (
    RearrangementEpisode,
    RearrangementObjectSpec,
    RearrangementSpec,
)

def load_replay_data(filter_path):
    # p = 'data/new_checkpoints/b9KSrZfBpB5WC5gcCUtmMX/replays/map_object/6T4DVDAzwC2ZzkEtGeoerS/'
    # data/new_checkpoints/b9KSrZfBpB5WC5gcCUtmMX/videos/map_object/dPKBtXvPV6S2RyA8HVckGZ
    if len(filter_path)!=0 and os.path.exists(filter_path):
        return pd.read_pickle(filter_path)
    return None

@registry.register_dataset(name="RearrangementDataset-v0")
class RearrangementDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads the Rearrangement dataset."""
    episodes: List[RearrangementEpisode]
    object_templates: Dict = attr.ib(default={}, validator=not_none_validator)
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)


    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, filter_scenes_path = ""
    ) -> None:
        deserialized = json.loads(json_str)
        self.replay_df = load_replay_data(filter_scenes_path)

        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "object_templates" in deserialized:
            self.object_templates = deserialized["object_templates"]

        # self.object_templates = deserialized['object_templates']
        for i, episode in enumerate(deserialized["episodes"]):
            episode['episode_id'] = str(episode['episode_id'])

            if self.replay_df is not None and not self.replay_df[
                (self.replay_df['episode_id'] == episode['episode_id']) &
                (self.replay_df['scene_id'] == episode['scene_id'])
            ].empty:
                continue

            episode_obj = RearrangementEpisode(**episode)
            # episode_obj.episode_id = str(episode.episode_id)

            if scenes_dir is not None:
                if episode_obj.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode_obj.scene_id = episode_obj.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode_obj.scene_id = os.path.join(
                    scenes_dir, episode_obj.scene_id
                )

            for i, obj in enumerate(episode_obj.objects):
                idx = obj["object_handle"]
                if type(idx) is not str:
                    template = episode_obj.object_templates[idx]
                    obj["object_handle"] = template["object_handle"]
                episode_obj.objects[i] = RearrangementObjectSpec(**obj)

            for i, goal in enumerate(episode_obj.goals):
                episode_obj.goals[i] = RearrangementSpec(**goal)

            self.episodes.append(episode_obj)

        print(f"Episode Length: {len(self.episodes)}")
