from habitat.core.registry import registry
from habitat.core.simulator import Simulator


def _try_register_rearrangement_sim():
    try:
        from habitat.sims.rearrangement.actions import (  # noqa: F401
            RearrangementSimV0ActionSpaceConfiguration,
        )
        from habitat.sims.rearrangement.rearrangement_simulator import (
            RearrangementSim,  # noqa: F401
        )

    except ImportError as e:
        habitat_sim_import_error = e
        @registry.register_simulator(name="RearrangementSim-v0")
        class HabitatSimImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise habitat_sim_import_error
