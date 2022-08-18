from pathlib import Path
from pprint import pprint
from per_point_pate.studies.parameters import DataParameters, ExperimentPlan, ModelParameters, PATEExperimentPlan, ResourceParameters


def test_experiment_plan_load():
    plan = ExperimentPlan.load(
        params_path=Path(__file__).parent / 'test_parameters.yaml',
        data_dir=Path('/tmp/datadir'),
        out_dir=Path('/tmp/out_dir'),
    )

    assert isinstance(plan.resources, ResourceParameters)
    assert isinstance(plan.data, DataParameters)
    assert isinstance(plan.models, ModelParameters)
    assert isinstance(plan.pate, PATEExperimentPlan)


def test_experiment_plan_num_derived():
    plan = ExperimentPlan.load(
        params_path=Path(__file__).parent / 'test_parameters.yaml',
        data_dir=Path('/tmp/datadir'),
        out_dir=Path('/tmp/out_dir'),
    )
    class DataFactoryStub:
        def n_private_per_label(self, seed): 
            return {l: 6000 for l in range(10)}
    experiment_prms = plan.derive_experiment_parameters(
        data_factory=DataFactoryStub()
    )
    assert len(experiment_prms) == 27
    pprint(experiment_prms)