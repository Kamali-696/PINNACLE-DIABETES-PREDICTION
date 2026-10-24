# Model artifacts

The active API does **not** load artifacts from this directory. It loads the two independently
trained local XGBoost pipelines from:

- `../../local_model_results/run_20261024T003644Z/ehr/`
- `../../local_model_results/run_20261024T003644Z/pima/`

For each dataset, the API also loads the matching fitted preprocessing artifact and metadata
from `../../processed/<dataset>/`. The federated logistic regression experiment is not used
as a fallback. See [`../API.md`](../API.md) for the runtime artifact checks and setup.

Any legacy files remaining under this directory are not referenced by the current API.
