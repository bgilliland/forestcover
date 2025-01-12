from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from _model.config.core import config
from _model.processing.features import TrigTransformer

price_pipe = make_pipeline(
    (TrigTransformer(variables=config.model_settings.trig_transform)),
    # (CosTransformer(variables=config.model_settings.trig_transform)),
    RandomForestClassifier(
        n_estimators=config.model_settings.n_estimators,
        min_samples_split=config.model_settings.min_samples_split,
        min_samples_leaf=config.model_settings.min_samples_leaf,
        max_samples=config.model_settings.max_samples,
        max_leaf_nodes=config.model_settings.max_leaf_nodes,
        max_features=config.model_settings.max_features,
        max_depth=config.model_settings.max_depth,
    ),
)
