from hyperopt import fmin, tpe, hp
from train import train_agaemd
import urllib.request
import urllib.error

"""
space = {
    "lr": hp.uniform("learning_rate", 1e-4, 5e-4),
    "dropout": hp.uniform("dropout", 0.4, 0.8),
    "mat_weight_coef": hp.uniform("mat_weight_coef", 0.5, 0.9),
    "embd1_features": hp.choice("embd1_features", [64, 128, 256]),
    "embd2_features": hp.choice("embd2_features", [64, 128, 256]),
    "embd3_features": hp.choice("embd3_features", [64, 128, 256])
}

best = fmin(
    fn=train_agaemd(),
    space=space,
    algo=tpe.suggest(),
    max_evals=100
)
"""

# print(best)
a = urllib.request.urlopen("http://www.baidu.com")
print(a)