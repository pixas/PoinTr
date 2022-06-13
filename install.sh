#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python -m pip install -e .

# NOTE: For GRNet 

# Cubic Feature Sampling
cd $HOME/extensions/cubic_feature_sampling
python -m pip install -e .


# Gridding & Gridding Reverse
cd $HOME/extensions/gridding
python -m pip install -e .

# Gridding Loss
cd $HOME/extensions/gridding_loss
python -m pip install -e .


