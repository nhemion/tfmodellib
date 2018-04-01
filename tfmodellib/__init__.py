# Copyright 2018 Nikolas Hemion. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tfmodellib.tfmodel import TFModel, TFModelConfig, graph_def, docsig
from tfmodellib.linreg import LinReg, LinRegConfig, build_linreg_graph
from tfmodellib.mlp import MLP, MLPConfig, build_mlp_graph
from tfmodellib.autoencoder import AutoEncoder, AutoEncoderConfig, build_autoencoder_graph
from tfmodellib.vae import VAE, VAEConfig, build_vae_graph, variational_loss
from tfmodellib.cae2d import CAE2d, CAE2dConfig, build_cae_2d_graph
# if sys.version_info >= (3,0):
#     from tfmodellib.cdvae2d import CDVAE2d, CDVAE2dConfig
