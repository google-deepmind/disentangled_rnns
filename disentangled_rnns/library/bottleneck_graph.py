# Copyright 2026 DeepMind Technologies Limited.
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

"""Bottleneck graphs of DisRNNs: build, plot, and compare them.

A DisRNN's "bottleneck graph" has a node for each input, each open latent, and
the output, and an edge wherever information can flow through an open
bottleneck. `get_bottleneck_graph` builds it as a networkx DiGraph.
`plot_bottleneck_graph` draws it with Graphviz. `graphs_isomorphic`,
`disrnn_isomorphic_to_graph`, and `disrnns_isomorphic` compare graphs, so that
you can test whether a DisRNN has learned the structure of a known model.

Only single-subject DisRNNs (`disrnn.DisRnnConfig`) are currently supported.
"""

import collections
from collections import abc
from typing import Any

from disentangled_rnns.library import disrnn
from disentangled_rnns.library import rnn_utils
import graphviz
from IPython import display as ipython_display
import networkx as nx
import numpy as np


def threshold_bottlenecks(
    params: rnn_utils.RnnParams, open_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Threshold the bottlenecks of a DisRNN to binarize them.

  Args:
    params: DisRNN parameters.
    open_threshold: Threshold under which bottlenecks are considered to be open.

  Returns:
    A tuple of numpy arrays, one for each bottleneck type (latent, update net
    obs, update net latent, choice net), with 1s indicating open bottlenecks
    and 0s indicating closed bottlenecks.
  """
  latent_bottlenecks = disrnn.reparameterize_sigma(
      params['hk_disentangled_rnn']['latent_sigma_params']
  )
  update_obs_bottlenecks = disrnn.reparameterize_sigma(
      params['hk_disentangled_rnn']['update_net_obs_sigma_params']
  )
  update_latent_bottlenecks = disrnn.reparameterize_sigma(
      params['hk_disentangled_rnn']['update_net_latent_sigma_params']
  )
  choice_bottlenecks = disrnn.reparameterize_sigma(
      params['hk_disentangled_rnn']['choice_net_sigma_params']
  )

  latent_bottlenecks_open = np.where(latent_bottlenecks < open_threshold, 1, 0)
  update_obs_bottlenecks_open = np.where(
      update_obs_bottlenecks < open_threshold, 1, 0
  )
  update_latent_bottlenecks_open = np.where(
      update_latent_bottlenecks < open_threshold, 1, 0
  )
  choice_bottlenecks_open = np.where(choice_bottlenecks < open_threshold, 1, 0)

  return (
      latent_bottlenecks_open,
      update_obs_bottlenecks_open,
      update_latent_bottlenecks_open,
      choice_bottlenecks_open,
  )


def get_bottleneck_graph(
    disrnn_config: disrnn.DisRnnConfig,
    params: rnn_utils.RnnParams,
    open_threshold: float = 0.5,
    sort_latents: bool = True,
) -> nx.DiGraph:
  """Build the bottleneck graph of a DisRNN.

  Args:
    disrnn_config: Config of the DisRNN. Must be a `disrnn.DisRnnConfig` (not a
      subclass) with `x_names` set, one name per input.
    params: Params of the DisRNN, as returned by `rnn_utils.train_network`.
    open_threshold: Bottlenecks with sigma below this are treated as open.
    sort_latents: If True, number latents by increasing latent bottleneck sigma
      (most open first), matching `plotting.plot_bottlenecks` and
      `plotting.plot_update_rules`. If False, number them by parameter index.

  Returns:
    A networkx DiGraph. Each node is keyed by its name and has attributes
    `name`, `kind` ('input', 'latent', or 'output'), and `index` (position
    within its kind). Input nodes are named by `disrnn_config.x_names`, latents
    are named "Latent 1", "Latent 2", etc., and the single output node (the
    DisRNN's output MLP) is named "Output". Closed latents are omitted, since
    their values are replaced by noise wherever they are read; so are inputs
    and the output if none of their connections to open latents are open.

  Raises:
    TypeError: If `disrnn_config` and `params` were passed in swapped order.
    NotImplementedError: If `disrnn_config` is a subclass of `DisRnnConfig`,
      such as a multisubject or neural-activity config.
    ValueError: If `disrnn_config.x_names` is unset or does not have one name
      per input, or if an input name coincides with another input, latent, or
      output name.
  """
  if isinstance(disrnn_config, abc.Mapping) or isinstance(
      params, disrnn.DisRnnConfig
  ):
    raise TypeError(
        'Expected arguments in the order (disrnn_config, params), but got'
        f' disrnn_config of type {type(disrnn_config).__name__} and params of'
        f' type {type(params).__name__}. Were they passed in swapped order?'
    )

  if type(disrnn_config) is not disrnn.DisRnnConfig:  # pylint: disable=unidiomatic-typecheck
    raise NotImplementedError(
        'Bottleneck graphs support only single-subject DisRNNs'
        f' (disrnn.DisRnnConfig), not {type(disrnn_config).__name__}.'
    )

  (
      latent_bottlenecks_open,
      update_obs_bottlenecks_open,
      update_latent_bottlenecks_open,
      choice_bottlenecks_open,
  ) = threshold_bottlenecks(params, open_threshold)
  n_inputs = update_obs_bottlenecks_open.shape[0]
  n_latents = update_latent_bottlenecks_open.shape[0]

  ##############
  # NAME NODES #
  ##############

  input_names = disrnn_config.x_names
  if input_names is None or len(input_names) != n_inputs:
    raise ValueError(
        f'disrnn_config.x_names must have one name per input ({n_inputs}), but'
        f' it is {input_names}.'
    )
  output_name = 'Output'

  # Number latents, optionally by bottleneck openness.
  if sort_latents:
    latent_sigmas = np.array(
        disrnn.reparameterize_sigma(
            params['hk_disentangled_rnn']['latent_sigma_params']
        )
    )
    latent_rank = np.empty(n_latents, dtype=int)
    latent_rank[np.argsort(latent_sigmas)] = np.arange(n_latents)
  else:
    latent_rank = np.arange(n_latents)
  latent_names = [f'Latent {latent_rank[i] + 1}' for i in range(n_latents)]

  # Check that all node names are unique.
  all_names = input_names + latent_names + [output_name]
  duplicate_names = sorted(
      name
      for name, count in collections.Counter(all_names).items()
      if count > 1
  )
  if duplicate_names:
    raise ValueError(
        f'Bottleneck graph node names must be unique, but {duplicate_names} are'
        ' repeated. Latents are named "Latent 1", "Latent 2", etc. and the'
        ' output is named "Output". Rename the clashing entries in'
        ' disrnn_config.x_names.'
    )

  g = nx.DiGraph()

  #############
  # ADD NODES #
  #############

  # Add input nodes
  for i, name in enumerate(input_names):
    g.add_node(name, name=name, kind='input', index=i)

  # Add latent nodes
  for i, name in enumerate(latent_names):
    g.add_node(name, name=name, kind='latent', index=int(latent_rank[i]))

  # Add the single output node
  g.add_node(output_name, name=output_name, kind='output', index=0)

  #############
  # ADD EDGES #
  #############

  # A closed latent's value is replaced by noise wherever it is read (by the
  # update nets and the choice net), so it carries no information. Edges into
  # or out of a closed latent are therefore omitted.
  latent_is_open = latent_bottlenecks_open == 1

  # inputs to latents
  for i in range(n_inputs):
    for j in np.argwhere(update_obs_bottlenecks_open[i] == 1).flatten():
      if latent_is_open[j]:
        g.add_edge(input_names[i], latent_names[j])

  # latents to other latents
  for i in np.argwhere(latent_is_open).flatten():
    for j in np.argwhere(update_latent_bottlenecks_open[i] == 1).flatten():
      if i != j and latent_is_open[j]:
        g.add_edge(latent_names[i], latent_names[j])

  # latents to themselves: considered open if the latent itself is open
  for i in np.argwhere(latent_is_open).flatten():
    g.add_edge(latent_names[i], latent_names[i])

  # latents to output
  for i in np.argwhere(choice_bottlenecks_open == 1).flatten():
    if latent_is_open[i]:
      g.add_edge(latent_names[i], output_name)

  # Remove nodes that do not participate in at least one edge.
  g.remove_nodes_from(list(nx.isolates(g)))

  return g


def plot_bottleneck_graph(
    disrnn_config: disrnn.DisRnnConfig,
    params: rnn_utils.RnnParams,
    open_threshold: float = 0.5,
    sort_latents: bool = True,
) -> ipython_display.SVG:
  """Displays the DisRNN bottleneck graph and returns it as an SVG.

  Like the `plotting.plot_*` functions, this both displays and returns the
  plot. Assign the result (e.g. `_ = plot_bottleneck_graph(...)`) when calling
  it on the last line of a notebook cell, to avoid rendering it twice.

  Rendering needs the Graphviz `dot` program, which Colab has preinstalled.
  Elsewhere, install Graphviz (e.g. `apt-get install graphviz` or
  `brew install graphviz`); see https://graphviz.org/download/.

  Args:
    disrnn_config: Config of the DisRNN. See `get_bottleneck_graph`.
    params: Params of the DisRNN.
    open_threshold: Bottlenecks with sigma below this are treated as open.
    sort_latents: If True, number latents by increasing latent bottleneck sigma,
      matching `plotting.plot_bottlenecks` and `plotting.plot_update_rules`.

  Returns:
    The rendered graph.

  Raises:
    graphviz.ExecutableNotFound: If the Graphviz `dot` program is not installed.
  """
  bottleneck_graph = get_bottleneck_graph(
      disrnn_config, params, open_threshold, sort_latents
  )
  node_kind = nx.get_node_attributes(bottleneck_graph, 'kind')
  node_index = nx.get_node_attributes(bottleneck_graph, 'index')

  # Graphviz IDs are plain strings, with the node name as the label, since the
  # `graphviz` package reads a colon in an edge endpoint as a port.
  dot_id = {n: f'node{i}' for i, n in enumerate(bottleneck_graph)}

  # Non-strict so a bidirectional latent pair is drawn as two edges.
  dot = graphviz.Digraph(strict=False)

  #############
  # ADD NODES #
  #############

  # Inputs: light green ovals
  input_nodes = sorted(
      (n for n in bottleneck_graph if node_kind[n] == 'input'),
      key=node_index.get,
  )
  for n in input_nodes:
    dot.node(
        dot_id[n], label=n, style='filled', fillcolor='#90EE90', shape='oval'
    )

  # Latents: light blue boxes, in index order
  latent_nodes = sorted(
      (n for n in bottleneck_graph if node_kind[n] == 'latent'),
      key=node_index.get,
  )
  for n in latent_nodes:
    dot.node(
        dot_id[n], label=n, style='filled', fillcolor='#ADD8E6', shape='box'
    )

  # Output: light red oval
  output_nodes = [n for n in bottleneck_graph if node_kind[n] == 'output']
  for n in output_nodes:
    dot.node(
        dot_id[n], label=n, style='filled', fillcolor='#FFB6C1', shape='oval'
    )

  #############
  # ADD EDGES #
  #############

  for src, dst in bottleneck_graph.edges():
    is_latent_edge = node_kind[src] == 'latent' and node_kind[dst] == 'latent'
    if is_latent_edge and node_index[src] > node_index[dst]:
      # dot puts a same-rank edge's tail left of its head, so a right-to-left
      # edge would reorder the latents. Store it left-to-right and put the
      # arrowhead at the tail instead.
      dot.edge(dot_id[dst], dot_id[src], dir='back')
    else:
      dot.edge(dot_id[src], dot_id[dst])

  # Invisible chain that pins latents to index order within their rank.
  for left, right in zip(latent_nodes[:-1], latent_nodes[1:]):
    dot.edge(dot_id[left], dot_id[right], style='invis', weight='1000')

  for subgraph_name, nodes, rank in [
      ('input_nodes', input_nodes, 'source'),
      ('latent_nodes', latent_nodes, 'same'),
      ('output_nodes', output_nodes, 'sink'),
  ]:
    with dot.subgraph(name=subgraph_name) as subgraph:
      subgraph.attr(rank=rank)
      for n in nodes:
        subgraph.node(dot_id[n])

  ##########
  # RENDER #
  ##########

  # SVG keeps labels as text; PNG shows missing-glyph boxes.
  svg = ipython_display.SVG(dot.pipe(format='svg', engine='dot'))
  ipython_display.display(svg)
  return svg


def graphs_isomorphic(g1: nx.DiGraph, g2: nx.DiGraph) -> bool:
  """Check if two bottleneck graphs are isomorphic.

  Every node in both graphs must have a `kind` attribute ('input', 'latent', or
  'output') and a `name` attribute. Nodes match if:
  - both are latents, whatever their names, so that different latent units can
    fulfill the same role; or
  - both are inputs, or both are outputs, and they have the same name.

  Other attributes, such as `index`, are ignored.

  Args:
    g1: The first bottleneck graph.
    g2: The second bottleneck graph.

  Returns:
    True if the graphs are isomorphic under the node matching rule above.
  """

  def node_match(x: dict[str, Any], y: dict[str, Any]) -> bool:
    if x['kind'] != y['kind']:
      return False
    elif x['kind'] == 'latent':
      return True
    else:
      return x['name'] == y['name']

  return nx.is_isomorphic(g1, g2, node_match=node_match)


def _check_input_names(
    graph: nx.DiGraph,
    graph_description: str,
    disrnn_config: disrnn.DisRnnConfig,
    config_description: str,
) -> None:
  """Raise if an input of `graph` is not an input of a DisRNN config.

  A DisRNN's bottleneck graph omits inputs whose bottlenecks are all closed, so
  an input missing from the DisRNN's graph may just be closed. An input missing
  from its config's `x_names`, though, can never appear, so no parameters could
  make the graphs isomorphic.

  Args:
    graph: A bottleneck graph whose input nodes are checked.
    graph_description: Description of `graph`, for the error message.
    disrnn_config: Config whose `x_names` must include every input of `graph`.
    config_description: Description of `disrnn_config`, for the error message.

  Raises:
    ValueError: If any input node of `graph` is not in `disrnn_config.x_names`.
  """
  x_names = disrnn_config.x_names or []
  missing_names = sorted(
      attrs['name']
      for _, attrs in graph.nodes(data=True)
      if attrs['kind'] == 'input' and attrs['name'] not in x_names
  )
  if missing_names:
    raise ValueError(
        f'{graph_description} has inputs {missing_names} that are not in the'
        f' x_names of {config_description} ({x_names}), so no parameters can'
        ' make the graphs isomorphic. If these inputs mean the same things as'
        ' inputs with other names, rename them in the DisRNN config. Inputs'
        ' encoded differently (e.g. one-hot vs. scalar) cannot match.'
    )


def disrnn_isomorphic_to_graph(
    disrnn_config: disrnn.DisRnnConfig,
    params: rnn_utils.RnnParams,
    reference_graph: nx.DiGraph,
    open_threshold: float = 0.5,
) -> bool:
  """Check if a DisRNN's bottleneck graph is isomorphic to a reference graph.

  Use this to test whether a trained DisRNN learned the structure of a known
  model. The reference graph's input nodes must be named as in
  `disrnn_config.x_names`; its latents may have any names; its output node must
  be named "Output". See `graphs_isomorphic` for how nodes are matched.

  Args:
    disrnn_config: Config of the DisRNN.
    params: Params of the DisRNN.
    reference_graph: Bottleneck graph of the reference model. Every node needs
      `name` and `kind` attributes.
    open_threshold: Bottlenecks with sigma below this are treated as open.

  Returns:
    True if the graphs are isomorphic. False if they are not, although some
    parameters could have made them so.

  Raises:
    TypeError: If `disrnn_config` and `params` were passed in swapped order.
    ValueError: If an input of `reference_graph` is not in
      `disrnn_config.x_names`, so that no parameters could make the graphs
      isomorphic.
  """
  if isinstance(disrnn_config, abc.Mapping) or isinstance(
      params, disrnn.DisRnnConfig
  ):
    raise TypeError(
        'Expected arguments in the order (disrnn_config, params), but got'
        f' disrnn_config of type {type(disrnn_config).__name__} and params of'
        f' type {type(params).__name__}. Were they passed in swapped order?'
    )
  _check_input_names(
      reference_graph, 'The reference graph', disrnn_config, 'the DisRNN'
  )
  g_disrnn = get_bottleneck_graph(disrnn_config, params, open_threshold)
  return graphs_isomorphic(reference_graph, g_disrnn)


def disrnns_isomorphic(
    disrnn_config1: disrnn.DisRnnConfig,
    params1: rnn_utils.RnnParams,
    disrnn_config2: disrnn.DisRnnConfig,
    params2: rnn_utils.RnnParams,
    open_threshold: float = 0.5,
) -> bool:
  """Check if two DisRNNs have isomorphic bottleneck graphs.

  Args:
    disrnn_config1: Config of the first DisRNN.
    params1: Params of the first DisRNN.
    disrnn_config2: Config of the second DisRNN.
    params2: Params of the second DisRNN.
    open_threshold: Bottlenecks with sigma below this are treated as open.

  Returns:
    True if the two DisRNNs' bottleneck graphs are isomorphic. False if they
    are not, although some parameters could have made them so.

  Raises:
    ValueError: If either DisRNN's graph has an open input that is not in the
      other DisRNN's `x_names`, so that no parameters for the other DisRNN
      could make the graphs isomorphic.
  """
  g1 = get_bottleneck_graph(disrnn_config1, params1, open_threshold)
  g2 = get_bottleneck_graph(disrnn_config2, params2, open_threshold)
  _check_input_names(
      g1, 'The bottleneck graph of DisRNN 1', disrnn_config2, 'DisRNN 2'
  )
  _check_input_names(
      g2, 'The bottleneck graph of DisRNN 2', disrnn_config1, 'DisRNN 1'
  )
  return graphs_isomorphic(g1, g2)
