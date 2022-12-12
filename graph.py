import networkx as nx
from collections import Counter
from torch_geometric.utils import from_networkx
import sys

class GraphConverter:

  def __init__(self, max_vocab_size):
    self.max_vocab_size = max_vocab_size
    self.instr_map = {}
    self.num_instrs_in_map = 0
  
  def lookup_instr_idx(self, instruction_text):
    if instruction_text not in self.instr_map:
        self.instr_map[instruction_text] = \
            min(self.num_instrs_in_map,self.max_vocab_size)
        self.num_instrs_in_map += 1

        if self.num_instrs_in_map == self.max_vocab_size:
            print("Reached max instruction embedding size!!", file=sys.stderr)

    return self.instr_map[instruction_text]

  def to_multiplicity_graph(self, programl: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Representation:
      Nodes:
          ProGraML Node Text Field
      Edges:
          ProGraML Edges
        metadata:
          flow type
          position
          multiplicity of edges
    """
    G = nx.MultiDiGraph()
    counter = Counter()

    for u, v, i in programl.edges:
        u_text = programl.nodes[u].get("text", "")
        v_text = programl.nodes[v].get("text", "")

        u_i = self.lookup_instr_idx(u_text)
        v_i = self.lookup_instr_idx(v_text)

        flow = programl.edges[(u, v, i)]["flow"]
        pos = programl.edges[(u, v, i)]["position"]

        counter[(u_i, v_i, flow, pos)] += 1

    for (u_i, v_i, flow, pos), cnt in counter.items():
      G.add_edge(u_i, v_i, flow=flow, pos=pos, mltpcty=cnt)
      G.nodes[u_i]["x"] = u_i
      G.nodes[v_i]["x"] = v_i

    return G
  
  def get_value(self, programl):
    value = 0
    for u, v, _ in programl.edges:
        u_text = programl.nodes[u].get("text", "")
        v_text = programl.nodes[v].get("text", "")

        u_i = self.lookup_instr_idx(u_text)
        v_i = self.lookup_instr_idx(v_text)

        value += (u_i + v_i)
    return value

  def to_pyg(self, programl):
    mult_g = self.to_multiplicity_graph(programl)
    data = from_networkx(
      mult_g,
      group_node_attrs = ["x"],
      group_edge_attrs = ["flow", "pos", "mltpcty"]
    )
    return data
