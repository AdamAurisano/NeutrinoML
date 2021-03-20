import torch, matplotlib as mpl
import matplotlib.pyplot as plt, matplotlib.collections as mc

class GraphPlot:
  def __init__(self):
    self.format()

  def format(self):
    mpl.rc('font', family='normal', weight='bold', size=36)

  def get_lines(self, g, score):
    '''Take a graph object and return a list of LineCollection objects, one per class'''
    wire = g.x[:,1]
    time = g.x[:,2]
    lines = [ [ [ wire[edge[0]], time[edge[0]] ], [ wire[edge[1]], time[edge[1]] ] ] for edge in g.edge_index.T ]
    lines_class = [ [], [], [], [] ]
    colours = ['gainsboro', 'red', 'green', 'blue' ]
    for l, y in zip(lines, score):
      lines_class[y].append(l)
    return [ mc.LineCollection(lines_class[i], colors=colours[i], linewidths=2, zorder=1) for i in range(len(colours)) ]

  def plot_edge_score(self, trainer, graph):
    fig, ax = plt.subplots(figsize=[16,9])
    trainer.model.eval()
    with torch.no_grad():
      y = trainer.model(graph.to(trainer.device)).argmax(dim=1)
    lcs = self.get_lines(graph, y)
    for lc in lcs: ax.add_collection(lc)
    ax.autoscale()
    plt.xlabel("Wire")
    plt.ylabel("Time tick")
    plt.tight_layout()

  def plot_edge_truth(self, graph):
    fig, ax = plt.subplots(figsize=[16,9])
    lcs = self.get_lines(graph, graph.y)
    for lc in lcs: ax.add_collection(lc)
    ax.autoscale()
    plt.xlabel("Wire")
    plt.ylabel("Time tick")
    plt.tight_layout()

  def plot_edge_diff(self, trainer, graph):
    fig, ax = plt.subplots(figsize=[16,9])
    trainer.model.eval()
    with torch.no_grad():
      y = trainer.model(graph.to(trainer.device)).argmax(dim=1)
    y = (y != graph.y)
    lcs = self.get_lines(graph, y)
    for lc in lcs: ax.add_collection(lc)
    ax.autoscale()
    plt.tight_layout()

