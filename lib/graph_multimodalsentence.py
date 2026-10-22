# -*- coding: utf-8 -*-
"""
Graph representation of tag-augmented (multimodal) sentences.

A sentence such as

    "{f: smile} Hello [excited] how are {g: Wave} you {t: Second.Person} ?"

is turned into a graph G = (V, E, l_f, l_s) where

* each node is one unimodal cue (gesture ``g``, facial expression ``f``,
  prosody ``i`` written ``[...]``, linguistic marker ``t``);
* each node carries the embedding of a textual description of the cue
  (feature space, l_f);
* the structure matrix C(i, j) is the number of words separating the two cues,
  0 when they overlap in time (structure space, l_s).

Time is measured in word units. The extent tau of a cue depends on its modality:

* prosody and linguistic markers: 1 word;
* facial expression: until the next facial expression (or the end of the sentence);
* gesture: ceil(duration_in_seconds * WORDS_PER_SECOND).

The graphs are compared with the Fused Gromov-Wasserstein distance
(see ``ot_distances.Fused_Gromov_Wasserstein_distance`` and ``FGW.py``).
"""

import io
import math
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import imageio
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.sparse.csgraph import shortest_path

from FGW import fgw_barycenters
# Shared with ot_distances, which catches graph.NoAttrMatrix.
from graph import NoAttrMatrix


#%% Constants

DEFAULT_GESTURE_FILE = Path(__file__).resolve().parent.parent / "data" / "gestures.csv"

# Average English speaking rate (about 160 words per minute), used to convert
# gesture durations from seconds to words.
WORDS_PER_SECOND = 2.7

# {f: ...}, {g: ...}, {t: ...} or [prosody]
TAG_PATTERN = re.compile(r"\{([fgt]):\s*([^}]*)\}|\[([^\]]*)\]")
# Any other {...} block is not a cue and must not be counted as words.
OTHER_BRACES_PATTERN = re.compile(r"\{[^}]*\}")
# A word, keeping contractions together ("that's" is one word).
WORD_PATTERN = re.compile(r"\w+(?:['’]\w+)*")

MODALITY_NAMES = {"f": "facial", "g": "gesture", "i": "prosody", "t": "linguistic"}
MODALITY_COLORS = {"f": "tab:blue", "g": "tab:green", "i": "tab:orange", "t": "tab:purple"}


#%% Parsing

@dataclass
class Cue:
    """A unimodal cue of a sentence, located on the word axis.

    The cue is active on the words ``onset, ..., onset + duration - 1``.
    """
    modality: str
    value: str
    onset: int
    duration: int
    description: str

    @property
    def end(self):
        return self.onset + self.duration

    @property
    def label(self):
        return f"{self.modality}:{self.value}"


def _gesture_key(name):
    return re.sub(r"\s+", "", name).lower()


@lru_cache(maxsize=None)
def load_gesture_table(path=DEFAULT_GESTURE_FILE):
    """Returns a dict {gesture key: (description, duration in seconds)}."""
    table = pd.read_csv(path, decimal=",")
    return {
        _gesture_key(name): (description, float(seconds))
        for name, description, seconds in zip(
            table["Name"], table["Description"], table["length in seconds"]
        )
    }


def count_words(text):
    """Number of words in a text segment that contains no cue tag."""
    return len(WORD_PATTERN.findall(OTHER_BRACES_PATTERN.sub(" ", text)))


def remove_tags(text):
    """Returns the plain transcript, without any tag."""
    text = OTHER_BRACES_PATTERN.sub(" ", TAG_PATTERN.sub(" ", text))
    return " ".join(WORD_PATTERN.findall(text))


def describe_gesture(value, gesture_table):
    """Returns (description, duration in words) of a gesture."""
    key = _gesture_key(value)
    if key not in gesture_table:
        warnings.warn(f"Gesture '{value}' not found in the gesture table, "
                      "using its name as description and a duration of 1 word.")
        return f"Use a gesture {value}", 1
    description, seconds = gesture_table[key]
    return description, max(1, math.ceil(seconds * WORDS_PER_SECOND))


def parse_cues(text, gesture_table):
    """Extracts the cues of a tag-augmented sentence.

    Returns
    -------
    cues : list of Cue, in order of appearance
    n_words : int, number of words of the sentence
    """
    cues = []
    position = 0  # number of words read so far
    last_end = 0
    for match in TAG_PATTERN.finditer(text):
        position += count_words(text[last_end:match.start()])
        last_end = match.end()

        if match.group(3) is not None:
            modality, value = "i", match.group(3).strip()
        else:
            modality, value = match.group(1), match.group(2).strip()

        if modality == "g":
            onset = position
            description, duration = describe_gesture(value, gesture_table)
        elif modality == "f":
            # Duration is set below, once the next facial expression is known.
            onset, duration = position, None
            description = f"Use a facial expression {value}"
        elif modality == "i":
            # Prosodic tags apply to the word that follows them.
            onset, duration = position, 1
            description = f"Talk with a {value} intonation"
        else:
            # Linguistic markers are written after the word they annotate.
            onset, duration = max(position - 1, 0), 1
            description = f"Use a linguistic feature {value.replace('.', ' ')}"

        cues.append(Cue(modality, value, onset, duration, description))

    n_words = position + count_words(text[last_end:])

    # Persistence rule: a facial expression lasts until the next one is
    # activated, or until the end of the sentence.
    facial = [cue for cue in cues if cue.modality == "f"]
    for current, following in zip(facial, facial[1:] + [None]):
        stop = following.onset if following is not None else n_words
        current.duration = max(stop - current.onset, 0)

    return cues, n_words


def temporal_gap(cue1, cue2):
    """Number of words separating two cues, 0 if they overlap."""
    return max(0, max(cue1.onset, cue2.onset) - min(cue1.end, cue2.end))


#%% Graph

class Graph():
    """Graph of a multimodal sentence.

    Attributes
    ----------
    nx_graph : networkx.Graph
        Nodes are the cues. Node attributes: ``label``, ``modality``,
        ``description``, ``onset``, ``duration`` and ``attr_name`` (embedding of
        the description). Edges link every pair of cues with their temporal
        ``gap`` in words.
    cues : list of Cue
    sentence : str
    n_words : int
    metadata : dict
        Free information about the sentence (level, task, gender, model...).
    C : ndarray, shape (n_nodes, n_nodes)
        The structure matrix, computed by ``distance_matrix``.
    """

    def __init__(self, nx_graph=None, gesture_file=DEFAULT_GESTURE_FILE):
        self.nx_graph = nx.Graph(nx_graph) if nx_graph is not None else nx.Graph()
        self.gesture_file = gesture_file
        self.name = None
        self.sentence = None
        self.cues = []
        self.n_words = 0
        self.metadata = {}
        self.C = None
        self.name_struct_dist = None

    @classmethod
    def from_sentence(cls, sentence, emb, **metadata):
        """Builds the graph of a sentence. ``emb`` must have an ``encode`` method
        (e.g. a SentenceTransformer)."""
        graph = cls()
        graph.add_sentence(sentence, emb, **metadata)
        return graph

    @classmethod
    def from_barycenter(cls, features, C):
        """Builds a graph from an FGW barycenter (features, structure matrix)."""
        graph = cls()
        for i, feature in enumerate(features):
            graph.nx_graph.add_node(i, label=str(i), attr_name=feature)
        graph.C = np.asarray(C)
        graph.name_struct_dist = "barycenter"
        return graph

    def add_sentence(self, sentence, emb, **metadata):
        """Fills the graph with the cues of ``sentence``. A graph holds a single sentence."""
        if self.sentence is not None:
            raise ValueError("This graph already holds a sentence, create a new Graph.")

        self.sentence = sentence
        self.metadata = metadata
        self.cues, self.n_words = parse_cues(sentence, load_gesture_table(self.gesture_file))

        for node, cue in enumerate(self.cues):
            self.nx_graph.add_node(
                node,
                label=cue.label,
                modality=cue.modality,
                description=cue.description,
                onset=cue.onset,
                duration=cue.duration,
                attr_name=emb.encode(cue.description),
            )
        for i, cue_i in enumerate(self.cues):
            for j in range(i + 1, len(self.cues)):
                self.nx_graph.add_edge(i, j, gap=temporal_gap(cue_i, self.cues[j]))
        self.C = None

    def nodes(self):
        """Returns the nodes of the graph with their attributes."""
        return dict(self.nx_graph.nodes())

    def edges(self):
        """Returns the edges of the graph."""
        return self.nx_graph.edges()

    def values(self):
        """Returns the list of the node features (embeddings)."""
        return list(nx.get_node_attributes(self.nx_graph, "attr_name").values())

    def all_matrix_attr(self, return_invd=False):
        """Returns the features as a matrix, and optionally a dict {node: row}."""
        features, invd = [], {}
        for row, (node, attributes) in enumerate(self.nx_graph.nodes.items()):
            if "attr_name" not in attributes:
                raise NoAttrMatrix
            features.append(attributes["attr_name"])
            invd[node] = row
        if return_invd:
            return np.array(features), invd
        return np.array(features)

    def distance_matrix(self, method="temporal", force_recompute=False, **kwargs):
        """Returns the structure matrix C of the graph.

        Parameters
        ----------
        method : 'temporal' or 'shortest_path'
            - temporal: C(i, j) = number of words separating cues i and j
              (0 if they overlap). Requires a graph built from a sentence.
            - shortest_path: weighted shortest path along the edges ('weight'
              attribute), for graphs built from an adjacency matrix (not for
              sentence graphs). 'shortest_path_weight' is accepted as an alias.
        force_recompute : bool
            If False, the stored matrix is returned when it exists.
        """
        if self.C is not None and not force_recompute:
            return self.C

        if method == "temporal":
            if self.sentence is None:
                raise ValueError("The temporal structure requires a graph built from a sentence.")
            n = len(self.cues)
            C = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    C[i, j] = C[j, i] = temporal_gap(self.cues[i], self.cues[j])
        elif method in ("shortest_path", "shortest_path_weight"):
            if self.sentence is not None:
                raise ValueError("Sentence graphs use method='temporal'.")
            A = nx.adjacency_matrix(self.nx_graph, weight="weight")
            C = shortest_path(A, directed=False)
            if np.isinf(C).any():
                finite = C[np.isfinite(C)]
                C[np.isinf(C)] = 10 * finite.max() if finite.size else 1
        else:
            raise ValueError(f"Unknown method '{method}'.")

        self.C = C
        self.name_struct_dist = method
        return self.C

    #%% Animation

    def generate_animation_timeline(self):
        """Returns, for every word of the sentence, the list of active cue indices."""
        return [
            [k for k, cue in enumerate(self.cues) if cue.onset <= word < cue.end]
            for word in range(self.n_words)
        ]

    def create_sentence_gif(self, filename="gesture_animation.gif", fps=0.5, seed=42):
        """Saves a GIF showing, word by word, which cues are active."""
        if self.sentence is None:
            raise ValueError("The animation requires a graph built from a sentence.")
        words = remove_tags(self.sentence).split()
        pos = nx.spring_layout(self.nx_graph, seed=seed)
        labels = nx.get_node_attributes(self.nx_graph, "label")

        frames = []
        for word, active in enumerate(self.generate_animation_timeline()):
            fig, ax = plt.subplots(figsize=(8, 6))
            inactive = [k for k in self.nx_graph.nodes if k not in active]
            nx.draw_networkx_nodes(self.nx_graph, pos, nodelist=inactive, ax=ax, node_size=1500, alpha=0.3,
                                   node_color=[MODALITY_COLORS[self.cues[k].modality] for k in inactive])
            nx.draw_networkx_nodes(self.nx_graph, pos, nodelist=active, ax=ax, node_size=1700,
                                   edgecolors="black",
                                   node_color=[MODALITY_COLORS[self.cues[k].modality] for k in active])
            nx.draw_networkx_labels(self.nx_graph, pos, labels=labels, ax=ax, font_size=9)

            # Sentence with the current word in red.
            text = ax.text(0, 1.02, " ".join(words[:word]) + " ", transform=ax.transAxes, fontsize=11)
            text = ax.annotate(words[word] + " ", xycoords=text, xy=(1, 0), color="red", fontsize=11)
            ax.annotate(" ".join(words[word + 1:]), xycoords=text, xy=(1, 0), fontsize=11)
            ax.axis("off")

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=80)
            plt.close(fig)
            frames.append(imageio.imread(buffer.getvalue()))

        imageio.mimsave(filename, frames, duration=1000 / fps)
        return filename


#%% Barycenter

def barycenter_size(graphs):
    """Number of nodes of a barycenter: median number of nodes of the graphs
    (rounded half up)."""
    return int(np.floor(np.median([len(g.nodes()) for g in graphs]) + 0.5))


def compute_barycenter(graphs, alpha, N=None, max_iter=100, verbose=False):
    """FGW barycenter of a group of sentence graphs, with uniform weights.

    Parameters
    ----------
    graphs : list of Graph
        Graphs without any node are ignored.
    alpha : float
        Trade-off parameter of FGW.
    N : int, optional
        Number of nodes of the barycenter. Default: median number of nodes of the graphs.

    Returns
    -------
    Graph holding the barycenter features and structure matrix.
    """
    graphs = [g for g in graphs if len(g.nodes()) > 0]
    if not graphs:
        raise ValueError("No non-empty graph to average.")
    if N is None:
        N = barycenter_size(graphs)

    features, C, _ = fgw_barycenters(
        N,
        Ys=[g.all_matrix_attr() for g in graphs],
        Cs=[g.distance_matrix() for g in graphs],
        ps=[np.ones(len(g.nodes())) / len(g.nodes()) for g in graphs],
        lambdas=np.ones(len(graphs)) / len(graphs),
        alpha=alpha,
        max_iter=max_iter,
        verbose=verbose,
    )
    return Graph.from_barycenter(features, C)


#%% Barycenter utilities

def find_thresh(C, inf=0.5, sup=3, step=10):
    """Finds the threshold under which values of C are considered as connected nodes.

    The threshold is found by a line search between ``inf`` and ``sup`` with
    ``step`` values. The best one minimizes the error between C and the shortest
    path matrix of the thresholded adjacency matrix.
    """
    dist = []
    search = np.linspace(inf, sup, step)
    for thresh in search:
        Cprime = sp_to_adjency(C, 0, thresh)
        SC = shortest_path(Cprime, method="D")
        SC[SC == float("inf")] = 100
        dist.append(np.linalg.norm(SC - C))
    return search[np.argmin(dist)], dist


def sp_to_adjency(C, threshinf=0.2, threshsup=1.8):
    """Thresholds a structure matrix into an adjacency matrix: values in
    [threshinf, threshsup) are set to 1, others to 0."""
    H = np.zeros_like(C)
    np.fill_diagonal(H, np.diagonal(C))
    C = C - H
    C = np.minimum(np.maximum(C, threshinf), threshsup)
    C[C == threshsup] = 0
    C[C != 0] = 1
    return C


#%% Drawing

def graph_colors(nx_graph, vmin=0, vmax=7):
    """Node colors: by modality for sentence graphs, by mean feature value otherwise."""
    cpick = cm.ScalarMappable(norm=mcol.Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
    colors = []
    for node, attributes in nx_graph.nodes.items():
        if "modality" in attributes:
            colors.append(MODALITY_COLORS[attributes["modality"]])
        else:
            colors.append(cpick.to_rgba(int(np.mean(attributes["attr_name"]) * 10000)))
    return colors


def draw_rel(G, draw=True, shiftx=0, shifty=0, return_pos=False, with_labels=True,
             swipy=False, swipx=False, vmin=0, vmax=7, seed=None):
    """Draws a networkx graph, optionally shifted or mirrored."""
    pos = nx.spring_layout(G, seed=seed)
    for v in pos.values():
        v[0] += shiftx
        v[1] += shifty
        if swipx:
            v[0] = -v[0]
        if swipy:
            v[1] = -v[1]

    colors = graph_colors(G, vmin=vmin, vmax=vmax)
    if with_labels:
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, "label"), node_color=colors)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "gap"))
    else:
        nx.draw(G, pos, with_labels=False, node_color=colors)
    if draw:
        plt.show()
    if return_pos:
        return pos


def draw_transp(G1, G2, transp, shiftx=1, shifty=0, thresh=0.09, swipy=False, swipx=False,
                vmin=0, vmax=7, with_labels=True):
    """Draws two graphs and the transport plan ``transp`` between their nodes."""
    pos1 = draw_rel(G1.nx_graph, draw=False, return_pos=True, vmin=vmin, vmax=vmax, with_labels=with_labels)
    pos2 = draw_rel(G2.nx_graph, draw=False, shiftx=shiftx, shifty=shifty, return_pos=True,
                    swipx=swipx, swipy=swipy, vmin=vmin, vmax=vmax, with_labels=with_labels)
    _, invd1 = G1.all_matrix_attr(return_invd=True)
    _, invd2 = G2.all_matrix_attr(return_invd=True)
    for k1, p1 in pos1.items():
        for k2, p2 in pos2.items():
            mass = transp[invd1[k1], invd2[k2]]
            if mass > thresh:
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", alpha=mass / np.max(transp), lw=2)


#%% Timeline drawing of sentence graphs

# Lanes from top to bottom.
MODALITY_ORDER = ["f", "g", "i", "t"]
LANE_GAP = 0.6


def timeline_layout(graph):
    """Positions of the cues on a timeline: x is the middle of the cue span (in
    words), y the lane of its modality. A lane is split into rows so that cues
    of the same modality never overlap.

    Returns
    -------
    pos : dict {node: (x, y)}
    lanes : dict {modality: (y_bottom, y_top)}
    height : float, total height of the lanes
    """
    rows, n_rows = {}, {}
    for modality in MODALITY_ORDER:
        row_ends = []  # end of the last cue of each row
        indexed = [(k, cue) for k, cue in enumerate(graph.cues) if cue.modality == modality]
        for k, cue in sorted(indexed, key=lambda kc: kc[1].onset):
            span_end = cue.onset + max(cue.duration, 1)  # a zero-length cue still takes one word
            for row, end in enumerate(row_ends):
                if end <= cue.onset:
                    row_ends[row] = span_end
                    break
            else:
                row = len(row_ends)
                row_ends.append(span_end)
            rows[k] = row
        n_rows[modality] = max(len(row_ends), 1)

    lanes, y = {}, 0.0
    for modality in reversed(MODALITY_ORDER):
        lanes[modality] = (y, y + n_rows[modality])
        y += n_rows[modality] + LANE_GAP
    height = y - LANE_GAP

    pos = {}
    for k, cue in enumerate(graph.cues):
        y_top = lanes[cue.modality][1]
        pos[k] = (cue.onset + max(cue.duration, 1) / 2, y_top - rows[k] - 0.5)
    return pos, lanes, height


def draw_sentence_graph(graph, ax=None, y_offset=0, words_position="bottom", show_edges=True, title=None):
    """Draws a sentence graph on a timeline.

    Each cue is a node in the lane of its modality, with a bar showing its
    extent in words. The words of the transcript are written along the time
    axis. With ``show_edges``, cues that overlap in time (C = 0) are linked.

    Returns
    -------
    pos : dict {node: (x, y)}, positions including ``y_offset``
    """
    if graph.sentence is None:
        raise ValueError("Timeline drawing requires a graph built from a sentence.")
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, 0.6 * graph.n_words), 3.5))

    pos, lanes, height = timeline_layout(graph)
    pos = {k: (x, y + y_offset) for k, (x, y) in pos.items()}
    n_cells = max(graph.n_words, max((cue.end for cue in graph.cues), default=0), 1)

    # Lanes and word grid
    for modality, (bottom, top) in lanes.items():
        ax.axhspan(bottom + y_offset - 0.1, top + y_offset + 0.1, color=MODALITY_COLORS[modality], alpha=0.06, lw=0)
        ax.text(-0.3, (bottom + top) / 2 + y_offset, MODALITY_NAMES[modality],
                ha="right", va="center", fontsize=9, color=MODALITY_COLORS[modality])
    for x in range(n_cells + 1):
        ax.plot([x, x], [y_offset - 0.1, y_offset + height + 0.1], color="0.9", lw=0.5, zorder=0)

    ax.set_xlim(-2.5, n_cells + 0.5)
    words = remove_tags(graph.sentence).split()
    y_words = y_offset - 0.3 if words_position == "bottom" else y_offset + height + 0.3
    for k, word in enumerate(words):
        ax.text(k + 0.5, y_words, word, ha="center", va="top" if words_position == "bottom" else "bottom",
                fontsize=8)

    # Overlap edges
    if show_edges:
        C = graph.distance_matrix()
        for i in range(len(graph.cues)):
            for j in range(i + 1, len(graph.cues)):
                if C[i, j] == 0:
                    ax.plot(*zip(pos[i], pos[j]), color="0.75", lw=0.8, zorder=1)

    # Cues. A label that would overlap the previous one on the same row goes below the node.
    char_width = 7.5 * 0.6 / 72 * ax.figure.dpi  # approximate width of a character, in pixels
    pixels_per_word = ax.get_window_extent().width / (n_cells + 3)
    label_end = {}  # right end of the last label above each row
    for k in sorted(range(len(graph.cues)), key=lambda k: pos[k][0]):
        cue = graph.cues[k]
        x, y = pos[k]
        color = MODALITY_COLORS[cue.modality]
        ax.add_patch(plt.Rectangle((cue.onset, y - 0.18), max(cue.duration, 0.1), 0.36,
                                   color=color, alpha=0.3, lw=0, zorder=2))
        ax.scatter([x], [y], s=60, color=color, edgecolors="black", linewidths=0.6, zorder=3)
        half_width = len(cue.value) * char_width / pixels_per_word / 2
        if x - half_width < label_end.get(y, -np.inf):
            ax.text(x, y - 0.22, cue.value, ha="center", va="top", fontsize=7.5, zorder=4)
        else:
            ax.text(x, y + 0.22, cue.value, ha="center", va="bottom", fontsize=7.5, zorder=4)
            label_end[y] = x + half_width

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    return pos


def draw_structure_matrix(graph, ax=None, cmap="viridis"):
    """Heatmap of the structure matrix C (number of words between cues)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    C = graph.distance_matrix()
    labels = [cue.label for cue in graph.cues]
    image = ax.imshow(C, cmap=cmap)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{C[i, j]:g}", ha="center", va="center", fontsize=7,
                    color="white" if C[i, j] < C.max() / 2 else "black")
    plt.colorbar(image, ax=ax, label="gap (words)")
    return ax


def draw_coupling(graph1, graph2, transp, ax=None, thresh=0.1, distance=None):
    """Draws two sentence graphs on timelines, one above the other, linked by
    the transport plan ``transp`` (rows: nodes of graph1, columns: graph2).

    Links whose mass is below ``thresh`` times the largest mass are hidden; the
    others have a width and opacity proportional to their mass.
    """
    _, _, height2 = timeline_layout(graph2)
    if ax is None:
        width = max(8, 0.6 * max(graph1.n_words, graph2.n_words))
        _, ax = plt.subplots(figsize=(width, 7))

    offset1 = height2 + 2.5
    pos2 = draw_sentence_graph(graph2, ax=ax, y_offset=0, words_position="bottom", show_edges=False)
    pos1 = draw_sentence_graph(graph1, ax=ax, y_offset=offset1, words_position="top", show_edges=False)

    mass_max = transp.max()
    for i in range(transp.shape[0]):
        for j in range(transp.shape[1]):
            weight = transp[i, j] / mass_max
            if weight >= thresh:
                ax.plot(*zip(pos1[i], pos2[j]), color="black", lw=0.5 + 2.5 * weight,
                        alpha=0.15 + 0.6 * weight, zorder=1)

    n_cells = max(graph1.n_words, graph2.n_words, 1)
    ax.set_xlim(-2.5, n_cells + 0.5)
    if distance is not None:
        ax.set_title(f"FGW distance = {distance:.4f}", fontsize=11, pad=18)
    return ax


def draw_coupling_matrix(graph1, graph2, transp, ax=None, cmap="Greys"):
    """Heatmap of the transport plan, with the cue labels of both graphs."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(transp, cmap=cmap)
    ax.set_xticks(range(len(graph2.cues)), [cue.label for cue in graph2.cues], rotation=90, fontsize=8)
    ax.set_yticks(range(len(graph1.cues)), [cue.label for cue in graph1.cues], fontsize=8)
    ax.set_xlabel("graph 2")
    ax.set_ylabel("graph 1")
    plt.colorbar(image, ax=ax, label="transported mass")
    return ax
