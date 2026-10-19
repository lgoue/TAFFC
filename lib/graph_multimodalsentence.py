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

# Shared with ot_distances, which catches graph.NoAttrMatrix.
from graph import NoAttrMatrix


#%% Constants

DEFAULT_GESTURE_FILE = Path(__file__).resolve().parent.parent / "data" / "gestures.csv"

# Average English reading rate (Rayner et al., 2009), used to convert gesture
# durations from seconds to words.
WORDS_PER_SECOND = 2.0

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
