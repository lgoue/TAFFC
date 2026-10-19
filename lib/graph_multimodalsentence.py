# -*- coding: utf-8 -*-


from ast import pattern

import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
import re
import ot
import time
from scipy.sparse.csgraph import shortest_path
from scipy import sparse
import copy
from numpy import linalg as LA
import matplotlib.colors as mcol
from matplotlib import cm
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
import re
import matplotlib.pyplot as plt
import imageio
import os
import pandas as pd
class NoAttrMatrix(Exception):
    pass

class NoPathException(Exception):
    pass

"""
Summarizes all the methods and classes related to graphs
"""

#%%
class Graph():
    """ Graph is a class that model all the graphs used in the experiments.
    
    Attributes
    ----------    
    nx_graph : a networkx graph, optionnal
               The networkx graph
    C : ndarray
        The structure matrix of the graph. Initalize at None
    name_struct_dist : string
                       The name of the method used to compute the structure matrix
    name : string, 
           Name of the graph because life without name has no meaning. 
    """
    def __init__(self,nx_graph=None):
        if nx_graph is not None:
            self.nx_graph=nx.DiGraph(nx_graph)
        else:
            self.nx_graph=nx.DiGraph()
        self.name='A graph as no name'
        self.log={}
        self.log['pertoperdistance']=[]
        self.log['pathtime']=[]
        self.log['attridist']=[]
        self.n_sentences=0
        self.sentences = []
        self.wordperminutes=120
        self.gesture_data = pd.read_csv("C:\\Users\\lucie\\OneDrive\\Documents\\TrustCalibration\\FGW\\data\\gestures.csv")
        self.n_nodes=0
        #removespaces from the gesture names in the gesture_data dataframe
        self.gesture_data['Name'] = self.gesture_data['Name'].str.replace(' ', '')
        self.gesture_data['Name'] = self.gesture_data['Name'].str.lower()
        #convert "5,345" into 5.345 in the gesture_data dataframe
        self.gesture_data['length in seconds'] = self.gesture_data['length in seconds'].str.replace(',', '.').astype(float)
        self.C=None
        
        self.name_struct_dist='No struct name for now'

    def add_sentence(self, sentence, emb,sentence_id=None,level=None,study=None,task=None,gender=None,model=None):

        if sentence_id is None:
            sentence_id = self.n_sentences
            self.n_sentences+=1
            self.sentences.append(sentence)
            self.add_sentence_to_graph(sentence, sentence_id,level,study,task,gender,model,emb)

    def clean_tags(self, text):
        # rtransforme [intonation] into {i:intonation}
        text = re.sub(r'\[([^\]]+)\]', r'{i:\1}', text)
    
        #transform {g: wave} into {g:wave}
        text = re.sub(r'\{g: ([^\}]+)\}', r'{g:\1}', text)
    
        #transform {g: wave deeply} into {g:wavedeeply}
        text = re.sub(r'\{g: ([^\}]+)\}', r'{g:\1}', text)
    
        text = re.sub(r'\{f: ([^\}]+)\}', r'{f:\1}', text)
    
        text = re.sub(r'\{f: ([^\}]+)\}', r'{f:\1}', text)
        text = re.sub(r'\{t: ([^\}]+)\}', r'{t:\1}', text)
    
        text = re.sub(r'\{t: ([^\}]+)\}', r'{t:\1}', text)
    
    
        #remove punctuation from the text, except for the modality tags
        text = re.sub(r'[^\w\s\{\}:]', '', text)
    
        # remove capitailization
        text = text.lower()

        return text
    
    def remove_tags(self, text):
        # Remove all tags from the text
        text = re.sub(r'\{[fgti]:\s*([^}]*)\}', '', text)
        text = re.sub(r'\[([^\]]*)\]', '', text)
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip()

    
    def add_sentence_to_graph(self, text, sentence_id, level, study, task, gender, model, emb):
        # 1. Parse tags into events with absolute positions
        pattern = r'(\{([fgt]):\s*([^}]*)\}|\[([^\]]*)\])'
        events = []
        current_word_pos = 0
        last_pos = 0
        len_sentence =0
        for match in re.finditer(pattern, text):
            intervening_text = self.clean_tags(text[last_pos:match.start()].strip())
            words_before = len(intervening_text.split()) if intervening_text else 0
            current_word_pos += words_before
            len_sentence +=len(intervening_text)
            if match.group(1).startswith('{'):
                tag_type = match.group(2)
                value = match.group(3).strip()
            else:
                tag_type = 'i'
                value = match.group(4).strip()
            
            tag_id = f"{tag_type}:{value}"
            
            # Duration Logic
            if tag_type == 'g':
                print(value)
                try:
                    duration_in_seconds = self.gesture_data[self.gesture_data['Name'] == value.replace(' ', '').lower()]['length in seconds'].values[0]
                except IndexError:
                    print(f"Gesture not found for value: {value}")
                    duration_in_seconds = 0
                duration_in_words = int((duration_in_seconds/60) * self.wordperminutes)
                end_pos = current_word_pos + duration_in_words
            elif tag_type == 'f':
                # Set to float('inf') initially; we will truncate this when the next 'f' appears
                end_pos = float('inf')
            else:
                # 'i' and 't' last for 1 word
                end_pos = current_word_pos + 1
            
            events.append({
                'tag': tag_id, 
                'start': current_word_pos, 
                'end': end_pos,
                'value': value, 
                'type': tag_type
            })
            
            last_pos = match.end()

        # 2. Handle 'f' tag persistence
        # Loop through events to find where one 'f' tag ends and another begins
        for i in range(len(events)):
            if events[i]['type'] == 'f':
                # Look ahead for the next 'f' tag
                for j in range(i + 1, len(events)):
                    if events[j]['type'] == 'f':
                        # The current 'f' ends exactly where the next 'f' starts
                        events[i]['end'] = events[j]['start']
                        break
                # If no subsequent 'f' is found, it lasts until the end of the sentence
                # We calculate the total word count of the sentence to cap it
                if events[i]['end'] == float('inf'):
                    # The sentence ends after the last event's end (or start if it was inf)
                    # To be safe, we find the max finite end position
                    all_ends = [e['end'] for e in events if e['end'] != float('inf')]
                    sentence_end = max(all_ends) if all_ends else current_word_pos
                    events[i]['end'] = sentence_end

        # 3. Map unique tags to nodes (per sentence)
        tag_to_node = {}
        for event in events:
            tag = event['tag']
            if tag not in tag_to_node:
                node_id = self.n_nodes
                self.n_nodes += 1
                
                if event['type'] == 'g':
                    try:
                        desc = self.gesture_data[self.gesture_data['Name'] == event['value'].replace(' ', '').lower()]['Description'].values[0]
                    except IndexError:
                        print(f"Gesture not found for value: {event['value']}")
                        desc = "Unknown gesture"
                elif event['type'] == 'i':
                    desc = "Talk with a " + event['value'] + " intonation"
                elif event['type'] == 'f':
                    desc = "Use a facial expression " + event['value']
                elif event['type'] == 't':
                    desc = "Use a linguistic feature " + event['value'].replace('.', ' ')
         
                
                self.nx_graph.add_node(node_id, label=tag, attr_name=emb.encode(desc))
                tag_to_node[tag] = node_id

        # 4. Create edges based on temporal distance
        last_finished_tag_past = [False]*len(events)
        for i in range(len(events)):
            current_ev = events[i]
            curr_node = tag_to_node[current_ev['tag']]
            
            for j in range(i):
                prev_ev = events[j]
                prev_node = tag_to_node[prev_ev['tag']]
                
                if curr_node == prev_node:
                    continue
                
                # Weight = distance between end of previous and start of current
                distance = max(0, current_ev['start'] - prev_ev['end']+1)
                distance_corrected=current_ev['start'] - prev_ev['start']
                if distance ==0 :
                    self.nx_graph.add_edge(
                        prev_node, 
                        curr_node, 
                        time=distance,
                        weight=distance/len_sentence,
                        sentence_id=sentence_id, 
                        level=level, 
                        tudy=study, 
                        task=task, 
                        gender=gender, 
                        model=model
                    )
                elif last_finished_tag_past[j]==False:
                    self.nx_graph.add_edge(
                        prev_node, 
                        curr_node, 
                        time=distance_corrected,
                        weight =distance_corrected/len_sentence,
                        sentence_id=sentence_id, 
                        level=level, 
                        tudy=study, 
                        task=task, 
                        gender=gender, 
                        model=model
                    )
                    last_finished_tag_past[j]=True

    def __eq__(self, other) : 
        return self.nx_graph == other.nx_graph

    def __hash__(self):
        return hash(str(self))

    def characterized(self):
        if self.name!='A graph as no name':
            return self.name
        else:
            return self

    def nodes(self):
        """ returns the vertices of a graph """
        return dict(self.nx_graph.nodes())

    def edges(self):
        """ returns the edges of a graph """
        return self.nx_graph.edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.nodes():
            self.nx_graph.add_node(vertex)

    def values(self):
        """ returns a list of all the features of the graph"""
        return [v for (k,v) in nx.get_node_attributes(self.nx_graph,'attr_name').items()]

    def add_nodes(self, nodes):
        self.nx_graph.add_nodes_from(nodes)

    def add_edge(self, edge,weight=1):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        (vertex1, vertex2) = tuple(edge)
        self.nx_graph.add_edge(vertex1,vertex2,weight=weight)

    def add_one_attribute(self,node,attr,attr_name='attr_name'):
        self.nx_graph.add_node(node,attr_name=attr)

    def add_attibutes(self,attributes):
        attributes=dict(attributes)
        for node,attr in attributes.items():
            self.add_one_attribute(node,attr)

    def get_attr(self,vertex):
        return self.nx_graph.node[vertex]
    
    def reshaper(self,x):
        try:
            a=x.shape[1]
            return x
        except IndexError:
            return x.reshape(-1,1)

    def distance_matrix(self,method='shortest_path',changeInf=True,maxvaluemulti=10,force_recompute=False): 
        """ Compute the structure matrix of the graph. 
        It aims at comparing nodes between them using a notion of similarity defined by the "method" parameter
        
        Parameters
        ----------
        method : string, default shortest_path. choices : shortest_path, square_shortest_path, weighted_shortest_path, adjency, harmonic_distance
               The method used to compute the structure matrix of the graph :
                   - shortest_path : compute all the shortest_path between the nodes
                   - square_shortest_path : same but squared 
                   - weighted_shortest_path : compute the shortest path of the weighted graph with weights the distances between the features of the nodes
                   - adjency : compute the adjency matrix of the graph
                   - harmonic_distance : harmonic distance between the nodes
        changeInf : bool
                    If true when the graph has disconnected parts it replaces inf distances by a maxvaluemulti times the largest value of the structure matrix
        force_recompute : force to recompute de distance matrix. If False the matrix is computed only if not already compute or if the method used for computing it changes       
        Returns
        -------
        C : ndarray, shape (n_nodes,n_nodes)
            The structure matrix of the graph
        Set also the attribute C of the graph if C does not exist or if force_recompute is True 
        """
        start=time.time()
        if (self.C is None) or force_recompute:
            if method=="shortest_path_weight":
                A=nx.adjacency_matrix(self.nx_graph,weight='weight' )
            else:
                A=nx.adjacency_matrix(self.nx_graph)
            if method=='harmonic_distance':
                A=A.astype(np.float32)
                D=np.sum(A,axis=0)
                L=np.diag(D)-A

                ones_vector=np.ones(L.shape[0])
                fL=np.linalg.pinv(L) 

                C=np.outer(np.diag(fL),ones_vector)+np.outer(ones_vector,np.diag(fL))-2*fL
                C=np.array(C)
                print(C)
            if method=='shortest_path':           
                C=shortest_path(A) 
            
            if method=='shortest_path_weight':              
                C=shortest_path(A)
          
                
            if method=='square_shortest_path':
                C=shortest_path(A)
                C=C**2
                
            if method=='adjency':
                return A.toarray()
                
            if method=='weighted_shortest_path':
                d=self.reshaper(np.array([v for (k,v) in nx.get_node_attributes(self.nx_graph,'attr_name').items()]))
                D= ot.dist(d,d)
                D_sparse=sparse.csr_matrix(D)
                print(A.shape,D_sparse.shape)
                T=A.multiply(D_sparse)
                C=shortest_path(T)
                
            if changeInf==True:
                C[C==float('inf')]=maxvaluemulti*np.max(C[C!=float('inf')]) # à voir
                
            self.C=C
            self.name_struct_dist=method
            end=time.time()
            self.log['allStructTime']=(end-start)
            return self.C

        else :
            end=time.time()
            self.log['allStructTime']=(end-start)
            return self.C


    def all_matrix_attr(self,return_invd=False):
        d=dict((k, v) for k, v in self.nx_graph.nodes.items())
        x=[]
        invd={}
        try :
            j=0
            for k,v in d.items():
                x.append(v['attr_name'])
                invd[k]=j
                j=j+1
            if return_invd:
                return np.array(x),invd
            else:
                return np.array(x)
        except KeyError:
            raise NoAttrMatrix
        


    def generate_animation_timeline(self, text):
        """Creates a state-by-state timeline for visualization. DOES NOT modify the graph."""
        pattern = r'(\{([fgt]):\s*([^}]*)\}|\[([^\]]*)\])'
        events = []
        last_pos = 0
        
        for match in re.finditer(pattern, text):
            intervening_text = text[last_pos:match.start()].strip()
            words_before = len(intervening_text.split()) if intervening_text else 0
            
            full_match = match.group(1)
            if full_match.startswith('{'):
                tag_type = match.group(2)
                value = match.group(3).strip()
            else:
                tag_type = 'i'
                value = match.group(4).strip()
            
            events.append({'type': tag_type, 'value': value, 'gap': words_before})
            last_pos = match.end()
        
        trailing_text = text[last_pos:].strip()
        final_gap = len(trailing_text.split()) if trailing_text else 0

        timeline = []
        state = {'f': None, 'g': None, 'i': None,'t': None, 'time': None}
        expirations = {'f': float('inf'), 'g': 0, 'i': 0, 't': 0}
        global_clock = 0
        
        for event in events:
            for _ in range(event['gap']):
                for m in ['g', 'i', 't']:
                    if global_clock >= expirations[m]: state[m] = None
                state['time'] = global_clock
                timeline.append(state.copy())
                global_clock += 1
            
            t_type = event['type']
            val = event['value']
            if t_type == 'f':
                state['f'] = val
                expirations['f'] = float('inf')
            elif t_type == 'g':
                state['g'] = val
                try:
                    duration_in_seconds = self.gesture_data[self.gesture_data['Name'] == val.replace(' ', '').lower()]['length in seconds'].values[0]
                except IndexError:
                    print(f"Gesture '{val}' not found in gesture_data. Setting duration to 0.")
                    duration_in_seconds = 2
                duration_in_words = int((duration_in_seconds/60) * self.wordperminutes)
                expirations['g'] = global_clock + duration_in_words
            elif t_type == 'i':
                state['i'] = val
                expirations['i'] = global_clock + 1
            elif t_type == 't':
                state['t'] = val
                expirations['t'] = global_clock + 1
            
            timeline.append(state.copy())
            
        for _ in range(final_gap):
            for m in ['g', 'i', 't']:
                if global_clock >= expirations[m]: state[m] = None
            timeline.append(state.copy())
            global_clock += 1
            
        return timeline

    def create_sentence_gif(self, text, filename="gesture_animation.gif"):
        timeline = self.generate_animation_timeline(text)
        text = self.clean_tags(text)
        text = self.remove_tags(text)

        all_unique_tags = []
        for state in timeline:
            for m in ['f', 'g', 'i', 't']:
                if state[m] is not None:
                    val = state[m]
                    if val not in [tag[0] for tag in all_unique_tags]:
                        all_unique_tags.append((val,{"modality":m}))
        
        nodes_list =all_unique_tags
        G_layout = nx.Graph()
        G_layout.add_nodes_from(nodes_list) 
        pos = nx.spring_layout(G_layout, seed=42)
        
        frames = []
        last_t = -1
        for state in timeline:
            
            t = state['time']
            print(t,last_t)
            if t!=last_t and last_t>=0:
                 # The first word, created with text().
                color_text = ax.text(-0.4, 1.2,' '.join(text.split()[:last_t]), color="black", fontsize=12,verticalalignment='bottom')
                # Subsequent words, positioned with annotate(), relative to the preceding one.
                color_text = ax.annotate(
                ' '+text.split()[last_t]+' ', xycoords=color_text, xy=(1, 0), fontsize=12,color="red",verticalalignment='bottom')  
                color_text = ax.annotate(
                ' '.join(text.split()[last_t+1:]), xycoords=color_text, xy=(1, 0), fontsize=12,color="black",verticalalignment='bottom')  
           
                plt.title(f"Graph Visualization", fontsize=12)
                plt.axis('off')
            
                # Save frame and convert to uint8
                plt.savefig('frame.png', bbox_inches='tight', dpi=80)
                img = plt.imread('frame.png')
            
                # THE FIX: Convert from float [0,1] to int [0,255]
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
            
                
                frames.append(img)
                
            plt.close()
            last_t = t
            ax = plt.figure().add_subplot(xticks=[], yticks=[])
            G_vis = nx.Graph()
            G_vis.add_nodes_from(nodes_list)
            active_now = set(state.values())
            inactive_nodes = [n for n in nodes_list if n[0] not in active_now]
            active_nodes = [n for n in nodes_list if n[0] in active_now]
            
            #change color with attribute modality
            node_colors  = []
            for n in active_nodes:
                if n[1]['modality'] == 'f':
                    node_colors.append('blue')
                elif n[1]['modality'] == 'g':
                    node_colors.append('green')
                elif n[1]['modality'] == 'i':
                    node_colors.append('coral')
                elif n[1]['modality'] == 't':
                    node_colors.append('purple')
                else:
                    node_colors.append('orange')

            node_colors_inactive = []
            for n in inactive_nodes:
                if n[1]['modality'] == 'f':
                    node_colors_inactive.append('lightblue')
                elif n[1]['modality'] == 'g':
                    node_colors_inactive.append('lightgreen')
                elif n[1]['modality'] == 'i':
                    node_colors_inactive.append('lightcoral')
                elif n[1]['modality'] == 't':
                    node_colors_inactive.append('thistle')
                else:
                    node_colors_inactive.append('lightgrey')
            nx.draw_networkx_nodes(G_vis, pos, nodelist=[n[0] for  n in inactive_nodes], 
                                   node_color=node_colors_inactive, node_size=1500, alpha=0.3)
            
            nx.draw_networkx_nodes(G_vis, pos, nodelist=[n[0] for n in active_nodes], 
                                   node_color=node_colors, node_size=1700, alpha=1.0, edgecolors='black')
            
            nx.draw_networkx_labels(G_vis, pos, font_size=10, font_weight='bold')
            
            
            
    # The first word, created with text().
        color_text = ax.text(-0.4, 1.2,' '.join(text.split()[:t]), color="black", fontsize=12,verticalalignment='bottom')
        # Subsequent words, positioned with annotate(), relative to the preceding one.
        color_text = ax.annotate(
        ' '+text.split()[t]+' ', xycoords=color_text, xy=(1, 0), fontsize=12,color="red",verticalalignment='bottom')  
        color_text = ax.annotate(
        ' '.join(text.split()[t+1:]), xycoords=color_text, xy=(1, 0), fontsize=12,color="black",verticalalignment='bottom')  
           
        plt.title(f"Graph Visualization", fontsize=12)
        plt.axis('off')
            
        # Save frame and convert to uint8
        plt.savefig('frame.png', bbox_inches='tight', dpi=80)
        img = plt.imread('frame.png')
            
        # THE FIX: Convert from float [0,1] to int [0,255]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
            
                
        frames.append(img)
                
        plt.close()        
            
           
            
            
            
            
        imageio.mimsave(filename, frames, fps=0.5)
        return filename
            
#%%

def find_thresh(C,inf=0.5,sup=3,step=10):
    """ Trick to find the adequate thresholds from where value of the C matrix are considered close enough to say that nodes are connected
        Tthe threshold is found by a linesearch between values "inf" and "sup" with "step" thresholds tested. 
        The optimal threshold is the one which minimizes the reconstruction error between the shortest_path matrix coming from the thresholded adjency matrix 
        and the original matrix.
    Parameters
    ----------
    C : ndarray, shape (n_nodes,n_nodes)
            The structure matrix to threshold
    inf : float
          The beginning of the linesearch
    sup : float
          The end of the linesearch
    step : integer 
            Number of thresholds tested        
    """
    dist=[]
    search=np.linspace(inf,sup,step)
    for thresh in search:
        Cprime=sp_to_adjency(C,0,thresh)
        SC=shortest_path(Cprime,method='D')
        SC[SC==float('inf')]=100
        dist.append(np.linalg.norm(SC-C))
    return search[np.argmin(dist)],dist

def sp_to_adjency(C,threshinf=0.2,threshsup=1.8):
    """ Thresholds the structure matrix in order to compute an adjency matrix. 
    All values between threshinf and threshsup are considered representing connected nodes and set to 1. Else are set to 0    
    Parameters
    ----------
    C : ndarray, shape (n_nodes,n_nodes)
        The structure matrix to threshold
    threshinf : float
        The minimum value of distance from which the new value is set to 1
    threshsup : float
        The maximum value of distance from which the new value is set to 1
    Returns
    -------
    C : ndarray, shape (n_nodes,n_nodes)
        The threshold matrix. Each element is in {0,1}
    """
    H=np.zeros_like(C)
    np.fill_diagonal(H,np.diagonal(C))
    C=C-H
    C=np.minimum(np.maximum(C,threshinf),threshsup)
    C[C==threshsup]=0
    C[C!=0]=1   
    
    return C 

def relabel_graph_order(graph):

    relabel_dict_={}
    graph_node_list=list(graph.nodes())
    for i in range(len(graph_node_list)):
        relabel_dict_[graph_node_list[i]]=i
        i+=1

    inv_relabel_dict_={v:k for k,v in relabel_dict_.items()}

    graph_relabel=nx.relabel_nodes(graph,relabel_dict_)

    return graph_relabel,inv_relabel_dict_

def wl_labeling(graph,h=2,tohash=True):
    """ Computes the Weisfeler-Lehman labeling for all nodes
    Parameters
    ----------
    graph : Graph
            The Graph to relabel
    h : integer
          The number of iteration of the Weisfeler-Lehman coloring. See [4]
    tohash : bool, optionnal
          Wether to hash the concatenated labeled
    Returns
    -------
    graphs : Graph, 
        The relabeled graph

    References
    ----------
    .. [4] Nils M. Kriege and Pierre{-}Louis Giscard and Richard C. Wilson
        "On Valid Optimal Assignment Kernels and Applications to Graph Classification"
        Advances in Neural Information Processing Systems 29 (NIPS). 2016.

    """
    niter=1
    final_graph=nx.Graph(graph)

    graph_relabel,inv_relabel_dict_=relabel_graph_order(final_graph)
    l_aux = list(nx.get_node_attributes(graph_relabel,'attr_name').values())
    labels = np.zeros(len(l_aux), dtype=np.int32)

    adjency_list = list([list(x[1].keys()) for x in graph_relabel.adjacency()]) #adjency list à l'ancienne comme version 1.0 de networkx
    for j in range(len(l_aux)):
        labels[j] = l_aux[j]

    new_labels = copy.deepcopy(l_aux)

    while niter<=h:

        labeled_graph=nx.Graph(final_graph)

        graph_relabel,inv_relabel_dict_=relabel_graph_order(final_graph)

        l_aux = list(nx.get_node_attributes(graph_relabel,'attr_name'+str(niter-1)).values())

        adjency_list = list([list(x[1].keys()) for x in graph_relabel.adjacency()]) #adjency list à l'ancienne comme version 1.0 de networkx

        for v in range(len(adjency_list)):
        # form a multiset label of the node v of the i'th graph
        # and convert it to a string

            prev_neigh=np.sort([labels[adjency_list[v]]][-1])

            long_label = np.concatenate((np.array([[labels[v]][-1]]),prev_neigh))
            long_label_string = ''.join([str(x) for x in long_label])
            #print('Type_labels before',type(labels))
            new_labels[v] =long_label_string
            #print('Type_labels after',type(labels))

        labels = np.array(copy.deepcopy(new_labels))

        dict_={inv_relabel_dict_[i]:labels[i] for i in range(len(labels))}

        nx.set_node_attributes(labeled_graph,dict_,'attr_name'+str(niter))
        niter+=1
        final_graph=nx.Graph(labeled_graph)

    dict_values={} # pas sûr d'ici niveau de l'ordre des trucs
    for k,v in final_graph.nodes().items():
        hashed=sorted([str(x) for x in v.values()], key=len)

        if tohash :
            dict_values[k]=np.array([hash(x) for x in hashed])
        else:
            dict_values[k]=np.array(hashed)

    graph2=nx.Graph(graph)
    nx.set_node_attributes(graph2,dict_values,'attr_name')

    return graph2


def graph_colors(nx_graph,vmin=0,vmax=7):
    cnorm = mcol.Normalize(vmin=vmin,vmax=vmax)
    cpick = cm.ScalarMappable(norm=cnorm,cmap='viridis')
    cpick.set_array([])
    val_map = {}
    for k,v in nx.get_node_attributes(nx_graph,'attr_name').items():
        print(k,int(np.mean(v)*10000))
        val_map[k]=cpick.to_rgba(int(np.mean(v)*10000))
    colors=[]
    for node in nx_graph.nodes():
        colors.append(val_map[node])
    return colors

def draw_rel(G,draw=True,shiftx=0,shifty=0,return_pos=False,with_labels=True,swipy=False,swipx=False,vmin=0,vmax=7):

    pos=nx.spring_layout(G)
    
    if shiftx!=0 or shifty!=0:
        for k,v in pos.items():
            # Shift the x values of every node by shiftx,shifty
            if shiftx!=0:
                v[0] = v[0] +shiftx
            if shifty!=0:
                v[1] = v[1] +shifty
            if swipy:
                v[1]=-v[1]
            if swipx:
                v[0]=-v[0]

    #map the colors of the nodes based on their 'modality' attribute
    colors=graph_colors(G,vmin=vmin,vmax=vmax)
    if with_labels:
        nx.draw(G,pos,with_labels=True,labels=nx.get_node_attributes(G,'label'),node_color = colors)
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'time'))
    else:
        nx.draw(G,pos,with_labels=False,node_color = colors)
    if draw:
        plt.show()
    if return_pos :
        return pos





def draw_transp(G1,G2,transp,shiftx=1,shifty=0,thresh=0.09,swipy=False,swipx=False,vmin=0,vmax=7,with_labels=True):
    pos1=draw_rel(G1.nx_graph,draw=False,return_pos=True,vmin=vmin,vmax=vmax,with_labels=with_labels)
    pos2=draw_rel(G2.nx_graph,draw=False,shiftx=shiftx,shifty=shifty,return_pos=True,swipx=swipx,swipy=swipy,vmin=vmin,vmax=vmax,with_labels=with_labels)
    _,invd1=G1.all_matrix_attr(return_invd=True)
    _,invd2=G2.all_matrix_attr(return_invd=True)
    for k1,v1 in pos1.items():
        for k2,v2 in pos2.items():
            if (transp[invd1[k1],invd2[k2]]>thresh):
                plt.plot([pos1[k1][0], pos2[k2][0]]
                         , [pos1[k1][1], pos2[k2][1]], 'k--'
                         , alpha=transp[invd1[k1],invd2[k2]]/np.max(transp),lw=2)



