# -*- coding: utf-8 -*-


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
        
        self.n_nodes=0
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
    
        #remove punctuation from the text, except for the modality tags
        text = re.sub(r'[^\w\s\{\}:]', '', text)
    
        # remove capitailization
        text = text.lower()
        return text

    def add_sentence_to_graph(self, text, sentence_id,level,study,task,gender,model,emb):

        pattern = r'(\{([fgt]):\s*([^}]*)\}|\[([^\]]*)\])'
        tokens = []
        last_pos = 0
        word_count=0
        count_from_last_f = 0
        for match in re.finditer(pattern, text):
            intervening_text = self.clean_tags(text[last_pos:match.start()]).strip()
            
            word_count = 1+len(intervening_text.split()) if intervening_text else 1
        
            full_match = match.group(1)
            if full_match.startswith('{'):
                tag_type = match.group(2)
                value = match.group(3).strip()
            else:
                tag_type = 'i'
                value = match.group(4).strip()
            
            tokens.append({'type': tag_type, 'value': value, 'word_count': word_count})
            last_pos = match.end()

        current_f = None
        previous_tag_node = None
        
        for token in tokens:
            val = token['value']
            t_type = token['type']
            word_count = token['word_count']
            count_from_last_f += word_count
            if t_type == 'f':
                
                self.nx_graph.add_node(self.n_nodes, layer='top', label=val, modality = 'facial_expression',attr_name= emb.encode("Use a facial expression " + val))
                if current_f is not None:
                    f_weight = 1.0 
                    self.nx_graph.add_edge(current_f, self.n_nodes, time=count_from_last_f,  sentence_id=sentence_id,level=level,study=study,task=task,gender=gender,model=model)
        
                

                current_f = self.n_nodes 
                count_from_last_f = 0
                self.n_nodes+=1
            else:
                # To aggregate across sentences, we use the general tag value as the node ID
                # instead of f::val, unless you want the 'f' context to distinguish nodes.
                # Based on the 'average weight' requirement, we assume 'Talking 3' is the same node 
                # regardless of which sentence it appeared in.
                node_id = self.n_nodes
                dict_modality = {'f': 'facial_expression', 'g': 'gesture', 't': 'text', 'i': 'intonation'}
                self.nx_graph.add_node(node_id, layer='sub', label=val,modality=dict_modality.get(t_type),attr_name= emb.encode(self.gesture_data[self.gesture_data['Name'] == val.replace(' ', '')]['Description'].values[0] if t_type == 'g' else "Talk with a " + val + " intonation" if t_type == 'i' else "Use a " + val if t_type == 'f' else "Use a " + val))
                self.n_nodes+=1
                if current_f is not None:
                    self.nx_graph.add_edge(current_f, node_id, time = 1,sentence_id=sentence_id,level=level,study=study,task=task,gender=gender,model=model)
                    self.nx_graph.add_edge( node_id,current_f, time = 1,sentence_id=sentence_id,level=level,study=study,task=task,gender=gender,model=model)
            
                if previous_tag_node is not None:
                    weight = 1.0 
                    if word_count == 1:
                        self.nx_graph.add_edge(previous_tag_node, node_id, time=word_count,  sentence_id=sentence_id,level=level,study=study,task=task,gender=gender,model=model)
                        self.nx_graph.add_edge(node_id, previous_tag_node,time=word_count,  sentence_id=sentence_id,level=level,study=study,task=task,gender=gender,model=model)
                    else:
                       self.nx_graph.add_edge(previous_tag_node, node_id,time=word_count, weight=weight,  sentence_id=sentence_id,level=level,study=study,task=task,gender=gender,model=model)
                
                previous_tag_node = node_id

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
        self.nx_graph.add_edge(vertex1,vertex2,time=weight)

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
                A=nx.adjacency_matrix(self.nx_graph,weight='time' )
            else:
                A=nx.adjacency_matrix(self.nx_graph,weight='time')
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
                # normalize the distance matrix by the maximum value
                l=LA.norm(C, np.inf)
                print(C)
                print(l)
                
            if method=='square_shortest_path':
                C=shortest_path(A)
                C=C**2
                
            if method=='adjency':
                return A.toarray()
                
            if method=='weighted_shortest_path':
                d=self.reshaper(np.array([v for (k,v) in nx.get_node_attributes(self.nx_graph,'attr_name').items()]))
                D= ot.dist(d,d)
                D_sparse=sparse.csr_matrix(D)
                C=shortest_path(A.multiply(D_sparse))
                
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
        val_map[k]=cpick.to_rgba(int(np.mean(v)*10000))
    colors=[]
    for node in nx_graph.nodes():
        print(node)
        colors.append(val_map[node])
    return colors

def draw_rel(G,draw=True,shiftx=0,shifty=0,return_pos=False,with_labels=True,swipy=False,swipx=False,vmin=0,vmax=7):

    pos=nx.kamada_kawai_layout(G)
    
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
    
    color_dict = {'facial_expression': 'red', 'gesture': 'blue', 'intonation': 'green', 'text': 'orange'}
    colors = [color_dict.get(G.nodes[node].get('modality'), 'black') for node in G.nodes()]
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



