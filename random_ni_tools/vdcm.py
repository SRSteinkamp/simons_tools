import numpy as np
import networkx as nx
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib import cm
import matplotlib as mpl

def spm_dcm_parser(DCM_path, nested = None):
    MAT = loadmat(DCM_path)
    
    if nested is None:
        pass
    else:
        MAT = MAT[nested]
    
    A = []; B = []; C = []; D = []; DCM = []; nreg = None; n_u = None
    if nested is None:
        DCM = MAT['DCM'][0][0]
        A = DCM['a']
        nreg = A.shape[0]

        try:
            B = DCM['b']
            n_u = B.shape[2]
            C = DCM['c']
            D = DCM['d']
        except:
            B = DCM['b'][0][0]
            n_u = B.shape[2]

            A = DCM['a'][0][0]
            nreg = A.shape[0]
            C = DCM['c'][0][0]
            D = DCM['d'][0][0] 
            print(nreg)
    
    

    if np.sum(D) == 0:
        non_linear = 0
    
    if nested is None:
        try:
            A_post = DCM['Ep'][0][0]['A']
            B_post = DCM['Ep'][0][0]['B']
            C_post = DCM['Ep'][0][0]['C']
            D_post = DCM['Ep'][0][0]['D']
            fit = 1
        except:
            fit = 0
            print('Not fitted yet')

        try:
            A_var = DCM['Cp'][0][0]['A']
            B_var = DCM['Cp'][0][0]['B']
            C_var = DCM['Cp'][0][0]['C']
            D_var = DCM['Cp'][0][0]['D']
            #fit = 1
        except:
            #fit = 0
            print('Not fitted yet')
        
    else:
        try:
            A_post = MAT['Ep'][0][0]['A']
            B_post = MAT['Ep'][0][0]['B']
            C_post = MAT['Ep'][0][0]['C']
            D_post = MAT['Ep'][0][0]['D']
            fit = 1
        except:
            fit = 0
            print('Not fitted yet')

        try:
            A_var = MAT['Cp'][0][0]['A']
            B_var = MAT['Cp'][0][0]['B']
            C_var = MAT['Cp'][0][0]['C']
            D_var = MAT['Cp'][0][0]['D']
            #fit = 1
        except:
            #fit = 0
            print('Not fitted yet')
        
    voi_names = []
    
    if nreg is None:
        nreg = A_post.shape[0]
        
    if n_u is None:
        n_u = C_post.shape[1]

    try: 
        for i_name in range(nreg):
            voi_names.append(DCM['xY'][0][i_name]['name'][0])
    except:
        for i_name in range(nreg):
            voi_names.append('reg_' + np.str(i_name))
    
    inp_names = []
    try: 
        for i_input in range(n_u):
            inp_names.append(DCM['U'][0][0]['name'][0,i_input][0])
    except:
        for i_input in range(n_u):
            inp_names.append('input_' + np.str(i_input))
    
    TS = []
    try:
        for i_input in range(nreg):
            TS.append(DCM['xY'][0][i_input]['u'])
    except:
        pass
    
    
    xyz = np.zeros((nreg, 3))
    
    try:
        for i_reg in range(nreg):
            xyz[i_reg,:] = DCM['xY'][0][i_reg]['xyz'].squeeze()
    except:
        print('no region specification found')
    
    out_dict = {'A': A, 'B': B, 'C': C, 'D': D, 'names': voi_names, 
                'inputs': inp_names, 'non_linear': non_linear, 'fit': fit,
               'xyz': xyz}
    

    if fit == 1:
        out_dict['A_post'] = A_post
        out_dict['B_post'] = B_post
        out_dict['C_post'] = C_post
        out_dict['D_post'] = D_post
    
    try:
        out_dict['A_var'] = A_var
        out_dict['B_var'] = B_var
        out_dict['C_var'] = C_var
        out_dict['D_var'] = D_var
    except:
        pass
    
    out_dict['TS'] = TS
            
    
    return out_dict, DCM

def dictionary_to_nx(out_dict, mat_name='A', m_idx=0):
    if out_dict[mat_name].ndim == 2:
        G = nx.DiGraph(out_dict[mat_name].T)
    elif out_dict[mat_name].ndim == 3:
        G = nx.DiGraph(out_dict[mat_name][:,:,m_idx].T)
        
    pos = out_dict['xyz']
     #Enforce symmetry of nodes
    half_pos = np.int(pos.shape[0] / 2)
    pos[half_pos:,0] = pos[:half_pos,0] * -1
    pos[half_pos:,1] = pos[:half_pos,1]
    label_dict = {}
    pos_dict = {}
    for i_label in range(out_dict[mat_name].shape[0]):
        label_dict[i_label] = out_dict['names'][i_label]
        pos_dict[i_label] = pos[i_label,:2]
    pos = pos_dict
    return G, pos, label_dict

def draw_network(G,pos,ax, edge_alpha=0.5, edge_color='k',
                 node_r=1.3, node_alpha=0.5, node_color='k' , LW = 1, 
                 label=False, rad2=0.1, linestyler= '--', MS=15, edge_cmap=None, offset =0.02,
                cbar=False):
    
    if edge_cmap is None:
        cbar = False
    
    if edge_cmap is not None:
        ws = []
        for (u,v,d) in G.edges(data=True):
            ws.append(d['weight'])
        
        norm = mpl.colors.Normalize(vmin=-1 * np.max([np.abs(np.max(ws)), 
                                                      np.abs(np.min(ws))]),
                                    vmax=1* np.max([np.abs(np.max(ws)), 
                                                 np.abs(np.min(ws))]))
    
    
    
    for n in G:
        c=Circle(pos[n],radius=node_r,alpha=node_alpha, color=node_color)
        ax.add_patch(c)
        G.node[n]['patch']=c
        x,y=pos[n]
    seen={}
    for (u,v,d) in G.edges(data=True):
        n1=G.node[u]['patch']
        n2=G.node[v]['patch']
        #rad=0.1
        if edge_cmap is not None:
            edge_color = edge_cmap(norm(d['weight']))
        
        if (v,u) in seen and edge_cmap is None:
            #rad=seen.get((u,v))
            #rad=(rad+np.sign(rad)*rad)*-1
            e, dd = seen[(v,u)]
            e.set(arrowstyle='<|-|>', linestyle='-', lw=LW+0.25)
        
        elif (v,u) in seen:

            e = FancyArrowPatch(np.array(n1.center) - ((np.array(n1.center) - np.array(n2.center))/2) , np.array(n2.center), patchA=n1, patchB=n2,
                                arrowstyle= '-|>',
                                connectionstyle='arc3', #'arc3,rad=%s'%rad, 
                                linestyle = '-',
                                mutation_scale=MS, shrinkA=2, shrinkB=2,
                                lw=LW,
                                alpha=edge_alpha,
                                color=edge_color)
            e1, dd = seen[(v,u)]
            e1 =  FancyArrowPatch(np.array(n1.center) - ((np.array(n1.center) - np.array(n2.center))/2) , np.array(n1.center) , patchA=n2, patchB=n1,
                                arrowstyle= '-|>',
                                connectionstyle='arc3', #'arc3,rad=%s'%rad, 
                                linestyle = '-',
                                mutation_scale=MS, shrinkA=2, shrinkB=2,
                                lw=LW,
                                alpha=edge_alpha,
                                color=edge_cmap(norm(dd)))
                
            seen[(u,v)]=[e, d['weight']]
            seen[(v,u)]=[e1, dd]
            
        else:
        
            e = FancyArrowPatch(n1.center, n2.center, patchA=n1, patchB=n2,
                                arrowstyle= '-|>',
                                connectionstyle='arc3', #'arc3,rad=%s'%rad, 
                                linestyle = linestyler,
                                mutation_scale=MS, shrinkA=2, shrinkB=2,
                                lw=LW,
                                alpha=edge_alpha,
                                color=edge_color)
            
            seen[(u,v)]=[e, d['weight']]
        
    text_dict = {'x': [], 'y': [], 'd': [], 'text': []}
    for ee, dd in seen.values():

        ax.add_patch(ee)
        if label is True:
            pp = ee.get_path().vertices
            pp = pp[1]
            if not pp[0] == pp[1]:
                text_dict['text'].append(plt.text(pp[0],pp[1], '%s' % (np.round(dd,2)), fontdict={'fontsize': 20}, axes=ax))
                text_dict['x'].append(pp[0])
                text_dict['y'].append(pp[1])
                text_dict['d'].append(dd)
    
    if cbar == True:
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
        sm._A = []
        plt.colorbar(sm, orientation='horizontal', fraction=0.05, pad=0.01)
        
    return ax#, [n1,n2]

def normalize_coordinates(pos):
    # Unstack pos dictionary:
    df_from_dict = pd.DataFrame.from_dict(pos).copy().T
    
    # min minimum: 
    df_min = df_from_dict.min()
    
    df_from_dict = df_from_dict - df_min
    df_max = df_from_dict.max()
    df_from_dict = df_from_dict / df_max

    
    # Reduce max distances along dimension:    
    
    
    dict_out = df_from_dict.T.to_dict('list')
    return dict_out

def hack_BMA(out_dict, name_list):
    out_dict['B_post'] = out_dict['B_post'][0][0]
    out_dict['A_post'] = out_dict['A_post'][0][0]
    out_dict['B_var'] = out_dict['B_var'][0][0]
    out_dict['A_var'] = out_dict['A_var'][0][0]
    out_dict['xyz'] = np.zeros((len(name_list), 2))
    out_dict['names'] = name_list
    
    return out_dict

def draw_simple_network(out_dict, A_mat,  ax, color_A, B_mat=None,pos=None,  B_idx=0, LW=3, node_r=0.2, MS=1):
    
    if pos is None:
        G, pos, label_dict = dictionary_to_nx(out_dict, A_mat)
    else:
        G, _,  label_dict = dictionary_to_nx(out_dict, A_mat)
    
    if B_mat is not None:
        G2, _, _ = dictionary_to_nx(out_dict, B_mat, B_idx)
        more_mats = False
        #more_mats = False
    else:
        more_mats = False
        
    if more_mats == True:
        e_idx = [] 
        for (u,v) in G.edges:
            if (u,v) in G2.edges:
                e_idx.append((u,v))
        for (u,v) in e_idx:
            G.remove_edge(u,v)
            #G2.add_edge(u,v)
    
    #pos = normalize_coordinates(pos)
    nx.draw_networkx_nodes(G, pos, node_color=[0.8, 0.8, 0.8], 
                           ax=ax, node_size=12000, linewidths=6, edgecolors='k')
    
    if color_A == True:
        ax = draw_network(G, pos, ax, node_alpha=0,edge_alpha=1,edge_color=[0.1,0.1,0.1], LW=LW, node_r=node_r, 
                     linestyler='--', MS=MS, label=True, edge_cmap=cm.Grays, cbar=False) #cm.Spectral, cbar=True)
    
    else:
        ax = draw_network(G, pos, ax, node_alpha=0,edge_alpha=0.4,edge_color=[0.1,0.1,0.1], 
                     LW=LW, node_r=node_r, linestyler='--', MS=MS) 
        
    
    if B_mat is not None:
        ax = draw_network(G2, pos, ax, node_alpha=0, edge_color='b', edge_alpha=0.9, 
                     node_r =node_r, LW=LW + 4, linestyler='-', MS=MS, label=False, edge_cmap=cm.RdYlBu, cbar=False)

    
    nx.draw_networkx_labels(G, pos, labels=label_dict, ax=ax, node_alpha=0, font_size=22)

    ax.axis('off')
    
    return ax