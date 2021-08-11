import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
# from PIL import Image
import streamlit.components.v1 as components
# import matplotlib.pyplot as plt
import pickle
import networkx as nx
from pyvis.network import Network
from pyvis import network as net
# load model 
import joblib
import bz2
import pickle
import _pickle as cPickle
from vat_func import treeDb,decompress_pickle,similarToken_client,similarToken2_client,similarityInput,corrInput

st.set_page_config(layout='wide')

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data
def draw_graph(npwp):
    A = decompress_pickle('A.pbz2')
    tree = decompress_pickle('tree.pbz2')
    G = decompress_pickle('G.pbz2')
    wp = decompress_pickle('wp.pbz2')
    db_wp = pickle.load(open("d_wp.p","rb"))
    idx = db_wp[db_wp.npwp == npwp].index.values[0]
    wp_awal = db_wp[db_wp.index.isin([idx])].npwp.values 
    idx = tree.query(A[idx,:].toarray(), k=10)[1][0]
    idx = wp[wp.index.isin(idx)].npwp.values
    nodesList = []
    for item in idx :
        for each in nx.shortest_path(G,wp_awal[0],item):
            nodesList.append(each)
    # nodes_between_set = {node for path in gen for node in path}
    # nodes_between_set
    H = G.subgraph(set(nodesList))
    # make a pyvis network
    pyvis_graph = net.Network(notebook=False, directed=True)
    # for each node and its attributes in the networkx graph
    for node,node_attrs in H.nodes(data=True):
        node_attrs['size'] = 5
        if node in idx : 
            node_attrs['size'] = 10
            node_attrs['color'] = 'green'
        if node == wp_awal : 
            node_attrs['size'] = 20
            node_attrs['color'] = 'red'

        pyvis_graph.add_node(str(node),**node_attrs)
    # for each edge and its attributes in the networkx graph
    for source,target,edge_attrs in H.edges(data=True):
    # edge_attrs['value'] = 0.5
        edge_attrs['arrowStrikethrough'] = True
        # add the edge
        pyvis_graph.add_edge(str(source),str(target),**edge_attrs)
    # return and also save
    hidden_net = []
    for item in H.nodes:
        if item not in idx :hidden_net.append(item)
#     return pyvis_graph.show('graph.html')
    return pyvis_graph.show('/tmp/graph.html')
def gen_list(npwp):
    A = decompress_pickle('A.pbz2')
    tree = decompress_pickle('tree.pbz2')
    G = decompress_pickle('G.pbz2')
    wp = decompress_pickle('wp.pbz2')
    db_wp = pickle.load(open("d_wp.p","rb"))
    idx = db_wp[db_wp.npwp == npwp].index.values[0]
    wp_awal = db_wp[db_wp.index.isin([idx])].npwp.values 
    idx = tree.query(A[idx,:].toarray(), k=10)[1][0]
    idx = wp[wp.index.isin(idx)].npwp.values
    nodesList = []
    for item in idx :
        for each in nx.shortest_path(G,wp_awal[0],item):
            nodesList.append(each)
    # nodes_between_set = {node for path in gen for node in path}
    # nodes_between_set
    H = G.subgraph(set(nodesList))
    # make a pyvis network
    pyvis_graph = net.Network(notebook=False, directed=True)
    # for each node and its attributes in the networkx graph
    for node,node_attrs in H.nodes(data=True):
        node_attrs['size'] = 5
        if node in idx : 
            node_attrs['size'] = 10
            node_attrs['color'] = 'green'
        if node == wp_awal : 
            node_attrs['size'] = 20
            node_attrs['color'] = 'red'

        pyvis_graph.add_node(str(node),**node_attrs)
    # for each edge and its attributes in the networkx graph
    for source,target,edge_attrs in H.edges(data=True):
    # edge_attrs['value'] = 0.5
        edge_attrs['arrowStrikethrough'] = True
        # add the edge
        pyvis_graph.add_edge(str(source),str(target),**edge_attrs)
    # return and also save
    hidden_net = []
    for item in H.nodes:
        if item not in idx :hidden_net.append(item)
    return hidden_net,idx


def main():
    """App by Gaspol"""
    menu = ["Network Analysis","Similarity Analysis","Classifier"]
    
    choice = st.sidebar.selectbox("Select Menu", menu)
    tsk = pd.read_csv('transaksi_wp.csv')
        
    if choice == "Similarity Analysis":
        db = decompress_pickle('db_exim.pbz2')
        db['sim_score'] = db['score']*100
        db = db[['NPWP','sim_score']]
        # db['sim_score'] = db['sim_score'].astype('int')
        tsk = tsk[['NPWP','NAMA','score']]
        # db['score'] = db['score'].astype('str')+" %"
        # db.sort_values(by='score',ascending=False)
        db = pd.merge(db,tsk,how='left',on="NPWP")
        c1,c2 = st.beta_columns((1,1))
        with c1:
            st.dataframe(db)
            listwp = db['NPWP'].tolist()
            # listwp = decompress_pickle('listwp')
        with c2:
            wplist = st.selectbox('Pilih WP',listwp)
            # idx = db.index([wplist])
            wp = db[db['NPWP'].isin([wplist])]
            idx = wp.index.values.tolist()
            # num = st.write(idx[0])
            if st.button('generate_similarity'):
                out = corrInput(idx[0])
                # out = list(out.values())
                st.write(out)

    elif choice == "Network Analysis":
        st.title("VAT Fraud Network Analysis")
        wpsample = [565148728502555,302335091102555,569092136088555]
        st.write(f'Sample WP: {wpsample}')
        npwp = st.text_input('Masukkan_NPWP:')
        if st.button('Generate Network'):
            c1,c2 = st.beta_columns((1,1))
            with c1:
                draw_graph(npwp)
#                 HtmlFile = open("graph.html", 'r', encoding='utf-8')
                HtmlFile = open("/tmp/graph.html", 'r', encoding='utf-8')
                source_code = HtmlFile.read() 
#                 print(source_code)
                components.html(source_code,height = 550,width=650)
#                 components.html(draw_graph(npwp),height = 550,width=650)
            with c2:
                glist = gen_list(npwp)
                blue = glist[0]
                green = list(glist[1])
                tsk['NPWP'] = tsk['NPWP'].astype('str')
                # tsk['SOURCE'] = tsk['SOURCE'].astype('str')
                # tsk = tsk[['SOURCE','TARGET']]
                # st.dataframe(tsk)
                # st.write(f'df: {dfl}')
                
                # st.write(f'blue: {blue}')
                st.subheader('WP dengan Pola transaksi Sama (hijau)')
                st.write(f'List Lengkap: {green}')
                st.subheader('Tersedia dalam daftar WP')
                gr_jl = tsk[tsk['NPWP'].isin(green)]
                # gr_jl = tsk[tsk['SOURCE'].isin(green)]
                # gr_bl = tsk[tsk['TARGET'].isin(green)]
                # gr = gr_jl.append(gr_bl,ignore_index=True)
                st.dataframe(gr_jl)
                st.subheader('Transaksi Hidden Network (biru)')
                bl = tsk[tsk['NPWP'].isin(blue)]
                # bl_jl = tsk[tsk['SOURCE'].isin(blue)]
                # bl_bl = tsk[tsk['TARGET'].isin(blue)]
                # bl = bl_jl.append(bl_bl,ignore_index=True)
                st.dataframe(bl)


    elif choice == "Classifier":
        st.title("Item classifier with Unsupervised + Supervised Learning")


if __name__=='__main__':
    main()
