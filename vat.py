import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
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
    pyvis_graph = net.Network(notebook=False, directed=True,height='700px', width='100%', )
    # bgcolor='#222222',font_color='white'
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
        # edge_attrs['value'] = 0.05
        edge_attrs['arrowStrikethrough'] = True
        # add the edge
        pyvis_graph.add_edge(str(source),str(target),**edge_attrs)
    # return and also save
    hidden_net = []
    for item in H.nodes:
        if item not in idx :hidden_net.append(item)
    return pyvis_graph.show('graph.html')
    # return pyvis_graph.show('/tmp/graph.html')
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
    # menu = ["Input-Output Correlation","Network Analysis"]
    
    # choice = st.sidebar.selectbox("Select Menu", menu)
    tsk = pd.read_csv('transaksi_wp.csv')
    tsk.fillna(0, inplace=True)
    tsk['nilai_transaksi'] = tsk['nilai_transaksi'].astype('int64')
    tsk['jumlah_transaksi'] = tsk['jumlah_transaksi'].astype('int')
    tsk['score'] = tsk['score'].astype('float')
    tsk = tsk.sort_values(by='score', ascending=False)
    # if choice == "Network Analysis":
    st.title("VAT Fraud Network Analysis")
    
    k1,k2 = st.beta_columns((2,3))
    with k1:
        s1 = tsk[tsk['skala_usaha'].isin([1])]
        s2 = tsk[tsk['skala_usaha'].isin([2])]
        s3 = tsk[tsk['skala_usaha'].isin([3])]
        s4 = tsk[tsk['skala_usaha'].isin([4])]
        fig = go.Figure()
        fig.add_trace(go.Box(y=s1['score'],name='skala 1'))
        fig.add_trace(go.Box(y=s2['score'],name='skala 2'))
        fig.add_trace(go.Box(y=s3['score'],name= 'skala 3'))
        fig.add_trace(go.Box(y=s4['score'],name= 'skala 4'))
        fig.update_layout(width=500,height=550,title="Sebaran score untuk masing-masing Skala Usaha")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
    with k2:
        # wpsample = [565148728502555,302335091102555,569092136088555]
        st.markdown('')
        st.markdown('')
        st.markdown('')
        skala = st.selectbox('Pilih Skala Usaha WP',[1,2,3,4])
        wpsample = tsk[tsk['skala_usaha'].isin([skala])]
        wpsample = wpsample[wpsample['score']>45.0]
        wpsample = wpsample[['NAMA','NPWP','nilai_transaksi','score']]
        st.dataframe(wpsample)
        # st.write(f'Sample WP: {wpsample}')
    npwp = st.text_input('Masukkan_NPWP:')
    if st.button('Generate Network'):
        c1,c2 = st.beta_columns((2,3))
        with c2:
            draw_graph(npwp)
            HtmlFile = open("graph.html", 'r', encoding='utf-8')
            # HtmlFile = open("/tmp/graph.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
#                 print(source_code)
            components.html(source_code,height = 700,width=700)
#                 components.html(draw_graph(npwp),height = 550,width=650)
        with c1:
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
            st.write(f'Daftar WP Lengkap:')
            st.write(f'{green}')
            st.write('WP yang tersedia dalam Tabel WP')
            gr_jl = tsk[tsk['NPWP'].isin(green)]
            st.dataframe(gr_jl)
            st.subheader('Transaksi Hidden Network (biru)')
            bl = tsk[tsk['NPWP'].isin(blue)]
            st.dataframe(bl)


    # elif choice == "Clustering":
    #     st.title("Item classifier with Semi-Supervised Learning")


if __name__=='__main__':
    main()
