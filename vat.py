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
    menu = ["Verdict","VATFraud"]
    
    choice = st.sidebar.selectbox("Select Menu", menu)
        
    if choice == "Verdict":
        # st.subheader("Prediction from Model")
#         st.title("MachineLearning Analytics App")
        st.subheader("Verdict Prediction with Machine Learning")
        # iris= Image.open('iris.png')

        model= open("vmodel.pkl", "rb")
        knn=joblib.load(model)

        st.subheader("Features")
        #Intializing
        c1,c2 = st.beta_columns((1,1))
        with c1:
            sl = st.number_input(label="FP Lengkap",value=1,min_value=0, max_value=1, step=1)
            sw = st.number_input(label="FP Tepat Waktu",value=1,min_value=0, max_value=1, step=1)
            pl = st.number_input(label="Keterangan FP Sesuai",value=0,min_value=0, max_value=1, step=1)
            dm1 = st.number_input(label="FP Diganti Dibatalkan",value=1,min_value=0, max_value=1, step=1)
            dm2 = st.number_input(label="FP Tidak Double Kredit",value=1,min_value=0, max_value=1, step=1)
        with c2:
            dm0 = st.number_input(label="Lawan PKP",value=1,min_value=0, max_value=1, step=1)
            dm3 = st.number_input(label="Lawan Disanksi",value=1,min_value=0, max_value=1, step=1)
            dm4 = st.number_input(label="Lawan Lapor",value=1,min_value=0, max_value=1, step=1)
            dm5 = st.number_input(label="Minta Tanggung Jawab Lawan",value=1,min_value=0, max_value=1, step=1)
            pw = st.number_input(label="PPN telah dibayar",value=0,min_value=0, max_value=1, step=1)

        if st.button("Click Here to Classify"):
            dfvalues = pd.DataFrame(list(zip([sl],[sw],[pl],[pw])),columns =['lengkap', 'tepatwaktu', 'ketsesuai', 'adapembayaran'])
            input_variables = np.array(dfvalues[['lengkap', 'tepatwaktu', 'ketsesuai', 'adapembayaran']])
            prediction = knn.predict(input_variables)
            if prediction == 'ditolak':
                st.subheader('Prediksi Hasil Verdict')
                st.title('Permohonan Banding Ditolak')
            elif prediction =='sebagian':
                st.subheader('Prediksi Hasil Verdict')
                st.title('Permohonan Banding Diterima Sebagian')
            else:
                st.subheader('Prediksi Hasil Verdict')
                st.title('Permohonan Banding Diterima Seluruhnya')
    
    elif choice == "VATFraud":
        st.title("VAT Fraud Network Analysis")
        wpsample = [565148728502555,302335091102555,569092136088555]
        st.write(f'Sample WP: {wpsample}')
        npwp = st.text_input('Masukkan_NPWP:')
        if st.button('Draw_graph'):
            c1,c2 = st.beta_columns((1,1))
            with c1:
                draw_graph(npwp)
                HtmlFile = open("/tmp/graph.html", 'r', encoding='utf-8')
                source_code = HtmlFile.read() 
#                 print(source_code)
                components.html(source_code,height = 550,width=650)
#                 components.html(draw_graph(npwp),height = 550,width=650)
            with c2:
                glist = gen_list(npwp)
                blue = glist[0]
                green = glist[1]
                st.write(f'green: {green}')
                st.write(f'blue: {blue}')

if __name__=='__main__':
    main()
