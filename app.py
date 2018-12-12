import os
from flask import Flask, render_template,jsonify,request,redirect,url_for,json,Response,send_file,send_from_directory

app = Flask(__name__)


import tensorflow as tf
import json
import numpy as np
from PIL import Image,ImageOps 
import requests
from io import BytesIO
import psycopg2
import sys
import os
import urllib.parse

cat_bread={'0':'Especial','1':'Tradicional','2':'Brioche','3':'Especiales','4':'Sin Pan'}
cat_price={'0':'Medio','1':'Bajo','2':'Medio-Bajo','3':'Medio','4':'Medio-Alto','5':'Alto'}
cat_side={'0':'Especial','1':'Papas','2':'Vegetales','3':'sin guarnicion'}
cat_style={'0':'Comedor','1':'Puestito','2':'Fast Food','3':'Cafeteria','4':'Comedor','5':'Bistro'}

def urlencode(str):
  return urllib.parse.quote(str)


def urldecode(str):
  return urllib.parse.unquote(str)

def conv2d(x, W,name,padd,strid=[1,1,1,1]):
    #El stride de esa función no reduce el tamaño de la imagen
    return tf.nn.conv2d(x, W, strides=strid, padding=padd,name=name)

def maxpool2d(x,ks,st):
    #           El st de esta función reduce la imagen a la mitad
    return tf.nn.max_pool(x, ksize=ks, strides=st, padding='SAME')
  
def reset_graph():
    #Limpiamos la gráfic
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def deep_neural_convolutional_class_BNB(
    batch_size=20,
    image_size=[200,200],
    Drop_prob=1.0,
    learning_rate = 1e-3,
    n_nodes_hl0 = 2000,
    n_nodes_hl1 = 1000,
    n_nodes_hl2 = 500,
    n_nodes_hl3 = 100,
    n_classes=2
    ):
    
    reset_graph()
    #Place holder de entrada 
    x= tf.placeholder(tf.float32,[batch_size,image_size[0],image_size[1],3], name='placeholder_img_entrada')
  
    #Diccionario de pesos convolucionales 
    with tf.name_scope('pesos_bias') as scope1:
        weigths={"w_conv1":tf.Variable(tf.random_normal([5,5,3,32]),name='Pesos_1_32'),
                 "w_conv2":tf.Variable(tf.random_normal([5,5,32,64]),name='Pesos_1_64'),
                 "w_conv3":tf.Variable(tf.random_normal([3,3,64,128]),name='Pesos_1_128'),   
                 "w_conv4":tf.Variable(tf.random_normal([5,5,128,256]),name='Pesos_1_256'),
                }
        #Diccionario de bias
        biases={"b_conv1":tf.Variable(tf.random_normal([32]),name='Bias_1_32'),
                "b_conv2":tf.Variable(tf.random_normal([64]),name='Bias_1_64'),
                "b_conv3":tf.Variable(tf.random_normal([128]),name='Bias_1_128'),
                "b_conv4":tf.Variable(tf.random_normal([256]),name='Bias_1_256'),
               }

    #Extractor de características
    with tf.name_scope('capas_conv') as scope2:
        conv1=tf.nn.relu(conv2d(x,weigths["w_conv1"],'Capa_Conv_1','SAME')+biases["b_conv1"],name='Func_relu_1')
        conv1=tf.nn.dropout(conv1,Drop_prob)
        conv1=maxpool2d(conv1,ks=[1,2,2,1],st=[1,2,2,1])
        #imagen resultante de 100x100x32
        #print(conv1)

        conv2=tf.nn.relu(conv2d(conv1,weigths["w_conv2"],'Capa_Conv_2','SAME')+biases["b_conv2"],name='Func_relu_2')
        conv2=tf.nn.dropout(conv2,Drop_prob)
        conv2=maxpool2d(conv2,ks=[1,2,2,1],st=[1,2,2,1])
        #imagen resultante de 50x50x64
        #print(conv2)

        conv3=tf.nn.relu(conv2d(conv2,weigths["w_conv3"],'Capa_Conv_3','VALID')+biases["b_conv3"],name='Func_relu_3')
        conv3=tf.nn.dropout(conv3,Drop_prob)
        conv3=maxpool2d(conv3,ks=[1,2,2,1],st=[1,2,2,1])
        #imagen resultante de 24x24x128
        #print(conv3)

        conv4=tf.nn.relu(conv2d(conv3,weigths["w_conv4"],'Capa_Conv_4','SAME')+biases["b_conv4"],name='Func_relu_4')
        conv4=tf.nn.dropout(conv4,Drop_prob)
        conv4=maxpool2d(conv4,ks=[1,2,2,1],st=[1,2,2,1])
        #imagen resultante de 12x12x256
        #print(conv4)

        #Embeding, son las caracteristicas fonales que se pasarán al MLP o red completamente conectada para clasifiacar
        embdeding=tf.reshape(conv4,[batch_size,12*12*256],name='Embeding')
        #print(embdeding)
    
    #Red perceptron, declaración de capas, son diccionarios de pesos y bias.
    with tf.name_scope('capas_clasificador') as scope3:
        hidden_0_layer = {'weights':tf.Variable(tf.random_normal([12*12*256, n_nodes_hl0]),name='Capa_oculta_pesos_0'),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl0]),name='Capa_oculta_bias_0')}

        hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl0, n_nodes_hl1]),'Capa_oculta_pesos_1'),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl1]),name='Capa_oculta_bias_1')}

        hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]),'Capa_oculta_pesos_2'),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl2]),name='Capa_oculta_bias_2')}
        
        hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]),'Capa_oculta_pesos_3'),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl3]),name='Capa_oculta_bias_3')}

        output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]),'Capa_salida_pesos'),
                        'biases':tf.Variable(tf.random_normal([n_classes]),name='Capa_salida_bias'),}
    
    #W*P + B 
    with tf.name_scope('op_clasificador') as scope4:
        
        l0 = tf.add(tf.matmul(embdeding,hidden_0_layer['weights'],name='Matmul_l0'), hidden_0_layer['biases'],name='Suma_Pesos_Bias_0')
        l0 = tf.nn.relu(l0,name='l0_relu_0')

        l1 = tf.add(tf.matmul(l0,hidden_1_layer['weights'],name='Matmul_l1'), hidden_1_layer['biases'],name='Suma_Pesos_Bias_1')
        l1 = tf.nn.relu(l1,name='l1_relu_1')

        l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights'],name='Matmul_l2'), hidden_2_layer['biases'],name='Suma_Pesos_Bias_2')
        l2 = tf.nn.relu(l2,name='l2_relu_2')
        
        l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights'],name='Matmul_l3'), hidden_3_layer['biases'],name='Suma_Pesos_Bias_3')
        l3 = tf.nn.relu(l3,name='l3_relu_3')

        output = tf.matmul(l3,output_layer['weights'],name='Matmul_out') + output_layer['biases']
    
    # Declarando la funcion de costo y entrenamiento
    #Reduce mean, reduce la dimension del tensor en un promedio es decir hace el promedio del costo o error
    
    
  
    return dict(
        x = x,
        embeding=conv4,
        output=output,
        saver = tf.train.Saver()
       
    )

def check_net_BNB(g, test, checkpoint):
   with tf.Session() as sess:
        g['saver'].restore(sess, checkpoint)       
        feed_dict={g['x']: test.tolist()} #La arquitectura recibe una matriz,lol. (tamaño batch, ancho, alto,3) debes suministrarlo asi
        preds = sess.run([g['output']], feed_dict)
        return preds

def deep_neural_convolutional_class_4B(
    batch_size=20,
    image_size=[200,200],
    Drop_prob=1.0,
    learning_rate = 1e-3,
    n_nodes_hl0 = 2000,
    n_nodes_hl1 = 1000,
    n_nodes_hl2 = 500,
    n_nodes_hl3 = 100,
    n_classes=17
    ):
    
    reset_graph()
    #Place holder de entrada 
    x= tf.placeholder(tf.float32,[batch_size,image_size[0],image_size[1],3], name='placeholder_img_entrada')
  
    #Diccionario de pesos convolucionales 
    with tf.name_scope('pesos_bias') as scope1:
        weigths={"w_conv1":tf.Variable(tf.random_normal([5,5,3,32]),name='Pesos_1_32'),
                 "w_conv2":tf.Variable(tf.random_normal([5,5,32,64]),name='Pesos_1_64'),
                 "w_conv3":tf.Variable(tf.random_normal([3,3,64,128]),name='Pesos_1_128'),   
                 "w_conv4":tf.Variable(tf.random_normal([5,5,128,256]),name='Pesos_1_256'),
                 "w_conv5":tf.Variable(tf.random_normal([5,5,256,512]),name='Pesos_1_512'),
                 "w_conv6":tf.Variable(tf.random_normal([5,5,512,1024]),name='Pesos_1_1024'),
                }
        #Diccionario de bias
        biases={"b_conv1":tf.Variable(tf.random_normal([32]),name='Bias_1_32'),
                "b_conv2":tf.Variable(tf.random_normal([64]),name='Bias_1_64'),
                "b_conv3":tf.Variable(tf.random_normal([128]),name='Bias_1_128'),
                "b_conv4":tf.Variable(tf.random_normal([256]),name='Bias_1_256'),
                "b_conv5":tf.Variable(tf.random_normal([512]),name='Bias_1_512'),
                "b_conv6":tf.Variable(tf.random_normal([1024]),name='Bias_1_1024'),
               }

    #Extractor de características
    with tf.name_scope('capas_conv') as scope2:
        conv1=tf.nn.relu(conv2d(x,weigths["w_conv1"],'Capa_Conv_1','SAME')+biases["b_conv1"],name='Func_relu_1')
        conv1=tf.nn.dropout(conv1,Drop_prob)
        conv1=maxpool2d(conv1,ks=[1,2,2,1],st=[1,2,2,1])
        #imagen resultante de 100x100x32
        #print(conv1)

        conv2=tf.nn.relu(conv2d(conv1,weigths["w_conv2"],'Capa_Conv_2','SAME')+biases["b_conv2"],name='Func_relu_2')
        conv2=tf.nn.dropout(conv2,Drop_prob)
        conv2=maxpool2d(conv2,ks=[1,2,2,1],st=[1,2,2,1])
        #imagen resultante de 50x50x64
        #print(conv2)

        conv3=tf.nn.relu(conv2d(conv2,weigths["w_conv3"],'Capa_Conv_3','VALID')+biases["b_conv3"],name='Func_relu_3')
        conv3=tf.nn.dropout(conv3,Drop_prob)
        conv3=maxpool2d(conv3,ks=[1,2,2,1],st=[1,2,2,1])
        #imagen resultante de 24x24x128
        #print(conv3)

        conv4=tf.nn.relu(conv2d(conv3,weigths["w_conv4"],'Capa_Conv_4','SAME')+biases["b_conv4"],name='Func_relu_4')
        conv4=tf.nn.dropout(conv4,Drop_prob)
        conv4=maxpool2d(conv4,ks=[1,2,2,1],st=[1,2,2,1])
        #imagen resultante de 12x12x256
        #print(conv4)

        conv5=tf.nn.relu(conv2d(conv4, weigths["w_conv5"],'Capa_Conv_5','SAME')+biases["b_conv5"],name='Func_relu_5')
        conv5=tf.nn.dropout(conv5,Drop_prob)
        conv5=maxpool2d(conv5,ks=[1,2,2,1],st=[1,2,2,1])
        #vector para clasificar de 6x6x512
        #print(conv5)
        
        #Embeding, son las caracteristicas fonales que se pasarán al MLP o red completamente conectada para clasifiacar
        embdeding=tf.reshape(conv5,[batch_size,6*6*512],name='Embeding')
        #print(embdeding)
    
    #Red perceptron, declaración de capas, son diccionarios de pesos y bias.
    with tf.name_scope('capas_clasificador') as scope3:
        hidden_0_layer = {'weights':tf.Variable(tf.random_normal([6*6*512, n_nodes_hl0]),name='Capa_oculta_pesos_0'),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl0]),name='Capa_oculta_bias_0')}

        hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl0, n_nodes_hl1]),'Capa_oculta_pesos_1'),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl1]),name='Capa_oculta_bias_1')}

        hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]),'Capa_oculta_pesos_2'),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl2]),name='Capa_oculta_bias_2')}
        
        hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]),'Capa_oculta_pesos_3'),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl3]),name='Capa_oculta_bias_3')}

        output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]),'Capa_salida_pesos'),
                        'biases':tf.Variable(tf.random_normal([n_classes]),name='Capa_salida_bias'),}
    
    #W*P + B 
    with tf.name_scope('op_clasificador') as scope4:
        
        l0 = tf.add(tf.matmul(embdeding,hidden_0_layer['weights'],name='Matmul_l0'), hidden_0_layer['biases'],name='Suma_Pesos_Bias_0')
        l0 = tf.nn.relu(l0,name='l0_relu_0')

        l1 = tf.add(tf.matmul(l0,hidden_1_layer['weights'],name='Matmul_l1'), hidden_1_layer['biases'],name='Suma_Pesos_Bias_1')
        l1 = tf.nn.relu(l1,name='l1_relu_1')

        l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights'],name='Matmul_l2'), hidden_2_layer['biases'],name='Suma_Pesos_Bias_2')
        l2 = tf.nn.relu(l2,name='l2_relu_2')
        
        l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights'],name='Matmul_l3'), hidden_3_layer['biases'],name='Suma_Pesos_Bias_3')
        l3 = tf.nn.relu(l3,name='l3_relu_3')

        output = tf.matmul(l3,output_layer['weights'],name='Matmul_out') + output_layer['biases']
    
    
    
    
    
    return dict(
        x = x,
        embeding=conv5,
        output=output,
        saver = tf.train.Saver(),
    )

def check_net_4B(g, test, checkpoint):
   with tf.Session() as sess:
        g['saver'].restore(sess, checkpoint)       
        feed_dict={g['x']: test.tolist()}
        preds = sess.run([g['output']], feed_dict)
        return preds

def get_4B_desc(data):
    g=deep_neural_convolutional_class_4B(batch_size = 1)#Batch de prueba
    pred=check_net_4B(g,data,"./4B3.ckpt")
    #print(pred[0][0])
    precio=pred[0][:,0:5]
    estilo=pred[0][:,5:11]
    pan=pred[0][:,11:15]
    guarnicion=pred[0][:,15:19]

    class_precio=str(list(precio[0]).index(max(precio[0])))
    class_estilo=str(list(estilo[0]).index(max(estilo[0])))
    class_pan=str(list(pan[0]).index(max(pan[0])))
    class_guar=str(list(guarnicion[0]).index(max(guarnicion[0])))

    if class_guar=='0':
        class_guar='3'

    print(class_precio,class_estilo,class_pan,class_guar)
    
    aux='Ya veo!😉 Detecto que tu hamburguesa 🍔 tiene un precio '+cat_price[class_precio]+', pan '+cat_bread[class_pan]+ ' y el acompañamiento de '+cat_side[class_guar]+' me indican que la puedes encontrar en un restaurante 🍽️ tipo '+cat_style[class_estilo]
    return({'descripcion':aux})

def get_4B(data):
    g=deep_neural_convolutional_class_4B(batch_size = 1)#Batch de prueba
    pred=check_net_4B(g,data,"./4B3.ckpt")

	#print(pred[0][0])

    precio=pred[0][:,0:5]
    estilo=pred[0][:,5:11]
    pan=pred[0][:,11:15]
    guarnicion=pred[0][:,15:19]

    class_precio=str(list(precio[0]).index(max(precio[0])))
    class_estilo=str(list(estilo[0]).index(max(estilo[0])))
    class_pan=str(list(pan[0]).index(max(pan[0])))
    class_guar=str(list(guarnicion[0]).index(max(guarnicion[0])))

	#for Testing motives, im gonna hardcode tsome values, my current bd doesnt contains restaurants thath fits all class combinations, the future Me is goping to complete it, maybe tomorrow
    class_pan=str(2)
    class_estilo=str(4)
 
    #print(class_precio,class_estilo,class_pan,class_guar)
    
    con = None
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Faus DEL FURTURO: AGREGAR GEOLOCALIZACION AGREGAR GEOLOCALIZACION   AGREGAR GEOLOCALIZACION   AGREGAR GEOLOCALIZACION   AGREGAR GEOLOCALIZACION AGREGAR GEOLOCALIZACION   AGREGAR GEOLOCALIZACION   
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    query ="select * from restaurants res, restside rs, restbread rb, restprice rp where res.style="+class_estilo+" and res.id = rs.idres and res.id = rb.idres and"
    query=query +" res.id = rp.idres and rb.idbread="+class_pan+" and rp.idprice="+class_precio+" and rs.idside="+class_guar+" ORDER BY random() LIMIT 4"
    #PRINT(query)

    results=[]
    aux={}
    con = psycopg2.connect("host='stampy.db.elephantsql.com' dbname='cxvdrtcw' user='cxvdrtcw' password='a6TWePHVobuuP2KcwTPbqFAWGZzQrSdE'")
    cur = con.cursor()
    cur.execute(query)
     
    while True:
        row = cur.fetchone()
        if row == None:
            break


        aux.update({'id': str(row[0]),'nombre': str(row[1]),'ubicacion': str(row[2]),'direccion': str(row[3]),'telefono': str(row[4]),'horario': str(row[5]),'rating': str(row[6])})
        results.append(aux)
        aux={}
    if(con):
        con.close()
    return({'restaurantes':results})
#----------------------

def get_4B_geo(data,l1,l2):
    g=deep_neural_convolutional_class_4B(batch_size = 1)#Batch de prueba
    pred=check_net_4B(g,data,"./4B3.ckpt")

    #print(pred[0][0])

    precio=pred[0][:,0:5]
    estilo=pred[0][:,5:11]
    pan=pred[0][:,11:15]
    guarnicion=pred[0][:,15:19]

    class_precio=str(list(precio[0]).index(max(precio[0])))
    class_estilo=str(list(estilo[0]).index(max(estilo[0])))
    class_pan=str(list(pan[0]).index(max(pan[0])))
    class_guar=str(list(guarnicion[0]).index(max(guarnicion[0])))
    if class_guar=='0':
        class_guar='3'
    #for Testing motives, im gonna hardcode tsome values, my current bd doesnt contains restaurants thath fits all class combinations, the future Me is goping to complete it, maybe tomorrow
    #class_pan=str(2)
    #class_estilo=str(4)
 
    #print(class_precio,class_estilo,class_pan,class_guar)
    
    con = None
    q_count="select count(*)"
    q_count= q_count+"from restaurants res, restside rs, restbread rb, restprice rp where res.style="+class_estilo+" and res.id = rs.idres and res.id = rb.idres and" 
    q_count= q_count+" res.id = rp.idres and rb.idbread="+class_pan+" and rp.idprice="+class_precio+" and rs.idside="+class_guar

    #print(q_count)
    query ="select *, 111.045 * DEGREES(ACOS(COS(RADIANS("+l1+"))* COS(RADIANS(lat))* COS(RADIANS(long) - RADIANS("+l2+"))+ SIN(RADIANS("+l1+"))* SIN(RADIANS(lat)))) AS distance_in_km "
    query=query+"from restaurants res, restside rs, restbread rb, restprice rp where res.style="+class_estilo+" and res.id = rs.idres and res.id = rb.idres and"
    query=query +" res.id = rp.idres and rb.idbread="+class_pan+" and rp.idprice="+class_precio+" and rs.idside="+class_guar+" ORDER BY  distance_in_km ASC limit 4"
    
    results=[]
    aux={}
    con = psycopg2.connect("host='stampy.db.elephantsql.com' dbname='cxvdrtcw' user='cxvdrtcw' password='a6TWePHVobuuP2KcwTPbqFAWGZzQrSdE'")
    cur1 = con.cursor()
    cur1.execute(q_count)

    while True:
        row1=cur1.fetchone()
        if row1==None:
            break
        if str(row1[0])=="0":
            query ="select *, 111.045 * DEGREES(ACOS(COS(RADIANS("+l1+"))* COS(RADIANS(lat))* COS(RADIANS(long) - RADIANS("+l2+"))+ SIN(RADIANS("+l1+"))* SIN(RADIANS(lat)))) AS distance_in_km "   
            query=query+"from restaurants ORDER BY  distance_in_km ASC limit 4"
    #print(query)
    cur = con.cursor()
    cur.execute(query)
     
    while True:
        row = cur.fetchone()
        if row == None:
            break

        url_google="https://www.google.com/maps/search/?api=1&query=Google&query_place_id="+str(row[0])
        aux.update({'id': str(row[0]),'nombre': str(row[1]),'ubicacion': str(row[2]),'direccion': str(row[3]),'telefono': str(row[4]),'horario': str(row[5]),'rating': str(row[6]),'url': str(row[10]),'google':url_google})
        results.append(aux)
        aux={}
    if(con):
        con.close()
    return({'restaurantes':results})
#----------------------

@app.route('/')
def index():
    return jsonify({'Deep': 'Deep Dishes'})

@app.route('/deepdeep/<some_kind_of_url>')
def rna(some_kind_of_url):

    url ="https://gus-raw-assets-dev.s3.amazonaws.com/"+some_kind_of_url
    print(url)
    #url= "https://raw.githubusercontent.com/maverick-faus/Files/master/"+some_kind_of_url

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')
    im = img.resize((200,200),Image.ANTIALIAS)
    img_data=np.array(im)

    data0=[]
    data0.append(img_data)
    data1=np.array(data0)
    g=deep_neural_convolutional_class_BNB(batch_size = 1)#Batch de prueba
    pred=check_net_BNB(g,data1,"./BNB1.ckpt")

    vector=pred[0][0].tolist()
    class_BNB=vector.index(max(vector))
    if class_BNB==0:
        restaurantes=get_4B(data1)
        y=jsonify(restaurantes)
        return(y)
    else:
        return(jsonify({'error':'Parece que eso no es una hambueguesa 😬 Trata con otra fotografía por favor.'}))

@app.route('/deepdeep/desc/<some_kind_of_url>')
def rna1(some_kind_of_url):

    url ="https://gus-raw-assets-dev.s3.amazonaws.com/"+some_kind_of_url

    #url= "https://raw.githubusercontent.com/maverick-faus/Files/master/"+some_kind_of_url
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')
    im = img.resize((200,200),Image.ANTIALIAS)
    img_data=np.array(im)

    data0=[]
    data0.append(img_data)
    data1=np.array(data0)
    g=deep_neural_convolutional_class_BNB(batch_size = 1)#Batch de prueba
    pred=check_net_BNB(g,data1,"./BNB1.ckpt")

    vector=pred[0][0].tolist()
    class_BNB=vector.index(max(vector))

    if class_BNB==0:
        restaurantes=get_4B_desc(data1)
        y=jsonify(restaurantes)
        print("Ahi va la descripcion.")
        return(y)
    else:
        return(jsonify({'error':'Parece que eso no es una hambueguesa 😬 Trata con otra fotografía por favor.'}))


@app.route('/deepdeep/geo/<some_kind_of_url>/<lat1>/<long1>')
def rna2(some_kind_of_url,lat1,long1):
    
    url ="https://gus-raw-assets-dev.s3.amazonaws.com/"+some_kind_of_url

    #url= "https://raw.githubusercontent.com/maverick-faus/Files/master/"+some_kind_of_url
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')
    im = img.resize((200,200),Image.ANTIALIAS)
    img_data=np.array(im)

    data0=[]
    data0.append(img_data)
    data1=np.array(data0)
    g=deep_neural_convolutional_class_BNB(batch_size = 1)#Batch de prueba
    pred=check_net_BNB(g,data1,"./BNB1.ckpt")

    vector=pred[0][0].tolist()
    class_BNB=vector.index(max(vector))

    if class_BNB==0:
        restaurantes=get_4B_geo(data1,lat1,long1)
        y=jsonify(restaurantes)
        print(restaurantes)
        return(y)
    else:
        return(jsonify({'error':'Parece que eso no es una hambueguesa 😬 Trata con otra fotografía por favor.'}))


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
