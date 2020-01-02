import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import argparse
import tensorflow as tf
import schedule
import pymysql
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus, unquote
import json
from airDaemon import daemon

class Air(daemon):
    def __init__(self):
        daemon.__init__(self, "/tmp/airDaemon.pid")
        self.batch_size = 1
        self.timesteps = 5
        self.data_dim = 144
        self.cell_size = 400

    def convertFormat(self, csvData, look_back=5):
        dictionary = {}
        for i in range(len(csvData)-look_back):
            rowX = []
            for timestep in range(5):
                for col in range(csvData.shape[1]-1):
                    rowX.append(csvData[i+timestep, col+1])
            dictionary[csvData[i+look_back-1,0]] = np.array(rowX).astype(np.float32).reshape([self.batch_size, self.timesteps, self.data_dim])
        return dictionary

    def getKey(self):
        now = datetime.now()
        timestamp = now.strftime("%m-%d %H:00:00")
        year = now.year % 2019 % 4 + 2012

        return str(year)+"-"+timestamp

    def insertPoint(self, table, point):
        now = datetime.now()
        if table == "prediction":
            afterEight = now + timedelta(hours=8)
            timestamp = afterEight.strftime("%Y-%m-%d %H:00:00")
        else:
            timestamp = now.strftime("%Y-%m-%d %H:00:00")
        conn = pymysql.connect(host="13.209.65.208", port=3306, user="root", password="", db="air", charset="utf8")
        curs = conn.cursor()
        sql = "insert into "+table+"(timestamp, pm10) values (%s, %s)"
        curs.execute(sql, (timestamp,str(point)))
        conn.commit()
        conn.close()

    def predictPM10(self, args, dictionary):
        print("predictPM10")
        graph = tf.Graph()
        with graph.as_default():
            X = tf.placeholder(tf.float32, [None, self.timesteps, self.data_dim], name="X")
            X_seqs = tf.unstack(X, axis=1)

            encoder = tf.nn.rnn_cell.LSTMCell(self.cell_size, use_peepholes=True, activation=tf.nn.relu, name="encoder")
            decoder = tf.nn.rnn_cell.LSTMCell(self.cell_size, use_peepholes=True, activation=tf.nn.relu, name="decoder")

            dec_w = tf.Variable(tf.truncated_normal([self.cell_size, 1], dtype=tf.float32), name='dec_w')
            dec_b = tf.Variable(tf.constant(0.1, shape=[1], dtype=tf.float32), name='dec_b')

            _, dec_states = tf.contrib.rnn.static_rnn(encoder, X_seqs, dtype=tf.float32)

            dec_input = tf.zeros([self.batch_size, 1], dtype=tf.float32)
            dec_outputs = []
            for step in range(8):
                dec_input, dec_states = decoder(dec_input, dec_states)
                dec_input = tf.matmul(dec_input, dec_w) + dec_b
                dec_outputs.append(dec_input)

            dec_outputs = dec_outputs[::-1]
            dec_outputs = tf.transpose(dec_outputs, [1,0,2]) #(1, steps, 1)

            dec_outputs = tf.squeeze(tf.squeeze(dec_outputs, [2]), [0])

            saver = tf.train.Saver()
            checkpoint_dir = "/home/hoyy/airGraph/model/"+args.model
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        with tf.Session(graph=graph) as sess:
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                pred = sess.run(dec_outputs, feed_dict={X: dictionary[self.getKey()]})
                sess.close()
                self.insertPoint("prediction", pred[7])

            else:
                print("not model")

    def getPM10(self):
        print("getPM10")
        decode_key = unquote("%2BWjxc8LGHK4rSYpwaYOGgSvw8kPrJTJ6FCQof2x8p4wbNnwNnOZ7XhCd6Zy%2B4WwShJXzTw8mMmM8cfd6DSKS0A%3D%3D")
        url = "http://openapi.airkorea.or.kr/openapi/services/rest/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
        queryParams = "?" + urlencode({quote_plus("ServiceKey"):decode_key, quote_plus("numOfRows"):"1", quote_plus("pageNo"):"1"
                                          , quote_plus("stationName"):"도봉구", quote_plus("dataTerm"):"DAILY", quote_plus("ver"):"1.3", quote_plus("_returnType"):"json"})
        request = Request(url + queryParams)
        request.get_method = lambda: "GET"
        responseBody = urlopen(request).read()
        jsonString = responseBody.decode("utf-8")
        jsonDict = json.loads(jsonString)
        nowPM10 = jsonDict["list"][0]["pm10Value"]
        self.insertPoint("stream", nowPM10)

    def run(self):
        csvData = pd.read_csv('/home/hoyy/airGraph/data/dataset.csv', encoding='ISO-8859-1')
        csvData = csvData.values

        dictionary = self.convertFormat(csvData)
        schedule.every().hour.at(":12").do(self.predictPM10, args, dictionary)
        schedule.every().hour.at(":12").do(self.getPM10)

        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="HDFS path to save/load model", default="AE_LSTM_Model")
    args = parser.parse_args()

    air = Air()
    air.start()
